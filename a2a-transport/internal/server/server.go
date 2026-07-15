// Package server provides the A2A gRPC server implementation.
package server

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"sync/atomic"

	pb "github.com/batmunkhcom/a2a-transport/proto/a2a/v1/gen"

	"github.com/batmunkhcom/a2a-transport/internal/bridge"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// BackpressureConfig controls flow control thresholds.
type BackpressureConfig struct {
	MaxQueueDepth int32
	SlowDownRatio float64 // 0.0–1.0
	StopRatio     float64
}

// Server implements the A2AService gRPC server.
type Server struct {
	pb.UnimplementedA2AServiceServer

	addr       string
	grpcServer *grpc.Server
	bridge     *bridge.Conn
	bpConfig   BackpressureConfig

	queueDepth    int64 // atomic
	pluginsLoaded int32
	uptimeSeconds int64
}

// New creates a new A2A gRPC server.
func New(addr string, bridgeConn *bridge.Conn, bpConf BackpressureConfig) *Server {
	return &Server{
		addr:     addr,
		bridge:   bridgeConn,
		bpConfig: bpConf,
	}
}

// Start runs the gRPC server.
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.addr, err)
	}

	s.grpcServer = grpc.NewServer(
		grpc.MaxConcurrentStreams(1000),
		grpc.MaxRecvMsgSize(100*1024*1024),
		grpc.MaxSendMsgSize(100*1024*1024),
	)
	pb.RegisterA2AServiceServer(s.grpcServer, s)

	log.Printf("gRPC server listening on %s", s.addr)
	return s.grpcServer.Serve(lis)
}

// Stop gracefully shuts down the server.
func (s *Server) Stop() {
	if s.grpcServer != nil {
		s.grpcServer.GracefulStop()
	}
}

// SendTensor handles a single tensor request.
func (s *Server) SendTensor(ctx context.Context, req *pb.TensorRequest) (*pb.TensorResponse, error) {
	depth := atomic.AddInt64(&s.queueDepth, 1)
	defer atomic.AddInt64(&s.queueDepth, -1)

	if signal := s.checkBackpressure(int32(depth)); signal != pb.BackpressureStatus_RESUME {
		return &pb.TensorResponse{
			Accepted:     false,
			ErrorMessage: fmt.Sprintf("backpressure: %s", signal.String()),
			ErrorCode:    8, // RESOURCE_EXHAUSTED
		}, nil
	}

	meta := req.GetMetadata()
	if meta == nil {
		return nil, status.Error(codes.InvalidArgument, "missing metadata")
	}

	// Forward to Python ML Core via bridge
	cmd := &bridge.Command{
		Op:      "PROCESS",
		AgentID: meta.GetSourceModel(),
		Label:   meta.GetSemanticLabel(),
		Dtype:   meta.GetTensorDtype(),
		Dim:     int32(meta.GetSourceLayer()),
	}

	resp, outData, err := s.bridge.Send(cmd, req.GetTensorData())
	if err != nil {
		return nil, status.Errorf(codes.Internal, "bridge error: %v", err)
	}

	return &pb.TensorResponse{
		Accepted:   resp.Status == "ok",
		Metadata:   meta,
		TensorData: outData,
	}, nil
}

// StreamTensors handles bidirectional streaming.
func (s *Server) StreamTensors(stream pb.A2AService_StreamTensorsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		meta := req.GetMetadata()
		cmd := &bridge.Command{
			Op:      "PROCESS",
			AgentID: meta.GetSourceModel(),
			Label:   meta.GetSemanticLabel(),
			Dtype:   meta.GetTensorDtype(),
			Dim:     int32(meta.GetSourceLayer()),
		}

		resp, outData, err := s.bridge.Send(cmd, req.GetTensorData())
		if err != nil {
			return status.Errorf(codes.Internal, "bridge error: %v", err)
		}

		if err := stream.Send(&pb.TensorResponse{
			Accepted:   resp.Status == "ok",
			Metadata:   meta,
			TensorData: outData,
		}); err != nil {
			return err
		}
	}
}

// HealthCheck returns server health status.
func (s *Server) HealthCheck(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	statusStr := "ok"
	_, err := s.bridge.HealthCheck()
	if err != nil {
		statusStr = "degraded"
	}

	return &pb.HealthResponse{
		Status:        statusStr,
		PluginsLoaded: uint32(s.pluginsLoaded),
		UptimeSeconds: uint64(s.uptimeSeconds),
		Version:       "0.1.0",
	}, nil
}

// Discover returns available agents.
func (s *Server) Discover(ctx context.Context, req *pb.DiscoverRequest) (*pb.DiscoverResponse, error) {
	return &pb.DiscoverResponse{
		Agents: []*pb.AgentInfo{
			{
				AgentId: "go-transport",
				Address: s.addr,
		Capabilities: []*pb.CapabilityInfo{
			{Name: "hidden_state", Dtype: "fp32", HiddenDim: 768},
			{Name: "error_context", Dtype: "fp32", HiddenDim: 768},
		},
			},
		},
	}, nil
}

// SetPluginsLoaded sets the reported plugin count.
func (s *Server) SetPluginsLoaded(n int32) { s.pluginsLoaded = n }

// SetUptime sets the reported uptime.
func (s *Server) SetUptime(n int64) { s.uptimeSeconds = n }

func (s *Server) checkBackpressure(depth int32) pb.BackpressureStatus {
	threshold := s.bpConfig.MaxQueueDepth
	if threshold <= 0 {
		return pb.BackpressureStatus_RESUME
	}
	ratio := float64(depth) / float64(threshold)
	if ratio >= s.bpConfig.StopRatio && s.bpConfig.StopRatio > 0 {
		return pb.BackpressureStatus_PAUSE
	}
	if ratio >= s.bpConfig.SlowDownRatio && s.bpConfig.SlowDownRatio > 0 {
		return pb.BackpressureStatus_SLOW_DOWN
	}
	return pb.BackpressureStatus_RESUME
}
