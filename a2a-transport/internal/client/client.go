// Package client provides the A2A gRPC client for Go.
package client

import (
	"context"
	"fmt"
	"time"

	pb "github.com/batmunkhcom/a2a-transport/proto/a2a/v1/gen"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is a high-concurrency A2A gRPC client.
type Client struct {
	conn   *grpc.ClientConn
	client pb.A2AServiceClient
	target string
}

// New creates a new A2A client.
func New(target string) (*Client, error) {
	conn, err := grpc.NewClient(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(100*1024*1024),
			grpc.MaxCallSendMsgSize(100*1024*1024),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", target, err)
	}

	return &Client{
		conn:   conn,
		client: pb.NewA2AServiceClient(conn),
		target: target,
	}, nil
}

// Close the client connection.
func (c *Client) Close() error {
	return c.conn.Close()
}

// SendTensor sends a single tensor and returns the result.
func (c *Client) SendTensor(
	ctx context.Context,
	sourceModel, targetModel, label string,
	data []byte,
	tensorDtype string,
	shape []uint32,
) ([]byte, *pb.TensorResponse, error) {
	req := &pb.TensorRequest{
		Metadata: &pb.TensorMetadata{
			SourceModel:   sourceModel,
			TargetModel:   targetModel,
			SemanticLabel: label,
			TensorDtype:   tensorDtype,
			TensorShape:   shape,
			Timestamp:     uint64(time.Now().UnixMicro()),
		},
		TensorData: data,
	}

	resp, err := c.client.SendTensor(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("send tensor: %w", err)
	}

	if !resp.GetAccepted() {
		return nil, resp, fmt.Errorf("server rejected: %s", resp.GetErrorMessage())
	}

	return resp.GetTensorData(), resp, nil
}

// HealthCheck checks if the server is healthy.
func (c *Client) HealthCheck(ctx context.Context) (*pb.HealthResponse, error) {
	return c.client.HealthCheck(ctx, &pb.HealthRequest{})
}

// Discover lists available agents.
func (c *Client) Discover(ctx context.Context) ([]*pb.AgentInfo, error) {
	resp, err := c.client.Discover(ctx, &pb.DiscoverRequest{})
	if err != nil {
		return nil, err
	}
	return resp.GetAgents(), nil
}

// StreamTensors opens a bidirectional tensor stream.
func (c *Client) StreamTensors(ctx context.Context) (pb.A2AService_StreamTensorsClient, error) {
	return c.client.StreamTensors(ctx)
}
