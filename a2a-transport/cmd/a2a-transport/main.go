// Package main is the entry point for the A2A Go transport layer.
//
// Usage:
//
//	a2a-transport --addr 0.0.0.0:50051 --bridge /tmp/a2a-ml.sock
package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/batmunkhcom/a2a-transport/internal/bridge"
	"github.com/batmunkhcom/a2a-transport/internal/server"
)

func main() {
	addr := flag.String("addr", "0.0.0.0:50051", "gRPC listen address")
	bridgePath := flag.String("bridge", "/tmp/a2a-ml.sock", "Unix socket path to Python ML Core")
	maxQueueDepth := flag.Int("max-queue", 1000, "Max queue depth before backpressure")
	slowRatio := flag.Float64("slow-ratio", 0.5, "Queue ratio for SLOW_DOWN signal")
	stopRatio := flag.Float64("stop-ratio", 0.8, "Queue ratio for STOP signal")
	flag.Parse()

	// Connect to Python ML Core
	bc, err := bridge.NewConn(*bridgePath)
	if err != nil {
		log.Fatalf("Failed to connect to Python ML Core at %s: %v", *bridgePath, err)
	}
	defer bc.Close()
	log.Printf("Connected to Python ML Core: %s", *bridgePath)

	// Health check
	if _, err := bc.HealthCheck(); err != nil {
		log.Printf("WARNING: Python ML Core health check failed: %v", err)
	} else {
		log.Println("Python ML Core health: ok")
	}

	// Start gRPC server
	srv := server.New(*addr, bc, server.BackpressureConfig{
		MaxQueueDepth: int32(*maxQueueDepth),
		SlowDownRatio: *slowRatio,
		StopRatio:     *stopRatio,
	})

	go func() {
		if err := srv.Start(); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	log.Printf("A2A Go Transport started on %s (bridge: %s)", *addr, *bridgePath)

	// Wait for shutdown signal
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	log.Println("Shutting down...")
	srv.Stop()
	bc.Close()
	log.Println("A2A Go Transport stopped")
}
