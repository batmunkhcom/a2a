// Package bridge provides Unix domain socket communication
// between the Go transport layer and Python ML Core.
package bridge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

// Command sent from Go to Python ML Core.
type Command struct {
	Op       string  `json:"op"`       // EXTRACT, INJECT, PROJECT, HEALTH
	AgentID  string  `json:"agent_id"`
	Label    string  `json:"label"`
	Dtype    string  `json:"dtype"`
	Dim      int32   `json:"dim"`
	Token    string  `json:"token,omitempty"`
}

// Response from Python ML Core.
type Response struct {
	Status     string  `json:"status"`     // ok, error
	Error      string  `json:"error,omitempty"`
	Dim        int32   `json:"dim"`
	Dtype      string  `json:"dtype"`
	LatencyUs  int64   `json:"latency_us"`
}

// Conn manages a persistent Unix domain socket connection to Python ML Core.
type Conn struct {
	mu     sync.Mutex
	conn   net.Conn
	addr   string
}

// NewConn creates a new Unix socket connection.
func NewConn(socketPath string) (*Conn, error) {
	conn, err := net.DialTimeout("unix", socketPath, 5*time.Second)
	if err != nil {
		return nil, fmt.Errorf("bridge dial: %w", err)
	}
	return &Conn{conn: conn, addr: socketPath}, nil
}

// Close the connection.
func (c *Conn) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Send sends a command with tensor data and returns the response.
func (c *Conn) Send(cmd *Command, data []byte) (*Response, []byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Encode command as JSON
	cmdBytes, err := json.Marshal(cmd)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal command: %w", err)
	}

	// Wire format: [4B cmd len][cmd JSON][4B data len][data bytes]
	header := make([]byte, 4+len(cmdBytes)+4)
	binary.BigEndian.PutUint32(header[0:4], uint32(len(cmdBytes)))
	copy(header[4:], cmdBytes)
	binary.BigEndian.PutUint32(header[4+len(cmdBytes):], uint32(len(data)))

	// Send header + data
	if _, err := c.conn.Write(header); err != nil {
		return nil, nil, fmt.Errorf("write header: %w", err)
	}
	if len(data) > 0 {
		if _, err := c.conn.Write(data); err != nil {
			return nil, nil, fmt.Errorf("write data: %w", err)
		}
	}

	// Read response: [4B resp len][resp JSON][4B data len][data bytes]
	lenBuf := make([]byte, 4)
	if _, err := io.ReadFull(c.conn, lenBuf); err != nil {
		return nil, nil, fmt.Errorf("read resp len: %w", err)
	}
	respLen := binary.BigEndian.Uint32(lenBuf)

	respBytes := make([]byte, respLen)
	if _, err := io.ReadFull(c.conn, respBytes); err != nil {
		return nil, nil, fmt.Errorf("read resp body: %w", err)
	}

	var resp Response
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		return nil, nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if resp.Status != "ok" {
		return &resp, nil, fmt.Errorf("ml core error: %s", resp.Error)
	}

	// Read response data length
	if _, err := io.ReadFull(c.conn, lenBuf); err != nil {
		return &resp, nil, nil
	}
	dataLen := binary.BigEndian.Uint32(lenBuf)
	if dataLen == 0 {
		return &resp, nil, nil
	}

	outData := make([]byte, dataLen)
	if _, err := io.ReadFull(c.conn, outData); err != nil {
		return &resp, nil, fmt.Errorf("read resp data: %w", err)
	}

	return &resp, outData, nil
}

// HealthCheck pings the Python ML Core.
func (c *Conn) HealthCheck() (*Response, error) {
	resp, _, err := c.Send(&Command{Op: "HEALTH"}, nil)
	return resp, err
}
