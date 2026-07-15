# Wire Protocol Specification

**Version:** 0.1.0

## Overview

A2A uses **gRPC** for transport with **Protocol Buffers** for message schema and
**FlatBuffers** for tensor binary encoding. The protocol is defined in
`a2a/protocol/a2a.proto`.

---

## Service Definition

```protobuf
service A2AService {
  // Send a tensor to an agent and receive a response
  rpc SendTensor (TensorRequest) returns (TensorResponse);

  // Stream tensors in both directions
  rpc StreamTensors (stream TensorRequest) returns (stream TensorResponse);

  // Health check
  rpc Health (HealthRequest) returns (HealthResponse);

  // Discover available agents and their capabilities
  rpc Discover (DiscoverRequest) returns (DiscoverResponse);
}
```

---

## Message Types

### TensorRequest

```protobuf
message TensorRequest {
  string agent_id = 1;          // Target agent identifier
  string label = 2;             // Semantic label (e.g., "error_context")
  bytes data = 3;               // FlatBuffers-encoded tensor
  int32 dim = 4;                // Tensor dimension (for validation)
  string dtype = 5;             // Data type: "fp32", "fp16", "bf16"
  string mesh_id = 6;           // Mesh identifier for routing
  string token = 7;             // JWT authentication token
  int64 seq_id = 8;             // Monotonic sequence number (ordering)
  int64 timestamp = 9;          // Unix timestamp (ms)
}
```

### TensorResponse

```protobuf
message TensorResponse {
  string agent_id = 1;
  string label = 2;
  bytes data = 3;
  int32 dim = 4;
  string dtype = 5;
  int64 seq_id = 6;
  int64 latency_us = 7;         // Processing latency (microseconds)
  Status status = 8;            // Response status
}
```

### Status

```protobuf
message Status {
  int32 code = 1;               // 0 = OK, see error codes
  string message = 2;
}
```

### BackpressureSignal

```protobuf
message BackpressureSignal {
  enum Signal {
    RESUME = 0;
    SLOW_DOWN = 1;
    STOP = 2;
  }
  Signal signal = 1;
  int32 queue_depth = 2;
  int32 threshold = 3;
  string agent_id = 4;
}
```

### HealthRequest / HealthResponse

```protobuf
message HealthRequest {
  string check_type = 1;        // "live" or "ready"
}

message HealthResponse {
  string status = 1;            // "ok" or "degraded"
  int32 plugins_loaded = 2;
  int64 uptime_seconds = 3;
}
```

### DiscoverRequest / DiscoverResponse

```protobuf
message DiscoverRequest {}

message DiscoverResponse {
  repeated AgentInfo agents = 1;
}

message AgentInfo {
  string agent_id = 1;
  string model = 2;
  string model_family = 3;
  int32 hidden_dim = 4;
  string dtype = 5;
  repeated string labels = 6;
}
```

---

## FlatBuffers Tensor Encoding

Tensors are serialized as FlatBuffers with this layout:

```
┌─────────────────────────────────────┐
│ 4 bytes    │ CRC32 checksum         │
│ 4 bytes    │ Data length (N)        │
│ 2 bytes    │ Dimension              │
│ 2 bytes    │ Dtype (0=fp32, 1=fp16,  │
│            │        2=bf16)         │
│ N bytes    │ Raw tensor data        │
└─────────────────────────────────────┘
```

**Serialization:** `a2a/transport/codec.py`

```python
from a2a.transport.codec import FlatBuffersCodec

codec = FlatBuffersCodec()
encoded = codec.encode(tensor, dtype="fp32")       # returns bytes
decoded = codec.decode(encoded)                    # returns np.ndarray
```

**Validation:**
- CRC32 checksum verified on decode
- Dimension and dtype validated against declared values
- Raises `A2ACodecError` on mismatch

---

## Error Codes

| Code | gRPC Status | Description |
|---|---|---|
| 0 | OK | Success |
| 1 | CANCELLED | Operation cancelled |
| 2 | UNKNOWN | Unknown error |
| 3 | INVALID_ARGUMENT | Bad tensor shape/dtype |
| 4 | DEADLINE_EXCEEDED | Request timeout |
| 5 | NOT_FOUND | Agent not found |
| 6 | ALREADY_EXISTS | Duplicate agent registration |
| 7 | PERMISSION_DENIED | Auth failed (JWT) |
| 8 | RESOURCE_EXHAUSTED | Rate limit exceeded |
| 9 | FAILED_PRECONDITION | Plugin not initialized |
| 10 | ABORTED | Concurrent modification |
| 11 | OUT_OF_RANGE | Tensor dimension mismatch |
| 12 | UNIMPLEMENTED | Plugin doesn't handle label |
| 13 | INTERNAL | Internal server error |
| 14 | UNAVAILABLE | Service unavailable |
| 15 | DATA_LOSS | Checksum mismatch (CRC32) |
| 16 | UNAUTHENTICATED | Missing or invalid JWT |

Source: `a2a/protocol/errors.py`

---

## Authentication Flow

```
Client                                          Server
  │                                               │
  │  1. JWT Create (HS256)                        │
  │  claims: {sub, exp, mesh_id, agent_id}        │
  │                                               │
  │  2. TensorRequest + token ──────────────────► │
  │                                               │ 3. Validate JWT
  │                                               │    - Check signature
  │                                               │    - Check expiry
  │                                               │    - Check mesh_id
  │                                               │
  │  4. TensorResponse ◄───────────────────────   │
  │                                               │
```

---

## Rate Limiting

Token bucket algorithm per `(agent_id, label)` pair:

- **Bucket capacity:** configurable (default: 100 tokens)
- **Refill rate:** configurable per agent (default: 10/sec)
- **Exceeded:** `RESOURCE_EXHAUSTED` (code 8) returned

---

## Backpressure

Server monitors queue depth and signals:

| Queue Depth | Signal | Client Action |
|---|---|---|
| < 50% threshold | RESUME | Normal rate |
| 50-80% threshold | SLOW_DOWN | Halve send rate |
| > 80% threshold | STOP | Pause, retry with backoff |

---

## Wire Latency Budget

| Component | Expected (μs) | Max (μs) |
|---|---|---|
| Tensor extract | 50 | 200 |
| Projection (MLP) | 100 | 500 |
| FlatBuffers encode | 10 | 50 |
| gRPC round-trip (local) | 200 | 1000 |
| FlatBuffers decode | 10 | 50 |
| Projection (reverse) | 100 | 500 |
| Tensor inject | 50 | 200 |
| **Total** | **~520** | **~2500** |

Vs. text-based API: 50–200 ms → **10–100× improvement**
