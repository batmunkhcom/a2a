# Architecture

## System Overview

```
                         ┌─────────────────────────────────┐
                         │         A2A Runtime              │
                         │  ┌───────────────────────────┐  │
  a2a.yaml ─────────────►│  │ Config Loader             │  │
                         │  └───────────┬───────────────┘  │
                         │              │                  │
                         │  ┌───────────▼───────────────┐  │
                         │  │ Plugin Manager            │  │
                         │  │  ┌──────┐ ┌──────┐       │  │
                         │  │  │LogRdr│ │CodeFx│       │  │
                         │  │  └──────┘ └──────┘       │  │
                         │  └───────────────────────────┘  │
                         │              │                  │
                         │  ┌───────────▼───────────────┐  │
                         │  │ Semantic Router           │  │
                         │  └───────────┬───────────────┘  │
                         │              │                  │
  Client ◄───────────────│  ┌───────────▼───────────────┐  │
                         │  │ gRPC Transport            │  │
                         │  │  ┌─────────────────────┐ │  │
                         │  │  │ FlatBuffers Codec   │ │  │
                         │  │  └─────────────────────┘ │  │
                         │  └───────────────────────────┘  │
                         │              │                  │
                         │  ┌───────────▼───────────────┐  │
                         │  │ Security Layer            │  │
                         │  │  JWT │ mTLS │ Rate Limit │  │
                         │  └───────────────────────────┘  │
                         └─────────────────────────────────┘
```

---

## Component Details

### 1. Config Layer (`a2a.config`)

**Files:** `schema.py`, `loader.py`

- **A2AConfig** — Pydantic v2 model with full validation
- Config search order: `A2A_CONFIG` env → `./a2a.yaml` → `~/.config/a2a/` → `/etc/a2a/`
- Sub-models: `MeshConfig`, `ModelConfig`, `PluginEntry`, `RouteConfig`, `SecurityConfig`, `TransportConfig`

### 2. Plugin System (`a2a.agent`)

**Files:** `base.py`, `manager.py`, `router.py`

- **BasePlugin** — Abstract base class with initialize/process_tensor/shutdown lifecycle
- **PluginRegistry** — Global plugin registry with capability discovery
- **PluginManager** — Dynamic `importlib` loading, plugin lifecycle management
- **SemanticRouter** — Routes tensors by `(agent_id, label)` pair with fallback strategies

### 3. Wire Protocol (`a2a.protocol`, `a2a.transport`)

**Files:** `a2a.proto`, `server.py`, `client.py`, `codec.py`, `errors.py`

- **Protobuf** — 11 message types (TensorRequest, TensorResponse, BackpressureSignal, etc.)
- **gRPC** — Synchronous server + client with health and discovery RPCs
- **FlatBuffers Codec** — Schema-free tensor serialization with CRC32 integrity check
- **Error Codes** — Standard error codes mapped to gRPC status codes

### 4. Tensor Engine (`a2a.tensor`)

**Files:** `extractor.py`, `injector.py`, `dtype.py`, `serializer.py`

- **Extractor** — Forward hook–based hidden state extraction, 4 pooling strategies (last, mean, max, cls)
- **Injector** — Two methods: prefix injection and cross-attention injection
- **Dtype** — Pure-Python FP32↔FP16↔BF16 conversion with NaN/Inf/L2 validation
- **Serializer** — Safetensors format with binary fallback

### 5. Projection (`a2a.projection`)

**Files:** `adapter.py`, `trainer.py`, `dataset.py`, `auto_trainer.py`, `registry.py`

- **ProjectionModel** — 3 variants (A: simple MLP, B: residual MLP, C: Res+LayerNorm)
- **ProjectionTrainer** — InfoNCE, MSE, Cosine similarity losses
- **ProjectionPairDataset** — Pre-computed positive/negative latent pairs
- **AutoTrainer** — Runtime projection training loop
- **ProjectionRegistry** — Persistent model cache with checksum validation

### 6. Security (`a2a.security`, `a2a.monitoring`)

**Files:** `auth.py`, `tls.py`, `rate_limiter.py`, `metrics.py`

- **JWT** — HS256 token creation and validation with mesh_id binding
- **mTLS** — Mutual TLS configuration builder
- **TokenBucket** — Token bucket rate limiter (per-agent, per-route)
- **BackpressureController** — Queue depth monitoring with SLOW_DOWN/RESUME signals
- **Prometheus** — Counters, histograms, gauges with text renderer

### 7. Utilities (`a2a.utils`)

**Files:** `logging.py`

- **JSONFormatter** — Structured JSON logging (timestamp, level, logger, message, exception)

---

## Data Flow

```
Source Model (Llama)
      │
      ▼ forward hook
  ┌─────────────┐
  │  Extractor   │ → hidden_states: (1, seq, 4096)
  └──────┬──────┘
         │ pooling (last/mean/max/cls)
         ▼
  tensor: (1, 4096) float32
         │
         ▼ projection (if cross-model)
  ┌──────────────┐
  │ ProjectionModel│ → projected: (1, 5120)
  └──────┬───────┘
         │
         ▼ FlatBuffers codec + CRC32
  ┌──────────────┐
  │  gRPC Client  │ → TensorRequest(agent_id, label, data)
  └──────┬───────┘
         │ network
         ▼
  ┌──────────────┐
  │  gRPC Server  │ → authenticate, rate check, route
  └──────┬───────┘
         │
         ▼ projection (reverse, if needed)
  ┌──────────────┐
  │   Injector    │ → prefix or cross-attention
  └──────┬───────┘
         │
         ▼
Target Model (Mistral)
      │
      ▼
  generated output
```

---

## Threading Model

- **gRPC Server** — Thread-pool (10 workers), synchronous handlers
- **Plugin Processing** — Synchronous by default, async for I/O-heavy plugins
- **Tensor Ops** — CPU-bound, release GIL where possible (native C extensions)
- **Rate Limiting** — Thread-safe with `threading.Lock` in TokenBucket

---

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| grpcio | ≥1.60 | gRPC transport |
| protobuf | ≥4.25 | Wire protocol schema |
| typer | ≥0.9 | CLI framework |
| rich | ≥13.0 | Terminal output |
| pydantic | ≥2.5 | Configuration validation |
| pyyaml | ≥6.0 | YAML config parsing |
| flatbuffers | ≥23.5 | Binary codec |
| torch (optional) | ≥2.1 | ML tensor operations |
| transformers (optional) | ≥4.36 | Model loading |
| safetensors (optional) | ≥0.4 | Safe tensor serialization |
