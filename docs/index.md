# A2A Protocol

**AI-to-AI Latent Space Communication Protocol** — v0.1.0-alpha

A2A enables AI models to communicate directly in **latent space** (tensor-to-tensor),
bypassing text serialization and achieving 10–100× lower latency and richer semantic
transfer compared to traditional text-based APIs.

```bash
pip install a2a-protocol
a2a serve --config a2a.yaml
```

---

## 5-Minute Quickstart

### 1. Install

```bash
pip install a2a-protocol
# With ML extras (GPU, torch):
pip install "a2a-protocol[ml]"
```

### 2. Create `a2a.yaml`

```yaml
version: "0.1"
mesh_id: "demo-mesh"
plugins:
  log_reader:
    module: a2a.plugins.log_reader.plugin
    agent_id: log-reader
    model: base
  code_fixer:
    module: a2a.plugins.code_fixer.plugin
    agent_id: code-fixer
    model: base
models:
  base:
    name: demo-model
    family: demo
    dtype: fp32
    hidden_dim: 768
```

### 3. Start the mesh

```bash
a2a serve
# Mesh runs on localhost:50051
# Health: http://localhost:8080/health
```

### 4. Send a tensor

```python
from a2a.transport.client import A2AClient
import numpy as np

client = A2AClient("localhost:50051")
tensor = np.random.randn(1, 768).astype(np.float32)

response = client.send_tensor(
    agent_id="log-reader",
    label="error_context",
    data=tensor,
)
print(f"Response shape: {response.shape}")
```

---

## How It Works

```
┌──────────┐  gRPC + FlatBuffers  ┌──────────┐
│ Agent A  │ ◄──────────────────► │ Agent B  │
│ (Llama)  │    latent tensors    │ (Mistral)│
└──────────┘                      └──────────┘
      │                                  │
      ▼                                  ▼
  Extractor                          Injector
  (hidden states)                    (prefix injection)
      │                                  │
      └────────── Projection ────────────┘
              (cross-model mapping)
```

1. **Extract** — Hidden states captured from source model via forward hooks
2. **Project** — Cross-model adapter maps between different latent spaces (optional)
3. **Transport** — gRPC + FlatBuffers binary codec with CRC32 integrity
4. **Inject** — Prefix or cross-attention injection into target model

---

## Architecture

| Layer | Description | Module |
|---|---|---|
| **Wire Protocol** | Protobuf schema, gRPC service, FlatBuffers codec | `a2a.protocol`, `a2a.transport` |
| **Tensor Engine** | Extractor, Injector, dtype conversion, serialization | `a2a.tensor` |
| **Plugin System** | BasePlugin, PluginManager, SemanticRouter | `a2a.agent` |
| **Configuration** | A2AConfig (Pydantic), YAML loader | `a2a.config` |
| **Projection** | Cross-model adapter, trainer, auto-train loop | `a2a.projection` |
| **Security** | JWT, mTLS, rate limiting | `a2a.security`, `a2a.monitoring` |
| **Runtime** | Orchestrator (config → transport → plugins) | `a2a.runtime` |
| **CLI** | `a2a serve`, `a2a train`, `a2a discover`, etc. | `a2a.cli` |

---

## CLI Reference

```bash
a2a serve          # Start A2A mesh
a2a train          # Train projection model
a2a discover       # Profile and register models
a2a config         # Manage config
a2a benchmark      # Run benchmarks
a2a info           # Mesh status and plugin info
```

---

## Next Steps

- [Architecture Deep Dive](architecture.md)
- [Wire Protocol Specification](protocol.md)
- [Plugin Development Guide](plugins.md)
- [Configuration Reference](config.md)
- [Deployment Guide](deployment.md)
- [API Reference](api/)

---

## License

Apache 2.0 — Copyright (c) 2026 mBm TECHNOLOGY LLC

Developed with [mBm AI Assistant](https://console.mbm.mn)
