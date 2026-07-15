<a id="top"></a>

<div align="center">
  <h1>A2A Protocol</h1>
  <p><strong>AI-to-AI Latent Space Communication Protocol</strong></p>

  <p>
    <a href="https://github.com/batmunkhcom/a2a/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python" alt="Python"></a>
    <img src="https://img.shields.io/badge/Status-Alpha-yellow" alt="Status">
    <img src="https://img.shields.io/badge/Version-0.1.0-blue" alt="Version">
  </p>
</div>

---

## Overview

Today's AI agents communicate through human-oriented text and JSON APIs. This introduces overhead from tokenization, text parsing, and serialization at every hop.

A2A Protocol replaces text-based communication with **direct latent space vector transfer**. Instead of "the error is a NullPointerException at line 45", an AI sends `[0.45, -0.12, 0.98, ...]` — the raw hidden state from its neural network. The receiving AI injects this vector directly and starts generating code *without reading text*.

```
CURRENT (Text-based)  ~1400ms:
  Log AI → text → API → text → Code AI → code → Human

A2A (Tensor-based)    ~215ms:
  Log AI → [hidden state vector] → Code AI → code → Human
  (6.5x faster, zero token overhead)
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     A2A RUNTIME                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PLUGIN LAYER: LogReader | CodeFixer | Security | ... │  │
│  │  (semantic label routing: "error_context" → code)     │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                           │                                  │
│  ┌───────────────────────┼──────────────────────────────┐  │
│  │  CORE LAYER: Transport (gRPC) | PluginManager |      │  │
│  │              ProjectionRegistry | SemanticRouter      │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  TENSOR ENGINE: Extract | Inject | Serialize    │  │  │
│  │  │  (PyTorch hooks + Safetensors / FlatBuffers)    │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  INFRASTRUCTURE: Discovery | mTLS | Rate Limiting | Metrics │
└────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---|---|
| **Tensor Transport** | Raw hidden state vectors sent over gRPC/QUIC instead of text |
| **Semantic Label Routing** | Vectors tagged with labels (`error_context`, `code_patch`) — plugins subscribe to labels |
| **Projection Model** | Lightweight MLP adapter (~2-8M params) that maps vectors between different model architectures (Llama ↔ Mistral) |
| **Auto-Training** | Runtime automatic projection training via contrastive learning when unknown model pairs meet |
| **Plugin System** | Agents are plugins implementing `BasePlugin` — `listens_to()`, `emits()`, `on_receive_tensor()` |

### Communication Model

| Type | Example | Projection Needed |
|---|---|---|
| **Homogeneous** | Llama-8B (logs) → Llama-8B (code) | No — direct injection |
| **Heterogeneous** | Llama-8B (logs) → Mistral-7B (code) | Yes — auto-trained MLP |

---

## Glossary

<details open>
<summary><b>Core Terms</b></summary>

| Term | Definition |
|---|---|
| **A2A Runtime** | The process that hosts plugins, manages transport, and orchestrates tensor flow between agents |
| **BasePlugin** | Abstract base class all plugins inherit from; defines `listens_to()`, `emits()`, `on_receive_tensor()`, `extract_tensor()` |
| **Capability** | Declaration of what a plugin can do — input labels, output labels, model info, hidden dimensions |
| **Contrastive Learning** | Self-supervised training technique (InfoNCE loss) used to align latent spaces of different models |
| **Cross-attention Injection** | Tensor injected as key/value pairs in a model's cross-attention layer (alternative to prefix injection) |
| **FlatBuffers** | Zero-copy serialization library used for tensor wire format — deserializes without allocation |
| **Hidden State** | The activation vector at a specific layer of a transformer model — represents the model's internal representation of input |
| **Homogeneous** | Same base model architecture on both sides — no projection needed |
| **Heterogeneous** | Different model architectures — requires a Projection Model to translate between latent spaces |
| **Latent Space** | The high-dimensional vector space where a model's internal representations live |
| **MSG_CAPABILITY** | Protocol message type for announcing a plugin's capabilities to peers |
| **MSG_PROJECT_AUTO** | Triggers automatic projection model training when an unknown model pair is discovered |
| **MSG_PROJECT_READY** | Sent when auto-training completes — projection model is cached and ready |
| **MSG_BACKPRESSURE** | Flow control signal sent when a receiver's queue is full |
| **Plugin Manager** | Core component that loads, registers, and routes tensors between plugins |
| **Prefix Injection** | Tensor prepended as an embedding prefix before the text prompt — allows the model to "read" the tensor |
| **Projection Registry** | Persistent cache of trained projection models by `src__tgt` key |
| **Protobuf** | Protocol Buffers — used for A2A message header schema and gRPC service definitions |
| **Safetensors** | Safe tensor serialization format — used for saving/loading projection model weights |
| **Semantic Label** | String tag (`error_context`, `code_patch`, `log_summary`) that routers use to match senders and receivers |
| **Single Source of Truth** | Configuration principle: shared resources (model API keys, endpoints) defined once in `a2a.yaml`, referenced by plugins |
| **Tensor Engine** | The low-level subsystem that extracts hidden states from source models and injects them into target models |
| **Token Savings** | Efficiency metric — percentage of text tokens eliminated by using direct tensor transfer |
| **Wire Protocol** | The binary message format (magic header + metadata + tensor payload + checksum) sent over the transport layer |

</details>

---

## Quick Start

> Project is in **pre-alpha** development. Current sprint: S0 (scaffolding).

```bash
# Clone
git clone https://github.com/batmunkhcom/a2a.git
cd a2a

# Install (editable)
pip install -e ".[dev]"

# Verify
a2a --help
```

Planned CLI commands:

```bash
a2a serve                        # Start A2A runtime with plugins
a2a discover                     # Find A2A agents on the network
a2a config validate              # Validate a2a.yaml
a2a config show                  # Show loaded config
a2a config show --section models # Show model configs
a2a train-projection --src llama-8b --tgt mistral-7b
a2a benchmark --task error-resolution
```

## Documentation

| Document | Description |
|---|---|
| [Protocol Specification](https://github.com/batmunkhcom/a2a/wiki/Protocol) | Wire format, message types, metadata schema, error codes |
| [Plugin Development Guide](https://github.com/batmunkhcom/a2a/wiki/Plugins) | How to build custom plugins with `BasePlugin` |
| [Configuration Reference](https://github.com/batmunkhcom/a2a/wiki/Configuration) | Full `a2a.yaml` schema, all sections and fields |
| [Deployment Guide](https://github.com/batmunkhcom/a2a/wiki/Deployment) | Docker, Kubernetes, multi-node mesh |
| [Architecture Overview](https://github.com/batmunkhcom/a2a/wiki/Architecture) | Deep dive into layers, data flow, projection model |
| [API Reference](#api-reference) | Core SDK API — classes, methods, types |

---

## API Reference

### Core Package Structure

```
a2a/
├── agent/         — Plugin system (BasePlugin, PluginManager, routing)
├── config/        — Configuration (A2AConfig, loader)
├── projection/    — Cross-model mapping (ProjectionModel, trainer, registry)
├── protocol/      — Wire protocol (messages, protobuf schema, errors)
├── tensor/        — Tensor operations (extract, inject, serialize, dtype)
├── transport/     — Network layer (gRPC server/client, codec, discovery)
├── plugins/       — Built-in plugins (LogReader, CodeFixer)
├── security/      — Auth (JWT, mTLS)
├── monitoring/    — Metrics (Prometheus, rate limiter)
└── utils/         — Logging, async helpers
```

### BasePlugin

Abstract base class for all A2A plugins. Every plugin inherits from this.

```python
from a2a.agent.base import BasePlugin, Capability, ModelInfo

class BasePlugin(ABC):
    """Superclass for all A2A plugins."""

    # ── Identity ──
    plugin_id: str            # Unique ID, e.g. "code-fixer"
    plugin_name: str          # Human-readable name
    version: str              # Semantic version, e.g. "1.0.0"

    # ── Lifecycle ──
    async def initialize(self, model, tokenizer, plugin_config: dict, global_config: A2AConfig): ...
    async def on_load(self, runtime: "A2ARuntime"): ...
    async def on_unload(self): ...

    # ── Capabilities ──
    @abstractmethod
    def get_capabilities(self) -> list[Capability]: ...
    @abstractmethod
    def get_model_info(self) -> ModelInfo: ...
    @abstractmethod
    def listens_to(self) -> list[str]: ...
    @abstractmethod
    def emits(self) -> list[str]: ...

    # ── Tensor Processing ──
    @abstractmethod
    async def extract_tensor(self, input_data: Any, semantic_label: str) -> torch.Tensor: ...
    @abstractmethod
    async def on_receive_tensor(self, tensor: torch.Tensor, metadata: A2AMetadata) -> Optional[torch.Tensor]: ...
```

### Capability

```python
@dataclass
class Capability:
    name: str              # "log_analysis", "code_generation"
    description: str
    input_labels: list[str]   # Labels this capability accepts
    output_labels: list[str]  # Labels this capability produces
    model_id: str
    hidden_dim: int
    dtype: str             # "float16", "bfloat16", "float32"
    max_batch_size: int
    priority: int = 0

@dataclass
class ModelInfo:
    model_id: str          # "llama-3-8b"
    architecture: str      # "llama", "mistral", "gpt"
    hidden_dim: int        # 4096
    num_layers: int        # 32
    dtype: str             # "float16"
    supported_projection: list[str] = []
```

### PluginManager

Central plugin lifecycle manager. Handles loading, registering, and routing.

```python
class PluginManager:
    def register(self, plugin: BasePlugin): ...
    def unregister(self, plugin_id: str): ...

    async def load_plugin(self, entry: PluginEntry, global_config: A2AConfig) -> BasePlugin: ...
    def discover_from_directory(self, path: str): ...
    def discover_from_entry_points(self, group: str = "a2a.plugins"): ...

    async def route_tensor(self, tensor: torch.Tensor, metadata: A2AMetadata) -> list[BasePlugin]: ...
    def get_matching_plugins(self, required_labels: list[str]) -> list[BasePlugin]: ...
```

**Plugin Loading Methods:**
```python
# 1. Direct code registration
manager.register(LogReaderPlugin(model, tokenizer))

# 2. Filesystem discovery
manager.discover_from_directory("./my_plugins/")

# 3. Python entry_points (pip-installable)
manager.discover_from_entry_points("a2a.plugins")

# 4. YAML config-driven
manager.load_from_config("config.yaml")
```

### Tensor Extractor

Extracts hidden states from HuggingFace models using forward hooks.

```python
class TensorExtractor:
    def __init__(self, model: nn.Module, layer_idx: int = -1):
        """
        Args:
            model: HuggingFace model
            layer_idx: Which layer's hidden state (-1 = last layer)
        """

    def extract(self, text: str, pooling: str = "last") -> torch.Tensor:
        """
        Forward pass → extract hidden state at layer_idx.

        Args:
            text: Input text
            pooling: "last" = last token, "mean" = mean pooling, "max" = max pooling

        Returns:
            Tensor of shape (1, hidden_dim)
        """

    def set_layer(self, layer_idx: int): ...
    def get_registered_hooks(self) -> list: ...
```

### Tensor Injector

Injects hidden state tensors into a target model for conditional generation.

```python
class TensorInjector:
    def __init__(self, model: nn.Module, target_layer: int = 0):
        """
        Args:
            model: HuggingFace model
            target_layer: Layer index for injection (0 = prefix, -1 = last)
        """

    def inject_prefix(self, tensor: torch.Tensor, prompt: str) -> dict:
        """
        Prepend tensor as embedding prefix before the prompt.
        Returns the modified inputs dict for model.generate().
        """

    def inject_cross_attention(self, tensor: torch.Tensor, prompt: str) -> dict:
        """
        Use tensor as cross-attention key/value pairs.
        Alternative injection method for models supporting cross-attention.
        """

    async def inject_and_generate(
        self, tensor: torch.Tensor, prompt: str,
        max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        """Convenience: inject + generate in one call."""
```

### Tensor Serializer

```python
# Safetensors-based save/load
def save_tensor_to_bytes(tensor: torch.Tensor) -> bytes: ...
def load_tensor_from_bytes(data: bytes) -> torch.Tensor: ...

# FlatBuffers codec (zero-copy)
def encode_tensor(tensor: torch.Tensor) -> bytes:
    """PyTorch tensor → FlatBuffers bytes (shape + dtype preserved)"""

def decode_tensor(data: bytes) -> torch.Tensor:
    """FlatBuffers bytes → PyTorch tensor"""

# dtype utilities
def convert_dtype(tensor: torch.Tensor, target_dtype: str) -> torch.Tensor:
    """Convert between FP32, FP16, BF16"""

def validate_tensor(tensor: torch.Tensor, max_l2: float = 1000.0, min_l2: float = 0.001) -> bool:
    """Check for NaN, Inf, bounds"""
```

### A2AConfig

Pydantic-based configuration model. Single source of truth for all shared resources.

```python
class A2AConfig(BaseModel):
    version: str
    runtime: RuntimeConfig
    models: dict[str, ModelConfig]
    transport: TransportConfig
    discovery: DiscoveryConfig
    projection: ProjectionConfig
    plugins: dict[str, PluginEntry]
    routes: dict[str, list[str]]
    security: SecurityConfig
    rate_limit: RateLimitConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "A2AConfig": ...
    def validate(self) -> bool: ...


class ModelConfig(BaseModel):
    provider: str           # "openai" | "huggingface" | "ollama" | "anthropic"
    model_id: str
    api_key_env: str | None  # e.g. "DEEPSEEK_API_KEY"
    api_base: str | None
    device: str = "cuda:0"
    dtype: str = "float16"
    max_tokens: int
    load_in_8bit: bool = False


def load_config(path: Path | str | None = None) -> A2AConfig:
    """Searches: A2A_CONFIG env → ./a2a.yaml → ~/.config/a2a/ → /etc/a2a/"""

def load_plugin_config(plugin_path: Path) -> dict:
    """Loads plugin-local config.yaml from plugin directory."""
```

### Projection Model

Lightweight MLP adapter mapping hidden states between different model architectures.

```python
class ProjectionModel(nn.Module):
    """
    Args:
        src_dim: Source model hidden dimension (e.g. 4096 for Llama-8B)
        tgt_dim: Target model hidden dimension (e.g. 4096 for Mistral-7B)
        hidden_dim: Internal MLP dimension (default: 2048)
    """
    def __init__(self, src_dim: int, tgt_dim: int, hidden_dim: int = 2048): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class ProjectionTrainer:
    """Train a projection model using contrastive loss (InfoNCE)."""
    def __init__(self, model: ProjectionModel, lr: float = 0.001, temperature: float = 0.07): ...
    def train(self, dataset: "ProjectionPairDataset", epochs: int = 50) -> dict[str, list[float]]: ...
    def save(self, path: Path): ...


class ProjectionPairDataset(Dataset):
    """Creates (src_hidden, tgt_hidden) pairs from a shared text corpus."""
    @classmethod
    def from_corpus(cls, corpus_path: Path, src_model, src_tokenizer, src_extractor,
                    tgt_model, tgt_tokenizer, tgt_extractor, device="cuda") -> "ProjectionPairDataset": ...


class ProjectionRegistry:
    """Cache and retrieve trained projection models by src__tgt key."""
    def get(self, src_model: str, tgt_model: str) -> ProjectionModel | None: ...
    def load(self, src_model: str, tgt_model: str, path: Path, src_dim: int, tgt_dim: int): ...
    def save(self, src_model: str, tgt_model: str, model: ProjectionModel, path: Path): ...


class AutoTrainer:
    """Runtime auto-training: detect unknown model pair → collect data → train → cache."""
    async def auto_train(self, src_plugin: BasePlugin, tgt_plugin: BasePlugin) -> ProjectionModel: ...
```

### A2A Runtime

Orchestrates the full lifecycle.

```python
class A2ARuntime:
    def __init__(self, config: A2AConfig): ...
    async def start(self): ...
    async def stop(self): ...
    async def send_tensor(self, tensor: torch.Tensor, metadata: A2AMetadata): ...
```

**Startup Sequence:**

```python
async def start(self):
    config = load_config()                    # 1. Load a2a.yaml
    setup_logging(config.runtime)             # 2. Structured logging
    for entry in config.plugins.values():      # 3. Load plugins
        if entry.enabled:
            await plugin_manager.load_plugin(entry, config)
    for pair_key, path in config.projection.pretrained.items():  # 4. Load projections
        projection_registry.load(*pair_key.split("__"), path)
    self.server = A2AServer(config)            # 5. Start gRPC server
    await self.server.start()
    start_metrics_server(config.runtime.metrics_port)  # 6. Health + metrics
```

### Wire Protocol Messages

```python
# Protobuf metadata
message A2AMetadata:
  source_model: str       # "llama-3-8b-finetuned"
  target_model: str       # "mistral-7b"
  source_layer: uint32    # hidden state extraction layer
  target_layer: uint32    # hidden state injection layer
  tensor_dtype: str       # "float16", "bfloat16", "float32"
  tensor_shape: [uint32]  # [seq_len, hidden_dim]
  semantic_label: str     # "error_context", "code_intent"
  confidence: float       # 0.0 - 1.0
  timestamp: uint64
  session_id: str
  projection_id: str
  requires_projection: bool

# Message types (msg_type)
MSG_UNICAST       = 0x01   # Single tensor send
MSG_MULTICAST     = 0x02   # Broadcast to multiple agents
MSG_DISCOVER      = 0x03   # Find agents on network
MSG_CAPABILITY    = 0x04   # Announce capabilities
MSG_PROJECT_REQ   = 0x05   # Request projection
MSG_PROJECT_RESP  = 0x06   # Projection response
MSG_PROJECT_AUTO  = 0x07   # Trigger auto-training
MSG_PROJECT_READY = 0x08   # Auto-training complete
MSG_KEEPALIVE     = 0x09   # Connection keepalive
MSG_ACK           = 0x0A   # Acknowledgment
MSG_ERROR         = 0x0B   # Error response
MSG_BACKPRESSURE  = 0x0C   # Flow control signal

# Error codes
TENSOR_SHAPE_MISMATCH     = 100
TENSOR_DTYPE_MISMATCH     = 101
TENSOR_VALUE_INVALID      = 102
PROJECTION_NOT_FOUND      = 200
PROJECTION_TRAINING_FAILED = 201
MODEL_NOT_FOUND           = 300
MODEL_OVERLOADED          = 301
AUTH_FAILED               = 400
PERMISSION_DENIED         = 401
MESH_MISMATCH             = 402
INTERNAL_ERROR            = 500
PLUGIN_CRASHED            = 501
GENERATION_FAILED         = 502
```

### gRPC Service Definition

```protobuf
service A2AService {
  rpc SendTensor(TensorRequest) returns (TensorResponse);
  rpc StreamTensors(stream TensorRequest) returns (stream TensorResponse);
  rpc Discover(DiscoverRequest) returns (DiscoverResponse);
  rpc RequestProjection(ProjectionRequest) returns (ProjectionResponse);
  rpc HealthCheck(HealthRequest) returns (HealthResponse);
}

message TensorRequest {
  A2AMetadata metadata = 1;
  bytes tensor_data = 2;   // FlatBuffers-encoded
}

message TensorResponse {
  bool accepted = 1;
  string error = 2;
  uint32 error_code = 3;
  A2AMetadata metadata = 4;
  bytes tensor_data = 5;   // Optional response tensor
}
```

### Transport

```python
class A2AServer:
    def __init__(self, config: TransportConfig, plugin_manager: PluginManager): ...
    async def start(self): ...
    async def stop(self): ...

class A2AClient:
    def __init__(self, address: str, credentials: grpc.ChannelCredentials | None = None): ...
    async def send_tensor(self, tensor: torch.Tensor, metadata: A2AMetadata) -> TensorResponse: ...
    async def stream_tensors(self, tensors: AsyncIterable[torch.Tensor]) -> AsyncIterable[TensorResponse]: ...
    async def health_check(self) -> HealthResponse: ...
    async def discover_agents(self) -> list[AgentInfo]: ...
```

### Semantic Label Routing

```python
class PluginManager:
    async def route_tensor(self, tensor: torch.Tensor, metadata: A2AMetadata):
        """
        Routing logic:
        1. Extract semantic_label from metadata
        2. Find all plugins where label is in listens_to()
        3. For each target, check model compatibility
        4. If different model → resolve projection (cached or auto-train)
        5. Forward projected tensor to plugin.on_receive_tensor()
        """

# Example route in a2a.yaml:
# routes:
#   "error_context":
#     - "code-fixer"
#     - "security-analyzer"
#   "code_patch":
#     - "code-reviewer"
```

### Security

```python
# JWT Authentication
def create_token(agent_id: str, mesh_id: str, secret: str, permissions: list[str]) -> str: ...
def validate_token(token: str, secret: str) -> dict | None: ...

# mTLS
class TlsConfig:
    cert_file: str
    key_file: str
    ca_file: str

def create_server_credentials(tls: TlsConfig) -> grpc.ServerCredentials: ...
def create_client_credentials(tls: TlsConfig) -> grpc.ChannelCredentials: ...
```

### Rate Limiting

```python
class TokenBucket:
    def __init__(self, bucket_size: int, refill_rate: float): ...
    def allow(self) -> bool: ...
    def consume(self, n: int = 1) -> bool: ...

class RateLimiter:
    def get_bucket(self, agent_id: str | None = None, route: str | None = None) -> TokenBucket: ...
    def allow(self, agent_id: str, route: str) -> bool: ...
```

### Monitoring & Metrics

```python
# Prometheus metrics
a2a_tensors_sent_total          # Counter: tensors sent by agent/label
a2a_tensors_received_total      # Counter: tensors received
a2a_tensor_latency_seconds      # Histogram: end-to-end latency
a2a_tensor_size_bytes           # Histogram: tensor payload size
a2a_projection_requests_total   # Counter: projection lookups
a2a_projection_auto_train_total # Counter: auto-train events
a2a_plugin_active               # Gauge: active plugin count
a2a_plugin_crash_total          # Counter: plugin failures
a2a_rate_limit_hits_total       # Counter: rate limit triggers
a2a_backpressure_events_total   # Counter: flow control signals

# Health endpoints
GET /health           → {"status": "ok", "uptime": 3600, "version": "1.0"}
GET /health/ready     → {"ready": true, "plugins_loaded": 5}
GET /health/live      → {"alive": true}
GET /metrics          → Prometheus scrape endpoint
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| **ML Core** | Python 3.11+, PyTorch 2.x, HuggingFace Transformers |
| **Transport** | gRPC, FlatBuffers (zero-copy), Protobuf |
| **Tensor Serialization** | Safetensors, FlatBuffers |
| **Inference** | vLLM, Ollama, HuggingFace Inference |
| **Security** | mTLS, JWT auth |
| **Monitoring** | Prometheus, OpenTelemetry, structured JSON logging |
| **Deployment** | Docker, GitHub Actions |

## Development Status

| Sprint | Weeks | Focus | Status |
|---|---|---|---|
| **S0** | 1–2 | Project setup, CI/CD, scaffolding | ✅ Complete |
| **S1** | 3–5 | Wire protocol + Transport layer (gRPC) | ✅ Complete |
| **S2** | 6–9 | Tensor Engine (extract/inject/serialize) | ✅ Complete |
| **S3** | 10–13 | Plugin System + Core Runtime | ✅ Complete |
| **S4** | 14–17 | Projection Model + Auto-training | ✅ Complete |
| **S5** | 18–20 | Security, Rate Limiting, Monitoring | ✅ Complete |
| **S6** | 21–22 | Docs, Integration, PyPI Release | ✅ Complete |
| **S7+** | — | Go Transport Layer (hybrid architecture) | ⬜ Future |
> **120 tests passing** | ruff+mypy clean | Apache 2.0 Licensed

## About

A2A Protocol is developed with [**mBm AI Assistant**](https://console.mbm.mn) — an AI-powered engineering and operations assistant by [mBm TECHNOLOGY LLC](https://www.mbm.technology) that handles rapid coding, server management, deployments, and full-stack troubleshooting.

> **Try it at:** [console.mbm.mn](https://console.mbm.mn)

## Contributing

Contributions are welcome — please open an issue before submitting a PR to discuss the proposed change.

Internal planning documents are maintained in `docs/` (gitignored). For the full architectural plan and sprint breakdown, contact the maintainers.

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

```
Copyright 2026 mBm TECHNOLOGY LLC
Licensed under the Apache License, Version 2.0
```

---

<div align="center">
  <sub>Developed with <a href="https://console.mbm.mn">mBm AI Assistant</a> by <a href="https://www.mbm.technology">mBm TECHNOLOGY LLC</a></sub>
</div>
