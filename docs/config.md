# Configuration Reference

A2A uses a YAML configuration file (`a2a.yaml`) with Pydantic v2 validation.

Config search order: `A2A_CONFIG` env var → `./a2a.yaml` → `~/.config/a2a/a2a.yaml` → `/etc/a2a/a2a.yaml`

---

## Full Example

```yaml
version: "0.1"
mesh_id: "production-mesh"

server:
  host: "0.0.0.0"
  port: 50051
  max_workers: 20
  max_message_length: 104857600  # 100 MB

security:
  jwt_secret: "${A2A_JWT_SECRET}"
  jwt_algorithm: "HS256"
  jwt_expiry: 3600
  tls:
    enabled: true
    cert_file: "/etc/a2a/certs/server.crt"
    key_file: "/etc/a2a/certs/server.key"
    ca_file: "/etc/a2a/certs/ca.crt"
  mesh_whitelist:
    - "dev-mesh"
    - "staging-mesh"

rate_limiting:
  enabled: true
  default_capacity: 100
  default_refill_rate: 10.0
  per_agent:
    log-reader:
      capacity: 200
      refill_rate: 20.0
    code-fixer:
      capacity: 50
      refill_rate: 5.0

monitoring:
  prometheus_port: 9090
  health_port: 8080
  log_level: "INFO"

models:
  llama-8b:
    name: "meta-llama/Llama-3.1-8B"
    family: "llama"
    dtype: "fp16"
    hidden_dim: 4096
    num_layers: 32
    projection:
      variant: "b"
      hidden_size: 1024
      dropout: 0.1

  mistral-7b:
    name: "mistralai/Mistral-7B-v0.3"
    family: "mistral"
    dtype: "fp16"
    hidden_dim: 4096

plugins:
  log_reader:
    module: a2a.plugins.log_reader.plugin
    agent_id: log-reader
    model: llama-8b
    labels:
      - error_context
      - hidden_state
    config:
      max_lines: 1000

  code_fixer:
    module: a2a.plugins.code_fixer.plugin
    agent_id: code-fixer
    model: mistral-7b
    labels:
      - error_context
    config:
      context_window: 50

routes:
  - from: {agent_id: log-reader, label: error_context}
    to: {agent_id: code-fixer, label: error_context}
    projection: true
    bidirectional: false
```

---

## Schema Reference

### Top-Level

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `version` | str | Yes | — | Config schema version (`"0.1"`) |
| `mesh_id` | str | Yes | — | Unique mesh identifier |
| `server` | ServerConfig | No | defaults | gRPC server settings |
| `security` | SecurityConfig | No | disabled | Security settings |
| `rate_limiting` | RateLimitConfig | No | disabled | Rate limiting |
| `monitoring` | MonitorConfig | No | defaults | Monitoring |
| `models` | dict[str, ModelConfig] | Yes | — | Model definitions |
| `plugins` | dict[str, PluginEntry] | Yes | — | Plugin definitions |
| `routes` | list[RouteConfig] | No | [] | Routing table |

### ServerConfig

| Field | Type | Default |
|---|---|---|
| `host` | str | `"0.0.0.0"` |
| `port` | int | `50051` |
| `max_workers` | int | `10` |
| `max_message_length` | int | `104857600` |

### ModelConfig

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | str | Yes | HF model name or path |
| `family` | str | Yes | Model family for compatibility |
| `dtype` | str | No | `fp32`, `fp16`, `bf16` |
| `hidden_dim` | int | Yes | Hidden dimension |
| `num_layers` | int | No | Number of transformer layers |
| `projection` | ProjectionConfig | No | Projection model config |

### ProjectionConfig

| Field | Type | Default |
|---|---|---|
| `variant` | str | `"b"` |
| `hidden_size` | int | `1024` |
| `dropout` | float | `0.1` |

### PluginEntry

| Field | Type | Required | Description |
|---|---|---|---|
| `module` | str | Yes | Python module path |
| `agent_id` | str | Yes | Unique agent identifier |
| `model` | str | Yes | Model reference (key in `models`) |
| `labels` | list[str] | No | Semantic labels handled |
| `config` | dict | No | Plugin-specific config |

### RouteConfig

| Field | Type | Description |
|---|---|---|
| `from` | RouteEndpoint | Source `{agent_id, label}` |
| `to` | RouteEndpoint | Destination `{agent_id, label}` |
| `projection` | bool | Enable cross-model projection |
| `bidirectional` | bool | Route works both ways |

### SecurityConfig

| Field | Type | Default |
|---|---|---|
| `jwt_secret` | str | Required if security enabled |
| `jwt_algorithm` | str | `"HS256"` |
| `jwt_expiry` | int | `3600` |
| `tls` | TLSConfig | None |
| `mesh_whitelist` | list[str] | [] |

### RateLimitConfig

| Field | Type | Default |
|---|---|---|
| `enabled` | bool | `false` |
| `default_capacity` | int | `100` |
| `default_refill_rate` | float | `10.0` |
| `per_agent` | dict[str, AgentRateLimit] | {} |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `A2A_CONFIG` | Override config file path |
| `A2A_JWT_SECRET` | JWT signing secret (used in config via `${}`) |
| `A2A_LOG_LEVEL` | Override log level |
| `A2A_NO_ML` | Disable ML dependencies |

---

## CLI

```bash
# Validate config
a2a config validate --config a2a.yaml

# Show parsed config
a2a config show --config a2a.yaml

# Generate default config
a2a config init > a2a.yaml
```
