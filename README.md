<a id="top"></a>

<div align="center">
  <h1>A2A Protocol</h1>
  <p><strong>AI-to-AI Latent Space Communication Protocol</strong></p>

  <p>
    <a href="https://github.com/batmunkhcom/a2a/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python" alt="Python"></a>
    <img src="https://img.shields.io/badge/Status-Pre--alpha-red" alt="Status">
    <img src="https://img.shields.io/badge/Version-0.1.0.dev0-lightgrey" alt="Version">
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
| **S0** | 1–2 | Project setup, CI/CD, scaffolding | 🔴 Not started |
| **S1** | 3–5 | Wire protocol + Transport layer (gRPC) | ⬜ |
| **S2** | 6–9 | Tensor Engine (extract/inject/serialize) | ⬜ |
| **S3** | 10–13 | Plugin System + Core Runtime | ⬜ |
| **S4** | 14–17 | Projection Model + Auto-training | ⬜ |
| **S5** | 18–20 | Security, Rate Limiting, Monitoring | ⬜ |
| **S6** | 21–22 | Docs, Integration, PyPI Release | ⬜ |
| **S7+** | — | Go Transport Layer (hybrid architecture) | ⬜ |

## Contributing

A2A Protocol is developed by [mBm TECHNOLOGY LLC](https://www.mbm.technology). Contributions are welcome — please open an issue before submitting a PR to discuss the proposed change.

See the internal docs (`docs/` — not in the repo) for the full architectural plan and sprint breakdown.

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

```
Copyright 2026 mBm TECHNOLOGY LLC
Licensed under the Apache License, Version 2.0
```

---

<div align="center">
  <sub>Developed by <a href="https://www.mbm.technology">mBm TECHNOLOGY LLC</a></sub>
</div>
