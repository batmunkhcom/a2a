# A2A Protocol — Development Task Plan

> **Project:** A2A Protocol — AI-to-AI Latent Space Communication
> **Version:** 0.1.0.dev0 → 0.1.0
> **Total Duration:** 22 weeks (~110 working days)
> **Language:** Python 3.11+
> **License:** Apache 2.0

---

## Project Overview

A2A Protocol enables AI agents to communicate via **direct latent space vectors** instead of text. By eliminating tokenization and text parsing overhead, communication speed increases **6.5x**.

| Metric | Current (Text API) | A2A Protocol |
|---|---|---|
| Latency | ~1400ms | ~215ms |
| Bandwidth | ~2-4KB text | 8KB tensor |
| Tokens | ~700 tokens | 0 tokens |

---

## Table of Contents

1. [Sprint 0 — Project Setup & Scaffolding](#sprint-0--project-setup--scaffolding)
2. [Sprint 1 — Wire Protocol + Transport Layer](#sprint-1--wire-protocol--transport-layer)
3. [Sprint 2 — Tensor Engine](#sprint-2--tensor-engine)
4. [Sprint 3 — Plugin System + Core Runtime](#sprint-3--plugin-system--core-runtime)
5. [Sprint 4 — Projection Model + Auto-training](#sprint-4--projection-model--auto-training)
6. [Sprint 5 — Security, Rate Limiting, Monitoring](#sprint-5--security-rate-limiting-monitoring)
7. [Sprint 6 — Documentation, Integration, Release](#sprint-6--documentation-integration-release)
8. [Sprint 7+ — Go Transport Layer (Phase 2)](#sprint-7--go-transport-layer-phase-2)
9. [MVP Definition](#mvp-definition)

---

## Sprint 0 — Project Setup & Scaffolding

**Duration:** 2 weeks (10 days)
**Goal:** Runnable skeleton — `pip install -e ".[dev]" ; a2a --help` works

### Tasks

- [x] **S0-T1** — Initialize repository, configure pyproject.toml  `# 1 day`
  - Files: `pyproject.toml`, `.gitignore`, `requirements.txt`, `requirements-dev.txt`
  - **AC:** `pip install -e ".[dev]"` succeeds, `import a2a` works

- [ ] **S0-T2** — CI/CD pipeline (GitHub Actions)  `# 1 day`
  - Files: `.github/workflows/ci.yml`
  - **AC:** Push triggers lint + test workflow

- [ ] **S0-T3** — Pre-commit hooks (ruff + mypy)  `# 0.5 day`
  - Files: `.pre-commit-config.yaml`
  - **AC:** `pre-commit run --all-files` passes

- [ ] **S0-T4** — Package skeleton (all `__init__.py`, `_version.py`)  `# 1 day`
  - Files: `a2a/__init__.py`, `a2a/_version.py`, all subpackage `__init__.py`
  - **AC:** All `__init__.py` present, `from a2a.config import ...` works

- [ ] **S0-T5** — CLI skeleton (typer) + entry point  `# 1 day`
  - Files: `a2a/cli.py`
  - **AC:** `a2a --help`, `a2a serve --help`, `a2a discover --help`, `a2a config --help`

- [ ] **S0-T6** — README.md + developer guide  `# 0.5 day`
  - File: `README.md`
  - **AC:** Newcomer can run the project by following README

- [ ] **S0-T7** — Dockerfile skeleton  `# 0.5 day`
  - Files: `Dockerfile`, `.dockerignore`
  - **AC:** `docker build .` succeeds

### S0 Tests (5)

- [ ] `test_package.py` — `import a2a` works, version string present
- [ ] `test_cli.py` — CLI help prints without error
- [ ] `test_cli.py` — `a2a serve` prints stub message
- [ ] `test_cli.py` — `a2a config validate` prints stub
- [ ] `test_package.py` — All subpackage imports work

### S0 Package Skeleton Structure

```
a2a/
├── __init__.py              → __version__, public API exports
├── _version.py              → __version__ = "0.1.0.dev0"
├── cli.py                   → typer app (serve, discover, config, train, benchmark)
├── config/                  → A2AConfig, loader, defaults (filled in S3)
│   ├── __init__.py
│   ├── schema.py            → stub
│   ├── loader.py            → stub
│   └── defaults.py          → stub
├── transport/               → gRPC server/client, codec, discovery (filled in S1)
│   ├── __init__.py
│   ├── server.py            → stub
│   ├── client.py            → stub
│   ├── codec.py             → stub
│   └── discovery.py         → stub
├── tensor/                  → Extractor, injector, serializer, dtype (filled in S2)
│   ├── __init__.py
│   ├── extractor.py         → stub
│   ├── injector.py          → stub
│   ├── serializer.py        → stub
│   └── dtype.py             → stub
├── projection/              → Adapter, trainer, dataset, registry (filled in S4)
│   ├── __init__.py
│   ├── adapter.py           → stub
│   ├── trainer.py           → stub
│   ├── dataset.py           → stub
│   ├── auto_trainer.py      → stub
│   └── registry.py          → stub
├── agent/                   → BasePlugin, PluginManager, router (filled in S3)
│   ├── __init__.py
│   ├── base.py              → stub
│   ├── manager.py           → stub
│   ├── router.py            → stub
│   └── capabilities.py      → stub
├── protocol/                → Protobuf, messages, errors (filled in S1)
│   ├── __init__.py
│   ├── messages.py          → stub
│   ├── a2a.proto            → stub
│   └── errors.py            → stub
├── security/                → Auth, TLS (filled in S5)
│   ├── __init__.py
│   ├── auth.py              → stub
│   └── tls.py               → stub
├── monitoring/              → Metrics, rate limiter (filled in S5)
│   ├── __init__.py
│   ├── metrics.py           → stub
│   └── rate_limiter.py      → stub
├── plugins/                 → Built-in plugins (filled in S3)
│   ├── __init__.py
│   ├── log_reader/          → stub
│   └── code_fixer/          → stub
└── utils/                   → Logging, async helpers (filled in S3, S5)
    ├── __init__.py
    ├── logging.py           → stub
    └── async_utils.py       → stub
```

---

## Sprint 1 — Wire Protocol + Transport Layer

**Duration:** 3 weeks (15 days)
**Goal:** Two processes communicate via gRPC, exchanging tensors

### Tasks

- [ ] **S1-T1** — Write Protobuf schema (`a2a.proto`)  `# 1 day`
  - File: `a2a/protocol/a2a.proto`
  - **AC:** `protoc` compiles successfully, `a2a_pb2.py` + `a2a_pb2_grpc.py` generated
  - Content: TensorRequest, TensorResponse, A2AMetadata, Discover, Health, all RPCs

- [ ] **S1-T2** — Protobuf codegen setup + Makefile target  `# 0.5 day`
  - File: `Makefile` (proto target)
  - **AC:** `make proto` → generate, CI pipeline has proto drift check

- [ ] **S1-T3** — FlatBuffers tensor codec: `encode_tensor()`  `# 1 day`
  - File: `a2a/transport/codec.py`
  - **AC:** PyTorch tensor → bytes, shape + dtype preserved

- [ ] **S1-T4** — FlatBuffers tensor codec: `decode_tensor()`  `# 1 day`
  - File: `a2a/transport/codec.py`
  - **AC:** bytes → PyTorch tensor, roundtrip error-free (FP32, FP16, BF16, multi-dim)

- [ ] **S1-T5** — `codec.py` — edge cases: empty tensor, NaN/Inf, batch dim  `# 0.5 day`
  - **AC:** Empty tensor → error, NaN → error (validate mode), batch supported

- [ ] **S1-T6** — gRPC server: `A2AServicer` (SendTensor, HealthCheck)  `# 1.5 days`
  - File: `a2a/transport/server.py`
  - **AC:** Server starts listening, health check works

- [ ] **S1-T7** — gRPC server: `StreamTensors` (bidirectional stream)  `# 1 day`
  - **AC:** Multiple tensors streamed sequentially, server receives all

- [ ] **S1-T8** — gRPC client: `A2AClient` (SendTensor, HealthCheck)  `# 1 day`
  - File: `a2a/transport/client.py`
  - **AC:** Client sends tensor, server receives, accepted=true

- [ ] **S1-T9** — gRPC integration test: send-receive roundtrip  `# 1 day`
  - **AC:** Server+client in same process, tensor sent and received

- [ ] **S1-T10** — Error handling: `MSG_ERROR` + error propagation  `# 1 day`
  - File: `a2a/protocol/errors.py`
  - **AC:** Invalid dtype → error_code=101, missing tensor → error, details propagated

### S1 Tests (17)

- [ ] `test_encode_fp32_tensor`
- [ ] `test_encode_fp16_tensor`
- [ ] `test_encode_bf16_tensor`
- [ ] `test_encode_multidim_tensor`
- [ ] `test_encode_empty_tensor_raises`
- [ ] `test_decode_roundtrip_fp32`
- [ ] `test_decode_roundtrip_fp16`
- [ ] `test_decode_roundtrip_bf16`
- [ ] `test_decode_roundtrip_multidim`
- [ ] `test_decode_invalid_header_raises`
- [ ] `test_grpc_health_check`
- [ ] `test_grpc_send_tensor_large`
- [ ] `test_grpc_stream_tensors`
- [ ] `test_grpc_send_tensor_error_on_empty`
- [ ] `test_grpc_send_tensor_error_on_nan`
- [ ] `test_grpc_client_reconnect`
- [ ] `test_protobuf_metadata_serialization`

---

## Sprint 2 — Tensor Engine

**Duration:** 4 weeks (20 days)
**Goal:** Extract hidden states from HuggingFace models, inject into target models, generate
**Test Model:** `sshleifer/tiny-gpt2` (69MB, fast on CPU)

### Tasks

- [ ] **S2-T1** — `TensorExtractor`: HF model hidden state extraction (register_forward_hook)  `# 2 days`
  - File: `a2a/tensor/extractor.py`
  - **AC:** Input text → (batch, seq, hidden_dim) shaped tensor

- [ ] **S2-T2** — `TensorExtractor`: pooling strategies (last, mean, max)  `# 1 day`
  - **AC:** pool="last" → (1, hidden), pool="mean" → (1, hidden), different values

- [ ] **S2-T3** — `TensorExtractor`: layer selection (default -1, arbitrary layer)  `# 0.5 day`
  - **AC:** layer=0 → first layer, layer=-1 → last layer

- [ ] **S2-T4** — `TensorInjector`: prefix injection (vector → embedding prefix → generate)  `# 2 days`
  - File: `a2a/tensor/injector.py`
  - **AC:** Tensor + prompt → model.generate() → text, output relates to prompt

- [ ] **S2-T5** — `TensorInjector`: cross-attention injection variant  `# 1.5 days`
  - **AC:** Not `inputs_embeds`, but cross-attention key/value injection

- [ ] **S2-T6** — `dtype.py`: FP32↔FP16↔BF16 conversion + validate_tensor (NaN/Inf)  `# 0.5 day`
  - File: `a2a/tensor/dtype.py`
  - **AC:** `convert_dtype(tensor, "float16")` → FP16, NaN detection correct

- [ ] **S2-T7** — `serializer.py`: Safetensors save/load  `# 0.5 day`
  - File: `a2a/tensor/serializer.py`
  - **AC:** `save_to_bytes(tensor)` → bytes, `load_from_bytes(data)` → tensor, roundtrip

- [ ] **S2-T8** — Integration test: extract → encode → decode → inject → generate  `# 2 days`
  - **AC:** Full pipeline, semantic consistency (output relates to input)

### S2 Tests (17)

- [ ] `test_extract_shape_last_pooling`
- [ ] `test_extract_shape_mean_pooling`
- [ ] `test_extract_shape_max_pooling`
- [ ] `test_extract_different_layers_produce_different_output`
- [ ] `test_extract_long_text`
- [ ] `test_extract_empty_text_raises`
- [ ] `test_inject_and_generate_produces_output`
- [ ] `test_inject_with_different_prompt`
- [ ] `test_inject_tensor_shape_mismatch_handled`
- [ ] `test_convert_dtype_fp32_to_fp16`
- [ ] `test_convert_dtype_fp16_to_fp32`
- [ ] `test_validate_nan_rejected`
- [ ] `test_validate_inf_rejected`
- [ ] `test_validate_clean_tensor_accepted`
- [ ] `test_safetensors_save_load_roundtrip`
- [ ] `test_safetensors_multidim_tensor`
- [ ] `test_extract_to_encode_to_decode_to_inject_pipeline`

---

## Sprint 3 — Plugin System + Core Runtime

**Duration:** 4 weeks (20 days)
**Goal:** `a2a serve` loads 2 demo plugins, they communicate via tensors, code fix is produced

### Tasks

- [ ] **S3-T1** — `BasePlugin` abstract class implementation  `# 0.5 day`
  - File: `a2a/agent/base.py`
  - **AC:** plugin_id, plugin_name, version, listens_to, emits, get_capabilities, on_receive_tensor, extract_tensor, initialize, lifecycle hooks

- [ ] **S3-T2** — `Capability` + `ModelInfo` dataclass  `# 0.5 day`
  - File: `a2a/agent/capabilities.py`
  - **AC:** Plugin self-description + network announcement support

- [ ] **S3-T3** — `PluginManager`: plugin loading (importlib + inspect)  `# 1 day`
  - File: `a2a/agent/manager.py`
  - **AC:** `load_plugin(entry, global_config)` → plugin instance + initialize called

- [ ] **S3-T4** — `PluginManager`: plugin registration + label routing  `# 0.5 day`
  - **AC:** `register(plugin)` → `listens_to()` labels added to routing table

- [ ] **S3-T5** — `PluginManager`: `route_tensor(tensor, metadata)`  `# 1 day`
  - File: `a2a/agent/router.py`
  - **AC:** semantic_label → matching plugin(s) → `on_receive_tensor()` called

- [ ] **S3-T6** — `PluginManager`: plugin-local config loading (`config.yaml`)  `# 0.5 day`
  - **AC:** Plugin's `config.yaml` read and passed to `initialize()`

- [ ] **S3-T7** — `A2AConfig` Pydantic model implementation  `# 1 day`
  - File: `a2a/config/schema.py`
  - **AC:** `A2AConfig.from_yaml(path)` → validated config object

- [ ] **S3-T8** — Config loader: search order, env override  `# 0.5 day`
  - File: `a2a/config/loader.py`
  - **AC:** `A2A_CONFIG` env → `./a2a.yaml` → `~/.config/a2a/` → `/etc/a2a/`

- [ ] **S3-T9** — Write demo `a2a.yaml` config  `# 0.5 day`
  - **AC:** 3 sections present: models, plugins, routes

- [ ] **S3-T10** — `A2ARuntime` orchestrator  `# 1 day`
  - File: `a2a/runtime.py`
  - **AC:** `start()` → load config → load plugins → start transport

- [ ] **S3-T11** — `LogReaderPlugin` implementation  `# 1 day`
  - Files: `a2a/plugins/log_reader/plugin.py`, `prompts.py`, `config.yaml`
  - **AC:** Log text → `extract_tensor()` → error_context tensor

- [ ] **S3-T12** — `CodeFixerPlugin` implementation  `# 1 day`
  - Files: `a2a/plugins/code_fixer/plugin.py`, `prompts.py`, `config.yaml`
  - **AC:** error_context tensor → `on_receive_tensor()` → inject → generate → code patch

- [ ] **S3-T13** — End-to-end integration test  `# 1 day`
  - **AC:** Runtime fully loads, log → code fix flow succeeds

### S3 Tests (19)

- [ ] Plugin subclass with all abstract methods
- [ ] PluginManager.register() → label routes populated
- [ ] PluginManager.route_tensor() → correct plugin called
- [ ] PluginManager.route_tensor() → no matching label → empty
- [ ] PluginManager.load_plugin() with valid entry
- [ ] PluginManager.load_plugin() with invalid module
- [ ] A2AConfig.from_yaml() valid file
- [ ] A2AConfig.from_yaml() missing required field
- [ ] A2AConfig.from_yaml() invalid model reference
- [ ] A2AConfig.from_yaml() invalid dtype
- [ ] load_config() finds file in cwd
- [ ] load_config() falls back to home dir
- [ ] load_config() raises if no file found
- [ ] load_config() explicit path
- [ ] LogReaderPlugin.extract_tensor() returns tensor
- [ ] CodeFixerPlugin.on_receive_tensor() returns text
- [ ] LogReader → CodeFixer full flow (integration)
- [ ] A2ARuntime.start() loads all plugins
- [ ] Semantic label routing with multiple listeners

---

## Sprint 4 — Projection Model + Auto-training

**Duration:** 4 weeks (20 days)
**Goal:** Two different models (Llama ↔ Mistral) communicate via tensors

### Tasks

- [ ] **S4-T1** — `ProjectionModel` (3-layer MLP + Residual + LayerNorm)  `# 1 day`
  - File: `a2a/projection/adapter.py`
  - **AC:** `forward(src_tensor)` → (target_dim,) correct shape

- [ ] **S4-T2** — `ProjectionModel` variant A (Linear)  `# 0.5 day`
  - **AC:** Same dims → identity, different dims → linear mapping

- [ ] **S4-T3** — `ProjectionTrainer`: contrastive loss (InfoNCE)  `# 1.5 days`
  - File: `a2a/projection/trainer.py`
  - **AC:** Training loop → loss decreases, cosine similarity increases

- [ ] **S4-T4** — `ProjectionTrainer`: multi-objective loss (MSE + Cosine + Contrastive)  `# 0.5 day`
  - **AC:** Three loss components, weighted sum correct
  - `loss = contrastive_loss + 0.1 * mse_loss + 0.01 * cosine_loss`

- [ ] **S4-T5** — `ProjectionPairDataset`: corpus → (src_hidden, tgt_hidden) pairs  `# 2 days`
  - File: `a2a/projection/dataset.py`
  - **AC:** Each text line → hidden states from both models → paired

- [ ] **S4-T6** — `ProjectionPairDataset`: negative pair generation  `# 0.5 day`
  - **AC:** Different text hidden states → negative pair

- [ ] **S4-T7** — `auto_trainer.py`: runtime auto-training trigger  `# 1.5 days`
  - File: `a2a/projection/auto_trainer.py`
  - **AC:** Model pair auto-discovery → read corpus → generate pairs → train → save

- [ ] **S4-T8** — `ProjectionRegistry`: model cache + load/save  `# 0.5 day`
  - File: `a2a/projection/registry.py`
  - **AC:** `get(src, tgt)` → cached model, `load(path)` → safetensors

- [ ] **S4-T9** — PluginManager projection integration (auto-resolve on route)  `# 1 day`
  - **AC:** `route_tensor()` → different target model → resolve projection → forward → send

- [ ] **S4-T10** — gRPC service: `RequestProjection` RPC implementation  `# 0.5 day`
  - **AC:** Server accepts projection request, triggers auto_train

- [ ] **S4-T11** — E2E integration: Llama-8B → Mistral-7B cross-model flow  `# 2 days`
  - **AC:** Llama LogReader → Mistral CodeFixer, projection auto-trained

### S4 Tests (15)

- [ ] ProjectionModel forward produces correct shape
- [ ] ProjectionModel forward preserves batch dim
- [ ] LinearProjection variant correct shape
- [ ] Training loop reduces loss over epochs
- [ ] Training improves cosine similarity
- [ ] Multi-objective loss components all decrease
- [ ] Dataset.from_corpus produces correct pairs count
- [ ] Dataset.from_corpus source/target dimensions differ
- [ ] Negative pairs have lower similarity than positive
- [ ] Auto-trainer completes successfully
- [ ] Auto-trainer saves safetensors file
- [ ] Registry caches loaded projection
- [ ] Registry returns None for unknown pair
- [ ] PluginManager resolves projection automatically
- [ ] Cross-model full pipeline (Llama→Mistral)

---

## Sprint 5 — Security, Rate Limiting, Monitoring

**Duration:** 3 weeks (15 days)
**Goal:** Production-ready — mTLS, JWT auth, rate limiting, Prometheus, structured logging

### Tasks

- [ ] **S5-T1** — mTLS: server-side SSL credentials + gRPC server integration  `# 1 day`
  - File: `a2a/security/tls.py`
  - **AC:** Server only accepts connections with valid client cert

- [ ] **S5-T2** — mTLS: client-side SSL credentials  `# 1 day`
  - **AC:** Client connects with valid cert, key, ca

- [ ] **S5-T3** — JWT auth: token create + validate  `# 0.5 day`
  - File: `a2a/security/auth.py`
  - **AC:** `create_token(agent_id, mesh_id, secret)` → JWT, `validate_token(token)` → payload

- [ ] **S5-T4** — JWT: gRPC client metadata token attach  `# 0.5 day`
  - **AC:** Every gRPC call carries JWT in metadata header

- [ ] **S5-T5** — JWT: gRPC server interceptor → token validation  `# 1 day`
  - **AC:** Invalid token → PERMISSION_DENIED error

- [ ] **S5-T6** — TokenBucket rate limiter  `# 1 day`
  - File: `a2a/monitoring/rate_limiter.py`
  - **AC:** `allow(agent_id)` → True if token available, False otherwise

- [ ] **S5-T7** — Rate limiter: per-agent + per-route, gRPC interceptor  `# 1 day`
  - **AC:** Checked by agent ID + semantic label, exceeded → RESOURCE_EXHAUSTED

- [ ] **S5-T8** — Backpressure / Flow control: `MSG_BACKPRESSURE` + logic  `# 1.5 days`
  - **AC:** Queue depth > threshold → SLOW_DOWN signal, empty → RESUME

- [ ] **S5-T9** — Prometheus metrics: counters, histograms, gauges  `# 1 day`
  - File: `a2a/monitoring/metrics.py`
  - **AC:** `/metrics` endpoint → Prometheus scrapeable

- [ ] **S5-T10** — Structured JSON logging  `# 0.5 day`
  - File: `a2a/utils/logging.py`
  - **AC:** timestamp + level + event + extra fields → JSON line

- [ ] **S5-T11** — Health endpoints: `/health`, `/health/ready`, `/health/live`  `# 0.5 day`
  - **AC:** GET /health → `{"status":"ok","plugins_loaded":2,"uptime":3600}`

- [ ] **S5-T12** — Security + rate limit integration test  `# 1 day`
  - **AC:** mTLS + JWT + rate limit all working simultaneously

### S5 Tests (15)

- [ ] TokenBucket consumes token correctly
- [ ] TokenBucket refills over time
- [ ] TokenBucket denies when empty
- [ ] RateLimiter respects per-agent config
- [ ] JWT create generates valid token
- [ ] JWT validate returns correct payload
- [ ] JWT validate rejects expired token
- [ ] JWT validate rejects wrong secret
- [ ] JWT validate checks mesh_id
- [ ] gRPC server rejects unauthenticated request
- [ ] gRPC client with valid token accepted
- [ ] Rate limit enforced in gRPC interceptor
- [ ] Backpressure signal sent when queue full
- [ ] Health endpoint returns expected fields
- [ ] JSON logger produces valid JSON

---

## Sprint 6 — Documentation, Integration, Release

**Duration:** 2 weeks (10 days)
**Goal:** PyPI package + Docker image + full documentation + demo

### Tasks

- [ ] **S6-T1** — `docs/index.md` — Overview, Quickstart (5 min setup)  `# 0.5 day`
- [ ] **S6-T2** — `docs/architecture.md` — Detailed architecture  `# 0.5 day`
- [ ] **S6-T3** — `docs/protocol.md` — Wire protocol specification  `# 1 day`
- [ ] **S6-T4** — `docs/plugins.md` — Plugin development guide  `# 1 day`
- [ ] **S6-T5** — `docs/config.md` — Full `a2a.yaml` reference  `# 1 day`
- [ ] **S6-T6** — `docs/deployment.md` — Docker + K8s deployment  `# 0.5 day`
- [ ] **S6-T7** — `docs/api/` — API reference (auto-generated)  `# 1 day`
- [ ] **S6-T8** — Demo: `examples/basic_mesh/` (2 plugins)  `# 1 day`
- [ ] **S6-T9** — Demo: `examples/multi_model/` (3 plugins, 2 models)  `# 1.5 days`
- [ ] **S6-T10** — PyPI release: build + twine upload  `# 0.5 day`
- [ ] **S6-T11** — Docker image: multi-stage build, push to ghcr  `# 0.5 day`

### S6 Tests (5)

- [ ] `examples/basic_mesh/start.sh` runs without error
- [ ] Basic mesh: log → code fix pipeline
- [ ] Multi-model mesh: 3 agents with auto-projection
- [ ] Docker image starts and `/health` returns ok
- [ ] PyPI wheel installs and CLI works

---

## Sprint 7+ — Go Transport Layer (Phase 2)

**Duration:** TBD (after Python SDK stabilizes)

- [x] Go module setup (`go.mod`, `go.sum`) — named `a2a-transport`
- [x] FlatBuffers Go codec (encode/decode + CRC32)
- [x] gRPC server in Go (SendTensor, StreamTensors, HealthCheck, Discover, backpressure)
- [x] gRPC client in Go (connection pool, HealthCheck, Discover)
- [x] Python ML Core Unix socket bridge server (BridgeServer)
- [x] Go ↔ Python integration test (9 tests, wire protocol validated)
- [x] Benchmark: Go codec (10.5µs encode, 11.9µs decode on Intel N150)

---

## MVP Definition

**MVP = Sprint 0 → Sprint 3 (12 weeks)**

| # | Requirement | Sprint |
|---|---|---|
| 1 | Single model family A2A (Llama ↔ Llama) | S2+ |
| 2 | gRPC + FlatBuffers transport | S1 |
| 3 | Hidden state extraction (last layer) + Prefix injection | S2 |
| 4 | Plugin Manager + 2 plugins (LogReader, CodeFixer) | S3 |
| 5 | Log → Code fix demo working | S3 |
| 6 | Text API vs A2A benchmark | S3 |

**v1.0 = Sprint 0 → Sprint 6 (22 weeks)**

- MVP + Projection Model (heterogeneous support) + Auto-training + Security + Release

---

## Summary by Sprint

| Sprint | Weeks | Days | Tasks | Status |
|---|---|---|---|---|
| S0 | 1–2 | 10 | 7 | ✅ Complete |
| S1 | 3–5 | 15 | 10 | ✅ Complete |
| S2 | 6–9 | 20 | 8 | ✅ Complete |
| S3 | 10–13 | 20 | 13 | ✅ Complete |
| S4 | 14–17 | 20 | 11 | ✅ Complete |
| S5 | 18–20 | 15 | 12 | ✅ Complete |
| S6 | 21–22 | 10 | 11 | ✅ Complete |
| S7+ | — | — | 7 | ✅ Complete |
| **Total** | **22** | **110** | **79** | |

---

## File Quick Reference

| File | Sprint | Task |
|---|---|---|
| `pyproject.toml` | S0 | S0-T1 |
| `.github/workflows/ci.yml` | S0 | S0-T2 |
| `.pre-commit-config.yaml` | S0 | S0-T3 |
| `a2a/__init__.py`, `a2a/_version.py` | S0 | S0-T4 |
| `a2a/cli.py` | S0 | S0-T5 |
| `a2a/protocol/a2a.proto` | S1 | S1-T1 |
| `a2a/transport/codec.py` | S1 | S1-T3..T5 |
| `a2a/transport/server.py` | S1 | S1-T6..T7 |
| `a2a/transport/client.py` | S1 | S1-T8 |
| `a2a/protocol/errors.py` | S1 | S1-T10 |
| `a2a/tensor/extractor.py` | S2 | S2-T1..T3 |
| `a2a/tensor/injector.py` | S2 | S2-T4..T5 |
| `a2a/tensor/dtype.py` | S2 | S2-T6 |
| `a2a/tensor/serializer.py` | S2 | S2-T7 |
| `a2a/agent/base.py` | S3 | S3-T1 |
| `a2a/agent/capabilities.py` | S3 | S3-T2 |
| `a2a/agent/manager.py` | S3 | S3-T3..T6 |
| `a2a/agent/router.py` | S3 | S3-T5 |
| `a2a/config/schema.py` | S3 | S3-T7 |
| `a2a/config/loader.py` | S3 | S3-T8 |
| `a2a/runtime.py` | S3 | S3-T10 |
| `a2a/plugins/log_reader/` | S3 | S3-T11 |
| `a2a/plugins/code_fixer/` | S3 | S3-T12 |
| `a2a/projection/adapter.py` | S4 | S4-T1..T2 |
| `a2a/projection/trainer.py` | S4 | S4-T3..T4 |
| `a2a/projection/dataset.py` | S4 | S4-T5..T6 |
| `a2a/projection/auto_trainer.py` | S4 | S4-T7 |
| `a2a/projection/registry.py` | S4 | S4-T8 |
| `a2a/security/tls.py` | S5 | S5-T1..T2 |
| `a2a/security/auth.py` | S5 | S5-T3 |
| `a2a/monitoring/rate_limiter.py` | S5 | S5-T6 |
| `a2a/monitoring/metrics.py` | S5 | S5-T9 |
| `a2a/utils/logging.py` | S5 | S5-T10 |
| `a2a/transport/bridge_server.py` | S7 | S7-T5 |
| `a2a-transport/cmd/a2a-transport/main.go` | S7 | S7-T1..T4 |
| `a2a-transport/internal/server/server.go` | S7 | S7-T3 |
| `a2a-transport/internal/client/client.go` | S7 | S7-T4 |
| `a2a-transport/internal/codec/flatbuffers.go` | S7 | S7-T2 |
| `a2a-transport/internal/bridge/unixsocket.go` | S7 | S7-T5 |
| `docs/*` | S6 | S6-T1..T7 |
| `examples/basic_mesh/` | S6 | S6-T8 |
| `examples/multi_model/` | S6 | S6-T9 |
| `Dockerfile` | S6 | S6-T11 |

---

> **Last updated:** 2026-07-15
