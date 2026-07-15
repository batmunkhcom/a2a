# A2A Protocol — Хөгжүүлэлтийн Ажлын Төлөвлөгөө

> **Төсөл:** A2A Protocol — AI-to-AI Latent Space Communication
> **Хувилбар:** 0.1.0.dev0 → 0.1.0
> **Нийт хугацаа:** 22 долоо хоног (~110 ажлын өдөр)
> **Хэл:** Python 3.11+
> **License:** Apache 2.0

---

## Төслийн Тойм

AI agent-ууд хоорондоо текст биш, **latent space дахь шууд вектор**-оор харилцах протокол. Tokenization болон текст парсинг-ийн overhead-г бүрэн арилгаж, харилцааны хурдыг **6.5x** нэмэгдүүлнэ.

| Үзүүлэлт | Одоогийн (Text API) | A2A Protocol |
|---|---|---|
| Хурд | ~1400ms | ~215ms |
| Bandwidth | ~2-4KB текст | 8KB тензор |
| Token | 700 токен | 0 токен |

---

## Агуулга

1. [Sprint 0 — Project Setup & Scaffolding](#sprint-0--project-setup--scaffolding)
2. [Sprint 1 — Wire Protocol + Transport Layer](#sprint-1--wire-protocol--transport-layer)
3. [Sprint 2 — Tensor Engine](#sprint-2--tensor-engine)
4. [Sprint 3 — Plugin System + Core Runtime](#sprint-3--plugin-system--core-runtime)
5. [Sprint 4 — Projection Model + Auto-training](#sprint-4--projection-model--auto-training)
6. [Sprint 5 — Security, Rate Limit, Monitoring](#sprint-5--security-rate-limit-monitoring)
7. [Sprint 6 — Documentation, Integration, Release](#sprint-6--documentation-integration-release)
8. [Sprint 7+ — Go Transport Layer (2-р үе)](#sprint-7--go-transport-layer-2-р-үе)
9. [MVP Тодорхойлолт](#mvp-тодорхойлолт)

---

## Sprint 0 — Project Setup & Scaffolding

**Хугацаа:** 2 долоо хоног (10 өдөр)
**Зорилго:** `pip install -e ".[dev]" ; a2a --help` ажилладаг skeleton бэлэн

### Tasks

- [x] **S0-T1** — Repository үүсгэх, pyproject.toml тохируулах  `# 1 өдөр`
  - Файлууд: `pyproject.toml`, `.gitignore`, `requirements.txt`, `requirements-dev.txt`
  - **AC:** `pip install -e ".[dev]"` амжилттай, `import a2a` алдаагүй

- [ ] **S0-T2** — CI/CD pipeline (GitHub Actions)  `# 1 өдөр`
  - Файлууд: `.github/workflows/ci.yml`
  - **AC:** Push хийхэд lint + test workflow асах

- [ ] **S0-T3** — Pre-commit hooks (ruff + mypy)  `# 0.5 өдөр`
  - Файлууд: `.pre-commit-config.yaml`
  - **AC:** `pre-commit run --all-files` амжилттай

- [ ] **S0-T4** — Package skeleton (бүх `__init__.py`, `_version.py`)  `# 1 өдөр`
  - Файлууд: `a2a/__init__.py`, `a2a/_version.py`, бүх дэд пакежийн `__init__.py`
  - **AC:** Бүх `__init__.py` байх, `from a2a.config import ...` ажиллах

- [ ] **S0-T5** — CLI skeleton (typer) + entry point  `# 1 өдөр`
  - Файлууд: `a2a/cli.py`
  - **AC:** `a2a --help`, `a2a serve --help`, `a2a discover --help`, `a2a config --help`

- [ ] **S0-T6** — README.md + developer guide бичих  `# 0.5 өдөр`
  - Файл: `README.md`
  - **AC:** Шинэ хүн README уншихад project-г асааж чадах

- [ ] **S0-T7** — Dockerfile skeleton  `# 0.5 өдөр`
  - Файлууд: `Dockerfile`, `.dockerignore`
  - **AC:** `docker build .` амжилттай

### S0 Тестүүд (5)

- [ ] `test_package.py` — `import a2a` works, version string present
- [ ] `test_cli.py` — CLI help prints without error
- [ ] `test_cli.py` — `a2a serve` prints "Starting..." stub
- [ ] `test_cli.py` — `a2a config validate` prints stub
- [ ] `test_package.py` — All subpackage imports work

### S0 Package Skeleton Бүтэц

```
a2a/
├── __init__.py              → __version__, public API exports
├── _version.py              → __version__ = "0.1.0.dev0"
├── cli.py                   → typer app (serve, discover, config, train, benchmark)
├── config/                  → A2AConfig, loader, defaults (S3 бөглөнө)
│   ├── __init__.py
│   ├── schema.py            → stub
│   ├── loader.py            → stub
│   └── defaults.py          → stub
├── transport/               → gRPC server/client, codec, discovery (S1 бөглөнө)
│   ├── __init__.py
│   ├── server.py            → stub
│   ├── client.py            → stub
│   ├── codec.py             → stub
│   └── discovery.py         → stub
├── tensor/                  → Extractor, injector, serializer, dtype (S2 бөглөнө)
│   ├── __init__.py
│   ├── extractor.py         → stub
│   ├── injector.py          → stub
│   ├── serializer.py        → stub
│   └── dtype.py             → stub
├── projection/              → Adapter, trainer, dataset, registry (S4 бөглөнө)
│   ├── __init__.py
│   ├── adapter.py           → stub
│   ├── trainer.py           → stub
│   ├── dataset.py           → stub
│   ├── auto_trainer.py      → stub
│   └── registry.py          → stub
├── agent/                   → BasePlugin, PluginManager, router (S3 бөглөнө)
│   ├── __init__.py
│   ├── base.py              → stub
│   ├── manager.py           → stub
│   ├── router.py            → stub
│   └── capabilities.py      → stub
├── protocol/                → Protobuf, messages, errors (S1 бөглөнө)
│   ├── __init__.py
│   ├── messages.py          → stub
│   ├── a2a.proto            → stub
│   └── errors.py            → stub
├── security/                → Auth, TLS (S5 бөглөнө)
│   ├── __init__.py
│   ├── auth.py              → stub
│   └── tls.py               → stub
├── monitoring/              → Metrics, rate limiter (S5 бөглөнө)
│   ├── __init__.py
│   ├── metrics.py           → stub
│   └── rate_limiter.py      → stub
├── plugins/                 → Built-in plugins (S3 бөглөнө)
│   ├── __init__.py
│   ├── log_reader/          → stub
│   └── code_fixer/          → stub
└── utils/                   → Logging, async helpers (S3, S5 бөглөнө)
    ├── __init__.py
    ├── logging.py           → stub
    └── async_utils.py       → stub
```

---

## Sprint 1 — Wire Protocol + Transport Layer

**Хугацаа:** 3 долоо хоног (15 өдөр)
**Зорилго:** gRPC-ээр хоёр процесс хооронд тензор дамжуулдаг болох

### Tasks

- [ ] **S1-T1** — Protobuf schema бичих (`a2a.proto`)  `# 1 өдөр`
  - Файл: `a2a/protocol/a2a.proto`
  - **AC:** `protoc` compile амжилттай, `a2a_pb2.py` + `a2a_pb2_grpc.py` үүснэ
  - Агуулга: TensorRequest, TensorResponse, A2AMetadata, Discover, Health, бүх RPCs

- [ ] **S1-T2** — Protobuf кодогенерац setup + Makefile комманд  `# 0.5 өдөр`
  - Файл: `Makefile` (proto target)
  - **AC:** `make proto` → generate, CI pipeline-д proto drift check

- [ ] **S1-T3** — FlatBuffers tensor codec: `encode_tensor()`  `# 1 өдөр`
  - Файл: `a2a/transport/codec.py`
  - **AC:** PyTorch tensor → bytes, shape + dtype хадгалагдана

- [ ] **S1-T4** — FlatBuffers tensor codec: `decode_tensor()`  `# 1 өдөр`
  - Файл: `a2a/transport/codec.py`
  - **AC:** bytes → PyTorch tensor, roundtrip алдаагүй (FP32, FP16, BF16, multi-dim)

- [ ] **S1-T5** — `codec.py` — edge cases: хоосон tensor, NaN/Inf, batch dim  `# 0.5 өдөр`
  - **AC:** Хоосон tensor → алдаа, NaN → алдаа (validate mode), batch дэмжинэ

- [ ] **S1-T6** — gRPC server: `A2AServicer` (SendTensor, HealthCheck)  `# 1.5 өдөр`
  - Файл: `a2a/transport/server.py`
  - **AC:** Сервер асахад port сонсоно, health check ажиллана

- [ ] **S1-T7** — gRPC server: `StreamTensors` (bidirectional stream)  `# 1 өдөр`
  - **AC:** Олон тензор дараалан илгээх, сервер бүгдийг хүлээж авах

- [ ] **S1-T8** — gRPC client: `A2AClient` (SendTensor, HealthCheck)  `# 1 өдөр`
  - Файл: `a2a/transport/client.py`
  - **AC:** Клиентээс тензор илгээхэд сервер хүлээж авна, accepted=true

- [ ] **S1-T9** — gRPC integration test: send-receive roundtrip  `# 1 өдөр`
  - **AC:** Сервер+клиент нэг процесс дотор асах, тензор илгээгдэж буцаж ирнэ

- [ ] **S1-T10** — Error handling: `MSG_ERROR` + error propagation  `# 1 өдөр`
  - Файл: `a2a/protocol/errors.py`
  - **AC:** Буруу dtype → error_code=101, тензоргүй → error, details дамжина

### S1 Тестүүд (17)

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

**Хугацаа:** 4 долоо хоног (20 өдөр)
**Зорилго:** HuggingFace model-ийн hidden state гаргаж, өөр model-д залгаж, generate хийх
**Test Model:** `sshleifer/tiny-gpt2` (69MB, CPU дээр хурдан)

### Tasks

- [ ] **S2-T1** — `TensorExtractor`: HF model hidden state extraction (register_forward_hook)  `# 2 өдөр`
  - Файл: `a2a/tensor/extractor.py`
  - **AC:** Текст өгөхөд (batch, seq, hidden_dim) shaped tensor гарна

- [ ] **S2-T2** — `TensorExtractor`: pooling strategies (last, mean, max)  `# 1 өдөр`
  - **AC:** pool="last" → (1, hidden), pool="mean" → (1, hidden), өөр утгатай

- [ ] **S2-T3** — `TensorExtractor`: layer selection (default -1, arbitrary layer)  `# 0.5 өдөр`
  - **AC:** layer=0 → эхний давхарга, layer=-1 → сүүлийн

- [ ] **S2-T4** — `TensorInjector`: prefix injection (vector → embedding prefix → generate)  `# 2 өдөр`
  - Файл: `a2a/tensor/injector.py`
  - **AC:** Тензор + prompt → model.generate() → текст гарна, prompt-той холбоотой

- [ ] **S2-T5** — `TensorInjector`: cross-attention injection хувилбар  `# 1.5 өдөр`
  - **AC:** `inputs_embeds` биш, cross-attention key/value-р залгах

- [ ] **S2-T6** — `dtype.py`: FP32↔FP16↔BF16 хөрвүүлэлт + validate_tensor (NaN/Inf)  `# 0.5 өдөр`
  - Файл: `a2a/tensor/dtype.py`
  - **AC:** `convert_dtype(tensor, "float16")` → FP16, NaN илрүүлэлт зөв

- [ ] **S2-T7** — `serializer.py`: Safetensors save/load  `# 0.5 өдөр`
  - Файл: `a2a/tensor/serializer.py`
  - **AC:** `save_to_bytes(tensor)` → bytes, `load_from_bytes(data)` → tensor, roundtrip

- [ ] **S2-T8** — Integration test: extract → encode → decode → inject → generate  `# 2 өдөр`
  - **AC:** Full pipeline, semantic consistency (output нь input-тай хамааралтай)

### S2 Тестүүд (17)

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

**Хугацаа:** 4 долоо хоног (20 өдөр)
**Зорилго:** `a2a serve` → 2 demo plugin ачаалж, тензороор харилцаж, код засвар гарах

### Tasks

- [ ] **S3-T1** — `BasePlugin` abstract class бөглөх  `# 0.5 өдөр`
  - Файл: `a2a/agent/base.py`
  - **AC:** plugin_id, plugin_name, version, listens_to, emits, get_capabilities, on_receive_tensor, extract_tensor, initialize, lifecycle hooks

- [ ] **S3-T2** — `Capability` + `ModelInfo` dataclass  `# 0.5 өдөр`
  - Файл: `a2a/agent/capabilities.py`
  - **AC:** Plugin өөрийгөө тодорхойлох + сүлжээгээр зарлах

- [ ] **S3-T3** — `PluginManager`: plugin loading (importlib + inspect)  `# 1 өдөр`
  - Файл: `a2a/agent/manager.py`
  - **AC:** `load_plugin(entry, global_config)` → plugin instance + initialize дуудагдана

- [ ] **S3-T4** — `PluginManager`: plugin registration + label routing  `# 0.5 өдөр`
  - **AC:** `register(plugin)` → `listens_to()` label-ууд routing table-д нэмэгдэнэ

- [ ] **S3-T5** — `PluginManager`: `route_tensor(tensor, metadata)`  `# 1 өдөр`
  - Файл: `a2a/agent/router.py`
  - **AC:** semantic_label → тохирох plugin(ууд) → `on_receive_tensor()` дуудагдана

- [ ] **S3-T6** — `PluginManager`: plugin-local config loading (`config.yaml`)  `# 0.5 өдөр`
  - **AC:** Plugin-ийн сан доторх `config.yaml`-г уншиж `initialize()`-д дамжуулна

- [ ] **S3-T7** — `A2AConfig` Pydantic model бөглөх  `# 1 өдөр`
  - Файл: `a2a/config/schema.py`
  - **AC:** `A2AConfig.from_yaml(path)` → validate хийгдсэн config объект

- [ ] **S3-T8** — Config loader: хайлтын дараалал, env override  `# 0.5 өдөр`
  - Файл: `a2a/config/loader.py`
  - **AC:** `A2A_CONFIG` env → `./a2a.yaml` → `~/.config/a2a/` → `/etc/a2a/`

- [ ] **S3-T9** — `a2a.yaml` demo config бичих  `# 0.5 өдөр`
  - **AC:** models, plugins, routes гэсэн 3 section-той demo config

- [ ] **S3-T10** — `A2ARuntime` orchestrator  `# 1 өдөр`
  - Файл: `a2a/runtime.py`
  - **AC:** `start()` → config унших → plugin ачаалах → transport эхлүүлэх

- [ ] **S3-T11** — `LogReaderPlugin` бөглөх  `# 1 өдөр`
  - Файлууд: `a2a/plugins/log_reader/plugin.py`, `prompts.py`, `config.yaml`
  - **AC:** Лог текст → `extract_tensor()` → error_context тензор гарна

- [ ] **S3-T12** — `CodeFixerPlugin` бөглөх  `# 1 өдөр`
  - Файлууд: `a2a/plugins/code_fixer/plugin.py`, `prompts.py`, `config.yaml`
  - **AC:** error_context тензор → `on_receive_tensor()` → inject → generate → код patch

- [ ] **S3-T13** — End-to-end integration test  `# 1 өдөр`
  - **AC:** Runtime бүрэн ачаалагдаж, лог → код засвар flow амжилттай

### S3 Тестүүд (19)

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

**Хугацаа:** 4 долоо хоног (20 өдөр)
**Зорилго:** Llama ↔ Mistral хоёр өөр модель тензороор харилцах

### Tasks

- [ ] **S4-T1** — `ProjectionModel` (3-layer MLP + Residual + LayerNorm)  `# 1 өдөр`
  - Файл: `a2a/projection/adapter.py`
  - **AC:** `forward(src_tensor)` → (target_dim,) shape зөв

- [ ] **S4-T2** — `ProjectionModel` variant A (Linear)  `# 0.5 өдөр`
  - **AC:** dims ижил үед identity, ялгаатай үед шугаман mapping

- [ ] **S4-T3** — `ProjectionTrainer`: contrastive loss (InfoNCE)  `# 1.5 өдөр`
  - Файл: `a2a/projection/trainer.py`
  - **AC:** Training loop → loss буурна, cosine similarity өснө

- [ ] **S4-T4** — `ProjectionTrainer`: multi-objective loss (MSE + Cosine + Contrastive)  `# 0.5 өдөр`
  - **AC:** Гурван loss бүрэлдэхүүн, weighted sum зөв
  - `loss = contrastive_loss + 0.1 * mse_loss + 0.01 * cosine_loss`

- [ ] **S4-T5** — `ProjectionPairDataset`: корпус → (src_hidden, tgt_hidden) хосууд  `# 2 өдөр`
  - Файл: `a2a/projection/dataset.py`
  - **AC:** Текст мөр бүрээс хоёр model-ийн hidden state гаргаж хос үүсгэнэ

- [ ] **S4-T6** — `ProjectionPairDataset`: negative pair generation  `# 0.5 өдөр`
  - **AC:** Өөр өөр текстийн hidden state → negative pair

- [ ] **S4-T7** — `auto_trainer.py`: runtime auto-training trigger  `# 1.5 өдөр`
  - Файл: `a2a/projection/auto_trainer.py`
  - **AC:** Model pair auto-discovery → корпус унших → хос үүсгэх → сургах → хадгалах

- [ ] **S4-T8** — `ProjectionRegistry`: model cache + load/save  `# 0.5 өдөр`
  - Файл: `a2a/projection/registry.py`
  - **AC:** `get(src, tgt)` → cached model, `load(path)` → safetensors

- [ ] **S4-T9** — PluginManager-д projection integration (route үед auto-resolve)  `# 1 өдөр`
  - **AC:** `route_tensor()` → target өөр бол projection resolve → forward → илгээх

- [ ] **S4-T10** — gRPC service-д `RequestProjection` RPC бөглөх  `# 0.5 өдөр`
  - **AC:** Сервер projection хүсэлт хүлээж авах, auto_train trigger хийх

- [ ] **S4-T11** — E2E integration: Llama-8B → Mistral-7B cross-model flow  `# 2 өдөр`
  - **AC:** Llama дээрх LogReader → Mistral дээрх CodeFixer, projection auto-surgach

### S4 Тестүүд (15)

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

## Sprint 5 — Security, Rate Limit, Monitoring

**Хугацаа:** 3 долоо хоног (15 өдөр)
**Зорилго:** Үйлдвэрлэлд бэлэн — mTLS, JWT auth, rate limiting, Prometheus, structured logging

### Tasks

- [ ] **S5-T1** — mTLS: server-side SSL credentials + gRPC server залгах  `# 1 өдөр`
  - Файл: `a2a/security/tls.py`
  - **AC:** Сервер зөвхөн valid client cert-тэй холболт хүлээж авна

- [ ] **S5-T2** — mTLS: client-side SSL credentials  `# 1 өдөр`
  - **AC:** Клиент valid cert, key, ca-тай серверт холбогдоно

- [ ] **S5-T3** — JWT auth: token create + validate  `# 0.5 өдөр`
  - Файл: `a2a/security/auth.py`
  - **AC:** `create_token(agent_id, mesh_id, secret)` → JWT, `validate_token(token)` → payload

- [ ] **S5-T4** — JWT: gRPC client metadata-д token attach  `# 0.5 өдөр`
  - **AC:** gRPC call бүрд metadata header-т JWT token очно

- [ ] **S5-T5** — JWT: gRPC server interceptor → token validate  `# 1 өдөр`
  - **AC:** Token хүчингүй бол PERMISSION_DENIED error буцаана

- [ ] **S5-T6** — TokenBucket rate limiter  `# 1 өдөр`
  - Файл: `a2a/monitoring/rate_limiter.py`
  - **AC:** `allow(agent_id)` → bucket-д token байвал True, үгүй бол False

- [ ] **S5-T7** — Rate limiter: per-agent + per-route, gRPC interceptor  `# 1 өдөр`
  - **AC:** Agent ID + semantic label-аар check, хэтэрвэл RESOURCE_EXHAUSTED

- [ ] **S5-T8** — Backpressure / Flow control: `MSG_BACKPRESSURE` + логик  `# 1.5 өдөр`
  - **AC:** Queue depth>threshold → SLOW_DOWN дохио, queue хоосорвол RESUME

- [ ] **S5-T9** — Prometheus metrics: counters, histograms, gauges  `# 1 өдөр`
  - Файл: `a2a/monitoring/metrics.py`
  - **AC:** `/metrics` endpoint → Prometheus scrape боломжтой

- [ ] **S5-T10** — Structured JSON logging  `# 0.5 өдөр`
  - Файл: `a2a/utils/logging.py`
  - **AC:** timestamp + level + event + extra fields → JSON line

- [ ] **S5-T11** — Health check endpoint: `/health`, `/health/ready`, `/health/live`  `# 0.5 өдөр`
  - **AC:** GET /health → `{"status":"ok","plugins_loaded":2,"uptime":3600}`

- [ ] **S5-T12** — Security + rate limit integration test  `# 1 өдөр`
  - **AC:** mTLS + JWT + rate limit бүгд зэрэг ажиллах

### S5 Тестүүд (15)

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

**Хугацаа:** 2 долоо хоног (10 өдөр)
**Зорилго:** PyPI package + Docker image + бүрэн documentation + demo

### Tasks

- [ ] **S6-T1** — `docs/index.md` — Overview, Quickstart (5 минутанд ажиллуулах)  `# 0.5 өдөр`
- [ ] **S6-T2** — `docs/architecture.md` — Архитектурын дэлгэрэнгүй  `# 0.5 өдөр`
- [ ] **S6-T3** — `docs/protocol.md` — Wire protocol specification  `# 1 өдөр`
- [ ] **S6-T4** — `docs/plugins.md` — Plugin хөгжүүлэх гарын авлага  `# 1 өдөр`
- [ ] **S6-T5** — `docs/config.md` — `a2a.yaml` бүрэн reference  `# 1 өдөр`
- [ ] **S6-T6** — `docs/deployment.md` — Docker + K8s deployment  `# 0.5 өдөр`
- [ ] **S6-T7** — `docs/api/` — API reference (auto-generated)  `# 1 өдөр`
- [ ] **S6-T8** — Demo: `examples/basic_mesh/` (2 plugin)  `# 1 өдөр`
- [ ] **S6-T9** — Demo: `examples/multi_model/` (3 plugin, 2 модель)  `# 1.5 өдөр`
- [ ] **S6-T10** — PyPI release: build + twine upload  `# 0.5 өдөр`
- [ ] **S6-T11** — Docker image: multi-stage build, push to ghcr  `# 0.5 өдөр`

### S6 Тестүүд (5)

- [ ] `examples/basic_mesh/start.sh` runs without error
- [ ] Basic mesh: log → code fix pipeline
- [ ] Multi-model mesh: 3 agents with auto-projection
- [ ] Docker image starts and `/health` returns ok
- [ ] PyPI wheel installs and CLI works

---

## Sprint 7+ — Go Transport Layer (2-р үе)

**Хугацаа:** Төлөвлөгөөгүй (Python SDK тогтворжсоны дараа)

- [ ] Go модуль үүсгэх (`go.mod`, `go.sum`) — `a2a-transport` нэртэй
- [ ] FlatBuffers Go кодогенерац (`.fbs` schema → Go structs)
- [ ] gRPC/QUIC сервер Go дээр (Python ML Core-той unix socket-ээр харилцана)
- [ ] gRPC/QUIC клиент Go дээр (өндөр concurrency)
- [ ] Go ↔ Python integration test
- [ ] Benchmark: Go transport vs Python transport

---

## MVP Тодорхойлолт

**MVP = Sprint 0 → Sprint 3 (12 долоо хоног)**

| # | Шаардлага | Sprint |
|---|---|---|
| 1 | Нэг загварын гэр бүл доторх A2A (Llama ↔ Llama) | S2+ |
| 2 | gRPC + FlatBuffers тээвэрлэлт | S1 |
| 3 | Hidden state extraction (сүүлийн давхарга) + Prefix injection | S2 |
| 4 | Plugin Manager + 2 plugin (LogReader, CodeFixer) | S3 |
| 5 | Log → Code fix demo ажиллах | S3 |
| 6 | Text API vs A2A benchmark | S3 |

**v1.0 = Sprint 0 → Sprint 6 (22 долоо хоног)**

- MVP + Projection Model (heterogeneous дэмжлэг) + Auto-training + Security + Release

---

## Өдрийн Хураангуй

| Sprint | Долоо хоног | Өдөр | Даалгавар | Төлөв |
|---|---|---|---|---|
| S0 | 1-2 | 10 | Project setup, CI/CD, scaffolding | 🔄 In Progress |
| S1 | 3-5 | 15 | Wire protocol + Transport layer | ⬜ Pending |
| S2 | 6-9 | 20 | Tensor Engine | ⬜ Pending |
| S3 | 10-13 | 20 | Plugin System + Core Runtime | ⬜ Pending |
| S4 | 14-17 | 20 | Projection Model + Auto-training | ⬜ Pending |
| S5 | 18-20 | 15 | Security, Rate Limit, Monitoring | ⬜ Pending |
| S6 | 21-22 | 10 | Docs, Integration, Release | ⬜ Pending |
| S7+ | — | — | Go Transport Layer | ⬜ Pending |
| **Нийт** | **22** | **110** | **72 tasks** | |

---

## Файлын Хурдан Лавлах

| Файл | Sprint | Task |
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
| `docs/*` | S6 | S6-T1..T7 |
| `examples/basic_mesh/` | S6 | S6-T8 |
| `examples/multi_model/` | S6 | S6-T9 |
| `Dockerfile` | S6 | S6-T11 |

---

> **Сүүлийн шинэчлэл:** 2026-07-15
