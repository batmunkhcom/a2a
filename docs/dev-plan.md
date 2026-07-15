# A2A Protocol — Хөгжүүлэлтийн Нарийвчилсан Төлөвлөгөө

**Хувилбар:** 1.0 | **Нийт хугацаа:** 22 долоо хоног (~110 ажлын өдөр) | **Хэл:** Python 3.11+

---

## Хураангуй

| Sprint | Долоо хоног | Өдөр | Даалгавар | Төлөв |
|---|---|---|---|---|
| S0 | 1-2 | 10 | 7 | Project setup, CI/CD, scaffolding |
| S1 | 3-5 | 15 | 10 | Wire protocol + Transport layer |
| S2 | 6-9 | 20 | 8 | Tensor Engine |
| S3 | 10-13 | 20 | 13 | Plugin System + Core Runtime |
| S4 | 14-17 | 20 | 11 | Projection Model + Auto-training |
| S5 | 18-20 | 15 | 12 | Security, Rate Limit, Monitoring |
| S6 | 21-22 | 10 | 11 | Docs, Integration, Release |
| **Нийт** | **22w** | **110d** | **72** | |

---

## Sprint 0: Project Setup & Scaffolding

**Хугацаа:** 2 долоо хоног (10 өдөр)
**Зорилго:** `pip install -e ".[dev]" ; a2a --help` ажилладаг skeleton бэлэн байх

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Файлууд | Acceptance Criteria |
|---|---|---|---|---|---|
| **S0-T1** | Repository үүсгэх, pyproject.toml тохируулах | 1 | — | `pyproject.toml`, `setup.cfg`, `requirements.txt`, `requirements-dev.txt`, `.gitignore` | `pip install -e ".[dev]"` амжилттай, `import a2a` алдаагүй |
| **S0-T2** | CI/CD pipeline (GitHub Actions) | 1 | S0-T1 | `.github/workflows/ci.yml` | Push хийхэд lint + test workflow асах |
| **S0-T3** | Pre-commit hooks (ruff + mypy) | 0.5 | S0-T1 | `.pre-commit-config.yaml` | `pre-commit run --all-files` амжилттай |
| **S0-T4** | Package skeleton (бүх `__init__.py`, `_version.py`) | 1 | S0-T1 | `a2a/__init__.py`, `a2a/_version.py`, `a2a/config/__init__.py`, `a2a/transport/__init__.py`, `a2a/tensor/__init__.py`, `a2a/projection/__init__.py`, `a2a/agent/__init__.py`, `a2a/protocol/__init__.py`, `a2a/plugins/__init__.py`, `a2a/utils/__init__.py` | Бүх `__init__.py` файлууд байх, `from a2a.config import ...` ажиллах |
| **S0-T5** | CLI skeleton (click/typer) + entry point | 1 | S0-T4 | `a2a/cli.py`, `__init__.py`-д export | Terminal дээр `a2a --help` → help текст харагдах, `a2a serve --help`, `a2a config --help`, `a2a discover --help` |
| **S0-T6** | README.md + developer guide бичих | 0.5 | S0-T5 | `README.md` | Шинэ хүн README уншихад project-г асааж чадах |
| **S0-T7** | Dockerfile skeleton | 0.5 | S0-T1 | `Dockerfile`, `.dockerignore` | `docker build .` амжилттай |

### S0-T4 дэлгэрэнгүй — Package skeleton файлууд

```
a2a/
├── __init__.py              → from a2a._version import __version__
├── _version.py              → __version__ = "0.1.0.dev0"
├── cli.py                   → click.Group + subcommands (serve, discover, config, train, benchmark)
├── config/
│   ├── __init__.py          → empty
│   ├── schema.py            → stub (Pydantic models — Sprint 3-т бөглөнө)
│   ├── loader.py            → stub (load_config() — Sprint 3-т бөглөнө)
│   └── defaults.py          → stub (default values — Sprint 3-т бөглөнө)
├── transport/
│   ├── __init__.py
│   ├── server.py            → stub (A2AServer — Sprint 1-д бөглөнө)
│   ├── client.py            → stub (A2AClient — Sprint 1-д бөглөнө)
│   ├── codec.py             → stub (FlatBuffers encode/decode — Sprint 1-д бөглөнө)
│   └── discovery.py         → stub (Sprint 5-д бөглөнө)
├── tensor/
│   ├── __init__.py
│   ├── extractor.py         → stub (Sprint 2-т бөглөнө)
│   ├── injector.py          → stub (Sprint 2-т бөглөнө)
│   ├── serializer.py        → stub (Sprint 2-т бөглөнө)
│   └── dtype.py             → stub (Sprint 2-т бөглөнө)
├── projection/
│   ├── __init__.py
│   ├── adapter.py           → stub (Sprint 4-д бөглөнө)
│   ├── trainer.py           → stub (Sprint 4-д бөглөнө)
│   ├── dataset.py           → stub (Sprint 4-д бөглөнө)
│   ├── auto_trainer.py      → stub (Sprint 4-д бөглөнө)
│   └── registry.py          → stub (Sprint 4-д бөглөнө)
├── agent/
│   ├── __init__.py
│   ├── base.py              → stub (Sprint 3-т бөглөнө)
│   ├── manager.py           → stub (Sprint 3-т бөглөнө)
│   ├── router.py            → stub (Sprint 3-т бөглөнө)
│   └── capabilities.py      → stub (Sprint 3-т бөглөнө)
├── protocol/
│   ├── __init__.py
│   ├── messages.py          → stub (Sprint 1-д бөглөнө)
│   ├── a2a.proto            → stub (Sprint 1-д бөглөнө)
│   └── errors.py            → stub (Sprint 1-д бөглөнө)
├── security/
│   ├── __init__.py
│   ├── auth.py              → stub (Sprint 5-д бөглөнө)
│   └── tls.py               → stub (Sprint 5-д бөглөнө)
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py           → stub (Sprint 5-д бөглөнө)
│   └── tracing.py           → stub (Sprint 5-д бөглөнө)
├── plugins/
│   ├── __init__.py
│   ├── log_reader/
│   │   ├── __init__.py
│   │   ├── plugin.py        → stub (Sprint 3-т бөглөнө)
│   │   ├── config.yaml      → Sprint 3-т бөглөнө
│   │   └── prompts.py       → stub (Sprint 3-т бөглөнө)
│   └── code_fixer/
│       ├── __init__.py
│       ├── plugin.py        → stub (Sprint 3-т бөглөнө)
│       ├── config.yaml      → Sprint 3-т бөглөнө
│       └── prompts.py       → stub (Sprint 3-т бөглөнө)
└── utils/
    ├── __init__.py
    ├── logging.py           → stub (Sprint 5-д бөглөнө)
    └── async_utils.py        → stub (Sprint 3-т бөглөнө)
```

### S0 тестүүд

| ID | Тест | Файл |
|---|---|---|
| S0-TEST-1 | `import a2a` works, version string present | `tests/test_package.py` |
| S0-TEST-2 | CLI help prints without error | `tests/test_cli.py` |
| S0-TEST-3 | `a2a serve` prints "Starting..." (stub) | `tests/test_cli.py` |
| S0-TEST-4 | `a2a config validate` prints stub message | `tests/test_cli.py` |
| S0-TEST-5 | All subpackage imports work | `tests/test_package.py` |

### S0 хийсний дараах шалгалт

```bash
# Terminal дээр:
$ python -c "import a2a; print(a2a.__version__)"   # → 0.1.0.dev0
$ a2a --help                                        # → help текст
$ a2a --version                                     # → 0.1.0.dev0
$ a2a serve --help                                  # → serve subcommand help
$ a2a config validate                               # → stub message
$ a2a discover                                      # → stub message
$ pre-commit run --all-files                        # → passes
$ pytest tests/ -v                                  # → 5 tests green
```

---

## Sprint 1: Wire Protocol + Transport Layer

**Хугацаа:** 3 долоо хоног (15 өдөр)
**Зорилго:** gRPC-ээр хоёр процесс хооронд тензор дамжуулдаг болох

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Acceptance Criteria |
|---|---|---|---|---|
| **S1-T1** | Protobuf schema бичих (`a2a.proto`) | 1 | S0-T4 | `protoc` compile амжилттай, `a2a_pb2.py` + `a2a_pb2_grpc.py` үүснэ |
| **S1-T2** | Protobuf кодогенерац setup + Makefile комманд | 0.5 | S1-T1 | `make proto` → generate, CI pipeline-д proto drift check |
| **S1-T3** | FlatBuffers tensor codec: `encode_tensor()` | 1 | — | PyTorch tensor → bytes, shape + dtype хадгалагдана |
| **S1-T4** | FlatBuffers tensor codec: `decode_tensor()` | 1 | S1-T3 | bytes → PyTorch tensor, roundtrip алдаагүй (FP32, FP16, BF16, олон хэмжээст) |
| **S1-T5** | `codec.py` — edge cases: хоосон tensor, NaN/Inf, batch dim | 0.5 | S1-T4 | Хоосон tensor → алдаа, NaN → алдаа (validate mode), batch дэмжинэ |
| **S1-T6** | gRPC server: `A2AServicer` (SendTensor, HealthCheck) | 1.5 | S1-T2, S1-T4 | Сервер асахад port сонсоно, health check ажиллана |
| **S1-T7** | gRPC server: `StreamTensors` (bidirectional stream) | 1 | S1-T6 | Олон тензор дараалан илгээх, сервер бүгдийг хүлээж авах |
| **S1-T8** | gRPC client: `A2AClient` (SendTensor, HealthCheck) | 1 | S1-T6 | Клиентээс тензор илгээхэд сервер хүлээж авна, accepted=true |
| **S1-T9** | gRPC integration test: send-receive roundtrip | 1 | S1-T8 | Сервер+клиент нэг процесс дотор асах, тензор илгээгдэж буцаж ирнэ |
| **S1-T10** | Error handling: `MSG_ERROR` protobuf type + сервер тал error propagation | 1 | S1-T8 | Буруу dtype → error_code=101, тензоргүй хүсэлт → error, error_details дамжина |

### S1 тестүүд

| ID | Тест | Файл | Тоо |
|---|---|---|---|
| S1-TEST-1 | `test_encode_fp32_tensor` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-2 | `test_encode_fp16_tensor` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-3 | `test_encode_bf16_tensor` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-4 | `test_encode_multidim_tensor` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-5 | `test_encode_empty_tensor_raises` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-6 | `test_decode_roundtrip_fp32` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-7 | `test_decode_roundtrip_fp16` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-8 | `test_decode_roundtrip_bf16` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-9 | `test_decode_roundtrip_multidim` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-10 | `test_decode_invalid_header_raises` | `tests/transport/test_codec.py` | 1 |
| S1-TEST-11 | `test_grpc_health_check` | `tests/transport/test_grpc.py` | 1 |
| S1-TEST-12 | `test_grpc_send_tensor_large` | `tests/transport/test_grpc.py` | 1 |
| S1-TEST-13 | `test_grpc_stream_tensors` | `tests/transport/test_grpc.py` | 1 |
| S1-TEST-14 | `test_grpc_send_tensor_error_on_empty` | `tests/transport/test_grpc.py` | 1 |
| S1-TEST-15 | `test_grpc_send_tensor_error_on_nan` | `tests/transport/test_grpc.py` | 1 |
| S1-TEST-16 | `test_grpc_client_reconnect` | `tests/transport/test_grpc.py` | 1 |
| S1-TEST-17 | `test_protobuf_metadata_serialization` | `tests/protocol/test_messages.py` | 1 |
| **Нийт** | | | **17** |

### S1 хийсний дараах шалгалт

```bash
# Terminal дээр (2 terminal):
# Terminal 1:
$ a2a serve --port 9090                          # сервер асах

# Terminal 2:
$ a2a discover                                   # серверийг олж харах
$ python scripts/send_test_tensor.py             # demo tensor илгээх
# → Accepted: True, roundtrip verified

$ pytest tests/transport/ -v                     # 17 tests green
```

---

## Sprint 2: Tensor Engine

**Хугацаа:** 4 долоо хоног (20 өдөр)
**Зорилго:** HuggingFace model-ийн hidden state гаргаж, өөр model-д залгаж, үргэлжлүүлэн generate хийх

### Чухал тэмдэглэл

**Энэ sprint-д бодит HF model ашиглана.** Test-д `sshleifer/tiny-gpt2` (69MB) эсвэл `distilgpt2` (330MB) ашиглах. Үйлдвэрлэлд Llama-3-8B.

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Acceptance Criteria |
|---|---|---|---|---|
| **S2-T1** | `TensorExtractor`: HF model-ийн hidden state гаргах (register_forward_hook) | 2 | — | Текст өгөхөд (batch, seq, hidden_dim) shaped tensor гарна |
| **S2-T2** | `TensorExtractor`: pooling strategies (last, mean, max) | 1 | S2-T1 | pool="last" → (1, hidden), pool="mean" → (1, hidden), өөр өөр утгатай |
| **S2-T3** | `TensorExtractor`: layer selection (default -1, arbitrary layer) | 0.5 | S2-T1 | layer=0 → эхний давхарга, layer=-1 → сүүлийн, tokenizer context тохируулж өгөх |
| **S2-T4** | `TensorInjector`: prefix injection (vector → embedding prefix → generate) | 2 | S2-T1 | Тензор + prompt → model.generate() → текст гарна. Гарсан текст нь prompt-той холбоотой |
| **S2-T5** | `TensorInjector`: cross-attention injection хувилбар | 1.5 | S2-T4 | `inputs_embeds` биш, cross-attention key/value-р залгах |
| **S2-T6** | `dtype.py`: FP32↔FP16↔BF16 хөрвүүлэлт + validate_tensor (NaN/Inf check) | 0.5 | — | `convert_dtype(tensor, "float16")` → FP16, NaN илрүүлэлт зөв |
| **S2-T7** | `serializer.py`: Safetensors save/load tensor to/from bytes | 0.5 | — | `save_to_bytes(tensor)` → bytes, `load_from_bytes(data)` → tensor, roundtrip зөв |
| **S2-T8** | Tensor engine integration test: extract → encode → decode → inject → generate | 2 | S2-T4, S2-T7 | Full pipeline: текст → hidden → bytes → hidden → текст. Semantic consistency шалгах (output нь input-тай хамааралтай эсэх) |

### S2 тестүүд

| ID | Тест | Файл | Тоо |
|---|---|---|---|
| S2-TEST-1 | `test_extract_shape_last_pooling` | `tests/tensor/test_extractor.py` | 1 |
| S2-TEST-2 | `test_extract_shape_mean_pooling` | `tests/tensor/test_extractor.py` | 1 |
| S2-TEST-3 | `test_extract_shape_max_pooling` | `tests/tensor/test_extractor.py` | 1 |
| S2-TEST-4 | `test_extract_different_layers_produce_different_output` | `tests/tensor/test_extractor.py` | 1 |
| S2-TEST-5 | `test_extract_long_text` | `tests/tensor/test_extractor.py` | 1 |
| S2-TEST-6 | `test_extract_empty_text_raises` | `tests/tensor/test_extractor.py` | 1 |
| S2-TEST-7 | `test_inject_and_generate_produces_output` | `tests/tensor/test_injector.py` | 1 |
| S2-TEST-8 | `test_inject_with_different_prompt` | `tests/tensor/test_injector.py` | 1 |
| S2-TEST-9 | `test_inject_tensor_shape_mismatch_handled` | `tests/tensor/test_injector.py` | 1 |
| S2-TEST-10 | `test_convert_dtype_fp32_to_fp16` | `tests/tensor/test_dtype.py` | 1 |
| S2-TEST-11 | `test_convert_dtype_fp16_to_fp32` | `tests/tensor/test_dtype.py` | 1 |
| S2-TEST-12 | `test_validate_nan_rejected` | `tests/tensor/test_dtype.py` | 1 |
| S2-TEST-13 | `test_validate_inf_rejected` | `tests/tensor/test_dtype.py` | 1 |
| S2-TEST-14 | `test_validate_clean_tensor_accepted` | `tests/tensor/test_dtype.py` | 1 |
| S2-TEST-15 | `test_safetensors_save_load_roundtrip` | `tests/tensor/test_serializer.py` | 1 |
| S2-TEST-16 | `test_safetensors_multidim_tensor` | `tests/tensor/test_serializer.py` | 1 |
| S2-TEST-17 | `test_extract_to_encode_to_decode_to_inject_pipeline` | `tests/integration/test_tensor_pipeline.py` | 1 |
| **Нийт** | | | **17** |

### S2 хийсний дараах шалгалт

```bash
$ pytest tests/tensor/ -v                           # 16 tests green
$ pytest tests/integration/test_tensor_pipeline.py  # 1 test green
$ python scripts/demo_extract_inject.py             # "Текст → hidden → generate → output" ажиллана
```

---

## Sprint 3: Plugin System + Core Runtime

**Хугацаа:** 4 долоо хоног (20 өдөр)
**Зорилго:** `a2a serve` → PluginManager 2 demo plugin ачаалж, тензороор харилцаж, код засвар гарах

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Acceptance Criteria |
|---|---|---|---|---|
| **S3-T1** | `BasePlugin` abstract class (Sprint 0 stub-г бөглөх) | 0.5 | S0-T4 | plugin_id, plugin_name, version, listens_to, emits, get_capabilities, on_receive_tensor, extract_tensor, initialize, lifecycle hooks |
| **S3-T2** | `Capability` + `ModelInfo` dataclass | 0.5 | — | Давхар хэрэглэгдэнэ — plugin өөрийгөө тодорхойлох, сүлжээгээр зарлах |
| **S3-T3** | `PluginManager`: plugin loading (importlib + inspect) | 1 | S3-T1 | `load_plugin(entry, global_config)` → plugin instance + initialize дуудагдана |
| **S3-T4** | `PluginManager`: plugin registration + label routing | 0.5 | S3-T3 | `register(plugin)` → `listens_to()` label-ууд routing table-д нэмэгдэнэ |
| **S3-T5** | `PluginManager`: `route_tensor(tensor, metadata)` | 1 | S3-T4 | semantic_label → тохирох plugin(ууд) → `on_receive_tensor()` дуудагдана |
| **S3-T6** | `PluginManager`: plugin-local config loading (`config.yaml`) | 0.5 | S3-T3 | Plugin-ийн сан доторх `config.yaml`-г уншиж `initialize()`-д дамжуулна |
| **S3-T7** | `A2AConfig` Pydantic model (Sprint 0 stub-г бөглөх) | 1 | — | `A2AConfig.from_yaml(path)` → validate хийгдсэн config объект |
| **S3-T8** | `Config loader`: хайлтын дараалал, env override | 0.5 | S3-T7 | `A2A_CONFIG` env → `./a2a.yaml` → `~/.config/a2a/` → `/etc/a2a/` |
| **S3-T9** | `a2a.yaml` demo config бичих | 0.5 | S3-T7 | models, plugins, routes гэсэн 3 section-той demo config |
| **S3-T10** | `A2ARuntime` orchestrator | 1 | S3-T5, S3-T8 | `start()` → config унших → plugin-ууд ачаалах → transport эхлүүлэх |
| **S3-T11** | `LogReaderPlugin` (Sprint 0 stub-г бөглөх) | 1 | S3-T1, S2-T1 | Лог текст → `extract_tensor()` → error_context тензор гарна |
| **S3-T12** | `CodeFixerPlugin` (Sprint 0 stub-г бөглөх) | 1 | S3-T1, S2-T4 | error_context тензор → `on_receive_tensor()` → inject → generate → код patch |
| **S3-T13** | End-to-end integration test: `a2a serve` → log → code fix | 1 | S3-T10, S3-T11, S3-T12 | Runtime бүрэн ачаалагдаж, лог → код засвар flow амжилттай |

### S3-T3 дэлгэрэнгүй — Plugin loading механизм

```python
# PluginManager.load_plugin() flow:
1. importlib.import_module(entry.module)     # → module object
2. inspect.getmembers(module, inspect.isclass) → BasePlugin subclass хайх
3. plugin = found_class()                     # → instance үүсгэх
4. model_config = global_config.models[entry.model]  # model reference resolve
5. model = load_model(model_config)           # → HF model ачаалах
6. tokenizer = load_tokenizer(model_config)   # → HF tokenizer ачаалах
7. plugin_config = yaml.load(plugin_dir / "config.yaml")  # plugin-local config
8. await plugin.initialize(model, tokenizer, plugin_config, global_config)
9. self.register(plugin)
```

### S3-T10 дэлгэрэнгүй — A2ARuntime.start() flow

```python
async def start(self):
    # 1. Config ачаалах
    config = load_config()
    
    # 2. Logger тохируулах
    setup_logging(config.runtime.log_level, config.runtime.log_format)
    
    # 3. Plugin-ууд ачаалах
    for plugin_id, entry in config.plugins.items():
        if entry.enabled:
            await self.plugin_manager.load_plugin(entry, config)
    
    # 4. Projection registry ачаалах (хэрэв pretrained байвал)
    for pair_key, path in config.projection.pretrained.items():
        src, tgt = pair_key.split("__")
        self.projection_registry.load(src, tgt, path, src_dim, tgt_dim)
    
    # 5. Transport сервер эхлүүлэх
    self.server = A2AServer(config, self.plugin_manager)
    await self.server.start()
    
    # 6. Health check + metrics server эхлүүлэх
    start_metrics_server(config.runtime.metrics_port)
```

### S3 тестүүд

| ID | Тест | Файл | Тоо |
|---|---|---|---|
| S3-TEST-1 | Plugin subclass with all abstract methods | `tests/agent/test_base.py` | 1 |
| S3-TEST-2 | PluginManager.register() → label routes populated | `tests/agent/test_manager.py` | 1 |
| S3-TEST-3 | PluginManager.route_tensor() → correct plugin called | `tests/agent/test_manager.py` | 1 |
| S3-TEST-4 | PluginManager.route_tensor() → no matching label → empty | `tests/agent/test_manager.py` | 1 |
| S3-TEST-5 | PluginManager.load_plugin() with valid entry | `tests/agent/test_manager.py` | 1 |
| S3-TEST-6 | PluginManager.load_plugin() with invalid module | `tests/agent/test_manager.py` | 1 |
| S3-TEST-7 | A2AConfig.from_yaml() valid file | `tests/config/test_schema.py` | 1 |
| S3-TEST-8 | A2AConfig.from_yaml() missing required field | `tests/config/test_schema.py` | 1 |
| S3-TEST-9 | A2AConfig.from_yaml() invalid model reference | `tests/config/test_schema.py` | 1 |
| S3-TEST-10 | A2AConfig.from_yaml() invalid dtype | `tests/config/test_schema.py` | 1 |
| S3-TEST-11 | load_config() finds file in cwd | `tests/config/test_loader.py` | 1 |
| S3-TEST-12 | load_config() falls back to home dir | `tests/config/test_loader.py` | 1 |
| S3-TEST-13 | load_config() raises if no file found | `tests/config/test_loader.py` | 1 |
| S3-TEST-14 | load_config() explicit path | `tests/config/test_loader.py` | 1 |
| S3-TEST-15 | LogReaderPlugin.extract_tensor() returns tensor | `tests/plugins/test_log_reader.py` | 1 |
| S3-TEST-16 | CodeFixerPlugin.on_receive_tensor() returns text | `tests/plugins/test_code_fixer.py` | 1 |
| S3-TEST-17 | LogReader → CodeFixer full flow (integration) | `tests/integration/test_plugin_flow.py` | 1 |
| S3-TEST-18 | A2ARuntime.start() loads all plugins | `tests/integration/test_runtime.py` | 1 |
| S3-TEST-19 | Semantic label routing with multiple listeners | `tests/agent/test_manager.py` | 1 |
| **Нийт** | | | **19** |

### S3 хийсний дараах шалгалт

```bash
# Terminal дээр:
$ a2a serve                                           # runtime асах, plugin-ууд ачаалагдах
# → [INFO] Loaded plugin: log-reader
# → [INFO] Loaded plugin: code-fixer
# → [INFO] gRPC server listening on 0.0.0.0:9090

$ python scripts/demo_log_to_fix.py                    # Demo скрипт
# → Input:  "ERROR: NullPointerException at line 45"
# → Output: "Here's the fix: ..." (code patch)

$ pytest tests/agent/ tests/config/ tests/plugins/ tests/integration/ -v  # 19 tests green
```

---

## Sprint 4: Projection Model + Auto-training

**Хугацаа:** 4 долоо хоног (20 өдөр)
**Зорилго:** Llama ↔ Mistral хоёр өөр модель тензороор харилцах

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Acceptance Criteria |
|---|---|---|---|---|
| **S4-T1** | `ProjectionModel` (3-layer MLP + Residual + LayerNorm) | 1 | — | `forward(src_tensor)` → (target_dim,) shape зөв |
| **S4-T2** | `ProjectionModel` variant A (Linear) | 0.5 | S4-T1 | dims ижил үед identity, ялгаатай үед шугаман mapping |
| **S4-T3** | `ProjectionTrainer`: contrastive loss (InfoNCE) | 1.5 | S4-T1 | Training loop → loss буурна, epoch-оор cosine similarity өснө |
| **S4-T4** | `ProjectionTrainer`: multi-objective loss (MSE + Cosine + Contrastive) | 0.5 | S4-T3 | Гурван loss бүрэлдэхүүн, weighted sum зөв |
| **S4-T5** | `ProjectionPairDataset`: корпус → (src_hidden, tgt_hidden) хосууд | 2 | S2-T1 | Текст корпусын мөр бүрээс хоёр model-ийн hidden state гаргаж хос үүсгэнэ |
| **S4-T6** | `ProjectionPairDataset`: negative pair generation | 0.5 | S4-T5 | Өөр өөр текстийн hidden state → negative pair |
| **S4-T7** | `auto_trainer.py`: runtime auto-training trigger | 1.5 | S4-T3, S4-T5 | Хоёр plugin-ийн model pair auto-discovery → корпус унших → хос үүсгэх → сургах → хадгалах |
| **S4-T8** | `ProjectionRegistry`: model cache + load/save | 0.5 | S4-T1 | `get(src, tgt)` → cached model, `load(path)` → safetensors |
| **S4-T9** | PluginManager-д projection integration (send-ийн өмнө auto-resolve) | 1 | S4-T7 | `route_tensor()` → target model өөр бол projection resolve → forward → илгээх |
| **S4-T10** | gRPC service-д `RequestProjection` RPC бөглөх | 0.5 | S1-T2 | Сервер projection хүсэлт хүлээж авах, auto_train trigger хийх |
| **S4-T11** | End-to-end integration: Llama-8B → Mistral-7B cross-model flow | 2 | S4-T9 | Llama дээрх LogReader → Mistral дээрх CodeFixer, projection автоматаар сургагдаж ажиллах |

### S4-T5 дэлгэрэнгүй — ProjectionPairDataset.from_corpus()

```python
@classmethod
def from_corpus(cls, corpus_path, src_model, src_tokenizer, src_extractor,
                tgt_model, tgt_tokenizer, tgt_extractor, device="cuda"):
    with open(corpus_path) as f:
        texts = [line.strip() for line in f if line.strip()]
    
    src_hiddens, tgt_hiddens = [], []
    for text in tqdm(texts, desc="Generating hidden pairs"):
        # Source
        src_h = src_extractor.extract(text)
        # Target
        tgt_h = tgt_extractor.extract(text)
        src_hiddens.append(src_h.cpu())
        tgt_hiddens.append(tgt_h.cpu())
    
    return cls(torch.stack(src_hiddens), torch.stack(tgt_hiddens))
```

### S4-T9 дэлгэрэнгүй — Route үед auto-projection

```python
async def route_tensor(self, tensor, metadata):
    src_model = metadata.source_model
    targets = self.plugin_manager.get_targets(metadata.semantic_label)
    
    for plugin in targets:
        tgt_model = plugin.get_model_info().model_id
        
        # Хэрэв модель өөр бол projection хэрэгтэй
        if src_model != tgt_model:
            projection = self.projection_registry.get(src_model, tgt_model)
            
            if projection is None:
                # Auto-train trigger
                if metadata.requires_projection:
                    projection = await self._auto_train_and_wait(src_model, tgt_model)
                else:
                    # Projection олдохгүй → алдаа
                    raise ProjectionNotFoundError(src_model, tgt_model)
            
            tensor = projection(tensor.to(device)).to(tensor.dtype)
        
        result = await plugin.on_receive_tensor(tensor, metadata)
```

### S4 тестүүд

| ID | Тест | Файл | Тоо |
|---|---|---|---|
| S4-TEST-1 | ProjectionModel forward produces correct shape | `tests/projection/test_adapter.py` | 1 |
| S4-TEST-2 | ProjectionModel forward preserves batch dim | `tests/projection/test_adapter.py` | 1 |
| S4-TEST-3 | LinearProjection variant correct shape | `tests/projection/test_adapter.py` | 1 |
| S4-TEST-4 | Training loop reduces loss over epochs | `tests/projection/test_trainer.py` | 1 |
| S4-TEST-5 | Training improves cosine similarity | `tests/projection/test_trainer.py` | 1 |
| S4-TEST-6 | Multi-objective loss components all decrease | `tests/projection/test_trainer.py` | 1 |
| S4-TEST-7 | Dataset.from_corpus produces correct pairs count | `tests/projection/test_dataset.py` | 1 |
| S4-TEST-8 | Dataset.from_corpus source/target dimensions differ | `tests/projection/test_dataset.py` | 1 |
| S4-TEST-9 | Negative pairs have lower similarity than positive | `tests/projection/test_dataset.py` | 1 |
| S4-TEST-10 | Auto-trainer completes successfully | `tests/projection/test_auto_trainer.py` | 1 |
| S4-TEST-11 | Auto-trainer saves safetensors file | `tests/projection/test_auto_trainer.py` | 1 |
| S4-TEST-12 | Registry caches loaded projection | `tests/projection/test_registry.py` | 1 |
| S4-TEST-13 | Registry returns None for unknown pair | `tests/projection/test_registry.py` | 1 |
| S4-TEST-14 | PluginManager resolves projection automatically | `tests/integration/test_projection_flow.py` | 1 |
| S4-TEST-15 | Cross-model full pipeline (Llama→Mistral) | `tests/integration/test_cross_model.py` | 1 |
| **Нийт** | | | **15** |

### S4 хийсний дараах шалгалт

```bash
# Llama-8B модель дээр LogReader, Mistral-7B дээр CodeFixer ажиллаж байна:
$ a2a serve --config examples/multi_model/a2a.yaml
# → [INFO] Loaded plugin: log-reader (model: llama-8b)
# → [INFO] Loaded plugin: code-fixer (model: mistral-7b)
# → [INFO] Auto-training projection: llama-8b → mistral-7b (45 seconds)
# → [INFO] Projection ready: llama-8b__mistral-7b

$ python scripts/demo_cross_model.py
# → Log text → Llama hidden → Projection → Mistral hidden → Code fix output
# → Latency: ~250ms (vs ~1200ms text API)

$ pytest tests/projection/ tests/integration/test_cross_model.py -v  # 15 tests green
```

---

## Sprint 5: Security, Rate Limit, Monitoring

**Хугацаа:** 3 долоо хоног (15 өдөр)
**Зорилго:** Үйлдвэрлэлд бэлэн — mTLS, JWT auth, rate limiting, Prometheus metrics, structured logging

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Acceptance Criteria |
|---|---|---|---|---|
| **S5-T1** | mTLS: server-side SSL credentials үүсгэх, gRPC server-д залгах | 1 | S1-T6 | Сервер зөвхөн valid client cert-тэй холболт хүлээж авна |
| **S5-T2** | mTLS: client-side SSL credentials, secure channel үүсгэх | 1 | S5-T1 | Клиент valid cert, key, ca-тай серверт холбогдоно |
| **S5-T3** | JWT auth: token create + validate `security/auth.py` | 0.5 | — | `create_token(agent_id, mesh_id, secret)` → JWT. `validate_token(token, secret)` → payload / None |
| **S5-T4** | JWT: gRPC client metadata-д token attach хийх | 0.5 | S5-T3 | gRPC call хийх бүрд metadata header-т JWT token очно |
| **S5-T5** | JWT: gRPC server interceptor → token validate хийх | 1 | S5-T3, S1-T6 | Token хүчингүй бол PERMISSION_DENIED error буцаана |
| **S5-T6** | TokenBucket rate limiter `monitoring/rate_limiter.py` | 1 | — | `allow(agent_id)` → bucket-д token байвал True, үгүй бол False |
| **S5-T7** | Rate limiter: per-agent + per-route config, gRPC interceptor-т залгах | 1 | S5-T6 | Agent ID + semantic label-аар rate limit шалгагдана, хэтэрвэл RESOURCE_EXHAUSTED |
| **S5-T8** | Backpressure / Flow control: `MSG_BACKPRESSURE` protobuf type + логик | 1.5 | S1-T2 | Queue depth > threshold → sender-т SLOW_DOWN дохио, queue хоосорвол RESUME |
| **S5-T9** | Prometheus metrics: counters, histograms, gauges | 1 | — | `/metrics` endpoint -> Prometheus scrape боломжтой |
| **S5-T10** | Structured JSON logging `utils/logging.py` | 0.5 | — | `logging.info("event", extra={...})` → JSON line, timestamp + level + message |
| **S5-T11** | Health check endpoint: `/health`, `/health/ready`, `/health/live` | 0.5 | — | GET /health → `{"status":"ok","plugins_loaded":2,"uptime":3600}` |
| **S5-T12** | Security + rate limit integration test | 1 | S5-T1..S5-T8 | mTLS холболт + JWT auth + rate limit бүгд зэрэг ажиллах |

### S5 тестүүд

| ID | Тест | Файл | Тоо |
|---|---|---|---|
| S5-TEST-1 | TokenBucket consumes token correctly | `tests/monitoring/test_rate_limiter.py` | 1 |
| S5-TEST-2 | TokenBucket refills over time | `tests/monitoring/test_rate_limiter.py` | 1 |
| S5-TEST-3 | TokenBucket denies when empty | `tests/monitoring/test_rate_limiter.py` | 1 |
| S5-TEST-4 | RateLimiter respects per-agent config | `tests/monitoring/test_rate_limiter.py` | 1 |
| S5-TEST-5 | JWT create generates valid token | `tests/security/test_auth.py` | 1 |
| S5-TEST-6 | JWT validate returns correct payload | `tests/security/test_auth.py` | 1 |
| S5-TEST-7 | JWT validate rejects expired token | `tests/security/test_auth.py` | 1 |
| S5-TEST-8 | JWT validate rejects wrong secret | `tests/security/test_auth.py` | 1 |
| S5-TEST-9 | JWT validate checks mesh_id | `tests/security/test_auth.py` | 1 |
| S5-TEST-10 | gRPC server rejects unauthenticated request | `tests/integration/test_secure_grpc.py` | 1 |
| S5-TEST-11 | gRPC client with valid token accepted | `tests/integration/test_secure_grpc.py` | 1 |
| S5-TEST-12 | Rate limit enforced in gRPC interceptor | `tests/integration/test_secure_grpc.py` | 1 |
| S5-TEST-13 | Backpressure signal sent when queue full | `tests/integration/test_backpressure.py` | 1 |
| S5-TEST-14 | Health endpoint returns expected fields | `tests/test_health.py` | 1 |
| S5-TEST-15 | JSON logger produces valid JSON | `tests/utils/test_logging.py` | 1 |
| **Нийт** | | | **15** |

### S5 хийсний дараах шалгалт

```bash
# Metrics endpoint:
$ curl http://localhost:9091/metrics
# → a2a_tensors_sent_total{agent_id="log-reader",semantic_label="error_context"} 42
# → a2a_tensor_latency_seconds_count{...} 42

# Health check:
$ curl http://localhost:9091/health
# → {"status":"ok","uptime":3600,"version":"0.1.0","plugins_loaded":2}

# Rate limit test:
$ python scripts/flood_test.py --rate 200
# → First 100: accepted, next 100: rate limited (429 / RESOURCE_EXHAUSTED)

# mTLS test:
$ a2a discover                                # invalid cert → холбогдохгүй
$ a2a discover --cert ... --key ... --ca ...  # valid → agent list харагдана

$ pytest tests/security/ tests/monitoring/ tests/integration/test_secure_grpc.py -v  # 15 tests green
```

---

## Sprint 6: Documentation, Integration, Release

**Хугацаа:** 2 долоо хоног (10 өдөр)
**Зорилго:** PyPI package + Docker image + бүрэн documentation + demo examples

### Task Breakdown

| ID | Даалгавар | Өдөр | Хамаарал | Acceptance Criteria |
|---|---|---|---|---|
| **S6-T1** | `docs/index.md` — Overview, Quickstart (5 минутанд ажиллуулах) | 0.5 | — | Шинэ хүн дагахад 5 минутанд demo ажиллана |
| **S6-T2** | `docs/architecture.md` — Архитектурын дэлгэрэнгүй (plan.md-с гаргах) | 0.5 | — | Архитектурын бүх diagram + тайлбар |
| **S6-T3** | `docs/protocol.md` — Wire protocol specification (бүрэн reference) | 1 | — | Message types, metadata schema, error codes, versioning |
| **S6-T4** | `docs/plugins.md` — Plugin хөгжүүлэх гарын авлага | 1 | — | `BasePlugin` subclass хийх, config.yaml, тест хийх жишээ |
| **S6-T5** | `docs/config.md` — `a2a.yaml` бүрэн reference | 1 | — | Секц бүрийн талбар, default утга, valid options |
| **S6-T6** | `docs/deployment.md` — Docker + K8s deployment guide | 0.5 | — | `docker run`, `docker-compose`, `kubectl apply` ажиллана |
| **S6-T7** | `docs/api/` — API reference (BasePlugin, TensorEngine, Config) | 1 | — | Бүх public API-ийн docstring-с auto-generated |
| **S6-T8** | Demo examples: `examples/basic_mesh/` (2 plugin) | 1 | S3-T13 | `start.sh` → ажиллах 2 plugin-той demo, README-тэй |
| **S6-T9** | Demo examples: `examples/multi_model/` (3 plugin, 2 model) | 1.5 | S4-T11 | Llama+DeepSeek+Mistral 3 plugin-той, auto-projection |
| **S6-T10** | PyPI release: build + twine upload | 0.5 | — | `pip install a2a-protocol` → install + `a2a --help` |
| **S6-T11** | Docker image: multi-stage build, push to ghcr | 0.5 | — | `docker run ghcr.io/mbm/a2a-runtime` → runtime асах |

### S6 тестүүд

| ID | Тест | Файл | Тоо |
|---|---|---|---|
| S6-TEST-1 | `examples/basic_mesh/start.sh` runs without error | `tests/e2e/test_basic_mesh.py` | 1 |
| S6-TEST-2 | Basic mesh: log → code fix pipeline | `tests/e2e/test_basic_mesh.py` | 1 |
| S6-TEST-3 | Multi-model mesh: 3 agents with auto-projection | `tests/e2e/test_multi_model.py` | 1 |
| S6-TEST-4 | Docker image starts and `/health` returns ok | `tests/e2e/test_docker.py` | 1 |
| S6-TEST-5 | PyPI wheel installs and CLI works | `tests/e2e/test_pypi.py` | 1 |
| **Нийт** | | | **5** |

### S6 хийсний дараах шалгалт

```bash
# PyPI-с суулгах:
$ pip install a2a-protocol
$ a2a --version                                    # → 0.1.0

# Docker:
$ docker run -p 9090:9090 -p 9091:9091 \
    -v ./a2a.yaml:/etc/a2a/a2a.yaml \
    ghcr.io/mbm/a2a-runtime:0.1.0
# → Server running on 0.0.0.0:9090

# Quickstart:
$ cp examples/basic_mesh/a2a.yaml .
$ mkdir -p projections && echo "some log line" > projection_corpus.txt
$ a2a serve
# → [INFO] 2 plugins loaded, ready

# Documentation:
$ open docs/index.html                              # mkdocs serve

$ pytest tests/e2e/ -v                              # 5 tests green
```

---

## Sprint 7+ (2-р үе): Go Transport Layer

**Хугацаа:** Төлөвлөгөөгүй (Python SDK тогтворжсоны дараа)

| ID | Даалгавар | Тайлбар |
|---|---|---|
| G1 | Go модуль үүсгэх (`go.mod`, `go.sum`) | `a2a-transport` нэртэй |
| G2 | FlatBuffers Go кодогенерац | `.fbs` schema → Go structs |
| G3 | gRPC/QUIC сервер Go дээр бичих | Python ML Core-той unix socket-ээр харилцана |
| G4 | gRPC/QUIC клиент Go дээр бичих | Өндөр concurrency, бага latency |
| G5 | Go ↔ Python integration test | Go transport нь Python ML Core-руу тензор дамжуулж, хариу авна |
| G6 | Benchmark: Go transport vs Python transport | Latency, throughput, memory харьцуулалт |

---

## Хавсралт А: Mock Test Model

Тестэд жижиг, хурдан HF model ашиглах:

```python
# tests/conftest.py
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_MODEL_ID = "sshleifer/tiny-gpt2"  # 69MB, CPU дээр ч хурдан

@pytest.fixture(scope="session")
def test_model():
    return AutoModelForCausalLM.from_pretrained(TEST_MODEL_ID)

@pytest.fixture(scope="session")
def test_tokenizer():
    return AutoTokenizer.from_pretrained(TEST_MODEL_ID)
```

## Хавсралт Б: CI/CD Pipeline бүрэн

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push: {branches: [main, develop]}
  pull_request: {branches: [main]}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install ruff mypy
      - run: ruff check a2a/ tests/
      - run: mypy a2a/

  test:
    needs: lint
    strategy:
      matrix:
        python: ["3.11", "3.12"]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=a2a --cov-report=xml --timeout=60
      - uses: codecov/codecov-action@v4

  proto-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make proto && git diff --exit-code  # Generated code drift check

  e2e:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/e2e/ -v --timeout=300
```

## Хавсралт В: Файл бүрийн Sprint (хурдан lookup)

| Файл | Sprint | Task |
|---|---|---|
| `pyproject.toml` | S0 | S0-T1 |
| `.github/workflows/ci.yml` | S0 | S0-T2 |
| `.pre-commit-config.yaml` | S0 | S0-T3 |
| `a2a/__init__.py` | S0 | S0-T4 |
| `a2a/cli.py` | S0 | S0-T5 |
| `a2a/protocol/a2a.proto` | S1 | S1-T1 |
| `a2a/protocol/a2a_pb2.py` | S1 | S1-T2 (generated) |
| `a2a/transport/codec.py` | S1 | S1-T3, S4, T5 |
| `a2a/transport/server.py` | S1 | S1-T6, T7 |
| `a2a/transport/client.py` | S1 | S1-T8 |
| `a2a/protocol/errors.py` | S1 | S1-T10 |
| `a2a/tensor/extractor.py` | S2 | S2-T1, T2, T3 |
| `a2a/tensor/injector.py` | S2 | S2-T4, T5 |
| `a2a/tensor/dtype.py` | S2 | S2-T6 |
| `a2a/tensor/serializer.py` | S2 | S2-T7 |
| `a2a/agent/base.py` | S3 | S3-T1 |
| `a2a/agent/capabilities.py` | S3 | S3-T2 |
| `a2a/agent/manager.py` | S3 | S3-T3, T4, T5, T6 |
| `a2a/config/schema.py` | S3 | S3-T7 |
| `a2a/config/loader.py` | S3 | S3-T8 |
| `a2a.yaml` | S3 | S3-T9 |
| `a2a/runtime.py` | S3 | S3-T10 |
| `a2a/plugins/log_reader/plugin.py` | S3 | S3-T11 |
| `a2a/plugins/log_reader/prompts.py` | S3 | S3-T11 |
| `a2a/plugins/log_reader/config.yaml` | S3 | S3-T11 |
| `a2a/plugins/code_fixer/plugin.py` | S3 | S3-T12 |
| `a2a/plugins/code_fixer/prompts.py` | S3 | S3-T12 |
| `a2a/plugins/code_fixer/config.yaml` | S3 | S3-T12 |
| `a2a/projection/adapter.py` | S4 | S4-T1, T2 |
| `a2a/projection/trainer.py` | S4 | S4-T3, T4 |
| `a2a/projection/dataset.py` | S4 | S4-T5, T6 |
| `a2a/projection/auto_trainer.py` | S4 | S4-T7 |
| `a2a/projection/registry.py` | S4 | S4-T8 |
| `a2a/security/tls.py` | S5 | S5-T1, T2 |
| `a2a/security/auth.py` | S5 | S5-T3 |
| `a2a/monitoring/rate_limiter.py` | S5 | S5-T6 |
| `a2a/monitoring/metrics.py` | S5 | S5-T9 |
| `a2a/utils/logging.py` | S5 | S5-T10 |
| `docs/*` | S6 | S6-T1..T7 |
| `examples/basic_mesh/` | S6 | S6-T8 |
| `examples/multi_model/` | S6 | S6-T9 |
| `Dockerfile` | S6 | S6-T11 |
