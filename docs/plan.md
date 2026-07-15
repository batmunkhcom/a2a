# A2A Protocol — AI-to-AI Communication Protocol

## Overview

Одоогийн AI-ууд хоорондоо хүнд зориулагдсан текст/JSON-оор харилцдаг. A2A Protocol нь AI-ийн "тархины хэл" болох **Latent Space дахь векторуудыг** шууд дамжуулж, tokenization болон текст парсинг-ийн overhead-г бүрэн арилгана.

Жишээ: Log уншдаг AI алдааны мэдээллийг [0.45, -0.12, 0.98, ...] вектор болгон код засдаг AI-руу шууд дамжуулж, текст уншихгүйгээр шууд код бичиж эхэлнэ.

---

## Архитектурын Ерөнхий Тойм

### A2A Protocol-ийн Байршил

A2A Protocol нь AI agent-уудын хооронд шинэ төрлийн харилцааны суваг (communication channel) үүсгэнэ. Одоогийн текстэд суурилсан API дуудлагаас ялгаатай нь, A2A нь AI-ийн дотоод төлөв (hidden state)-ийг шууд дамжуулдаг.

```
ОДООГИЙН АРГА (Text-based):
┌──────────┐  текст    ┌────────┐  JSON/API   ┌──────────┐  текст    ┌──────────┐
│  Log     │ ────────→ │ OpenAI │ ──────────→ │  Код      │ ────────→ │  Хэрэг-  │
│  уншигч  │ "алдаа.." │  API   │ "засвар.."  │  бичигч   │ "код.."   │  лэгч    │
│  (AI)    │           │        │             │  (AI)     │           │  (хүн)   │
└──────────┘           └────────┘             └──────────┘           └──────────┘
     Text                     Text                   Text                   Text
     500ms tokenize           200ms API              500ms tokenize         200ms read
     ─────────────────────────────────────────────────────────────────────────────
     Нийт: ~1400ms


A2A PROTOCOL (Tensor-based):
┌──────────┐  [0.45, -0.12, 0.98, ...]  ┌──────────┐  текст    ┌──────────┐
│  Log     │ ───────────────────────────→│  Код      │ ────────→ │  Хэрэг-  │
│  уншигч  │   1024-dim vector (8KB)     │  бичигч   │ "код.."   │  лэгч    │
│  (AI)    │   gRPC direct transport     │  (AI)     │           │  (хүн)   │
└──────────┘                             └──────────┘           └──────────┘
     Hidden State                             Inject → Generate     Text
     10ms extract                             5ms + 200ms
     ────────────────────────────────────────────────────
     Нийт: ~215ms  (6.5x хурдан)
```

### Системийн Давхаргын Архитектур

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        A2A RUNTIME                                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    PLUGIN LAYER (Agent-ууд)                       │  │
│  │                                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
│  │  │ Log      │  │ Code     │  │ Security │  │ Custom   │  ...     │  │
│  │  │ Reader   │  │ Fixer    │  │ Analyzer │  │ Plugin   │         │  │
│  │  │ Plugin   │  │ Plugin   │  │ Plugin   │  │ Plugin   │         │  │
│  │  │          │  │          │  │          │  │          │         │  │
│  │  │ model:   │  │ model:   │  │ model:   │  │ model:   │         │  │
│  │  │ llama-8b │  │ deepseek │  │ mistral  │  │ (any)    │         │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘         │  │
│  │       │             │             │                               │  │
│  │       │  listens_to / emits  (semantic label routing)             │  │
│  │       └─────────────┼─────────────┘                               │  │
│  └─────────────────────┼─────────────────────────────────────────────┘  │
│                        │                                                │
│  ┌─────────────────────┼─────────────────────────────────────────────┐  │
│  │              CORE LAYER (Protocol Engine)                         │  │
│  │                                                                   │  │
│  │  ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌─────────────┐   │  │
│  │  │ Transport │  │ Plugin    │  │ Projection │  │  Semantic    │   │  │
│  │  │ (gRPC/    │  │ Manager   │  │ Registry   │  │  Router      │   │  │
│  │  │ QUIC)     │  │           │  │            │  │              │   │  │
│  │  └─────┬─────┘  └─────┬─────┘  └──────┬─────┘  └──────┬──────┘   │  │
│  │        │              │               │               │          │  │
│  │  ┌─────┴──────────────┴───────────────┴───────────────┴──────┐   │  │
│  │  │              TENSOR ENGINE                                 │   │  │
│  │  │  - Hidden State Extraction (PyTorch hooks)                 │   │  │
│  │  │  - Tensor Injection (prefix / cross-attention)             │   │  │
│  │  │  - Serialization (FlatBuffers / Safetensors)               │   │  │
│  │  │  - Projection Model forward pass                           │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   INFRASTRUCTURE LAYER                            │  │
│  │                                                                   │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐   │  │
│  │  │ Discovery│  │ Auth &    │  │ Rate     │  │ Health Check  │   │  │
│  │  │ (mDNS/   │  │ mTLS      │  │ Limiter  │  │ & Metrics     │   │  │
│  │  │ static)  │  │           │  │          │  │ (Prometheus)  │   │  │
│  │  └──────────┘  └───────────┘  └──────────┘  └───────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Plugin Хоорондын Тензор Урсгал (Semantic Label Routing)

```
                    listens_to: []          listens_to: ["error_context"]
                    emits: ["error_context", emits: ["code_patch",
                           "log_summary"]           "fix_explanation"]
                    ┌──────────────┐        ┌──────────────┐
    External  ────→ │  Log Reader  │ ─────→ │  Code Fixer  │ ────→ PR Creator
    (лог файл)      │  Plugin      │        │  Plugin      │       (disabled)
                    │              │        │              │
                    │ model:       │        │ model:       │
                    │ llama-8b     │        │ deepseek     │
                    └──────┬───────┘        └──────────────┘
                           │
                           │ "log_summary"
                           ▼
                    ┌──────────────┐
                    │  Security    │
                    │  Analyzer    │
                    │  Plugin      │
                    │              │
                    │ model:       │
                    │ mistral-7b   │
                    └──────────────┘
                    listens_to: ["log_summary", "error_context"]
                    emits: ["security_alert"]
```

### Харилцааны Загвар: Homogeneous vs Heterogeneous

```
HOMOGENEOUS (ижил base model — projection хэрэггүй):
┌─────────────┐          ┌─────────────┐
│ Llama-3-8B  │  tensor  │ Llama-3-8B  │
│ (fine-tuned │ ────────→│ (fine-tuned │
│  for logs)  │  шууд    │  for code)  │
└─────────────┘          └─────────────┘
    1024-dim                1024-dim
    hidden state            hidden state
    → ШУУД injection →      → шууд ойлгоно


HETEROGENEOUS (өөр model — Projection Model шаардлагатай):
┌─────────────┐          ┌──────────────┐          ┌─────────────┐
│ Llama-3-8B  │  tensor  │  Projection  │  tensor  │ Mistral-7B  │
│             │ ────────→│  Model       │ ────────→│             │
│  4096-dim   │          │  (сургагдсан) │          │  1024-dim   │
└─────────────┘          └──────────────┘          └─────────────┘
                             4096 → 1024
                             mapping (MLP)
                             ~2-8M параметр
```

### Projection Model — Auto-Training Flow

```
1. Discovery Phase:
   Agent A ──MSG_CAPABILITY──→ Agent B
   Agent A ←──MSG_CAPABILITY── Agent B
   → hidden_dim зөрүүтэй → auto-trigger

2. Data Collection Phase:
   Shared Corpus ──→ Agent A: hidden_A[i]
   (projection_   ──→ Agent B: hidden_B[i]
    corpus.txt)        → (hidden_A[i], hidden_B[i]) хосууд

3. Training Phase:
   ProjectionModel(hidden_A) → predicted_B
   Loss = ||predicted_B - hidden_B||² + contrastive
   ~1000-5000 хос, ~30-60sec GPU дээр

4. Ready Phase:
   MSG_PROJECT_READY → бүртгэгдэнэ → дараагийн удаа дахин сургахгүй
```

### Өгөгдлийн Урсгал — Бүрэн Цикл

```
1. External trigger (лог файл, HTTP request, event)
        │
2. Plugin.extract_tensor(input_text)
        │  → PyTorch forward pass → hidden state (N-dim vector)
        │
3. PluginManager.route_tensor(tensor, metadata)
        │  → semantic_label-аар target plugin-уудыг олох
        │
4. ProjectionModel.forward(tensor)     ← ХЭРЭВ src ≠ tgt model бол
        │  → src hidden → tgt hidden mapping
        │
5. Transport.send(tensor, target_address)
        │  → gRPC stream → FlatBuffers serialization → network
        │
6. Transport.receive() → deserialize → tensor
        │
7. Plugin.on_receive_tensor(tensor, metadata)
        │  → TensorInjector.inject(tensor, prompt)
        │  → model.generate() → output text
        │
8. Result (text output, next action, alert)
```

### Технологийн Стек

```
                    ┌─────────────────────────┐
                    │   Plugin Development     │
                    │   Python 3.11+           │
                    │   PyTorch 2.x + HF       │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────┴───────┐     ┌─────────┴─────────┐     ┌───────┴───────┐
│   Transport   │     │   ML Compute      │     │   Storage     │
│               │     │                   │     │               │
│ gRPC / QUIC   │     │ PyTorch           │     │ Safetensors   │
│ FlatBuffers   │     │ HuggingFace       │     │ Projection    │
│ Protobuf      │     │ Transformers      │     │ Registry (FS) │
│ mTLS          │     │ vLLM / Ollama     │     │ YAML Config   │
└───────────────┘     └───────────────────┘     └───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────┴───────┐     ┌─────────┴─────────┐     ┌───────┴───────┐
│   Models      │     │   Monitoring      │     │   DevOps      │
│               │     │                   │     │               │
│ Llama-3       │     │ Prometheus        │     │ Docker        │
│ Mistral       │     │ OpenTelemetry     │     │ GitHub Actions│
│ DeepSeek      │     │ Health endpoints  │     │ PyPI          │
│ GPT-4o        │     │ Structured logs   │     │               │
└───────────────┘     └───────────────────┘     └───────────────┘
```

---

## Table of Contents

0. [Архитектурын Ерөнхий Тойм](#архитектурын-ерөнхий-тойм)
1. [Технологийн стек: Python vs Go](#1-технологийн-стек-python-vs-go)
2. [Үе 1 — Судалгаа & Үндэслэл](#2-үе-1--судалгаа--үндэслэл)
3. [Үе 2 — Протоколын Спецификаци](#3-үе-2--протоколын-спецификаци)
4. [Үе 3 — Core SDK & Runtime (Python)](#4-үе-3--core-sdk--runtime-python)
5. [Үе 4 — Projection Model Сургалт](#5-үе-4--projection-model-сургалт)
6. [Үе 5 — Benchmark & Optimization](#6-үе-5--benchmark--optimization)
7. [Үе 6 — SDK, Docs, Integration](#7-үе-6--sdk-docs-integration)
8. [Plugin/Component Architecture](#8-plugincomponent-architecture)
9. [Single Source of Truth — Тохиргооны Архитектур](#9-single-source-of-truth--тохиргооны-архитектур)
10. [Security & Authentication](#10-security--authentication)
11. [Error Handling Protocol](#11-error-handling-protocol)
12. [Rate Limiting & Flow Control](#12-rate-limiting--flow-control)
13. [Protocol Versioning & Backward Compatibility](#13-protocol-versioning--backward-compatibility)
14. [Testing Strategy](#14-testing-strategy)
15. [Monitoring & Observability](#15-monitoring--observability)
16. [Scaling & Deployment Strategy](#16-scaling--deployment-strategy)
17. [Project Phases, Timeline & MVP](#17-project-phases-timeline--mvp)

---

## 1. Технологийн стек: Python vs Go

### Python яагаад илүү тохиромжтой вэ?

A2A Protocol-ийн **гол compute нь Neural Network дээр** явагдана:
- Hidden state extraction (forward pass through Transformer)
- Projection/Adapter model (жижиг MLP forward pass)
- Tensor injection → generation (Transformer forward + sampling)

Эдгээр нь бүгд **PyTorch**-ийн үндсэн домэйн. Python / PyTorch-гүйгээр:
- HuggingFace Transformers ажиллахгүй
- vLLM, Ollama, llama.cpp-тэй интеграц хийх боломжгүй
- CUDA/GPU compute ашиглах боломжгүй

### Go-г хаана ашиглаж болох вэ?

Хэрэв **service-oriented architecture** барих юм бол:

```
┌──────────────────────────────┐
│  A2A Transport Layer (Go)    │  ← gRPC сервер, тензор routing, discovery
│  - gRPC/QUIC server/client   │
│  - Connection pooling        │
│  - Service discovery (mDNS)  │
│  - FlatBuffers encode/decode │
│  - Load balancing            │
└──────────┬───────────────────┘
           │ localhost gRPC / unix socket
┌──────────▼───────────────────┐
│  A2A ML Runtime (Python)     │  ← PyTorch, HF Transformers, vLLM
│  - Hidden state extraction   │
│  - Projection model forward  │
│  - Tensor injection/generate │
│  - Model management          │
└──────────────────────────────┘
```

Go-ийн давуу тал:
- Өндөр concurrency (goroutines), сүлжээний I/O-д гайхалтай
- Бага санах ойн хэрэглээ
- Single binary деплой
- gRPC-ийн native кодогенераци

**Гэхдээ Go-ийн ML ecosystem маш сул:**
- gorgonia.org (чимхүүр сан, PyTorch шиг биш)
- ONNX Runtime Go bindings (хязгаарлагдмал)
- HuggingFace Transformers Go-д ажиллахгүй
- CUDA GPU acceleration байхгүй

### Санал болгох архитектур

**Холимог стек (Hybrid):**

| Давхарга | Хэл | Шалтгаан |
|---|---|---|
| Transport / gRPC Gateway | **Go** (эсвэл Rust) | Өндөр concurrency, бага latency сүлжээний давхарга |
| ML Core / Tensor Ops | **Python** | PyTorch + HF + CUDA |
| Projection Training | **Python** | PyTorch training loop |
| Agent SDK (dev API) | **Python** | ML engineer-ууд Python дээр ажилладаг |
| CLI Tool | **Go** эсвэл **Python** | Python (>click/typer) эсвэл Go (>cobra) |
| Benchmark Tool | **Go** | Нарийвчилсан latency хэмжилтэд Go илүү |

**Эхний үе шатанд зөвхөн Python ашиглах:**

Эхний 3-4 сард **100% Python** ашиглах нь зөв. Учир нь:
1. ML compute-г Python-гүй хийх боломжгүй
2. Go transport layer нь premature optimization
3. MVP-д gRPC серверийг Python-ийн `grpcio` сангаар маш сайн хийж болно
4. Протокол боловсронгуй болсны дараа Go transport layer нэмэх нь хялбар

**2-р хувилбарт Go руу шилжих:**
- Transport layer-ийн latency critical path-г Go-руу шилжүүлнэ
- Python ML Core-той unix socket эсвэл localhost gRPC-ээр харилцана
- FlatBuffers encode/decode-г Go дээр хийж Python-ийн GIL overhead-с зайлсхийнэ

### Шийдвэр: **Python first, Go second**

```
MVP (Үе 1-4):     Python 100%
Production (Үе 5-6): Python (ML Core) + Go (Transport)
```

---

## 2. Үе 1 — Судалгаа & Үндэслэл

### 2.1 Одоо байгаа судалгаа

- Neural Communication / Latent Space Communication чиглэлийн цааснууд
- Cross-model Latent Space Alignment судалгаанууд
- OpenAI, Anthropic, Google DeepMind-ийн multi-agent communication
- Contrastive Representation Learning (SimCLR, CLIP аргачлал)
- Llama, GPT, Mistral, Gemma зэрэг нээлттэй загваруудын hidden state extraction API

### 2.2 Техникийн нөөц / технологиуд

| Категори | Сонголтууд |
|---|---|
| Тээвэрлэлтийн протокол | gRPC, QUIC (HTTP/3), WebSocket |
| Вектор сериалчлал | Safetensors, FlatBuffers, Cap'n Proto |
| ML Framework | PyTorch, HuggingFace Transformers |
| Модел inference | vLLM, llama.cpp, Ollama, HF Inference |
| Үйлчилгээ ололт | gRPC reflection, Consul, mDNS, Custom registry |
| Inference backend | ONNX Runtime, Triton Inference Server |

### 2.3 Туршилтын загварууд

- **Homogeneous тест**: Llama-3-8B (log-reader) ↔ Llama-3-8B (code-fixer)
- **Heterogeneous тест**: Llama-3-8B ↔ Mistral-7B
- Hidden state source: Transformer-ийн сүүлийн давхаргын residual stream

---

## 3. Үе 2 — Протоколын Спецификаци

### 3.1 Wire Protocol (A2A Frame)

```
+------------------+------------------+------------------+------------------+
| Header (16 bytes)| Metadata (var)   | Tensor Payload   | Checksum (4)     |
+------------------+------------------+------------------+------------------+
| magic: 0xA2A0    | model_id         | dims, dtype,     | CRC32            |
| version: 1       | source_agent     | raw tensor data  |                  |
| msg_type: 0x01   | target_agent     |                  |                  |
| payload_len      | semantic_tag     |                  |                  |
+------------------+------------------+------------------+------------------+
```

### 3.2 Мессежийн төрлүүд (msg_type)

| Код | Төрөл | Тайлбар |
|---|---|---|
| 0x01 | `MSG_UNICAST` | Нэг AI → Нэг AI вектор дамжуулах |
| 0x02 | `MSG_MULTICAST` | Олон AI-д нэгэн зэрэг дамжуулах |
| 0x03 | `MSG_DISCOVER` | Сүлжээнд байгаа AI-уудыг илрүүлэх |
| 0x04 | `MSG_CAPABILITY` | AI-ийн чадварын мэдээлэл (model_id, dims, dtype) |
| 0x05 | `MSG_PROJECT_REQ` | Projection layer хүсэх |
| 0x06 | `MSG_PROJECT_RESP` | Projection параметрүүдийг хариу явуулах |
| 0x07 | `MSG_PROJECT_AUTO` | Автомат projection сургалт эхлүүлэх хүсэлт |
| 0x08 | `MSG_PROJECT_READY` | Projection model бэлэн болсон |
| 0x09 | `MSG_KEEPALIVE` | Холболтыг хадгалах |
| 0x0A | `MSG_ACK` | Баталгаажуулалт |

### 3.3 Meta-мэдээлэл (Protobuf schema)

Тэмдэглэл: Протоколын тодорхойлолтыг Protobuf-аар бичиж, gRPC-ийн кодогенерацтай нийцүүлнэ. Тензорын бодит дамжуулалтад FlatBuffers (zero-copy) ашиглах боловч мета өгөгдөл нь Protobuf message хэлбэртэй байна.

```protobuf
message A2AMetadata {
  string source_model = 1;       // "llama-3-8b-finetuned"
  string target_model = 2;       // "mistral-7b"
  uint32 source_layer = 3;       // hidden state авсан давхарга
  uint32 target_layer = 4;       // hidden state оруулах давхарга
  string tensor_dtype = 5;       // "float16", "bfloat16", "float32"
  repeated uint32 tensor_shape = 6; // [seq_len, hidden_dim]
  string semantic_label = 7;     // "error_context", "code_intent"
  float confidence = 8;          // 0.0 - 1.0
  uint64 timestamp = 9;
  string session_id = 10;
  string projection_id = 11;     // Хэрэглэж буй projection model-ийн ID
  bool requires_projection = 12; // Автомат projection хэрэгтэй эсэх
}
```

### 3.4 Хүлээн авагчид залгах аргууд

1. **Prefix injection**: Векторыг контекстийн prefix болгон оруулах (embedding layer-ийн дараа)
2. **Cross-attention injection**: Векторыг cross-attention-ийн key/value болгон ашиглах

---

## 4. Үе 3 — Core SDK & Runtime (Python)

### 4.1 Package бүтэц

```
a2a-protocol/
├── a2a/
│   ├── __init__.py
│   ├── transport/           # Сүлжээний давхарга
│   │   ├── __init__.py
│   │   ├── server.py        # gRPC/QUIC сервер
│   │   ├── client.py        # gRPC/QUIC клиент
│   │   ├── codec.py         # FlatBuffers encode/decode
│   │   └── discovery.py     # mDNS / service discovery
│   ├── tensor/              # Векторын давхарга
│   │   ├── __init__.py
│   │   ├── extractor.py     # Hidden state гаргах (HF hooks)
│   │   ├── injector.py      # Hidden state-г моделд шургуулах
│   │   ├── serializer.py    # Safetensors / binary serialization
│   │   └── dtype.py         # BF16/FP16/FP32 хөрвүүлэлт
│   ├── projection/          # Cross-model орчуулагч
│   │   ├── __init__.py
│   │   ├── adapter.py       # Projection model архитектур
│   │   ├── trainer.py       # Contrastive Learning сургалт
│   │   ├── auto_trainer.py  # Автомат сургалтын триггер
│   │   ├── dataset.py       # Сургалтын өгөгдөл
│   │   └── registry.py      # Сургасан проекцүүдийн бүртгэл
│   ├── agent/               # AI Agent SDK (plugin system)
│   │   ├── __init__.py
│   │   ├── base.py          # BasePlugin — бүх plugin-уудын суурь
│   │   ├── manager.py       # PluginManager — plugin-уудыг удирдах
│   │   ├── registry.py      # Plugin бүртгэл
│   │   ├── router.py        # Тензорыг зөв plugin-руу чиглүүлэх
│   │   └── capabilities.py  # Чадварын тодорхойлолт
│   ├── plugins/             # Суурилагдсан plugin-ууд
│   │   ├── __init__.py
│   │   ├── log_reader/      # Лог уншигч plugin
│   │   │   ├── __init__.py
│   │   │   ├── plugin.py    # LogReaderPlugin
│   │   │   └── prompts.py
│   │   └── code_fixer/      # Код засдаг plugin
│   │       ├── __init__.py
│   │       ├── plugin.py    # CodeFixerPlugin
│   │       └── prompts.py
│   ├── protocol/            # Протоколын тодорхойлолт
│   │   ├── __init__.py
│   │   ├── messages.py
│   │   ├── schema.fbs       # FlatBuffers схем
│   │   └── errors.py
│   └── benchmark/           # Гүйцэтгэлийн хэмжилт
│       ├── __init__.py
│       ├── latency.py
│       ├── accuracy.py
│       └── compare.py       # Текст API vs A2A харьцуулалт
├── examples/
│   ├── homogeneous/         # Ижил моделуудын жишээ
│   │   ├── log_reader.py
│   │   └── code_fixer.py
│   ├── heterogeneous/       # Өөр моделуудын жишээ
│   │   ├── adapter_train.py
│   │   └── cross_model_demo.py
│   └── plugins/             # Custom plugin жишээ
│       └── my_agent.py
├── tests/
├── proto/
│   └── a2a.proto
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 5. Үе 4 — Projection Model Сургалт

### 5.1 Өөр загвар хоорондын векторын үл ойлголцол: Шийдэл

#### Асуудал

Llama-3-8B-ийн `[0.45, -0.12, 0.98, ...]` вектор нь "алдаа гарлаа" гэсэн утгатай.
Харин Mistral-7B-ийн Latent Space-д яг тэр вектор нь "муур" эсвэл утгагүй цэг байж болно.
Учир нь: өөр архитектур, өөр өгөгдлөөр сургагдсан → тэдний Latent Space өөр өөр "зурагдсан".

#### Шийдэл: Гурван түвшний автомат тохируулга

```
ТҮВШИН 1                ТҮВШИН 2                ТҮВШИН 3
(Homogeneous)           (Pre-trained Proj)       (Runtime Auto-Learn)
                       
Хөрвүүлэлт хэрэггүй     Урьдчилан сургасан       Бодит цагт өөрөө сурна
Нэг base model          Projection Model         
Llama↔Llama             Llama↔Mistral             Үл мэдэгдэх model pair
                        (сургалттай)             (auto-trigger)
```

#### Түвшин 1: Homogeneous — шууд холбогдоно
Хоёр AI нэг base model-тэй бол (жишээ нь Llama-3-8B-finetuned-Log ↔ Llama-3-8B-finetuned-Code): **ямар ч хөрвүүлэлт хэрэггүй**. Hidden state-г шууд дамжуулахад л 100% ойлголцоно.

#### Түвшин 2: Pre-trained Projection — урьдчилан сургасан хөрвүүлэгч
Түгээмэл model pair-уудын (Llama↔Mistral, GPT↔Claude, г.м.) projection model-г **урьдчилан сургаж** registry-д хадгална. Discovery хийх үед target model-ийн ID-г шалгаад тохирох projection-г ачаална.

#### Түвшин 3: Runtime Auto-Learning — өөрөө сурдаг
Хэрэв үл мэдэгдэх model pair таарвал:

1. **Discovery phase**: Хоёр AI хоорондоо `MSG_CAPABILITY` солилцож, өөрсдийн model_id, hidden_dim, dtype-г мэдээлнэ.
2. **Auto-trigger**: `MSG_PROJECT_AUTO` илгээгдэнэ.
3. **Data collection**: Хоёр AI-д **ижилхэн текстүүд** өгч, тус тусад нь hidden state гаргана. Энэ нь projection сургах `(src_hidden, tgt_hidden)` хосуудыг үүсгэнэ.
4. **Online training**: Эдгээр хосуудыг ашиглан contrastive loss-ээр Projection model-г **онлайнаар** сургана (ихэвчлэн ~1000-5000 хос хангалттай).
5. **Ready signal**: `MSG_PROJECT_READY` илгээгдэж, сурсан projection model идэвхжинэ.
6. **Cache**: Сурсан projection-г registry-д хадгалж, дараагийн удаа дахин сургахгүй.

#### Projection Model Архитектур

```python
class ProjectionModel(nn.Module):
    """
    Source model-ийн hidden state → Target model-ийн hidden state
    
    Жижиг adapter: 2-3 давхаргат MLP + LayerNorm + Dropout
    Параметр: ~2-8M (маш жижиг, хурдан сурна)
    """
    def __init__(self, src_dim: int, tgt_dim: int, hidden_dim: int = 2048):
        self.norm_in = nn.LayerNorm(src_dim)
        self.fc1 = nn.Linear(src_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, tgt_dim)
        self.norm_out = nn.LayerNorm(tgt_dim)
        self.dropout = nn.Dropout(0.1)
```

#### Сургалтын өгөгдөл бэлтгэх

```
Алгоритм:
1. Dataset = [text_1, text_2, ..., text_N]
2. Source Model → text_i → hidden_state_src[i]  (~4096 dims)
3. Target Model → text_i → hidden_state_tgt[i]  (~1024-8192 dims)
4. Positive pair: (hidden_state_src[i], hidden_state_tgt[i])
5. Negative pair: (hidden_state_src[i], hidden_state_tgt[j≠i])
```

#### Алдагдлын функц (Multi-objective)

```python
def projection_loss(src_proj, tgt_actual, temperature=0.07):
    # 1. Contrastive loss (InfoNCE)
    # 2. MSE loss: ||projected_src - tgt_hidden||²
    # 3. Cosine similarity loss: 1 - cos(projected_src, tgt_hidden)
    return contrastive_loss + 0.1 * mse_loss + 0.01 * cosine_loss
```

#### Projection Model-ийн хувилбарууд

| Хувилбар | Нэр | Давхарга | Параметр | Хэрэглэх нөхцөл |
|---|---|---|---|---|
| A | Linear | W·x + b | src_dim × tgt_dim | dims ижил үед |
| B | 2-layer MLP | 2048 hidden | ~2-4M | dims ялгаатай үед |
| C | 3-layer MLP + Residual | 2048-1024 | ~6-8M | Хэцүү mapping |
| D | Tiny Transformer (2-layer) | 4-head attn | ~8-12M | Sequence-level |

---

## 6. Үе 5 — Benchmark & Optimization

### 6.1 Хэмжих үзүүлэлтүүд

| Үзүүлэлт | Хэмжих зүйл |
|---|---|
| **Latency (RTT)** | Вектор илгээх → хариу авах хугацаа (ms) |
| **Throughput** | Секундэд дамжих тензор тоо |
| **Serialization overhead** | Векторыг bytes руу хөрвүүлэх хугацаа |
| **Injection latency** | Hidden state-г target model-д залгах хугацаа |
| **Semantic fidelity** | Projection хийсэн векторын утга хадгалагдсан эсэх |
| **Task accuracy** | Text API-тай жиших |
| **Token savings** | Текстээр бичсэнтэй харьцуулахад хэмнэсэн токен % |
| **Bandwidth** | Текст vs Тензор дамжуулах сүлжээний ашиглалт |

### 6.2 Харьцуулалт: A2A vs Text API

```
Test case: Error Context дамжуулалт

Text API (одоогийн):
  Input tokens:   ~500  (алдааны лог + тайлбар)
  Output tokens:  ~200  (засвар хийх код)
  Total latency:  ~800ms
  Bandwidth:      ~2-4 KB

A2A Protocol:
  Tensor dims:    4096 × FP16 = 8KB
  Transport:      ~1-2ms (gRPC stream)
  Injection:      ~5-10ms (single forward pass)
  Generation:     ~200ms (код бичих)
  Total latency:  ~210ms
  Bandwidth:      8KB
  Speedup:        ~3.8x
  Token savings:  100% (no text at all)
```

### 6.3 Оновчлолууд

- **Tensor compression**: FP16 → FP8 (E4M3) quantization — latency 50% буурах
- **Zero-copy deserialization**: FlatBuffers нь файлаас шууд memory-map хийх боломжтой
- **Connection pooling**: gRPC connection reuse
- **Batched tensor transfer**: Олон векторыг нэг message-д цуглуулж явуулах
- **Pipeline parallelism**: Receive → Project → Inject → Generate

---

## 7. Үе 6 — SDK, Docs, Integration

### 7.1 Plugin хэлбэрээр SDK ашиглах (BasePlugin API)

```python
# plugins/code_fixer/plugin.py
from a2a.plugins.base import BasePlugin
from a2a.tensor.extractor import TensorExtractor
from a2a.tensor.injector import TensorInjector

class CodeFixerPlugin(BasePlugin):
    plugin_id = "code-fixer"
    plugin_name = "Code Fixer Agent"

    def initialize(self, model, tokenizer, plugin_config, global_config):
        self.model = model
        self.tokenizer = tokenizer
        self.plugin_config = plugin_config
        self.global_config = global_config
        self.injector = TensorInjector(model, target_layer=0)

    def listens_to(self):
        return ["error_context"]

    def emits(self):
        return ["code_patch"]

    async def on_receive_tensor(self, tensor, metadata):
        fixed_code = await self.injector.inject_and_generate(
            tensor,
            prompt="Fix the error shown in context:",
            max_tokens=self.plugin_config.get("max_output_tokens", 1024)
        )
        return fixed_code
```

### 7.2 CLI Tool

```bash
# a2a.yaml-с бүх тохиргоог уншиж ажиллах (plugin-уудыг ачаална)
a2a serve

# Тодорхой config файлаар
a2a serve --config /etc/a2a/production.yaml

# Сүлжээнд байгаа A2A agent-уудыг харах
a2a discover

# Projection model сургах
a2a train-projection --src llama-8b --tgt mistral-7b --data ./projection_corpus.txt

# Benchmark хийх (A2A vs Text API)
a2a benchmark --task error-resolution --iterations 100

# Config validate хийх
a2a config validate

# Идэвхтэй plugin-ууд, routes, model-уудыг харах
a2a config show
a2a config show --section models
a2a config show --section routes
```

### 7.3 Гадны хэрэгсэлтэй интеграц

- **LangChain**: `A2ATool` класс — LangChain agent-ууд A2A plugin-тай тензороор харилцах
- **CrewAI / AutoGen**: Multi-agent orchestration-д A2A сувгийг transport backend болгон залгах
- **vLLM**: OpenAI-compatible API дээр hidden state extraction endpoint нэмэх custom middleware
- **Ollama**: Modelfile-д A2A bridge layer оруулах
- **FastAPI**: A2ARuntime-г REST gateway-тай хослуулах (external trigger-т зориулж)

---

## 8. Plugin/Component Architecture

### 8.1 Суурь философи

A2A Protocol-ийн **core** (тээвэрлэлт, тензор routing, discovery) нь ямар ч plugin-аас хамаарахгүй, plugin-ууд нь core-г эвдэхгүйгээр залгагдах ёстой.

Зарчим: **Core нь үл мэдэх plugin-уудыг дэмжинэ** (Plugin-ууд нь core-ийн интерфейсийг хэрэгжүүлнэ).

### 8.2 Архитектурын тойм

```
┌──────────────────────────────────────────────────┐
│                 A2A RUNTIME (Core)                │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐   │
│  │Transport│  │Discovery │  │Projection Mgr  │   │
│  │(gRPC)   │  │(mDNS)    │  │(Auto-Learn)    │   │
│  └────┬────┘  └────┬─────┘  └───────┬────────┘   │
│       │            │               │             │
│  ┌────┴────────────┴───────────────┴────────┐    │
│  │           PLUGIN MANAGER                 │    │
│  │  - Plugin loading/unloading              │    │
│  │  - Capability registration               │    │
│  │  - Semantic label → Plugin routing       │    │
│  │  - Plugin lifecycle hooks                │    │
│  └────┬────────────┬──────────────┬─────────┘    │
│       │            │              │               │
└───────┼────────────┼──────────────┼───────────────┘
        │            │              │
   ┌────▼────┐  ┌────▼────┐  ┌─────▼─────┐
   │ Plugin  │  │ Plugin  │  │  Plugin   │
   │ Log     │  │ Code    │  │  Custom   │
   │ Reader  │  │ Fixer   │  │  Plugin   │
   └─────────┘  └─────────┘  └───────────┘
```

### 8.3 Plugin бүтэц

Plugin бүр нь дараах бүрэлдэхүүн хэсгээс тогтоно:

```
plugins/log_reader/
├── __init__.py
├── plugin.py          # Plugin класс (BasePlugin-оос удамшсан)
├── prompts.py         # Prompt template-ууд
├── config.yaml        # Plugin-ийн тохиргоо
└── README.md          # Plugin-ийн тайлбар
```

### 8.4 BasePlugin интерфейс

```python
from abc import ABC, abstractmethod

class BasePlugin(ABC):
    """
    Бүх A2A plugin-уудын суурь класс.
    Core A2A runtime нь зөвхөн энэ интерфейсээр plugin-тай харилцана.
    """
    
    # ── Plugin Identity ──
    @property
    @abstractmethod
    def plugin_id(self) -> str: ...
    
    @property
    @abstractmethod
    def plugin_name(self) -> str: ...
    
    @property
    @abstractmethod
    def version(self) -> str: ...
    
    # ── Initialization (PluginManager-аас дуудагдана) ──
    async def initialize(
        self,
        model,             # Loaded model object (HuggingFace, vLLM, etc.)
        tokenizer,         # Tokenizer object
        plugin_config: dict,      # Plugin-ийн дотоод config (plugins/<name>/config.yaml)
        global_config,     # A2AConfig — бүхэл системийн тохиргоо (a2a.yaml)
    ):
        """Plugin-г model, tokenizer, config-тэй холбох."""
        self.model = model
        self.tokenizer = tokenizer
        self.plugin_config = plugin_config
        self.global_config = global_config

    # ── Capabilities ──
    @abstractmethod
    def get_capabilities(self) -> list[Capability]: ...
    
    # ── Semantic Labels ──
    @abstractmethod
    def listens_to(self) -> list[str]: ...
    
    @abstractmethod
    def emits(self) -> list[str]: ...
    
    # ── Tensor Processing ──
    @abstractmethod
    async def on_receive_tensor(
        self,
        tensor: torch.Tensor,
        metadata: A2AMetadata
    ) -> Optional[torch.Tensor]: ...
    
    @abstractmethod
    async def extract_tensor(
        self,
        input_data: Any,
        semantic_label: str
    ) -> torch.Tensor: ...
    
    # ── Lifecycle Hooks ──
    async def on_load(self, runtime: "A2ARuntime"): ...
    async def on_unload(self): ...
    async def on_model_connected(self, model_id: str): ...
    async def on_model_disconnected(self, model_id: str): ...
    
    # ── Projection Support ──
    @abstractmethod
    def get_model_info(self) -> ModelInfo: ...
```

### 8.5 Capability тодорхойлолт

```python
@dataclass
class Capability:
    """Plugin-ийн чадварын мэдээлэл"""
    name: str           # "log_analysis", "code_generation"
    description: str
    input_labels: list[str]   # Хүлээж авах semantic label-үүд
    output_labels: list[str]  # Гаргах semantic label-үүд
    model_id: str
    hidden_dim: int
    dtype: str          # "float16"
    max_batch_size: int
    priority: int = 0   # Олон plugin ижил label сонсоход давуу эрх

@dataclass
class ModelInfo:
    """Plugin-ийн ашиглаж буй моделийн мэдээлэл"""
    model_id: str       # "llama-3-8b"
    architecture: str   # "llama", "mistral", "gpt"
    hidden_dim: int     # 4096
    num_layers: int     # 32
    dtype: str          # "float16"
    supported_projection: list[str] = field(default_factory=list)
```

### 8.6 Plugin Manager

```python
class PluginManager:
    """
    Plugin-уудыг ачаалах, бүртгэх, routing хийх.
    Core A2A runtime-ийн дотор ажиллана.
    """
    
    def __init__(self):
        self._plugins: dict[str, BasePlugin] = {}
        self._label_routes: dict[str, list[BasePlugin]] = {}  # label → plugins
        self._capability_registry: dict[str, Capability] = {}
    
    def register(self, plugin: BasePlugin):
        """Plugin-г бүртгэх"""
        self._plugins[plugin.plugin_id] = plugin
        for cap in plugin.get_capabilities():
            self._capability_registry[cap.name] = cap
        for label in plugin.listens_to():
            if label not in self._label_routes:
                self._label_routes[label] = []
            self._label_routes[label].append(plugin)
    
    def unregister(self, plugin_id: str):
        """Plugin-г хасах"""
        ...
    
    def route_tensor(self, tensor: torch.Tensor, metadata: A2AMetadata):
        """
        Semantic label-аар тохирох plugin-уудыг олж,
        тензорыг дамжуулах.
        """
        label = metadata.semantic_label
        targets = self._label_routes.get(label, [])
        return targets
    
    def discover_from_directory(self, path: str):
        """Тодорхой сангаас plugin-уудыг автоматаар илрүүлэх"""
        ...
    
    def discover_from_entry_points(self, group: str = "a2a.plugins"):
        """Python entry_points механизмаар plugin илрүүлэх"""
        ...
    
    def get_matching_plugins(self, required_labels: list[str]) -> list[BasePlugin]:
        """Шаардлагатай label-үүдэд тохирох plugin-уудыг буцаах"""
        ...
```

### 8.7 Plugin жишээ: LogReaderPlugin

```python
class LogReaderPlugin(BasePlugin):
    """
    Системийн лог уншиж, алдааг илрүүлэн,
    error_context векторыг гаргадаг plugin.
    """
    
    plugin_id = "log-reader"
    plugin_name = "Log Reader Agent"
    version = "1.0.0"
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.extractor = TensorExtractor(model, layer_idx=-1)
    
    def get_capabilities(self):
        return [
            Capability(
                name="log_analysis",
                description="Analyze system logs and extract error context",
                input_labels=[],
                output_labels=["error_context", "log_summary"],
                model_id="llama-3-8b",
                hidden_dim=4096,
                dtype="float16"
            )
        ]
    
    def listens_to(self) -> list[str]:
        return []  # Log reader зөвхөн external trigger-ээр ажиллана
    
    def emits(self) -> list[str]:
        return ["error_context", "log_summary"]
    
    async def extract_tensor(self, input_data: str, semantic_label: str) -> torch.Tensor:
        """Лог текстийг вектор болгох"""
        if semantic_label == "error_context":
            prompt = f"Analyze this error log:\n{input_data}"
            return self.extractor.extract(prompt)
        ...
    
    async def on_receive_tensor(self, tensor, metadata):
        # Log reader хүлээж авдаггүй
        return None
```

### 8.8 Plugin жишээ: CodeFixerPlugin

```python
class CodeFixerPlugin(BasePlugin):
    """
    error_context векторыг хүлээж авч,
    код засах шийдлийг гаргадаг plugin.
    """
    
    plugin_id = "code-fixer"
    plugin_name = "Code Fixer Agent"
    version = "1.0.0"
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.injector = TensorInjector(model, target_layer=0)
    
    def get_capabilities(self):
        return [
            Capability(
                name="code_generation",
                description="Generate code fixes from error context",
                input_labels=["error_context"],
                output_labels=["code_patch", "fix_explanation"],
                model_id="llama-3-8b",
                hidden_dim=4096,
                dtype="float16"
            )
        ]
    
    def listens_to(self) -> list[str]:
        return ["error_context"]  # error_context label-тэй тензорыг сонсоно
    
    def emits(self) -> list[str]:
        return ["code_patch", "fix_explanation"]
    
    async def on_receive_tensor(self, tensor, metadata):
        """error_context векторыг хүлээж аваад код бичих"""
        result = self.injector.inject_prefix(
            tensor,
            prompt="Fix the following error:"
        )
        code = self.injector.generate(result, max_tokens=512)
        return code
    
    async def extract_tensor(self, input_data, semantic_label):
        pass  # Code fixer өөрөө emit хийх нь ховор
```

### 8.9 Plugin ачаалах механизмууд

```python
# 1. Шууд кодоор бүртгэх
from a2a.agent.manager import PluginManager
from a2a.plugins.log_reader import LogReaderPlugin
from a2a.plugins.code_fixer import CodeFixerPlugin

manager = PluginManager()
manager.register(LogReaderPlugin(model, tokenizer))
manager.register(CodeFixerPlugin(model, tokenizer))

# 2. Сангаас автоматаар илрүүлэх
manager.discover_from_directory("./my_plugins/")

# 3. Python entry_points-оор (pip install хийхэд автоматаар)
# pyproject.toml:
# [project.entry-points."a2a.plugins"]
# log_reader = "a2a.plugins.log_reader:LogReaderPlugin"
# code_fixer = "a2a.plugins.code_fixer:CodeFixerPlugin"

manager.discover_from_entry_points("a2a.plugins")

# 4. YAML тохиргоогоор
# config.yaml:
# plugins:
#   - id: log-reader
#     module: a2a.plugins.log_reader
#     class: LogReaderPlugin
#     config: {model: llama-3-8b}
#   - id: my-custom
#     path: /home/user/my_plugin/
manager.load_from_config("config.yaml")
```

### 8.10 Semantic Label Routing — Plugin хоорондын харилцаа

```
┌──────────────┐  error_context   ┌──────────────┐
│  Log Reader   │ ───────────────→│  Code Fixer   │
│  (plugin)     │                  │  (plugin)     │
│              │                  │              │
│ emits:       │                  │ listens_to:  │
│ error_context│                  │ error_context │
└──────────────┘                  └──────┬───────┘
                                         │ code_patch
                                         ▼
                                  ┌──────────────┐
                                  │  PR Creator   │
                                  │  (plugin)     │
                                  │              │
                                  │ listens_to:  │
                                  │ code_patch   │
                                  └──────────────┘
```

Semantic label routing нь **цоохон мэдрэлийн систем шиг** ажиллана:
- Plugin бүр ямар label-үүдийг **сонсож** (listens_to), ямар label-үүдийг **гаргаж** (emits) байгаагаа зарлана
- Plugin Manager эдгээрийг холбож, тензорыг зөв plugin-руу чиглүүлнэ
- Шинэ plugin нэмэхэд бусад plugin-уудад өөрчлөлт орохгүй — зүгээр л шинэ listener болж нэмэгдэнэ

### 8.11 Plugin Discovery Protocol (Сүлжээнд)

Plugin-уудыг **локал сүлжээнд** ч илрүүлж болно:

```
1. Plugin өөрийн A2A Runtime-д бүртгүүлнэ
2. Runtime нь plugin-ийн Capability-г MSG_CAPABILITY болгон broadcast хийнэ
3. Сүлжээнд байгаа бусад Runtime-ууд үүнийг хүлээн авч plugin-ийн чадварыг мэднэ
4. Тензор илгээх үед semantic_label-аар тохирох remote plugin-руу чиглүүлнэ
```

Ингэснээр **Plugin = Distributed Agent** болж, нэг машин дээрх plugin нөгөө машин дээрх plugin-тай шууд тензороор харилцах боломжтой.

### 8.12 Plugin development workflow

Шинэ plugin бичихэд:

1. `BasePlugin`-оос удамшуулна
2. `plugin_id`, `listens_to()`, `emits()`-г тодорхойлно
3. `on_receive_tensor()` дотор логикоо бичнэ
4. Plugin-г Manager-д бүртгүүлнэ (entry_points, config file, эсвэл кодоор)
5. **Core A2A runtime-д ямар ч өөрчлөлт орохгүй**

---

## 9. Single Source of Truth — Тохиргооны Архитектур

### 9.1 Суурь зарчим

**Single Source of Truth гэдэг нь "бүх зүйл нэг файлд" гэсэн үг биш.** Харин:

> **Хуваалцсан нөөц (shared resource) нэг л газар тодорхойлогдож, бусад бүх компонент тэндээс reference хэлбэрээр уншина.**

Жишээ нь DeepSeek API ключ эсвэл Llama-3 загварын тохиргоо нэг л удаа тодорхойлогдож, 10 өөр plugin тэр нэг тодорхойлолтыг нэрээр нь зааж хэрэглэнэ. Хэрэв API ключ солигдвол нэг газар өөрчлөхөд бүх plugin автоматаар шинэ утгыг авна.

**Харин plugin-ийн өөрийн дотоод тохиргоо (max_log_lines, filter_levels, severity_threshold гэх мэт) нь тухайн plugin-ийн дотор л байна.** Учир нь тэр тохиргоо нь зөвхөн тэр plugin-д хамааралтай, хуваалцсан нөөц биш.

### 9.2 Файлын бүтэц

```
a2a-protocol/
├── a2a.yaml                        ← SINGLE SOURCE OF TRUTH (хуваалцсан нөөц)
│                                      - models (API key, provider, path)
│                                      - transport, discovery, projection
│                                      - plugin БҮРТГЭЛ (аль plugin ачаалах)
│                                      - semantic ROUTES (хэн хэнтэй ярих)
│
├── plugins/
│   └── log_reader/
│       ├── plugin.py               ← Plugin код
│       └── config.yaml             ← Plugin-ийн өөрийн тохиргоо
│                                      (max_log_lines, filter_levels...)
│
│   └── code_fixer/
│       ├── plugin.py
│       └── config.yaml             ← Plugin-ийн өөрийн тохиргоо
│                                      (max_output_tokens, languages...)
│
└── projections/
    └── llama-8b_to_mistral-7b.safetensors
```

**Хоёр түвшний тохиргоо:**

| Түвшин | Файл | Агуулга | Жишээ |
|---|---|---|---|
| **Shared (хуваалцсан)** | `a2a.yaml` | Бүх plugin-д нийтлэг нөөц | Model provider, API key env, transport port, routes |
| **Plugin-local (дотоод)** | `plugins/<name>/config.yaml` | Зөвхөн тухайн plugin-д хамааралтай | max_log_lines, filter_levels, severity_threshold |

**Зарчим: Plugin-ийн дотоод config нь `a2a.yaml`-руу орохгүй. Харин plugin-ийн `model: "deepseek"` гэх reference нь `a2a.yaml`-д байна.**

### 9.3 a2a.yaml — Хуваалцсан нөөц (Shared Resource)

```yaml
# =============================================================================
# A2A Protocol — SINGLE SOURCE OF TRUTH (хуваалцсан нөөцийн тохиргоо)
# Файлын байрлал: <project_root>/a2a.yaml  эсвэл  A2A_CONFIG env-р заана
# =============================================================================

version: "1.0"

# ═══════════════════════════════════════════════════════════════════════════
# 1. RUNTIME — Үндсэн runtime тохиргоо
# ═══════════════════════════════════════════════════════════════════════════
runtime:
  agent_id: "node-01"
  log_level: "info"
  log_format: "json"
  metrics_port: 9091
  tensor:
    default_dtype: "float16"
    default_layer: -1
    compression: "none"
    max_sequence_length: 4096
  session:
    ttl_seconds: 300
    max_concurrent: 100

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODELS — Хуваалцсан загваруудын тодорхойлолт
#    ★ DeepSeek API key энд НЭГ удаа тодорхойлогдоно.
#    ★ Бүх plugin эндээс model name-ээр reference хийж авна.
# ═══════════════════════════════════════════════════════════════════════════
models:
  deepseek:
    provider: "openai"                # DeepSeek OpenAI-compatible API
    model_id: "deepseek-chat"
    api_key_env: "DEEPSEEK_API_KEY"   # ★ ГАНЦ газар! env variable нэр
    api_base: "https://api.deepseek.com/v1"
    max_tokens: 65536
    dtype: "float16"

  llama-8b:
    provider: "huggingface"
    model_id: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda:0"
    dtype: "float16"
    load_in_8bit: false
    hf_token_env: "HF_TOKEN"          # ★ ГАНЦ газар!
    max_tokens: 4096

  mistral-7b:
    provider: "ollama"
    model_id: "mistral:7b-instruct"
    device: "cpu"
    dtype: "float16"
    max_tokens: 32768
    api_base: "http://localhost:11434"

  gpt-4o:
    provider: "openai"
    model_id: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"     # ★ ГАНЦ газар!
    max_tokens: 128000

  claude-3:
    provider: "anthropic"
    model_id: "claude-3-opus-20240229"
    api_key_env: "ANTHROPIC_API_KEY"  # ★ ГАНЦ газар!
    max_tokens: 200000

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSPORT — Сүлжээний тээвэрлэлт
# ═══════════════════════════════════════════════════════════════════════════
transport:
  protocol: "grpc"
  host: "0.0.0.0"
  port: 9090
  max_message_size_mb: 64
  serialization:
    format: "flatbuffers"
    zero_copy: true
  tls:
    enabled: false

# ═══════════════════════════════════════════════════════════════════════════
# 4. DISCOVERY — Service discovery
# ═══════════════════════════════════════════════════════════════════════════
discovery:
  mechanism: "static"
  static:
    peers:
      - id: "node-02"
        address: "192.168.1.10:9090"

# ═══════════════════════════════════════════════════════════════════════════
# 5. PROJECTION — Cross-model хөрвүүлэгч
# ═══════════════════════════════════════════════════════════════════════════
projection:
  auto_train:
    enabled: true
    min_pairs: 1000
    epochs: 50
    learning_rate: 0.001
    dataset:
      source: "file"
      path: "./projection_corpus.txt"
  pretrained:
    "llama-8b__mistral-7b": "./projections/llama-8b_to_mistral-7b.safetensors"
    "mistral-7b__llama-8b": "./projections/mistral-7b_to_llama-8b.safetensors"
  architecture:
    variant: "b"
    hidden_dim: 2048
    dropout: 0.1

# ═══════════════════════════════════════════════════════════════════════════
# 6. PLUGIN REGISTRY — Plugin БҮРТГЭЛ (plugin-ийн meta-мэдээлэл, model ref)
#    ★ Plugin-ийн дотоод тохиргоо ЭНД БАЙХГҮЙ.
#    ★ Зөвхөн: аль plugin ачаалах, ямар model ашиглах, config file нь хаана байх.
# ═══════════════════════════════════════════════════════════════════════════
plugins:
  log-reader:
    enabled: true
    module: "a2a.plugins.log_reader"
    class: "LogReaderPlugin"
    model: "llama-8b"              # → models.llama-8b reference
    priority: 10

  code-fixer:
    enabled: true
    module: "a2a.plugins.code_fixer"
    class: "CodeFixerPlugin"
    model: "deepseek"              # → models.deepseek reference ★
    priority: 10

  code-reviewer:
    enabled: true
    module: "a2a.plugins.code_reviewer"
    class: "CodeReviewerPlugin"
    model: "deepseek"              # → models.deepseek reference ★ (НЭГ source!)

  security-analyzer:
    enabled: true
    module: "a2a.plugins.security_analyzer"
    class: "SecurityAnalyzerPlugin"
    model: "mistral-7b"            # → models.mistral-7b reference
    priority: 5

  pr-creator:
    enabled: false
    module: "a2a.plugins.pr_creator"
    class: "PRCreatorPlugin"
    model: "gpt-4o"

# ═══════════════════════════════════════════════════════════════════════════
# 7. SEMANTIC ROUTES — Plugin хоорондын чиглүүлэлт
# ═══════════════════════════════════════════════════════════════════════════
routes:
  "error_context":
    - "code-fixer"
    - "security-analyzer"
  "code_patch":
    - "code-reviewer"
  "log_summary":
    - "security-analyzer"
```

### 9.4 Plugin-ийн дотоод тохиргоо (Plugin-local Config)

Plugin бүр өөрийн дотоод тохиргоог өөрийн сан доторх `config.yaml`-с уншина. Энэ тохиргоо нь **зөвхөн тухайн plugin-д хамааралтай** бөгөөд хуваалцсан нөөц биш:

```yaml
# plugins/log_reader/config.yaml
# Зөвхөн LogReader plugin-ийн дотоод тохиргоо (shared биш)

max_log_lines: 5000
include_stacktrace: true
filter_levels: ["ERROR", "FATAL", "WARN"]
context_lines: 10
log_patterns:
  - pattern: "NullPointerException"
    severity: "high"
  - pattern: "ConnectionTimeout"
    severity: "medium"
```

```yaml
# plugins/code_fixer/config.yaml
# Зөвхөн CodeFixer plugin-ийн дотоод тохиргоо

max_output_tokens: 1024
languages: ["python", "go", "typescript", "rust"]
auto_format: true
temperature: 0.2
system_prompt: "You are an expert code fixer. Respond with code only."
```

```yaml
# plugins/security_analyzer/config.yaml
severity_threshold: 7
scan_patterns:
  - "password"
  - "secret_key"
  - "BEGIN RSA PRIVATE KEY"
alert_webhook_url_env: "SECURITY_WEBHOOK_URL"  # env variable нэр
```

### 9.5 Концепцийн харьцуулалт

```
ХУВААЛЦСАН НӨӨЦ (a2a.yaml)              PLUGIN ДОТООД (plugins/<name>/config.yaml)
─────────────────────────────────────    ─────────────────────────────────────
models.deepseek.api_key_env              log_reader.max_log_lines
models.deepseek.api_base                 code_fixer.max_output_tokens
models.llama-8b.hf_token_env             security_analyzer.severity_threshold
transport.port                           code_fixer.languages
discovery.static.peers                   log_reader.filter_levels
routes.error_context                     code_fixer.system_prompt
                                          security_analyzer.scan_patterns

★ НЭГ удаа тодорхойлогдоно            ★ Зөвхөн тухайн plugin уншина
★ Бүх plugin reference хийж авна      ★ Бусад plugin-д хамааралгүй
★ Өөрчлөлт бүгдэд нөлөөлнө           ★ Өөрчлөлт ганц plugin-д нөлөөлнө
```

**DeepSeek жишээгээр:**
- `DEEPSEEK_API_KEY` env variable нэр `a2a.yaml` → `models.deepseek.api_key_env`-д **нэг удаа** бичигдэнэ
- `code-fixer` plugin: `model: "deepseek"` гэсэн reference
- `code-reviewer` plugin: `model: "deepseek"` гэсэн reference
- Хэрэв DeepSeek API base URL өөрчлөгдвөл `a2a.yaml`-ийн ганц мөр өөрчлөхөд хоёр plugin хоёулаа шинэ URL-г авна

### 9.6 Config Loader — Хоёр түвшний ачаалалт

```python
# a2a/config/loader.py
import os
from pathlib import Path
import yaml
from a2a.config.schema import A2AConfig, PluginConfig

def load_a2a_config(path: str | Path | None = None) -> A2AConfig:
    """
    a2a.yaml-г ачаалах (хуваалцсан нөөц).
    Системийн ганц entry point.
    """
    search_paths = [
        lambda: os.environ.get("A2A_CONFIG"),
        lambda: Path.cwd() / "a2a.yaml",
        lambda: Path.home() / ".config" / "a2a" / "a2a.yaml",
        lambda: Path("/etc/a2a/a2a.yaml"),
    ]
    if path:
        return A2AConfig.from_yaml(path)
    for fn in search_paths:
        p = fn()
        if p and Path(p).exists():
            return A2AConfig.from_yaml(p)
    raise ConfigNotFoundError("a2a.yaml not found")


def load_plugin_config(plugin_path: Path) -> dict:
    """
    Plugin-ийн дотоод тохиргоог ачаалах.
    plugin_path/../config.yaml эсвэл plugin_path/config.yaml
    """
    # Plugin модулийн хажууд байх config.yaml
    config_file = plugin_path.parent / "config.yaml"
    if not config_file.exists():
        # plugin.py-тэй нэг санд байх config.yaml
        config_file = plugin_path.parent.parent / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return {}  # config.yaml байхгүй бол хоосон


class PluginManager:
    def load_plugin(self, entry: PluginEntry, global_config: A2AConfig):
        """Plugin-г бүрэн ачаалах: model resolve + дотоод config унших"""
        
        # 1. Хуваалцсан нөөц: model тохиргоог a2a.yaml-с resolve хийх
        model_config = global_config.resolve_model(entry.model)
        model = load_model(model_config)           # API key энд орж ирнэ
        tokenizer = load_tokenizer(model_config)
        
        # 2. Plugin instance үүсгэх
        plugin = self._instantiate(entry.module, entry.class_)
        
        # 3. Plugin-ийн дотоод config-г plugin-ийн өөрийн сангаас унших
        plugin_config = load_plugin_config(
            Path(inspect.getfile(plugin.__class__))
        )
        
        # 4. Plugin-г initialize хийх
        plugin.initialize(
            model=model,
            tokenizer=tokenizer,
            plugin_config=plugin_config,       # plugins/log_reader/config.yaml
            global_config=global_config        # a2a.yaml (хэрэгтэй бол)
        )
        
        self.register(plugin)
```

### 9.7 Config override дараалал

```
Plugin-ийн тохиргооны эцсийн утга:

1. CLI flag / env variable          (override — хамгийн давуу)
2. plugins/<name>/config.yaml       (plugin-ийн дотоод тохиргоо)
3. a2a.yaml → plugins.<id>          (plugin бүртгэлийн мета өгөгдөл)
4. Code default                     (fallback)
```

Shared нөөцийн эцсийн утга:

```
1. env variable (DEEPSEEK_API_KEY, HF_TOKEN...)  (хамгийн давуу)
2. a2a.yaml → models.<id>                         (single source of truth)
3. Code default                                    (fallback)
```

### 9.8 Хориотой паттернүүд

| Хориотой | Зөв |
|---|---|
| `api_key = "sk-deepseek-xxx"` кодонд hardcode | `a2a.yaml` → `models.deepseek.api_key_env: "DEEPSEEK_API_KEY"` |
| Plugin-д `self.api_key = os.getenv("DEEPSEEK_KEY")` | Plugin `global_config`-с model reference-ээр дамжуулж авна |
| 3 plugin тус тусдаа DeepSeek endpoint бичсэн | `a2a.yaml`-д нэг л удаа, plugin-ууд reference хийх |
| `a2a.yaml`-д `log_reader.max_log_lines: 5000` | `plugins/log_reader/config.yaml`-д байх |
| Plugin-ийн дотоод config `a2a.yaml` руу орсон | Plugin өөрийн `config.yaml`-тай |

### 9.9 CLI

```bash
a2a serve                          # a2a.yaml + plugin config.yaml-уудыг уншина
a2a serve --config /etc/a2a/prod.yaml
a2a config validate                # a2a.yaml validate
a2a config show --section models   # Хуваалцсан model-уудыг харах
a2a config show --section plugins  # Plugin бүртгэл + тус бүрийн model ref
a2a config show --section routes   # Semantic route mapping
```

### 9.10 Deployment

```bash
# Хуваалцсан тохиргоог серверт хуулах (НЭГ файл)
scp a2a.yaml user@server:/etc/a2a/a2a.yaml

# Plugin-уудыг хуулах (plugin бүр өөрийн config.yaml-тэй)
rsync -av plugins/ user@server:/opt/a2a/plugins/

# API token-уудыг env variable-аар дамжуулах (хэзээ ч файлд бичихгүй!)
# Сервер дээр:
export DEEPSEEK_API_KEY="sk-deepseek-xxx"
export HF_TOKEN="hf_xxx"
a2a serve
```

---

## 10. Security & Authentication

### 10.1 Аюулгүй байдлын загвар

A2A mesh сүлжээнд зөвшөөрөлгүй agent орж ирж тензор дамжуулах, хортой вектор injection хийхээс сэргийлэх шаардлагатай. Гурван түвшний хамгаалалттай байна:

```
ТҮВШИН 1: Transport Security (mTLS)
    → Бүх gRPC холболт mutual TLS-ээр хамгаалагдана
    → Agent бүр өөрийн certificate-тай, CA-аар verify хийгдэнэ
    → Certificate-д agent_id, model_id мэдээлэл шифрлэгдсэн байна

ТҮВШИН 2: Agent Authentication (Token-based)
    → Agent бүр pre-shared token эсвэл JWT-ээр баталгаажина
    → Token нь gRPC metadata header-ээр дамжина
    → Agent-ийн эрх: read-only, send-only, full-access

ТҮВШИН 3: Tensor Validation (Sanity Check)
    → Хүлээн авсан тензорын shape, dtype, value range шалгагдана
    → NaN/Inf илрүүлэлт, bounds check
    → Semantic label-ийн хүлээгдэж буй tensor dimension шалгалт
```

### 10.2 Authentication Flow

```
1. Agent A → Agent B: Connect (gRPC channel)
2. Agent B → Agent A: Server certificate (mTLS handshake)
3. Agent A → Agent B: Client certificate + JWT token
4. Agent B: Verify certificate chain + validate JWT
5. Agent B: Check agent_id permissions (RBAC)
6. Agent B → Agent A: Connection established (or rejected)

JWT token-д:
{
  "agent_id": "log-reader-01",
  "model_id": "llama-8b",
  "permissions": ["MSG_UNICAST", "MSG_MULTICAST", "MSG_DISCOVER"],
  "exp": 1234567890,
  "mesh_id": "production-mesh"
}
```

### 10.3 Permission Model (RBAC)

| Role | MSG_UNICAST | MSG_MULTICAST | MSG_DISCOVER | TRAIN_PROJECTION |
|---|---|---|---|---|
| `reader` | ✗ | ✗ | ✓ | ✗ |
| `writer` | ✓ | ✗ | ✓ | ✗ |
| `agent` | ✓ | ✓ | ✓ | ✗ |
| `trainer` | ✓ | ✓ | ✓ | ✓ |
| `admin` | ✓ | ✓ | ✓ | ✓ |

### 10.4 Mesh Isolation (Multi-tenancy)

Нэг сүлжээнд олон тусдаа A2A mesh ажиллах боломжтой. Mesh-үүд хоорондоо харагдахгүй, харилцахгүй.

- `mesh_id` — mesh таних ID
- mesh хоорондын isolation нь TLS + token validation давхаргаар хангагдана
- Discovery (mDNS/registry) mesh_id-аар filter хийгдэнэ

### 10.5 Config дахь Security тохиргоо

```yaml
# a2a.yaml → security section
security:
  tls:
    enabled: true
    cert_file: "/etc/a2a/certs/node-01.crt"
    key_file: "/etc/a2a/certs/node-01.key"
    ca_file: "/etc/a2a/certs/ca.crt"
  
  auth:
    mechanism: "jwt"             # jwt | shared_token | none
    jwt_secret_env: "A2A_JWT_SECRET"
    token_expiry_minutes: 60
  
  rbac:
    default_role: "agent"
    role_assignments:
      "admin-*": "admin"
      "trainer-*": "trainer"
  
  mesh:
    mesh_id: "production-mesh"
    allow_cross_mesh: false
  
  tensor_validation:
    check_nan: true
    check_inf: true
    max_l2_norm: 1000.0
    min_l2_norm: 0.001
```

---

## 11. Error Handling Protocol

### 11.1 Алдааны мессежийн төрөл

Одоогийн протоколд алдааны мессеж байхгүй. Дараах `MSG_ERROR` төрлийг нэмнэ:

| Код | Төрөл | Тайлбар |
|---|---|---|
| 0x0B | `MSG_ERROR` | Алдааны хариу мессеж |

### 11.2 Алдааны кодууд (Error Codes)

```protobuf
message A2AError {
  uint32 error_code = 1;
  string message = 2;
  string source_agent = 3;
  uint64 request_timestamp = 4;
  map<string, string> details = 5;
}

// Error codes:
// 100 - TENSOR_SHAPE_MISMATCH     → Илгээгдсэн тензорын хэлбэр тохирохгүй
// 101 - TENSOR_DTYPE_MISMATCH     → dtype тохирохгүй
// 102 - TENSOR_VALUE_INVALID      → NaN/Inf утга илэрсэн
// 200 - PROJECTION_NOT_FOUND      → Тохирох projection model олдсонгүй
// 201 - PROJECTION_TRAINING_FAILED→ Projection auto-train амжилтгүй
// 300 - MODEL_NOT_FOUND           → Target model олдсонгүй
// 301 - MODEL_OVERLOADED          → Model overloaded, rate limit
// 400 - AUTH_FAILED               → Authentication амжилтгүй
// 401 - PERMISSION_DENIED         → Эрх хүрэлцэхгүй
// 402 - MESH_MISMATCH             → Mesh ID тохирохгүй
// 500 - INTERNAL_ERROR            → Дотоод алдаа (retry боломжтой)
// 501 - PLUGIN_CRASHED            → Plugin унасан
// 502 - GENERATION_FAILED         → Model generation амжилтгүй
```

### 11.3 Retry стратеги

```
Exponential Backoff:
  - Initial delay: 100ms
  - Max delay: 30s
  - Backoff multiplier: 2x
  - Max retries: 5
  - Retryable codes: 500 (INTERNAL), 301 (OVERLOADED)
  - Non-retryable codes: 400 (AUTH), 401 (PERMISSION), 402 (MESH_MISMATCH)
```

### 11.4 Plugin Crash Recovery

```
1. Plugin crash илрүүлэх (unhandled exception in on_receive_tensor)
2. Plugin-ийн error count нэмэгдэнэ
3. 1 минутанд 3+ crash → Plugin temporary disable
4. Admin alert (metrics-р дамжина)
5. Plugin restart (config-д auto_restart: true бол)
6. Crash log + stack trace → structured log-руу бичигдэнэ
```

---

## 12. Rate Limiting & Flow Control

### 12.1 Rate Limiting загвар

Agent бүр өөрийн хурдыг хязгаарлах ёстой. Rate limit нь гурван түвшинд хэрэгжинэ:

| Түвшин | Хязгаарлалт | Тайлбар |
|---|---|---|
| **Per-agent** | Agent бүр хязгаартай | `agent "log-reader" → max 100 tensor/sec` |
| **Per-route** | Semantic route бүр хязгаартай | `"error_context" route → max 50/sec` |
| **Global** | Бүх runtime нийт | `Общий → max 1000 tensor/sec` |

### 12.2 Token Bucket Algorithm

```
Token Bucket (per-agent, per-route):
  - bucket_size:    max burst size
  - refill_rate:    tokens per second
  - refill_interval: 100ms

Жишээ:
  agent "log-reader":
    bucket_size: 20
    refill_rate: 100/sec  → steady-state: 100 tensor/sec
    burst:       20 tensors in single burst
```

### 12.3 Flow Control (Backpressure)

Хүлээн авагч agent overload болсон үед:

```
1. Receiver → Sender: gRPC flow control (HTTP/2 window update)
2. Receiver → Sender: MSG_BACKPRESSURE (албан ёсны дохио)
   {
     "agent_id": "code-fixer",
     "queue_depth": 500,
     "status": "SLOW_DOWN",   // SLOW_DOWN | PAUSE | RESUME
     "suggested_rate": 10     // second-д хэдэн tensor хүлээж авах вэ
   }
3. Sender: Token bucket rate-г тохируулах
4. Receiver queue хоосроход → MSG_BACKPRESSURE(status=RESUME)
```

### 12.4 Config

```yaml
# a2a.yaml
rate_limit:
  enabled: true
  algorithm: "token_bucket"
  
  # Per-agent default
  default:
    bucket_size: 20
    refill_rate: 100       # tokens/sec
  
  # Agent-specific overrides
  agents:
    "log-reader":
      bucket_size: 50
      refill_rate: 200
    "code-fixer":
      bucket_size: 10
      refill_rate: 50      # Код бичих удаан → бага rate

  # Route-specific overrides
  routes:
    "error_context":
      bucket_size: 30
      refill_rate: 100
    "security_alert":
      bucket_size: 100      # Чухал alert-ууд burst-тэй
      refill_rate: 500
  
  flow_control:
    enabled: true
    max_queue_depth: 500     # Queue дүүрвэл backpressure
    backpressure_threshold: 0.8  # 80% дүүрэхэд дохио өгөх
```

---

## 13. Protocol Versioning & Backward Compatibility

### 13.1 Version Negotiation

A2A протоколын хувилбар `major.minor` форматаар явагдана. Холболт тогтоох үед хувилбарын negotiation хийгдэнэ:

```
1. Client → Server: MSG_HELLO { supported_versions: ["1.2", "1.1", "1.0"] }
2. Server → Client: MSG_HELLO_ACK { agreed_version: "1.1" }
   → Сервер хамгийн дээд нийтлэг хувилбарыг сонгоно
3. Хэрэв нийтлэг хувилбар олдохгүй → MSG_ERROR (VERSION_MISMATCH)
```

### 13.2 Compatibility Rules

| Хувилбарын өөрчлөлт | Дүрэм |
|---|---|
| **Major (1.0 → 2.0)** | Breaking changes: хуучин client-ууд ажиллахгүй |
| **Minor (1.0 → 1.1)** | Backward compatible: хуучин client шинэ server-тэй ажиллана |
| **Patch (1.0.0 → 1.0.1)** | Bug fix only, wire protocol өөрчлөгдөхгүй |

### 13.3 Deprecation стратеги

```
1. Шинэ feature-г minor version-д нэмнэ (1.1)
2. Хуучин feature-г DEPRECATED гэж тэмдэглэнэ (1.1 release notes)
3. 2 minor version-ийн дараа (1.3) хуучин feature-г REMOVE хийнэ
4. Deprecation warning-г MSG_ERROR (DEPRECATED) хэлбэрээр мэдэгдэнэ
```

### 13.4 Config versioning

`a2a.yaml`-ийн `version` талбар нь config schema-ийн хувилбар:
- Runtime нь config version-г шалгаж, тохирох schema migration хийх
- Schema migration scripts: `1.0→1.1`, `1.1→2.0`
- Unknown version → алдаа, migration path олдохгүй

---

## 14. Testing Strategy

### 14.1 Тестийн пирамид

```
          ╱ ╲
         ╱   ╲        E2E: Бүрэн mesh, олон agent, projection auto-train
        ╱     ╲       (~5%, удаан, CI-д nightly)
       ╱───────╲
      ╱         ╲     Integration: Plugin-to-plugin, gRPC client-server
     ╱           ╲    (~20%, CI-д PR бүрт)
    ╱─────────────╲
   ╱               ╲  Unit: Tensor ops, serializer, config parser
  ╱                 ╲ (~75%, хурдан, CI-д commit бүрт)
 ─────────────────────
```

### 14.2 Unit Tests (75%)

```python
# tests/tensor/test_extractor.py
class TestTensorExtractor:
    def test_extract_last_hidden_shape(self): ...
    def test_extract_specific_layer(self): ...
    def test_dtype_conversion_fp16_to_fp32(self): ...

# tests/protocol/test_codec.py
class TestFlatBuffersCodec:
    def test_encode_decode_roundtrip(self): ...
    def test_metadata_serialization(self): ...

# tests/config/test_schema.py
class TestConfigValidation:
    def test_valid_config(self): ...
    def test_missing_required_field(self): ...
    def test_invalid_model_reference(self): ...

# tests/projection/test_adapter.py
class TestProjectionModel:
    def test_forward_shape(self): ...
    def test_cosine_similarity_improves_during_training(self): ...
```

### 14.3 Integration Tests (20%)

```python
# tests/integration/test_plugin_communication.py
class TestPluginToPlugin:
    """LogReaderPlugin → CodeFixerPlugin (ижил model, projection хэрэггүй)"""
    async def test_error_context_flow(self):
        log_reader = create_plugin("log-reader")
        code_fixer = create_plugin("code-fixer")
        tensor = await log_reader.extract_tensor("NullPointerException at line 45", "error_context")
        result = await code_fixer.on_receive_tensor(tensor, metadata)
        assert "try" in result or "if" in result  # код шинжтэй

# tests/integration/test_grpc_transport.py
class TestGRPCTransport:
    async def test_agent_discovery(self): ...
    async def test_tensor_send_receive(self): ...
    async def test_backpressure_signal(self): ...

# tests/integration/test_projection_auto_train.py
class TestAutoProjection:
    """Llama → Mistral cross-model auto-training"""
    async def test_auto_trigger_and_train(self): ...
```

### 14.4 E2E Tests (5%)

```python
# tests/e2e/test_full_mesh.py
class TestFullMesh:
    """3 agent: log-reader(llama), code-fixer(deepseek), security(mistral)"""
    async def test_error_resolution_pipeline(self):
        # 1. Log reader лог унших
        # 2. Code fixer-руу error_context дамжуулах (projection: llama→deepseek)
        # 3. Security analyzer-руу log_summary дамжуулах (projection: llama→mistral)
        # 4. Code fixer гаргасан patch-г verify хийх
        ...
```

### 14.5 Benchmark Tests

```python
# tests/benchmark/test_latency.py
def test_text_api_vs_a2a_latency():
    """Ижил error context, text API (800ms) vs A2A (210ms)"""
    ...

def test_projection_overhead():
    """Projection model forward pass-ийн latency overhead"""
    ...
```

### 14.6 CI/CD Integration

- **GitHub Actions**: PR бүрт unit + integration тестүүд
- **Nightly**: E2E mesh тест + benchmark regression
- **Coverage target**: ≥ 80% (unit), ≥ 60% (integration)
- **Test matrix**: Python 3.11, 3.12 × Linux (GPU) + macOS (CPU)
- **Mock model**: `tests/fixtures/tiny_model/` — 2-layer dummy transformer, тест хурдасгах

---

## 15. Monitoring & Observability

### 15.1 Metrics (Prometheus)

```
# Tensor Flow Metrics
a2a_tensors_sent_total{agent_id, semantic_label}         # илгээсэн тензор
a2a_tensors_received_total{agent_id, semantic_label}      # хүлээн авсан
a2a_tensor_latency_seconds{agent_id, semantic_label}      # дамжуулалтын хугацаа
a2a_tensor_size_bytes{agent_id}                           # тензорын хэмжээ

# Projection Metrics
a2a_projection_requests_total{src_model, tgt_model}       # projection хүсэлт
a2a_projection_auto_train_total{src_model, tgt_model}     # auto-train тоо
a2a_projection_train_duration_seconds{src_model, tgt_model}

# Plugin Health
a2a_plugin_active{plugin_id}                              # идэвхтэй plugin
a2a_plugin_crash_total{plugin_id}                         # plugin crash тоо
a2a_plugin_error_total{plugin_id, error_code}             # алдааны тоо

# Transport Metrics
a2a_grpc_connections_active                               # идэвхтэй холболт
a2a_rate_limit_hits_total{agent_id}                       # rate limit хүрсэн
a2a_backpressure_events_total{agent_id, status}           # backpressure дохио
```

### 15.2 Health Check Endpoints

```
GET /health          → {"status": "ok", "uptime": 3600, "version": "1.0"}
GET /health/ready    → {"ready": true, "plugins_loaded": 5, "models_ready": 3}
GET /health/live     → {"alive": true}
GET /metrics         → Prometheus metrics
```

### 15.3 Distributed Tracing (OpenTelemetry)

Trace span бүр:
```
[trace: error-resolution-abc123]
  ├── span: log-reader.extract_tensor  [10ms]
  ├── span: projection.forward         [3ms]   (llama→deepseek)
  ├── span: transport.send             [2ms]   (gRPC)
  ├── span: transport.receive          [1ms]
  ├── span: code-fixer.inject          [5ms]
  └── span: code-fixer.generate        [200ms]
```

### 15.4 Structured Logging

```json
{
  "timestamp": "2026-07-07T12:00:00Z",
  "level": "info",
  "event": "tensor_forwarded",
  "trace_id": "abc123",
  "source_plugin": "log-reader",
  "target_plugin": "code-fixer",
  "semantic_label": "error_context",
  "tensor_shape": [1, 4096],
  "tensor_dtype": "float16",
  "projection_used": "llama-8b__deepseek",
  "latency_ms": 15
}
```

---

## 16. Scaling & Deployment Strategy

### 16.1 Scaling Model

```
SINGLE NODE                     MULTI-NODE
┌──────────────────┐            ┌──────────────────┐
│ A2A Runtime      │            │ A2A Runtime      │
│ ┌──┐ ┌──┐ ┌──┐  │            │ ┌──┐ ┌──┐ ┌──┐  │
│ │P1│ │P2│ │P3│  │    gRPC    │ │P4│ │P5│ │P6│  │
│ └──┘ └──┘ └──┘  │◄──────────►│ └──┘ └──┘ └──┘  │
└──────────────────┘            └──────────────────┘
  1 GPU (Llama)                   1 GPU (Mistral)
  plugins 1-3                     plugins 4-6
```

Plugin-ууд нь model-оор grouped:
- Нэг node дээр нэг model (GPU memory constraint)
- Олон plugin нэг model хуваалцаж болно (context switching)

### 16.2 Deployment Scenarios

| Scenario | Plugin тоо | Node тоо | Ашиглах үед |
|---|---|---|---|
| **Dev** | 2-3 plugin | 1 node (local) | Хөгжүүлэлт, тест |
| **Stage** | 3-5 plugin | 1-2 node | Интеграц тест |
| **Production** | 5-20 plugin | 3-10 node | Бодит ашиглалт |

### 16.3 Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY a2a/ /opt/a2a/a2a/
COPY plugins/ /opt/a2a/plugins/
COPY a2a.yaml /etc/a2a/a2a.yaml
COPY projections/ /etc/a2a/projections/

RUN pip install -e /opt/a2a/

EXPOSE 9090 9091

CMD ["a2a", "serve", "--config", "/etc/a2a/a2a.yaml"]
```

### 16.4 Kubernetes Deployment

```yaml
# k8s/a2a-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a2a-runtime
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: a2a
        image: a2a-protocol:latest
        ports:
        - containerPort: 9090
        - containerPort: 9091
        env:
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: a2a-secrets
              key: deepseek-key
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: a2a-runtime
spec:
  ports:
  - port: 9090
    name: grpc
  - port: 9091
    name: metrics
```

### 16.5 CI/CD Pipeline

```
Git Push → GitHub Actions:
  1. Lint (ruff) + type-check (mypy)
  2. Unit tests (parallel matrix)
  3. Integration tests (docker-compose)
  4. Build Docker image
  5. Push to container registry
  6. Deploy to staging (auto)
  7. E2E tests (staging)
  8. Deploy to production (manual approval)

Nightly:
  9. E2E mesh full test
  10. Benchmark regression test
  11. Security scan (bandit, safety, trivy)
```

### 16.6 Release Model

- **PyPI package**: `pip install a2a-protocol` — SDK + CLI
- **Docker image**: `ghcr.io/mbm/a2a-runtime` — Багцлагдсан runtime
- **Plugins**: `pip install a2a-plugin-log-reader` — Plugin-ууд тусдаа PyPI package
- **Semantic versioning**: `major.minor.patch` (1.0.0, 1.1.0, 2.0.0)

---

## 17. Үе шат, Цагийн Хуваарь & MVP

| Үе | Нэр | Хугацаа | Хуримтлагдсан |
|---|---|---|---|
| 1 | Судалгаа & Үндэслэл | 2-3 w | 3 w |
| 2 | Протоколын Спецификаци | 2-3 w | 6 w |
| 3 | Core SDK & Runtime (Python) | 4-6 w | 12 w |
| 4 | Projection Model Сургалт | 3-4 w | 16 w |
| 5 | Benchmark & Optimization | 2-3 w | 19 w |
| 6 | SDK, Docs, Integration | 2-3 w | 22 w |
| **Нийт** | | **~5-6 сар** | |

### MVP тодорхойлолт

Эхний хувилбарт:

1. **Нэг загварын гэр бүл** доторх A2A (Llama-3-8B ↔ Llama-3-8B)
2. **gRPC + FlatBuffers** тээвэрлэлт
3. **Hidden state extraction** (сүүлийн давхарга) + **Prefix injection**
4. **Plugin Manager** + 2 plugin (LogReader, CodeFixer)
5. **Log → Code fix demo**
6. **Text API vs A2A benchmark**

Хоёр дахь хувилбарт:
- Projection Model → Heterogeneous дэмжлэг
- Auto-Learning projection
- Go Transport Layer
