# Plugin Development Guide

## Overview

A2A plugins process latent tensors — they receive tensors from other agents,
apply transformations or reasoning, and return results. Plugins are dynamically
loaded from Python modules.

---

## Quick Start

### 1. Create a plugin module

```
my_mesh/
├── a2a.yaml
└── plugins/
    └── my_filter.py
```

### 2. Implement BasePlugin

```python
"""my_filter.py — Simple tensor filter plugin."""
import numpy as np
from a2a.agent.base import BasePlugin
from a2a.tensor.dtype import DtypeConverter


class NoiseFilterPlugin(BasePlugin):
    """Filters out low-magnitude tensors below a threshold."""

    def __init__(self):
        super().__init__(
            agent_id="noise-filter",
            model="base",
            labels=["hidden_state", "attention"],
        )
        self.threshold = 0.01

    def process_tensor(
        self,
        data: np.ndarray,
        label: str,
        source_agent: str,
    ) -> np.ndarray:
        """Zero out tensor elements below threshold."""
        mask = np.abs(data) < self.threshold
        data = data.copy()
        data[mask] = 0.0
        return data

    async def initialize(self) -> None:
        """Called once when the plugin is loaded."""
        self.threshold = float(self.config.get("threshold", 0.01))

    async def shutdown(self) -> None:
        """Cleanup before unloading."""
        pass
```

### 3. Register in `a2a.yaml`

```yaml
plugins:
  noise_filter:
    module: plugins.my_filter
    agent_id: noise-filter
    model: base
    labels:
      - hidden_state
      - attention
    config:
      threshold: 0.02
```

---

## BasePlugin API

```python
class BasePlugin(ABC):
    def __init__(
        self,
        agent_id: str,
        model: str,
        labels: list[str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.labels = labels or []
        self.config: dict[str, Any] = {}
        self._initialized = False

    @abstractmethod
    def process_tensor(
        self,
        data: np.ndarray,
        label: str,
        source_agent: str,
    ) -> np.ndarray:
        """Process an incoming tensor. Must return a tensor of the same shape.
        Called synchronously — must be thread-safe."""

    async def initialize(self) -> None:
        """Async initialization. Called once after loading."""

    async def shutdown(self) -> None:
        """Async cleanup. Called before unloading."""

    @property
    def initialized(self) -> bool:
        """Whether initialize() has completed."""

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Return plugin capabilities for discovery."""
```

---

## Lifecycle

```
load_module → instantiate → initialize() → process_tensor()* → shutdown()
```

1. **load_module** — PluginManager imports the module via `importlib`
2. **instantiate** — Plugin class constructor called
3. **initialize()** — Async init (config, model loading, connections)
4. **process_tensor()** — Called repeatedly for each incoming tensor
5. **shutdown()** — Async cleanup on mesh stop

---

## Tensor Processing

### Input

```python
def process_tensor(
    self,
    data: np.ndarray,        # shape: (batch, dim) or (dim,)
    label: str,              # semantic label from routing
    source_agent: str,       # agent_id of sender
) -> np.ndarray:             # must return same shape
```

### Shape Conventions

| Scenario | Input Shape |
|---|---|
| Single hidden state | `(dim,)` |
| Batched hidden states | `(batch, dim)` |
| Sequence of states | `(seq, batch, dim)` |

### Dtype Support

Always check and convert dtype:

```python
from a2a.tensor.dtype import DtypeConverter

def process_tensor(self, data, label, source_agent):
    data_fp32 = DtypeConverter.to_float32(data)
    # ... process ...
    return DtypeConverter.from_float32(result, data.dtype.name)
```

---

## Available Labels (by convention)

| Label | Semantics | Typical Shape |
|---|---|---|
| `hidden_state` | Last layer hidden state | `(dim,)` |
| `error_context` | Error/debug context tensor | `(batch, dim)` |
| `attention` | Attention map | `(heads, seq, seq)` |
| `logits` | Output logits | `(vocab_size,)` |
| `embedding` | Token embeddings | `(seq, dim)` |
| `kv_cache` | Key-value cache | `(layers, 2, heads, seq, head_dim)` |
| `gradient` | Gradient information | `(dim,)` |
| `uncertainty` | Uncertainty estimate | `(dim,)` |

---

## Best Practices

1. **Copy before mutate** — Input tensors may be shared. Call `.copy()` if modifying.
2. **Validate shape** — Check `data.ndim` and `data.shape[-1]` matches expected dim.
3. **Thread safety** — `process_tensor` is called from thread-pool. Use locks if needed.
4. **Stateless preferred** — Avoid mutable state in plugins. Use config for parameters.
5. **Fast path** — Avoid Python loops; use numpy vectorized ops.
6. **Error handling** — Raise `np` exceptions; framework converts to gRPC errors.

---

## Built-in Plugins

### LogReader (`a2a.plugins.log_reader`)

Parses log lines into tensor representations. Handles timestamps, severity,
and structured log fields.

**Labels:** `error_context`, `hidden_state`

### CodeFixer (`a2a.plugins.code_fixer`)

Receives error context tensors and generates fix suggestions.

**Labels:** `error_context`, `hidden_state`

---

## Testing Plugins

```python
import numpy as np
import pytest
from a2a.agent.manager import PluginManager

def test_my_plugin():
    mgr = PluginManager()
    plugin = mgr.load_plugin("plugins.my_filter", "test-filter")
    mgr.register(plugin, "test-filter", "base")

    tensor = np.random.randn(768).astype(np.float32)
    result = plugin.process_tensor(tensor, "hidden_state", "test-sender")

    assert result.shape == tensor.shape
    assert result.dtype == np.float32
```

See `tests/plugins/` for more examples.
