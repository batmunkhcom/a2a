# API Reference

## Package: `a2a`

```python
from a2a import __version__
```

### `a2a.runtime`

```python
class A2ARuntime:
    def __init__(self, config: A2AConfig) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def is_running(self) -> bool: ...
    @property
    def plugin_manager(self) -> PluginManager: ...
    @property
    def uptime_seconds(self) -> float: ...
```

---

## Transport

### `a2a.transport.client.A2AClient`

```python
class A2AClient:
    def __init__(self, target: str) -> None: ...
    def send_tensor(
        self, agent_id: str, label: str, data: np.ndarray
    ) -> np.ndarray: ...
    def health_check(self) -> dict[str, Any]: ...
    def discover(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...
```

### `a2a.transport.codec.FlatBuffersCodec`

```python
class FlatBuffersCodec:
    def encode(self, tensor: np.ndarray, dtype: str = "fp32") -> bytes: ...
    def decode(self, data: bytes) -> np.ndarray: ...
```

---

## Tensor Engine

### `a2a.tensor.extractor.TensorExtractor`

```python
class TensorExtractor:
    def __init__(self, model, strategy: str = "last") -> None: ...
    def extract(self, inputs) -> np.ndarray: ...
    @property
    def hidden_dim(self) -> int: ...
```

### `a2a.tensor.injector.TensorInjector`

```python
class TensorInjector:
    def __init__(self, model, method: str = "prefix") -> None: ...
    def inject(self, inputs, tensor: np.ndarray) -> Any: ...
    def clear(self) -> None: ...
```

### `a2a.tensor.dtype.DtypeConverter`

```python
class DtypeConverter:
    @staticmethod
    def to_float32(data: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def to_float16(data: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def to_bfloat16(data: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def from_float32(data: np.ndarray, target_dtype: str) -> np.ndarray: ...
    @staticmethod
    def validate(tensor: np.ndarray) -> bool: ...
```

### `a2a.tensor.serializer.TensorSerializer`

```python
class TensorSerializer:
    def save(self, tensor: np.ndarray, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> np.ndarray: ...
```

---

## Plugin System

### `a2a.agent.base.BasePlugin`

```python
class BasePlugin(ABC):
    agent_id: str
    model: str
    labels: list[str]
    config: dict[str, Any]

    @abstractmethod
    def process_tensor(
        self, data: np.ndarray, label: str, source_agent: str
    ) -> np.ndarray: ...

    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    @property
    def initialized(self) -> bool: ...
```

### `a2a.agent.base.PluginRegistry`

```python
class PluginRegistry:
    def register(self, plugin: BasePlugin) -> None: ...
    def unregister(self, agent_id: str) -> None: ...
    def get(self, agent_id: str) -> BasePlugin | None: ...
    def list_agents(self) -> list[str]: ...
    def discover(self) -> list[dict[str, Any]]: ...
```

### `a2a.agent.manager.PluginManager`

```python
class PluginManager:
    def __init__(self) -> None: ...
    def load_plugin(
        self, module_path: str, agent_id: str
    ) -> BasePlugin: ...
    def register(
        self, plugin: BasePlugin, agent_id: str, model: str
    ) -> None: ...
    def initialize_all(self) -> None: ...
    def shutdown_all(self) -> None: ...
    def route_tensor(
        self, agent_id: str, label: str, data: np.ndarray, source: str
    ) -> np.ndarray: ...
```

### `a2a.agent.router.SemanticRouter`

```python
class SemanticRouter:
    def add_route(self, route: RouteConfig) -> None: ...
    def resolve(self, agent_id: str, label: str) -> RouteConfig | None: ...
```

---

## Configuration

### `a2a.config.schema.A2AConfig`

```python
class A2AConfig(BaseModel):
    version: str
    mesh_id: str
    server: ServerConfig
    security: SecurityConfig
    rate_limiting: RateLimitConfig
    monitoring: MonitorConfig
    models: dict[str, ModelConfig]
    plugins: dict[str, PluginEntry]
    routes: list[RouteConfig]

class ModelConfig(BaseModel): ...
class PluginEntry(BaseModel): ...
class RouteConfig(BaseModel): ...
class SecurityConfig(BaseModel): ...
class RateLimitConfig(BaseModel): ...
```

### `a2a.config.loader`

```python
def load_config(path: str | None = None) -> A2AConfig: ...
def find_config() -> str: ...
```

---

## Projection

### `a2a.projection.adapter.ProjectionModel`

```python
class ProjectionModel:
    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        variant: str = "b",
        hidden_size: int = 1024,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(self, x: np.ndarray) -> np.ndarray: ...
```

### `a2a.projection.trainer.ProjectionTrainer`

```python
class ProjectionTrainer:
    def __init__(
        self, model: ProjectionModel, loss: str = "infonce"
    ) -> None: ...
    def train(
        self, pairs: list[tuple[np.ndarray, np.ndarray]], epochs: int = 10
    ) -> None: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> ProjectionTrainer: ...
```

### `a2a.projection.auto_trainer.AutoTrainer`

```python
class AutoTrainer:
    def __init__(self, runtime) -> None: ...
    async def run(self) -> None: ...
```

---

## Security

### `a2a.security.auth`

```python
def create_jwt(
    secret: str,
    mesh_id: str,
    agent_id: str,
    algorithm: str = "HS256",
    expiry: int = 3600,
) -> str: ...

def validate_jwt(
    token: str, secret: str, mesh_id: str | None = None
) -> dict[str, Any] | None: ...
```

### `a2a.monitoring.rate_limiter`

```python
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float) -> None: ...
    def consume(self, tokens: int = 1) -> bool: ...

class RateLimiter:
    def __init__(self, config: RateLimitConfig) -> None: ...
    def check(self, agent_id: str) -> bool: ...
```

### `a2a.monitoring.metrics`

```python
class MetricsRegistry:
    def increment(self, name: str, value: int = 1) -> None: ...
    def observe(self, name: str, value: float) -> None: ...
    def set_gauge(self, name: str, value: float) -> None: ...
    def get_counter(self, name: str) -> int: ...
    def get_histogram(self, name: str) -> dict: ...
    def get_gauge(self, name: str) -> float: ...
    def render_text(self) -> str: ...
    def record_tensor_sent(self, agent_id: str, label: str, size: int) -> None: ...
    def record_latency(self, agent_id: str, label: str, latency_s: float) -> None: ...

def get_metrics() -> MetricsRegistry: ...
```

---

## Exceptions

```python
class A2AError(Exception): ...              # Base
class A2AConfigError(A2AError): ...         # Config loading/validation
class A2ACodecError(A2AError): ...          # Tensor encode/decode
class A2AAuthError(A2AError): ...           # JWT validation failure
class A2ARateLimitError(A2AError): ...      # Rate limit exceeded
class A2APluginError(A2AError): ...         # Plugin loading/runtime
class A2ATransportError(A2AError): ...      # gRPC transport failure
```
