"""A2AConfig — Pydantic model for full A2A runtime configuration.

Single source of truth for all shared resources (models, transport,
discovery, projection, plugins, routes, security, rate limiting).

Load from yaml: A2AConfig.from_yaml("a2a.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ── Sub-models ─────────────────────────────────────────────────


class RuntimeConfig(BaseModel):
    agent_id: str = "a2a-node"
    log_level: str = "info"
    log_format: str = "json"
    metrics_port: int = 9091
    default_dtype: str = "float16"
    default_layer: int = -1
    compression: str = "none"
    max_sequence_length: int = 4096
    session_ttl_seconds: int = 300
    max_concurrent_sessions: int = 100


class ModelConfig(BaseModel):
    provider: str = "huggingface"  # huggingface | openai | ollama | anthropic | vllm
    model_id: str
    api_key_env: str | None = None  # env var name, e.g. "DEEPSEEK_API_KEY"
    api_base: str | None = None
    device: str = "cpu"
    dtype: str = "float16"
    max_tokens: int = 4096
    load_in_8bit: bool = False
    hf_token_env: str | None = None


class TransportConfig(BaseModel):
    protocol: str = "grpc"
    host: str = "0.0.0.0"
    port: int = 9090
    max_message_size_mb: int = 64
    serialization_format: str = "flatbuffers"
    zero_copy: bool = True
    tls_enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None


class DiscoveryConfig(BaseModel):
    mechanism: str = "static"  # static | mdns | consul | registry
    static_peers: list[dict[str, str]] = Field(default_factory=list)


class AutoTrainConfig(BaseModel):
    enabled: bool = True
    min_pairs: int = 1000
    epochs: int = 50
    learning_rate: float = 0.001
    dataset_source: str = "file"
    dataset_path: str = "./projection_corpus.txt"


class ProjectionArchitecture(BaseModel):
    variant: str = "b"  # a | b | c | d
    hidden_dim: int = 2048
    dropout: float = 0.1


class ProjectionConfig(BaseModel):
    auto_train: AutoTrainConfig = Field(default_factory=AutoTrainConfig)
    pretrained: dict[str, str] = Field(default_factory=dict)
    architecture: ProjectionArchitecture = Field(default_factory=ProjectionArchitecture)


class PluginEntryConfig(BaseModel):
    enabled: bool = True
    module: str
    class_name: str = ""
    model: str = ""
    priority: int = 10
    config_path: str = ""


class SecurityConfig(BaseModel):
    tls_enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None
    auth_mechanism: str = "none"  # none | jwt | shared_token
    jwt_secret_env: str | None = None
    token_expiry_minutes: int = 60
    default_role: str = "agent"
    mesh_id: str = "default"
    allow_cross_mesh: bool = False
    check_nan: bool = True
    check_inf: bool = True
    max_l2_norm: float = 1000.0
    min_l2_norm: float = 0.001


class AgentRateLimit(BaseModel):
    bucket_size: int = 20
    refill_rate: float = 100.0


class RateLimitConfig(BaseModel):
    enabled: bool = True
    algorithm: str = "token_bucket"
    default: AgentRateLimit = Field(default_factory=AgentRateLimit)
    agents: dict[str, AgentRateLimit] = Field(default_factory=dict)
    routes: dict[str, AgentRateLimit] = Field(default_factory=dict)
    max_queue_depth: int = 500
    backpressure_threshold: float = 0.8


# ── Main config model ──────────────────────────────────────────


class A2AConfig(BaseModel):
    """Root configuration loaded from a2a.yaml."""

    version: str = "1.0"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    plugins: dict[str, PluginEntryConfig] = Field(default_factory=dict)
    routes: dict[str, list[str]] = Field(default_factory=dict)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # ── Methods ────────────────────────────────────────────────

    def resolve_model(self, model_key: str) -> ModelConfig | None:
        """Resolve a model reference by key from the models section.

        Args:
            model_key: Key in the models dict, e.g. "deepseek".

        Returns:
            ModelConfig if found, None otherwise.

        Raises:
            ValueError: If the model key is not configured.
        """
        if model_key not in self.models:
            raise ValueError(
                f"Model '{model_key}' not found in config.models. "
                f"Available: {list(self.models.keys())}"
            )
        return self.models[model_key]

    def get_enabled_plugins(self) -> list[tuple[str, PluginEntryConfig]]:
        """Return all enabled plugin entries."""
        return [
            (pid, cfg) for pid, cfg in self.plugins.items() if cfg.enabled
        ]

    def validate_config(self) -> list[str]:
        """Return list of validation warnings. Empty list = valid."""
        warnings: list[str] = []

        # Check all plugin model references
        for pid, cfg in self.plugins.items():
            if cfg.enabled and cfg.model and cfg.model not in self.models:
                warnings.append(
                    f"Plugin '{pid}' references model '{cfg.model}' "
                    f"which is not defined in config.models"
                )

        # Check route targets reference existing plugins
        for label, targets in self.routes.items():
            for target in targets:
                if target not in self.plugins:
                    warnings.append(
                        f"Route '{label}' targets plugin '{target}' "
                        f"which is not defined in config.plugins"
                    )

        # Check transport port conflicts
        if self.transport.port == self.runtime.metrics_port:
            warnings.append("Transport port and metrics port are the same")

        return warnings

    @classmethod
    def from_yaml(cls, path: str | Path) -> A2AConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to a2a.yaml.

        Returns:
            Validated A2AConfig instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            pydantic.ValidationError: If the config is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Config must be a YAML mapping, got {type(data).__name__}")

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AConfig:
        """Create config from a dict (useful for tests)."""
        return cls.model_validate(data)
