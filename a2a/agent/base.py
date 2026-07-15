"""BasePlugin — abstract base class for all A2A plugins.

Every plugin (LogReader, CodeFixer, Security, etc.) inherits from BasePlugin.
The A2A Core Runtime communicates with plugins exclusively through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ── Data Structures ────────────────────────────────────────────


@dataclass
class Capability:
    """Declares what a plugin can do — announced over the network."""

    name: str
    description: str
    input_labels: list[str] = field(default_factory=list)
    output_labels: list[str] = field(default_factory=list)
    model_id: str = ""
    hidden_dim: int = 0
    dtype: str = "float32"
    max_batch_size: int = 1
    priority: int = 0


@dataclass
class ModelInfo:
    """Information about the model a plugin uses."""

    model_id: str = ""
    architecture: str = ""
    hidden_dim: int = 0
    num_layers: int = 0
    dtype: str = "float32"
    supported_projection: list[str] = field(default_factory=list)


@dataclass
class PluginEntry:
    """Registry entry for a plugin — loaded from a2a.yaml."""

    plugin_id: str
    module: str
    class_name: str
    model: str = ""
    enabled: bool = True
    priority: int = 10
    config_path: str = ""  # Path to plugin-local config.yaml


# ── BasePlugin ─────────────────────────────────────────────────


class BasePlugin(ABC):
    """Abstract base for all A2A plugins.

    Subclasses must implement:
    - plugin_id (property)
    - plugin_name (property)
    - version (property)
    - get_capabilities()
    - get_model_info()
    - listens_to()
    - emits()
    - on_receive_tensor()
    - extract_tensor()

    Lifecycle hooks (optional override):
    - initialize()   — called after model/tokenizer are assigned
    - on_load()      — called when runtime loads the plugin
    - on_unload()    — called when runtime unloads the plugin
    """

    # ── Identity ──────────────────────────────────────────────

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier, e.g. 'log-reader'."""
        ...

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Human-readable name, e.g. 'Log Reader Agent'."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version, e.g. '1.0.0'."""
        ...

    # ── Initialization ────────────────────────────────────────

    async def initialize(
        self,
        model: Any = None,
        tokenizer: Any = None,
        plugin_config: dict[str, Any] | None = None,
        global_config: Any = None,
    ) -> None:
        """Called by PluginManager after loading the plugin.

        Assigns the model, tokenizer, and configs that the plugin
        needs to operate. Plugins can override this to perform
        setup (e.g., initialize extractors/injectors).

        Args:
            model: Loaded model object (HF, vLLM, etc.)
            tokenizer: Tokenizer for the model.
            plugin_config: Plugin-local config from config.yaml.
            global_config: The full A2AConfig (a2a.yaml).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.plugin_config = plugin_config or {}
        self.global_config = global_config

    # ── Capabilities ──────────────────────────────────────────

    @abstractmethod
    def get_capabilities(self) -> list[Capability]:
        """Return this plugin's capabilities."""
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return information about the model this plugin uses."""
        ...

    @abstractmethod
    def listens_to(self) -> list[str]:
        """Semantic labels this plugin subscribes to.

        Example: ["error_context", "log_summary"]
        """
        ...

    @abstractmethod
    def emits(self) -> list[str]:
        """Semantic labels this plugin produces.

        Example: ["code_patch", "fix_explanation"]
        """
        ...

    # ── Tensor Processing ─────────────────────────────────────

    @abstractmethod
    async def on_receive_tensor(
        self,
        tensor: Any,
        metadata: Any,
    ) -> Any:
        """Called when a tensor with a matching semantic label arrives.

        Args:
            tensor: The received tensor (PyTorch tensor or raw bytes).
            metadata: TensorMetadata protobuf from the sender.

        Returns:
            Optional response tensor, or None if no response needed.
        """
        ...

    @abstractmethod
    async def extract_tensor(
        self,
        input_data: Any,
        semantic_label: str,
    ) -> Any:
        """Extract a tensor from input data for the given semantic label.

        Args:
            input_data: Raw input (text, binary, etc.).
            semantic_label: The label to extract for.

        Returns:
            A tensor (PyTorch or raw bytes).
        """
        ...

    # ── Lifecycle Hooks ───────────────────────────────────────

    async def on_load(self, runtime: Any) -> None:
        """Called when the runtime loads this plugin."""
        return

    async def on_unload(self) -> None:
        """Called when the runtime unloads this plugin."""
        return


# ── Plugin Registry ────────────────────────────────────────────


class PluginRegistry:
    """Tracks which plugins are available by semantic label."""

    def __init__(self) -> None:
        self._plugins: dict[str, BasePlugin] = {}
        self._label_subscribers: dict[str, list[BasePlugin]] = {}
        self._capabilities: dict[str, Capability] = {}

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin and index its labels."""
        pid = plugin.plugin_id
        self._plugins[pid] = plugin

        for cap in plugin.get_capabilities():
            self._capabilities[cap.name] = cap

        for label in plugin.listens_to():
            self._label_subscribers.setdefault(label, []).append(plugin)

    def unregister(self, plugin_id: str) -> bool:
        """Remove a plugin and clean up its label subscriptions."""
        plugin = self._plugins.pop(plugin_id, None)
        if plugin is None:
            return False

        # Remove from label subscriptions
        for label in plugin.listens_to():
            subs = self._label_subscribers.get(label, [])
            if plugin in subs:
                subs.remove(plugin)
            if not subs:
                self._label_subscribers.pop(label, None)

        # Remove capabilities
        for cap in plugin.get_capabilities():
            self._capabilities.pop(cap.name, None)

        return True

    def get(self, plugin_id: str) -> BasePlugin | None:
        return self._plugins.get(plugin_id)

    def get_subscribers(self, label: str) -> list[BasePlugin]:
        """Return all plugins subscribed to a semantic label."""
        return list(self._label_subscribers.get(label, []))

    def get_capability(self, name: str) -> Capability | None:
        return self._capabilities.get(name)

    @property
    def plugin_ids(self) -> list[str]:
        return list(self._plugins.keys())

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    @property
    def labels(self) -> list[str]:
        return list(self._label_subscribers.keys())
