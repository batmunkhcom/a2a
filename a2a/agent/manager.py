"""PluginManager — loads, registers, and routes tensors between plugins.

Handles:
- Plugin discovery via importlib, filesystem, entry_points.
- Model/tokenizer loading and assignment to plugins.
- Semantic label routing (which plugin receives which tensor).
- Plugin-local config loading from config.yaml.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml

from a2a.agent.base import BasePlugin, PluginEntry, PluginRegistry


class PluginManager:
    """Central manager for plugin lifecycle and tensor routing."""

    def __init__(self) -> None:
        self._registry = PluginRegistry()
        self._global_config: Any = None

    # ── Registration ───────────────────────────────────────────

    def register(self, plugin: BasePlugin) -> None:
        """Register a manually-created plugin instance."""
        self._registry.register(plugin)

    def unregister(self, plugin_id: str) -> bool:
        """Remove a plugin by ID."""
        return self._registry.unregister(plugin_id)

    # ── Plugin Loading ─────────────────────────────────────────

    def load_plugin(
        self,
        entry: PluginEntry,
        global_config: Any | None = None,
    ) -> BasePlugin:
        """Load a plugin from a PluginEntry specification.

        Uses importlib to load the module, finds the BasePlugin subclass,
        instantiates it, and calls initialize().

        Args:
            entry: PluginEntry with module, class_name, model config key.
            global_config: Full A2AConfig for model resolution.

        Returns:
            Initialized plugin instance.

        Raises:
            ImportError: If the module cannot be imported.
            TypeError: If no BasePlugin subclass found.
        """
        self._global_config = global_config

        # 1. Import the module
        try:
            module = importlib.import_module(entry.module)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import plugin module '{entry.module}': {exc}"
            ) from exc

        # 2. Find the BasePlugin subclass
        plugin_class = self._find_plugin_class(module, entry.class_name)
        if plugin_class is None:
            raise TypeError(
                f"No BasePlugin subclass named '{entry.class_name}' "
                f"found in module '{entry.module}'"
            )

        # 3. Instantiate the plugin
        plugin = plugin_class()

        # 4. Resolve model config
        model_config = None
        if global_config and entry.model:
            model_config = global_config.resolve_model(entry.model)

        # 5. Load plugin-local config
        plugin_config = self._load_plugin_config(module, entry)

        # 6. Initialize (plugin.initialize is async but safe to call
        #    synchronously during loading — it just assigns attributes)
        plugin.initialize(  # type: ignore[unused-coroutine]
            model=getattr(model_config, "model_id", None) if model_config else None,
            tokenizer=None,
            plugin_config=plugin_config,
            global_config=global_config,
        )

        # 7. Register
        self._registry.register(plugin)

        return plugin

    def discover_from_directory(self, path: str | Path) -> list[BasePlugin]:
        """Auto-discover and load plugins from a directory.

        Each subdirectory under `path` is treated as a plugin package.
        Looks for plugin.py or __init__.py containing a BasePlugin subclass.

        Args:
            path: Root directory containing plugin packages.

        Returns:
            List of loaded plugin instances.
        """
        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"Plugin directory not found: {path}")

        plugins: list[BasePlugin] = []
        for item in sorted(root.iterdir()):
            if item.is_dir() and not item.name.startswith("_"):
                plugin_path = item / "plugin.py"
                if plugin_path.exists():
                    module_name = f"{root.name}.{item.name}.plugin"
                    try:
                        spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
                            module_name, plugin_path
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
                            spec.loader.exec_module(module)
                            plugin = self._find_and_instantiate_plugin(module)
                            if plugin:
                                plugins.append(plugin)
                    except Exception:
                        continue

        return plugins

    # ── Tensor Routing ─────────────────────────────────────────

    async def route_tensor(
        self,
        tensor: Any,
        metadata: Any,
    ) -> list[Any]:
        """Route a tensor to all plugins subscribed to its semantic label.

        Args:
            tensor: Tensor data (PyTorch tensor or bytes).
            metadata: TensorMetadata protobuf with semantic_label.

        Returns:
            List of results from each plugin's on_receive_tensor().
        """
        label = getattr(metadata, "semantic_label", "")
        targets = self._registry.get_subscribers(label)

        results: list[Any] = []
        for plugin in targets:
            result = await plugin.on_receive_tensor(tensor, metadata)
            if result is not None:
                results.append(result)

        return results

    # ── Query Methods ──────────────────────────────────────────

    def get_plugin(self, plugin_id: str) -> BasePlugin | None:
        return self._registry.get(plugin_id)

    def get_subscribers(self, label: str) -> list[BasePlugin]:
        return self._registry.get_subscribers(label)

    def get_matching_plugins(self, required_labels: list[str]) -> list[BasePlugin]:
        """Find plugins whose listen labels match the required set."""
        matches: set[BasePlugin] = set()
        for label in required_labels:
            subscribers = self._registry.get_subscribers(label)
            matches.update(subscribers)
        return list(matches)

    @property
    def plugin_count(self) -> int:
        return self._registry.plugin_count

    @property
    def plugin_ids(self) -> list[str]:
        return self._registry.plugin_ids

    @property
    def labels(self) -> list[str]:
        return self._registry.labels

    # ── Internal ───────────────────────────────────────────────

    def _find_plugin_class(
        self, module: Any, class_name: str
    ) -> type[BasePlugin] | None:
        """Find a BasePlugin subclass in a module."""
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BasePlugin)
                and obj is not BasePlugin
                and obj.__name__ == class_name
            ):
                return obj
        return None

    def _find_and_instantiate_plugin(self, module: Any) -> BasePlugin | None:
        """Find the first BasePlugin subclass in a module and instantiate it."""
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                plugin: BasePlugin = obj()
                # Synchronous initialization during discovery
                import asyncio as _asyncio
                try:
                    _asyncio.get_running_loop()
                except RuntimeError:
                    _asyncio.run(plugin.initialize())
                self._registry.register(plugin)
                return plugin
        return None

    def _load_plugin_config(
        self, module: Any, entry: PluginEntry
    ) -> dict[str, Any]:
        """Load plugin-local config from config.yaml next to the module."""
        if entry.config_path:
            config_path = Path(entry.config_path)
            if config_path.exists():
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}

        # Try next to the module file
        try:
            module_path = Path(inspect.getfile(module))
            config_path = module_path.parent / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
        except (TypeError, OSError):
            pass

        return {}
