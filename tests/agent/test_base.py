"""Tests for BasePlugin, Capability, ModelInfo, and PluginRegistry."""


from a2a.agent.base import (
    BasePlugin,
    Capability,
    ModelInfo,
    PluginRegistry,
)


class MinimalPlugin(BasePlugin):
    """Minimal concrete plugin for testing."""

    @property
    def plugin_id(self) -> str:
        return "minimal"

    @property
    def plugin_name(self) -> str:
        return "Minimal Test Plugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    def get_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="test_cap",
                description="Test capability",
                input_labels=["input_label"],
                output_labels=["output_label"],
                model_id="test-model",
                hidden_dim=128,
            )
        ]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_id="test-model", hidden_dim=128)

    def listens_to(self) -> list[str]:
        return ["error_context", "log_summary"]

    def emits(self) -> list[str]:
        return ["code_patch"]

    async def on_receive_tensor(self, tensor: object, metadata: object) -> object:
        return f"processed: {metadata}"

    async def extract_tensor(self, input_data: object, semantic_label: str) -> object:
        return f"extracted: {semantic_label}"


# ── Tests ──────────────────────────────────────────────────────


def test_plugin_subclass_all_abstract_methods() -> None:
    plugin = MinimalPlugin()
    assert plugin.plugin_id == "minimal"
    assert plugin.plugin_name == "Minimal Test Plugin"
    assert plugin.version == "0.1.0"
    assert len(plugin.get_capabilities()) == 1
    assert plugin.listens_to() == ["error_context", "log_summary"]
    assert plugin.emits() == ["code_patch"]


def test_capability_dataclass() -> None:
    cap = Capability(
        name="test",
        description="desc",
        input_labels=["a"],
        output_labels=["b"],
        model_id="m1",
        hidden_dim=256,
    )
    assert cap.name == "test"
    assert cap.hidden_dim == 256


def test_model_info_dataclass() -> None:
    mi = ModelInfo(
        model_id="llama-8b",
        architecture="llama",
        hidden_dim=4096,
        num_layers=32,
        dtype="float16",
    )
    assert mi.hidden_dim == 4096
    assert mi.dtype == "float16"


def test_plugin_registry_register() -> None:
    registry = PluginRegistry()
    plugin = MinimalPlugin()
    registry.register(plugin)
    assert registry.get("minimal") is plugin
    assert registry.plugin_count == 1


def test_plugin_registry_label_routes_populated() -> None:
    registry = PluginRegistry()
    plugin = MinimalPlugin()
    registry.register(plugin)

    subs = registry.get_subscribers("error_context")
    assert len(subs) == 1
    assert subs[0] is plugin


def test_plugin_registry_unregister() -> None:
    registry = PluginRegistry()
    plugin = MinimalPlugin()
    registry.register(plugin)
    assert registry.plugin_count == 1

    assert registry.unregister("minimal") is True
    assert registry.plugin_count == 0
    assert registry.unregister("nonexistent") is False


def test_plugin_registry_labels() -> None:
    registry = PluginRegistry()
    registry.register(MinimalPlugin())
    assert "error_context" in registry.labels
    assert "log_summary" in registry.labels
