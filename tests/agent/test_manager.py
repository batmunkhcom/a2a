"""Tests for PluginManager — loading, registration, routing."""

import pytest

from a2a.agent.base import BasePlugin, Capability, ModelInfo
from a2a.agent.manager import PluginManager


class EchoPlugin(BasePlugin):
    """Simple plugin that echoes back what it receives."""

    @property
    def plugin_id(self) -> str:
        return "echo"

    @property
    def plugin_name(self) -> str:
        return "Echo"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> list[Capability]:
        return [Capability(name="echo_cap", description="Echo")]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_id="test-model")

    def listens_to(self) -> list[str]:
        return ["echo_in"]

    def emits(self) -> list[str]:
        return ["echo_out"]

    async def on_receive_tensor(self, tensor: object, metadata: object) -> object:
        return f"echo: {tensor}"

    async def extract_tensor(self, input_data: object, semantic_label: str) -> object:
        return str(input_data)


class MultiLabelPlugin(BasePlugin):
    """Plugin that listens to multiple labels."""

    @property
    def plugin_id(self) -> str:
        return "multi"

    @property
    def plugin_name(self) -> str:
        return "Multi"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> list[Capability]:
        return [Capability(name="multi_cap", description="Multi", hidden_dim=64)]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_id="test-model", hidden_dim=64)

    def listens_to(self) -> list[str]:
        return ["label_a", "label_b", "label_c"]

    def emits(self) -> list[str]:
        return ["label_d"]

    async def on_receive_tensor(self, tensor: object, metadata: object) -> object:
        return {"plugin": "multi", "data": tensor}

    async def extract_tensor(self, input_data: object, semantic_label: str) -> object:
        return str(input_data)


# ── Tests ──────────────────────────────────────────────────────


@pytest.fixture
def manager() -> PluginManager:
    return PluginManager()


def test_register_routes_populated(manager: PluginManager) -> None:
    plugin = EchoPlugin()
    manager.register(plugin)
    assert manager.plugin_count == 1
    assert manager.get_plugin("echo") is plugin


def test_route_tensor_correct_plugin_called(manager: PluginManager) -> None:
    import asyncio

    plugin = EchoPlugin()
    manager.register(plugin)

    # Create a simple metadata stub
    class Meta:
        semantic_label = "echo_in"

    result = asyncio.run(manager.route_tensor("hello", Meta()))
    assert len(result) == 1
    assert "echo: hello" in result[0]


def test_route_tensor_no_match_returns_empty(manager: PluginManager) -> None:
    import asyncio

    plugin = EchoPlugin()
    manager.register(plugin)

    class Meta:
        semantic_label = "nonexistent_label"

    result = asyncio.run(manager.route_tensor("data", Meta()))
    assert result == []


def test_plugin_manager_label_routing_with_multiple_listeners(manager: PluginManager) -> None:
    import asyncio

    p1 = EchoPlugin()
    p2 = MultiLabelPlugin()
    manager.register(p1)
    manager.register(p2)

    assert manager.plugin_count == 2

    class Meta:
        semantic_label = "label_a"

    result = asyncio.run(manager.route_tensor("test_data", Meta()))
    assert len(result) == 1
    assert result[0]["plugin"] == "multi"


def test_plugin_ids(manager: PluginManager) -> None:
    manager.register(EchoPlugin())
    manager.register(MultiLabelPlugin())
    assert set(manager.plugin_ids) == {"echo", "multi"}


def test_get_matching_plugins(manager: PluginManager) -> None:
    p1 = EchoPlugin()
    p2 = MultiLabelPlugin()
    manager.register(p1)
    manager.register(p2)

    matches = manager.get_matching_plugins(["echo_in", "label_a"])
    assert len(matches) == 2


def test_labels(manager: PluginManager) -> None:
    manager.register(EchoPlugin())
    manager.register(MultiLabelPlugin())
    labels = manager.labels
    assert "echo_in" in labels
    assert "label_a" in labels
