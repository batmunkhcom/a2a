"""Tests for example mesh configurations."""

import os

from a2a.config.loader import load_config


def _path(rel: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", rel)


class TestBasicMesh:
    def test_config_loads(self) -> None:
        config = load_config(_path("examples/basic_mesh/a2a.yaml"))
        assert config.runtime.agent_id == "basic-mesh"
        assert "repeater" in config.plugins
        assert "inverter" in config.plugins

    def test_models_defined(self) -> None:
        config = load_config(_path("examples/basic_mesh/a2a.yaml"))
        assert "base" in config.models
        assert config.models["base"].model_id == "demo-model"

    def test_plugins_have_modules(self) -> None:
        config = load_config(_path("examples/basic_mesh/a2a.yaml"))
        for _name, plugin in config.plugins.items():
            assert plugin.module
            assert plugin.class_name

    def test_route_connects_to_inverter(self) -> None:
        config = load_config(_path("examples/basic_mesh/a2a.yaml"))
        assert "hidden_state" in config.routes
        assert "inverter" in config.routes["hidden_state"]


class TestMultiModelMesh:
    def test_config_loads(self) -> None:
        config = load_config(_path("examples/multi_model/a2a.yaml"))
        assert config.runtime.agent_id == "multi-model-mesh"

    def test_has_three_plugins(self) -> None:
        config = load_config(_path("examples/multi_model/a2a.yaml"))
        assert len(config.plugins) == 3
        assert "summarizer" in config.plugins
        assert "normalizer" in config.plugins
        assert "mixer" in config.plugins

    def test_two_models(self) -> None:
        config = load_config(_path("examples/multi_model/a2a.yaml"))
        assert len(config.models) == 2
        assert "model-a" in config.models
        assert "model-b" in config.models

    def test_projection_configured(self) -> None:
        config = load_config(_path("examples/multi_model/a2a.yaml"))
        assert config.projection.architecture.variant == "b"
        assert config.projection.architecture.hidden_dim == 512

    def test_routes_configured(self) -> None:
        config = load_config(_path("examples/multi_model/a2a.yaml"))
        assert len(config.routes) >= 2
        assert "normalizer" in config.routes.get("hidden_state", [])
        assert "mixer" in config.routes.get("mixed_state", [])


class TestDemoConfig:
    def test_root_config_loads(self) -> None:
        config = load_config(_path("a2a.yaml"))
        assert config.version
        assert config.runtime.agent_id

    def test_root_config_has_plugins(self) -> None:
        config = load_config(_path("a2a.yaml"))
        assert len(config.plugins) > 0

    def test_root_config_has_models(self) -> None:
        config = load_config(_path("a2a.yaml"))
        assert len(config.models) >= 3

    def test_root_config_has_routes(self) -> None:
        config = load_config(_path("a2a.yaml"))
        assert len(config.routes) > 0
