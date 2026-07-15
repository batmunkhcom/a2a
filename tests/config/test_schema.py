"""Tests for A2AConfig Pydantic model."""

import pytest

from a2a.config.schema import A2AConfig


def test_config_from_dict_valid() -> None:
    data = {
        "version": "1.0",
        "models": {
            "test-model": {
                "provider": "huggingface",
                "model_id": "meta-llama/Test",
                "dtype": "float32",
                "max_tokens": 512,
            }
        },
        "plugins": {},
        "routes": {},
    }
    config = A2AConfig.from_dict(data)
    assert config.version == "1.0"
    assert config.models["test-model"].model_id == "meta-llama/Test"


def test_config_resolve_model() -> None:
    data = {
        "models": {
            "gpt": {"provider": "openai", "model_id": "gpt-4", "max_tokens": 4096}
        },
        "plugins": {},
        "routes": {},
    }
    config = A2AConfig.from_dict(data)
    model = config.resolve_model("gpt")
    assert model is not None
    assert model.model_id == "gpt-4"


def test_config_resolve_model_missing() -> None:
    data = {"models": {}, "plugins": {}, "routes": {}}
    config = A2AConfig.from_dict(data)
    with pytest.raises(ValueError, match="not found"):
        config.resolve_model("nonexistent")


def test_config_validate_model_reference() -> None:
    data = {
        "models": {},
        "plugins": {
            "code-fixer": {
                "enabled": True,
                "module": "pkg.plugin",
                "class_name": "Plugin",
                "model": "gpt-4",
            }
        },
        "routes": {},
    }
    config = A2AConfig.from_dict(data)
    warnings = config.validate_config()
    assert len(warnings) >= 1
    assert any("gpt-4" in w for w in warnings)


def test_config_validate_route_targets() -> None:
    data = {
        "models": {},
        "plugins": {
            "code-fixer": {
                "enabled": True,
                "module": "pkg.plugin",
                "class_name": "Plugin",
            }
        },
        "routes": {"error_context": ["nonexistent-plugin"]},
    }
    config = A2AConfig.from_dict(data)
    warnings = config.validate_config()
    assert len(warnings) >= 1
    assert any("nonexistent-plugin" in w for w in warnings)


def test_config_valid_no_warnings() -> None:
    data = {
        "models": {
            "llama": {"provider": "huggingface", "model_id": "meta/llama", "max_tokens": 512}
        },
        "plugins": {
            "log-reader": {
                "enabled": True,
                "module": "pkg.plugin",
                "class_name": "Plugin",
                "model": "llama",
            }
        },
        "routes": {"error_context": ["log-reader"]},
    }
    config = A2AConfig.from_dict(data)
    warnings = config.validate_config()
    assert warnings == []
