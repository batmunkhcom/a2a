"""Tests for config loader — search path, env override, explicit path."""

import tempfile
from pathlib import Path

import pytest

from a2a.config.loader import ConfigNotFoundError, load_config


def _write_temp_config(data: str) -> Path:
    """Write config to temp file, return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="a2a_test_"
    )
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def test_load_config_explicit_path() -> None:
    yaml = """
version: "1.0"
models:
  test:
    provider: openai
    model_id: gpt-4
    max_tokens: 100
plugins: {}
routes: {}
"""
    path = _write_temp_config(yaml)
    try:
        config = load_config(path)
        assert config.version == "1.0"
        assert config.models["test"].provider == "openai"
    finally:
        path.unlink()


def test_load_config_file_not_found() -> None:
    with pytest.raises((ConfigNotFoundError, FileNotFoundError)):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_from_cwd_env(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config found via A2A_CONFIG env variable."""
    yaml = """
version: "1.0"
models: {}
plugins: {}
routes: {}
"""
    cfg = tmp_path / "test_config.yaml"  # type: ignore[union-attr]
    cfg.write_text(yaml)
    monkeypatch.setenv("A2A_CONFIG", str(cfg))

    config = load_config()
    assert config.version == "1.0"


def test_load_config_from_working_directory(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Find ./a2a.yaml in cwd."""
    yaml = """
version: "1.0"
models: {}
plugins: {}
routes: {}
"""
    cfg = tmp_path / "a2a.yaml"  # type: ignore[union-attr]
    cfg.write_text(yaml)

    monkeypatch.chdir(tmp_path)
    # Unset A2A_CONFIG to test cwd fallback
    monkeypatch.delenv("A2A_CONFIG", raising=False)

    config = load_config()
    assert config.version == "1.0"
