"""Integration test: A2ARuntime startup and plugin loading."""

import tempfile
from pathlib import Path

import pytest

from a2a.runtime import A2ARuntime


@pytest.mark.asyncio
async def test_runtime_loads_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A2ARuntime.start() loads all plugins from config."""
    config_yaml = """
version: "1.0"
models:
  test-model:
    provider: "test"
    model_id: "test-id"
    max_tokens: 100
plugins:
  log-reader:
    enabled: true
    module: "a2a.plugins.log_reader.plugin"
    class_name: "LogReaderPlugin"
    model: "test-model"
  code-fixer:
    enabled: true
    module: "a2a.plugins.code_fixer.plugin"
    class_name: "CodeFixerPlugin"
    model: "test-model"
  disabled-plugin:
    enabled: false
    module: "a2a.plugins.code_fixer.plugin"
    class_name: "CodeFixerPlugin"
routes:
  error_context:
    - "code-fixer"
transport:
  protocol: "grpc"
  host: "localhost"
  port: 0
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="a2a_test_"
    ) as f:
        f.write(config_yaml)
        config_path = f.name

    try:
        monkeypatch.setenv("A2A_CONFIG", config_path)
        runtime = A2ARuntime()

        # Start runtime (plugins loaded, gRPC on random port)
        await runtime.start()

        assert runtime.plugin_manager.plugin_count == 2
        plugin_ids = sorted(runtime.plugin_manager.plugin_ids)
        assert "code-fixer" in plugin_ids
        assert "log-reader" in plugin_ids

        # Disabled plugin should NOT be loaded
        assert "disabled-plugin" not in plugin_ids

        await runtime.stop()
    finally:
        Path(config_path).unlink(missing_ok=True)
