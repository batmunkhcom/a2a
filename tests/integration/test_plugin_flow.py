"""Integration test: LogReaderPlugin → CodeFixerPlugin full flow."""

import pytest

from a2a.agent.manager import PluginManager
from a2a.plugins.code_fixer.plugin import CodeFixerPlugin
from a2a.plugins.log_reader.plugin import LogReaderPlugin


@pytest.mark.asyncio
async def test_log_reader_to_code_fixer_full_flow() -> None:
    """End-to-end: LogReader extracts → CodeFixer receives and responds."""
    manager = PluginManager()

    log_reader = LogReaderPlugin()
    code_fixer = CodeFixerPlugin()

    await log_reader.initialize()
    await code_fixer.initialize()

    manager.register(log_reader)
    manager.register(code_fixer)

    assert manager.plugin_count == 2

    # LogReader extracts error_context from a log line
    tensor = await log_reader.extract_tensor(
        "ERROR: NullPointerException at line 45", "error_context"
    )
    assert tensor is not None

    # CodeFixer receives the tensor and generates a fix
    class Meta:
        semantic_label = "error_context"
        source_model = "llama-8b"

    result = await code_fixer.on_receive_tensor(tensor, Meta())
    assert "code_patch" in result
    assert len(result["code_patch"]) > 0
