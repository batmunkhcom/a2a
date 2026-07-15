"""Tests for LogReaderPlugin."""

import pytest

from a2a.plugins.log_reader.plugin import LogReaderPlugin


@pytest.mark.asyncio
async def test_log_reader_extract_tensor_returns_dict() -> None:
    plugin = LogReaderPlugin()
    await plugin.initialize()
    result = await plugin.extract_tensor("ERROR: NullPointerException at line 45", "error_context")
    assert result is not None
    assert "shape" in result


@pytest.mark.asyncio
async def test_log_reader_emits() -> None:
    plugin = LogReaderPlugin()
    assert plugin.emits() == ["error_context", "log_summary"]


@pytest.mark.asyncio
async def test_log_reader_listens_to_empty() -> None:
    plugin = LogReaderPlugin()
    assert plugin.listens_to() == []


@pytest.mark.asyncio
async def test_log_reader_on_receive_returns_none() -> None:
    plugin = LogReaderPlugin()
    result = await plugin.on_receive_tensor("data", None)
    assert result is None


@pytest.mark.asyncio
async def test_log_reader_unknown_label_raises() -> None:
    plugin = LogReaderPlugin()
    await plugin.initialize()
    with pytest.raises(ValueError, match="Unknown"):
        await plugin.extract_tensor("data", "unknown_label")
