"""Tests for CodeFixerPlugin."""

import pytest

from a2a.plugins.code_fixer.plugin import CodeFixerPlugin


@pytest.mark.asyncio
async def test_code_fixer_on_receive_returns_text() -> None:
    plugin = CodeFixerPlugin()
    await plugin.initialize(
        plugin_config={
            "max_output_tokens": 64,
            "system_prompt": "Fix error",
        }
    )
    result = await plugin.on_receive_tensor(42, None)
    assert result is not None
    assert "code_patch" in result
    assert "fix_explanation" in result


@pytest.mark.asyncio
async def test_code_fixer_listens_to() -> None:
    plugin = CodeFixerPlugin()
    assert plugin.listens_to() == ["error_context"]


@pytest.mark.asyncio
async def test_code_fixer_emits() -> None:
    plugin = CodeFixerPlugin()
    assert plugin.emits() == ["code_patch", "fix_explanation"]


@pytest.mark.asyncio
async def test_code_fixer_extract_returns_none() -> None:
    plugin = CodeFixerPlugin()
    result = await plugin.extract_tensor("data", "any_label")
    assert result is None
