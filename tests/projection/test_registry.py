"""Tests for ProjectionRegistry."""

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.projection.adapter import ProjectionModel  # noqa: E402
from a2a.projection.registry import ProjectionRegistry  # noqa: E402


def test_registry_put_and_get() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ProjectionRegistry(cache_dir=tmpdir)
        model = ProjectionModel(src_dim=64, tgt_dim=128)
        reg.put("llama", "mistral", model, 64, 128)

        cached = reg.get("llama", "mistral")
        assert cached is not None
        assert reg.has("llama", "mistral")


def test_registry_returns_none_for_unknown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ProjectionRegistry(cache_dir=tmpdir)
        assert reg.get("unknown", "pair") is None
        assert not reg.has("unknown", "pair")


def test_registry_list_cached() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ProjectionRegistry(cache_dir=tmpdir)
        model = ProjectionModel(src_dim=32, tgt_dim=32)
        reg.put("a", "b", model, 32, 32)
        reg.put("c", "d", model, 32, 32)

        cached = reg.list_cached()
        assert "a__b" in cached
        assert "c__d" in cached


def test_registry_save_load_disk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ProjectionRegistry(cache_dir=tmpdir)
        model = ProjectionModel(src_dim=32, tgt_dim=32)
        reg.put("src", "tgt", model, 32, 32, save=True)

        # Verify file exists on disk
        expected_path = Path(tmpdir) / "src__tgt.safetensors"
        expected_pt = Path(tmpdir) / "src__tgt.pt"
        assert expected_path.exists() or expected_pt.exists()

        # Load back in new registry
        reg2 = ProjectionRegistry(cache_dir=tmpdir)
        loaded = reg2.get("src", "tgt")
        assert loaded is not None
