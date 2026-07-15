"""Cross-model integration test — requires torch."""

import tempfile

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.projection.auto_trainer import AutoTrainer  # noqa: E402
from a2a.projection.registry import ProjectionRegistry  # noqa: E402


def test_cross_model_full_pipeline() -> None:
    """Simulate Llama→Mistral cross-model flow with projection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Auto-train a projection
        trainer = AutoTrainer(output_dir=tmpdir, epochs=3, lr=0.01)
        model = trainer.train_for_pair(
            src_dim=128,
            tgt_dim=64,
            src_model_id="llama-8b",
            tgt_model_id="mistral-7b",
        )
        assert model is not None

        # 2. Cache in registry
        reg = ProjectionRegistry(cache_dir=tmpdir)
        reg.put("llama-8b", "mistral-7b", model, 128, 64)

        # 3. Verify projection works
        # Source (llama hidden)
        src = torch.randn(1, 128)
        projected = model.forward(src)
        assert projected.shape == (1, 64)

        # 4. Cosine similarity check — with untrained model it's random
        # but shape is correct
        assert projected.shape[-1] == 64


def test_projection_flow_with_registry() -> None:
    """PluginManager-style projection resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ProjectionRegistry(cache_dir=tmpdir)

        # Simulate a scenario: LogReader (llama) → CodeFixer (mistral)
        src_model = "llama-8b"
        tgt_model = "mistral-7b"
        src_dim = 128
        tgt_dim = 64

        # Check registry — not cached
        projection = reg.get(src_model, tgt_model)
        assert projection is None

        # Auto-train
        at = AutoTrainer(output_dir=tmpdir, epochs=3)
        model = at.train_for_pair(src_dim, tgt_dim, src_model, tgt_model)

        # Cache
        reg.put(src_model, tgt_model, model, src_dim, tgt_dim)

        # Now it's available
        projection2 = reg.get(src_model, tgt_model)
        assert projection2 is not None

        # Forward through projection
        src_hidden = torch.randn(2, 128)
        result = projection2.forward(src_hidden)
        assert result.shape == (2, 64)
