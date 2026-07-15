"""Tests for AutoTrainer — requires torch."""

import tempfile

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.projection.auto_trainer import AutoTrainer  # noqa: E402


def test_auto_trainer_completes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AutoTrainer(
            corpus_path=None, output_dir=tmpdir, epochs=3, lr=0.01
        )

        model = trainer.train_for_pair(
            src_dim=32,
            tgt_dim=32,
            src_model_id="test-src",
            tgt_model_id="test-tgt",
        )

        assert model is not None
        assert trainer.is_cached("test-src", "test-tgt")


def test_auto_trainer_saves_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AutoTrainer(output_dir=tmpdir, epochs=3)

        trainer.train_for_pair(64, 64, "a", "b")
        cache_path = trainer.get_cached_path("a", "b")
        assert cache_path.exists() or cache_path.with_suffix(".pt").exists()


def test_auto_trainer_not_cached_initially() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AutoTrainer(output_dir=tmpdir)
        assert not trainer.is_cached("unknown-src", "unknown-tgt")
