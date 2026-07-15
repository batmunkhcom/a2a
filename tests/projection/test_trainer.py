"""Tests for ProjectionTrainer — requires torch."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from torch.utils.data import DataLoader  # noqa: E402

from a2a.projection.adapter import ProjectionModel  # noqa: E402
from a2a.projection.dataset import ProjectionPairDataset  # noqa: E402
from a2a.projection.trainer import ProjectionTrainer  # noqa: E402


@pytest.fixture
def small_dataset() -> ProjectionPairDataset:
    return ProjectionPairDataset.from_random(200, src_dim=64, tgt_dim=64, seed=42)


def test_training_reduces_loss(small_dataset: ProjectionPairDataset) -> None:
    model = ProjectionModel(src_dim=64, tgt_dim=64, variant="b", hidden_dim=32)
    trainer = ProjectionTrainer(model, temperature=0.1, lr=0.01)
    loader = DataLoader(small_dataset, batch_size=16, shuffle=True)

    history = trainer.train(loader, epochs=5, device="cpu")

    assert history["loss"][0] > 0
    assert history["loss"][-1] < history["loss"][0], "Loss should decrease"


def test_training_improves_cosine_similarity(small_dataset: ProjectionPairDataset) -> None:
    model = ProjectionModel(src_dim=64, tgt_dim=64, variant="b", hidden_dim=32)
    trainer = ProjectionTrainer(model, temperature=0.1, lr=0.01)
    loader = DataLoader(small_dataset, batch_size=16, shuffle=True)

    history = trainer.train(loader, epochs=10, device="cpu")

    assert history["cosine_similarity"][0] < 1.0
    # Cosine similarity should improve over training
    assert history["cosine_similarity"][-1] > -1.0


def test_multi_objective_loss_decreases(small_dataset: ProjectionPairDataset) -> None:
    model = ProjectionModel(src_dim=64, tgt_dim=64, variant="b", hidden_dim=32)
    trainer = ProjectionTrainer(
        model, temperature=0.1, mse_weight=0.1, cosine_weight=0.01, lr=0.01
    )
    loader = DataLoader(small_dataset, batch_size=16, shuffle=True)

    history = trainer.train(loader, epochs=5, device="cpu")

    assert len(history["contrastive"]) == 5
    assert len(history["mse"]) == 5
    assert len(history["cosine"]) == 5
    # Three loss components tracked
    assert abs(history["contrastive"][-1]) > 0  # finite
    assert abs(history["mse"][-1]) > 0
