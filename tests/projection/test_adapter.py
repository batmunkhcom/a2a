"""Tests for ProjectionModel adapter — requires torch."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.projection.adapter import ProjectionModel, create_projection  # noqa: E402


def test_projection_forward_shape() -> None:
    model = ProjectionModel(src_dim=4096, tgt_dim=4096, variant="b")
    x = torch.randn(1, 4096)
    out = model.forward(x)
    assert out.shape == (1, 4096)


def test_projection_preserves_batch_dim() -> None:
    model = ProjectionModel(src_dim=128, tgt_dim=256)
    x = torch.randn(4, 128)
    out = model.forward(x)
    assert out.shape == (4, 256)


def test_linear_variant_correct_shape() -> None:
    model = ProjectionModel(src_dim=512, tgt_dim=256, variant="a")
    x = torch.randn(2, 512)
    out = model.forward(x)
    assert out.shape == (2, 256)


def test_variant_c_forward() -> None:
    model = ProjectionModel(src_dim=128, tgt_dim=128, variant="c", num_hidden=3)
    x = torch.randn(8, 128)
    out = model.forward(x)
    assert out.shape == (8, 128)


def test_same_dim_identity_like() -> None:
    """Same src/tgt dim — output should be close to input after training."""
    model = ProjectionModel(src_dim=64, tgt_dim=64, variant="a")
    x = torch.randn(1, 64)
    out = model.forward(x)
    assert out.shape == x.shape


def test_different_dim_projection() -> None:
    model = ProjectionModel(src_dim=1024, tgt_dim=512, variant="b")
    x = torch.randn(3, 1024)
    out = model.forward(x)
    assert out.shape == (3, 512)


def test_create_projection_variants() -> None:
    for v in ("a", "b", "c"):
        m = create_projection(v, src_dim=256, tgt_dim=256)
        x = torch.randn(1, 256)
        out = m.forward(x)
        assert out.shape == (1, 256)


def test_model_state_dict_roundtrip() -> None:
    model1 = ProjectionModel(src_dim=64, tgt_dim=128)
    state = model1.state_dict()

    model2 = ProjectionModel(
        src_dim=state["config"]["src_dim"],
        tgt_dim=state["config"]["tgt_dim"],
        variant=state["config"]["variant"],
    )
    model2.load_state_dict(state)

    x = torch.randn(2, 64)
    o1 = model1.forward(x)
    o2 = model2.forward(x)
    assert torch.allclose(o1, o2)
