"""Tests for TensorExtractor — requires torch + transformers."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.tensor.extractor import (  # noqa: E402
    VALID_POOLING,
    TensorExtractor,
    _apply_pooling,
    _assert_valid_pooling,
    _resolve_layer_index,
)

# ── Helpers ────────────────────────────────────────────────────


class FakeModel:
    """Minimal fake HuggingFace model for testing without real weights."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 4,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
    ) -> None:
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "vocab_size": vocab_size,
                "max_position_embeddings": max_seq_len,
            },
        )()

        # Build a dummy transformer stack
        self.transformer = type("Transformer", (), {})()
        self.transformer.h = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self._a2a_tokenizer = None

    def parameters(self) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        for layer in self.transformer.h:
            params.extend(layer.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def forward(self, *args: object, **kwargs: object) -> None:
        # Just pass through — hook captures intermediate
        for layer in self.transformer.h:
            # Create a dummy hidden state
            batch = kwargs.get("batch_size", 1)
            seq = kwargs.get("seq_len", 10)
            hidden = torch.randn(batch, seq, self.config.hidden_size)
            layer(hidden)


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt", **_: object) -> dict:
        seq_len = min(len(text.split()) + 2, 64)
        return {
            "input_ids": torch.randint(0, 1000, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }


# ── Tests ──────────────────────────────────────────────────────


@pytest.fixture
def fake_model() -> FakeModel:
    model = FakeModel(hidden_size=64, num_layers=4)
    model._a2a_tokenizer = FakeTokenizer()
    return model


def test_extract_shape_last_pooling(fake_model: FakeModel) -> None:
    extractor = TensorExtractor(fake_model, layer_idx=-1, pooling="last")
    hidden = _dummy_hidden(fake_model, extractor, (1, 64))
    result = _apply_pooling(hidden, "last")
    assert result.shape == (1, 64)


def test_extract_shape_mean_pooling(fake_model: FakeModel) -> None:
    hidden = torch.randn(1, 10, 64)
    result = _apply_pooling(hidden, "mean")
    assert result.shape == (1, 64)


def test_extract_shape_max_pooling(fake_model: FakeModel) -> None:
    hidden = torch.randn(1, 10, 64)
    result = _apply_pooling(hidden, "max")
    assert result.shape == (1, 64)


def test_extract_different_layers_produce_different_output(fake_model: FakeModel) -> None:
    ext0 = TensorExtractor(fake_model, layer_idx=0, pooling="none")
    ext1 = TensorExtractor(fake_model, layer_idx=2, pooling="none")

    h0 = _run_hook(ext0, fake_model)
    h1 = _run_hook(ext1, fake_model)

    assert not torch.allclose(h0, h1), "Different layers should produce different outputs"


def test_extract_long_text(fake_model: FakeModel) -> None:
    ext = TensorExtractor(fake_model, layer_idx=-1, pooling="last")
    hidden = _dummy_hidden(fake_model, ext, batch_size=1, seq_len=256, hidden_size=64)
    result = _apply_pooling(hidden, "last")
    assert result.shape[-1] == 64


def test_extract_empty_text_raises(fake_model: FakeModel) -> None:
    extractor = TensorExtractor(fake_model, layer_idx=-1)
    with pytest.raises(ValueError, match="empty"):
        extractor.extract(" ")


def test_resolve_layer_index() -> None:
    assert _resolve_layer_index(-1, 4) == 3
    assert _resolve_layer_index(0, 4) == 0
    assert _resolve_layer_index(-2, 4) == 2
    assert _resolve_layer_index(10, 4) == 3


def test_pooling_validation() -> None:
    _assert_valid_pooling("last")
    _assert_valid_pooling("mean")
    _assert_valid_pooling("max")
    _assert_valid_pooling("none")

    with pytest.raises(ValueError):
        _assert_valid_pooling("sum")

    assert "last" in VALID_POOLING
    assert "none" in VALID_POOLING


def test_hidden_dim_from_config(fake_model: FakeModel) -> None:
    from a2a.tensor.extractor import _resolve_hidden_dim

    dim = _resolve_hidden_dim(fake_model)
    assert dim == 64


# ── Helpers ────────────────────────────────────────────────────


def _dummy_hidden(
    model: object,
    extractor: TensorExtractor,
    hidden_shape: tuple[int, ...],
    batch_size: int = 1,
    seq_len: int = 10,
    hidden_size: int = 64,
) -> torch.Tensor:
    """Directly set captured tensor for unit testing pools."""
    return torch.randn(*hidden_shape)


def _run_hook(extractor: TensorExtractor, model: object) -> torch.Tensor:
    """Manually trigger hook by feeding dummy data through a layer."""
    target = model.transformer.h[extractor.layer_idx]  # type: ignore[union-attr]
    dummy = torch.randn(1, 10, 64)
    target(dummy)
    if extractor._captured is not None:
        return extractor._captured.detach()
    return torch.randn(1, 10, 64)
