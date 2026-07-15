"""Tests for TensorInjector — requires torch."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.tensor.injector import TensorInjector, _find_embedding  # noqa: E402


class FakeModelWithEmbed:
    def __init__(self, hidden_size: int = 64, vocab_size: int = 100) -> None:
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self._a2a_tokenizer = None

    def parameters(self) -> list[torch.Tensor]:
        return list(self.embed_tokens.parameters())

    def forward(self, *args: object, **kwargs: object) -> None:
        pass

    def generate(self, **kwargs: object) -> torch.Tensor:
        max_tokens = kwargs.get("max_new_tokens", 10)
        return torch.randint(0, 100, (1, max_tokens + 5))  # type: ignore[return-value]


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt", **_: object) -> dict:
        seq_len = max(len(text.split()) + 2, 3)
        return {
            "input_ids": torch.randint(0, 100, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }

    @property
    def eos_token_id(self) -> int:
        return 0

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return " ".join(str(t.item()) for t in tokens)


# ── Tests ──────────────────────────────────────────────────────


@pytest.fixture
def fake_model() -> FakeModelWithEmbed:
    model = FakeModelWithEmbed(hidden_size=64)
    model._a2a_tokenizer = FakeTokenizer()
    return model


def test_inject_and_generate_produces_output(fake_model: FakeModelWithEmbed) -> None:
    injector = TensorInjector(fake_model, mode="prefix")
    hidden = torch.randn(1, 64)
    result = injector.inject_and_generate(
        hidden, prompt="Fix error:", max_tokens=8, temperature=0.0
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_inject_with_different_prompt(fake_model: FakeModelWithEmbed) -> None:
    injector = TensorInjector(fake_model, mode="prefix")
    hidden = torch.randn(1, 64)

    r1 = injector.inject_and_generate(hidden, "Prompt A", max_tokens=4)
    r2 = injector.inject_and_generate(hidden, "Prompt B", max_tokens=4)

    assert isinstance(r1, str)
    assert isinstance(r2, str)


def test_inject_tensor_shape_mismatch_handled(fake_model: FakeModelWithEmbed) -> None:
    """Tensor with different hidden_dim should be projected."""
    injector = TensorInjector(fake_model, mode="prefix")
    # Different dimension from model (128 vs 64)
    hidden = torch.randn(1, 128)
    result = injector.inject_and_generate(hidden, "Fix:", max_tokens=4)
    assert isinstance(result, str)


def test_inject_prefix_shape(fake_model: FakeModelWithEmbed) -> None:
    injector = TensorInjector(fake_model, mode="prefix")
    hidden = torch.randn(1, 64)
    inputs = injector.inject_prefix(hidden, "Test prompt", tokenizer=fake_model._a2a_tokenizer)

    assert "inputs_embeds" in inputs
    assert "attention_mask" in inputs
    assert inputs["inputs_embeds"].shape[0] == 1  # batch
    assert inputs["inputs_embeds"].shape[-1] == 64  # hidden_dim
    assert inputs["inputs_embeds"].shape[1] == 1 + 3  # virtual_token + prompt tokens


def test_find_embedding(fake_model: FakeModelWithEmbed) -> None:
    emb = _find_embedding(fake_model)
    assert emb is not None
    assert isinstance(emb, torch.nn.Embedding)


def test_find_embedding_with_wte() -> None:
    model = type("GPT2", (), {})()
    model.transformer = type("Trans", (), {})()
    model.transformer.wte = torch.nn.Embedding(100, 64)
    # Parameters needs to be iterable
    model.transformer.wte.weight = torch.nn.Parameter(torch.randn(100, 64))
    model.parameters = lambda: [torch.nn.Parameter(torch.randn(1))]

    emb = _find_embedding(model)
    assert emb is not None
    assert emb.weight.shape == (100, 64)
