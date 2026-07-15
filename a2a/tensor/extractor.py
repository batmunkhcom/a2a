"""Hidden state extraction from HuggingFace transformer models.

Uses PyTorch forward hooks to capture internal activations at any layer.
"""

from __future__ import annotations

from typing import Any

# Lazy imports — only required when actually using the extractor
try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TensorExtractor:
    """Extracts hidden states from a HuggingFace transformer model.

    Uses register_forward_hook on a specific layer to intercept
    activations during the forward pass.

    Args:
        model: A HuggingFace PreTrainedModel (e.g. AutoModelForCausalLM).
        layer_idx: Which transformer layer to hook.
                   -1 = last hidden layer (default).
                    0 = embedding output.
                    N = Nth transformer block output.
        pooling: Default pooling strategy: "last", "mean", or "max".

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        extractor = TensorExtractor(model, layer_idx=-1)
        hidden = extractor.extract("Hello world")
        # hidden.shape = (1, hidden_dim)
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_idx: int = -1,
        pooling: str = "last",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "TensorExtractor requires PyTorch. Install with: pip install a2a-protocol[ml]"
            )

        self._model = model
        self._model_device = _get_model_device(model)
        self._hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._captured: torch.Tensor | None = None
        self._hidden_dim: int = _resolve_hidden_dim(model)
        self._num_layers: int = _resolve_num_layers(model)
        self._pooling = pooling

        actual_idx = _resolve_layer_index(layer_idx, self._num_layers)
        self._layer_idx = actual_idx
        self._attach_hook(actual_idx)

    # ── Public API ─────────────────────────────────────────────

    @property
    def hidden_dim(self) -> int:
        """Size of hidden state dimension."""
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the model."""
        return self._num_layers

    @property
    def layer_idx(self) -> int:
        """Currently hooked layer index."""
        return self._layer_idx

    def extract(
        self,
        text: str,
        *,
        pooling: str | None = None,
    ) -> torch.Tensor:
        """Run forward pass and extract hidden state.

        Args:
            text: Input text string.
            pooling: Override default pooling.
                     "last"  — last token's hidden state (default).
                     "mean"  — mean over all tokens.
                     "max"   — max over all tokens.

        Returns:
            Tensor of shape (1, hidden_dim) for pooled output,
            or (1, seq_len, hidden_dim) if pooling is "none".

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be non-empty")

        strategy = pooling if pooling is not None else self._pooling

        # Tokenize
        tokenizer = _get_tokenizer(self._model)
        inputs = tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            self._model(**inputs)

        if self._captured is None:
            raise RuntimeError(
                f"TensorExtractor hook did not capture output at layer {self._layer_idx}. "
                f"Check that the model forward pass uses the hooked module."
            )

        raw = self._captured.detach()
        self._captured = None

        return _apply_pooling(raw, strategy)

    def set_layer(self, layer_idx: int) -> None:
        """Change the hooked layer index.

        Args:
            layer_idx: New layer index (-1 for last, 0 for embedding, etc.)
        """
        actual = _resolve_layer_index(layer_idx, self._num_layers)
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._layer_idx = actual
        self._attach_hook(actual)

    def set_pooling(self, strategy: str) -> None:
        """Change the default pooling strategy."""
        _assert_valid_pooling(strategy)
        self._pooling = strategy

    def close(self) -> None:
        """Remove hook handle and free captured data."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured = None

    # ── Internal ──────────────────────────────────────────────

    @property
    def _device(self) -> torch.device:
        return self._model_device

    def _hook_fn(
        self,
        module: nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> None:
        """Forward hook callback — stores the output tensor."""
        if isinstance(output, tuple):
            # Some architectures return (hidden_states, ...)
            self._captured = output[0]
        else:
            self._captured = output

    def _attach_hook(self, layer_idx: int) -> None:
        target_module = _get_layer_module(self._model, layer_idx)
        self._hook_handle = target_module.register_forward_hook(self._hook_fn)  # type: ignore[arg-type]


# ── Helpers ────────────────────────────────────────────────────


def _get_model_device(model: Any) -> torch.device:
    """Detect device of the first model parameter."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _resolve_hidden_dim(model: Any) -> int:
    """Detect hidden dimension from model config."""
    config = getattr(model, "config", None)
    if config:
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "d_model"):
            return config.d_model  # e.g., T5
    # Fallback: inspect first weight
    for name, param in model.named_parameters():
        if "embed" in name.lower() and param.ndim == 2:
            return param.shape[1]
        if "lm_head" in name.lower() and param.ndim == 2:
            return param.shape[0]
    raise RuntimeError("Cannot determine hidden_dim from model")


def _resolve_num_layers(model: Any) -> int:
    """Detect number of transformer layers."""
    config = getattr(model, "config", None)
    if config:
        for attr in ("num_hidden_layers", "num_layers", "n_layer"):
            val = getattr(config, attr, None)
            if val is not None:
                return val
    count = sum(1 for _ in model.named_modules())
    return max(1, count)


def _resolve_layer_index(requested: int, num_layers: int) -> int:
    """Normalize layer index (supports negative indexing)."""
    if requested < 0:
        return max(0, num_layers + requested)
    if requested >= num_layers:
        return num_layers - 1
    return requested


def _get_layer_module(model: Any, layer_idx: int) -> nn.Module:
    """Find the module at the given transformer layer index."""
    # Common patterns for transformer block containers:

    # Pattern 1: model.transformer.h[N] (GPT-2, GPT-J, OPT)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]

    # Pattern 2: model.model.layers[N] (Llama, Mistral)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]

    # Pattern 3: model.model.decoder.layers[N] (T5, BART)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers[layer_idx]

    # Pattern 4: model.encoder.layer[N] (BERT)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer[layer_idx]

    # Pattern 5: Fallback — iterate all modules
    layer_counter = 0
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in ("block", "layer", "decoder_layer")):
            if layer_counter == layer_idx:
                return module
            layer_counter += 1

    raise RuntimeError(
        f"Cannot locate layer {layer_idx} in model. "
        f"Model has {_resolve_num_layers(model)} layers."
    )


def _get_tokenizer(model: Any) -> Any:
    """Retrieve tokenizer from model or raise helpful error."""
    tokenizer = getattr(model, "_a2a_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not set. Assign it via model._a2a_tokenizer = tokenizer "
            "before using TensorExtractor."
        )
    return tokenizer


def _apply_pooling(tensor: torch.Tensor, strategy: str) -> torch.Tensor:
    """Apply a pooling strategy to reduce sequence dimension."""
    # tensor shape: (batch, seq_len, hidden_dim)
    if strategy == "none":
        return tensor

    if strategy == "last":
        return tensor[:, -1, :]  # (batch, hidden_dim)

    if strategy == "mean":
        return tensor.mean(dim=1)  # (batch, hidden_dim)

    if strategy == "max":
        return tensor.max(dim=1).values  # (batch, hidden_dim)

    raise ValueError(f"Unknown pooling strategy: {strategy}. Use: last, mean, max, none")


def _assert_valid_pooling(strategy: str) -> None:
    valid = {"last", "mean", "max", "none"}
    if strategy not in valid:
        raise ValueError(f"Unknown pooling: {strategy}. Valid: {sorted(valid)}")


# ── Public helpers ─────────────────────────────────────────────

VALID_POOLING = frozenset({"last", "mean", "max", "none"})
