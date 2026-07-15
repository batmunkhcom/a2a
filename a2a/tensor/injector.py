"""Tensor injection into HuggingFace models for conditional generation.

Supports two injection modes:
- Prefix injection: prepend tensor as embedding prefix before text prompt.
- Cross-attention injection: use tensor as key/value in cross-attention layers.
"""

from __future__ import annotations

from typing import Any

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TensorInjector:
    """Injects latent vectors into a target model for generation.

    Args:
        model: A HuggingFace PreTrainedModel with generate() capability.
        target_layer: Layer index for cross-attention injection.
                      0 = prefix injection (into embeddings).
                      -1 = last layer cross-attention.
        mode: Injection mode — "prefix" or "cross_attention".

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        injector = TensorInjector(model)

        hidden = ...  # (1, hidden_dim) tensor from another model
        result = injector.inject_and_generate(
            hidden, prompt="Fix this error:",
            max_tokens=256
        )
    """

    def __init__(
        self,
        model: Any,
        *,
        target_layer: int = 0,
        mode: str = "prefix",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "TensorInjector requires PyTorch. Install with: pip install a2a-protocol[ml]"
            )

        self._model = model
        self._model_device = _get_device(model)
        self._target_layer = target_layer
        self._mode = mode

        # Detect embedding layer
        self._embedding: nn.Module | None = _find_embedding(model)
        if mode == "prefix" and self._embedding is None:
            raise RuntimeError("Cannot find embedding layer for prefix injection")

    # ── Injection methods ──────────────────────────────────────

    def inject_prefix(
        self,
        tensor: torch.Tensor,
        prompt: str,
        *_,
        tokenizer: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inject tensor as an embedding prefix before the text prompt.

        The tensor is treated as N virtual tokens prepended to the
        actual text tokens. This uses inputs_embeds instead of input_ids.

        Args:
            tensor: Hidden state vector of shape (1, hidden_dim) or
                    (1, seq_len, hidden_dim).
            prompt: Text prompt to append after the injected tensor.
            tokenizer: Optional tokenizer override.

        Returns:
            A dict with "inputs_embeds" and "attention_mask" keys,
            ready to pass to model.generate().
        """
        if self._embedding is None:
            raise RuntimeError("No embedding layer found — cannot use prefix injection")

        tok = tokenizer or _get_tokenizer(self._model)

        # Ensure proper shape
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)  # (1, hidden_dim) → (1, 1, hidden_dim)

        tensor = tensor.to(device=self._model_device, dtype=self._embedding.weight.dtype)

        # Project tensor if dimensions differ
        if tensor.shape[-1] != self._embedding.weight.shape[1]:
            tensor = _linear_project(tensor, self._embedding.weight.shape[1])

        # Tokenize prompt
        text_inputs = tok(prompt, return_tensors="pt", add_special_tokens=True)
        text_ids = text_inputs["input_ids"].to(self._model_device)
        text_embeds = self._embedding(text_ids)

        # Concatenate: [virtual_tokens | text_tokens]
        inputs_embeds = torch.cat([tensor, text_embeds], dim=1)

        # Build attention mask
        batch, virtual_len = tensor.shape[0], tensor.shape[1]
        _, text_len = text_embeds.shape[0], text_embeds.shape[1]
        attention_mask = torch.cat(
            [
                torch.ones(batch, virtual_len, device=self._model_device),
                text_inputs.get("attention_mask", torch.ones(batch, text_len)).to(
                    self._model_device
                ),
            ],
            dim=1,
        )

        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

    def inject_cross_attention(
        self,
        tensor: torch.Tensor,
        prompt: str,
        *_,
        tokenizer: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Use tensor as cross-attention key/value.

        This targets the encoder_hidden_states argument of decoder models
        (e.g., T5, BART, encoder-decoder architectures).

        For decoder-only models (GPT-2, Llama), this method wraps the
        tensor into a format compatible with cross-attention injection.

        Args:
            tensor: Hidden state of shape (1, hidden_dim) or (1, seq, hidden_dim).
            prompt: Text prompt for generation.
            tokenizer: Optional tokenizer override.

        Returns:
            A dict with "input_ids" and "encoder_hidden_states" keys.
        """
        tok = tokenizer or _get_tokenizer(self._model)
        tensor = tensor.to(device=self._model_device, dtype=torch.float32)

        # Expand: (1, hidden_dim) → (1, 1, hidden_dim)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)

        inputs = tok(prompt, return_tensors="pt").to(self._model_device)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "encoder_outputs": (tensor,),
        }

    def inject_and_generate(
        self,
        tensor: torch.Tensor,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tokenizer: Any | None = None,
    ) -> str:
        """Convenience: inject tensor + generate text.

        Args:
            tensor: Hidden state tensor to inject.
            prompt: Text prompt.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            tokenizer: Optional tokenizer override.

        Returns:
            Generated text string.
        """
        if self._mode == "prefix":
            gen_inputs = self.inject_prefix(tensor, prompt, tokenizer=tokenizer)
        elif self._mode == "cross_attention":
            gen_inputs = self.inject_cross_attention(tensor, prompt, tokenizer=tokenizer)
        else:
            raise ValueError(f"Unknown injection mode: {self._mode}")

        tok = tokenizer or _get_tokenizer(self._model)

        with torch.no_grad():
            outputs = self._model.generate(
                **gen_inputs,
                max_new_tokens=max_tokens,
                temperature=None if temperature <= 0 else temperature,
                do_sample=temperature > 0,
                pad_token_id=tok.eos_token_id or 0,
                eos_token_id=tok.eos_token_id,
            )

        # Decode only the newly generated tokens
        gen_inputs.pop("encoder_outputs", None)
        gen_inputs.pop("inputs_embeds", None)
        prompt_len = gen_inputs.get("input_ids", torch.tensor([[]])).shape[-1]

        new_tokens = outputs[0][prompt_len:] if prompt_len > 0 else outputs[0]
        return tok.decode(new_tokens, skip_special_tokens=True)

    # ── Helpers ─────────────────────────────────────────────────

    def close(self) -> None:
        """Release any held resources."""
        pass


# ── Helpers ────────────────────────────────────────────────────


def _find_embedding(model: Any) -> nn.Module | None:
    """Find the token embedding module in a HuggingFace model."""
    candidates = []

    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(k in name_lower for k in ("wte", "embed_tokens", "word_embeddings")):
            candidates.append((name, module))

    if not candidates:
        # Fallback: look for any Embedding layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                candidates.append((name, module))

    if candidates:
        return candidates[0][1]

    # Last resort: check common attribute paths
    for path in [
        "transformer.wte",
        "model.embed_tokens",
        "bert.embeddings.word_embeddings",
    ]:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj  # type: ignore[return-value]

    return None


def _get_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _get_tokenizer(model: Any) -> Any:
    tokenizer = getattr(model, "_a2a_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer not set. Assign via model._a2a_tokenizer = tokenizer."
        )
    return tokenizer


def _linear_project(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project tensor to target dimension using a simple linear layer."""
    src_dim = tensor.shape[-1]
    if src_dim == target_dim:
        return tensor

    proj = nn.Linear(src_dim, target_dim, bias=False).to(tensor.device, tensor.dtype)
    with torch.no_grad():
        proj.weight.zero_()
        proj.weight[:, : min(src_dim, target_dim)] = torch.eye(
            min(src_dim, target_dim), device=tensor.device, dtype=tensor.dtype
        )
    return proj(tensor)
