"""Projection Model — lightweight MLP adapter mapping hidden states between model architectures.

Variants:
- A (Linear): Single linear layer — src_dim → tgt_dim.
- B (3-layer MLP): 2 hidden layers + residual + LayerNorm (~2-4M params).
- C (Deep MLP): 3 hidden layers + residual (~6-8M params).
"""

from __future__ import annotations

from typing import Any

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F_  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ProjectionModel:
    """Lightweight MLP adapter for cross-model latent space mapping.

    Architecture: LayerNorm → Linear → GELU → Dropout → Linear → GELU →
                  Linear → LayerNorm (with residual skip).

    Args:
        src_dim: Source model hidden dimension (e.g. 4096 for Llama-8B).
        tgt_dim: Target model hidden dimension (e.g. 4096 for Mistral-7B).
        hidden_dim: Internal MLP dimension (default 2048).
        num_hidden: Number of hidden layers (2 = variant B, 3 = variant C).
        dropout: Dropout probability.
        variant: "a" (Linear), "b" (2-layer MLP), "c" (3-layer MLP).
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        hidden_dim: int = 2048,
        num_hidden: int = 2,
        dropout: float = 0.1,
        variant: str = "b",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("ProjectionModel requires PyTorch")

        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.hidden_dim = hidden_dim
        self.variant = variant

        layers: list[nn.Module] = []
        layers.append(nn.LayerNorm(src_dim))

        if variant == "a":
            # Single linear projection
            layers.append(nn.Linear(src_dim, tgt_dim, bias=False))
        else:
            # MLP with residual
            in_dim = src_dim
            for i in range(num_hidden):
                out_dim = hidden_dim if i < num_hidden - 1 else tgt_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < num_hidden - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                in_dim = out_dim

        layers.append(nn.LayerNorm(tgt_dim))

        self._net = nn.Sequential(*layers)
        self._dropout = nn.Dropout(dropout)

        # Residual projection (if src_dim != tgt_dim)
        if src_dim != tgt_dim:
            self._residual = nn.Linear(src_dim, tgt_dim, bias=False)
        else:
            self._residual = nn.Identity()

    def forward(self, x: Any) -> Any:
        """Forward pass: project src hidden → tgt hidden space.

        Args:
            x: Source hidden state tensor of shape (..., src_dim).

        Returns:
            Projected tensor of shape (..., tgt_dim).
        """
        out = self._net(x)
        residual = self._residual(x)
        return out + residual

    def parameters(self) -> Any:
        """Return model parameters for training."""
        return self._net.parameters()

    def train(self, mode: bool = True) -> ProjectionModel:
        """Set model to training mode."""
        self._net.train(mode)
        return self

    def eval(self) -> ProjectionModel:
        """Set model to evaluation mode."""
        self._net.eval()
        return self

    def to(self, device: Any) -> ProjectionModel:
        """Move model to device."""
        self._net.to(device)
        if not isinstance(self._residual, nn.Identity):
            self._residual.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return model state for saving."""
        return {
            "net": self._net.state_dict(),
            "residual": (
                self._residual.state_dict()
                if not isinstance(self._residual, nn.Identity)
                else {}
            ),
            "config": {
                "src_dim": self.src_dim,
                "tgt_dim": self.tgt_dim,
                "hidden_dim": self.hidden_dim,
                "variant": self.variant,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load model state."""
        self._net.load_state_dict(state["net"])
        if (
            "residual" in state
            and state["residual"]
            and not isinstance(self._residual, nn.Identity)
        ):
            self._residual.load_state_dict(state["residual"])


def create_projection(
    variant: str,
    src_dim: int,
    tgt_dim: int,
    **kwargs: Any,
) -> ProjectionModel:
    """Factory function for creating projection models.

    Args:
        variant: "a" (Linear), "b" (2-layer MLP), "c" (3-layer MLP).
        src_dim: Source model hidden dimension.
        tgt_dim: Target model hidden dimension.
        **kwargs: Passed to ProjectionModel.

    Returns:
        Configured ProjectionModel.
    """
    variant_map = {"a": 0, "b": 2, "c": 3}
    num_hidden = variant_map.get(variant, 2)
    return ProjectionModel(
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        num_hidden=num_hidden,
        variant=variant,
        **kwargs,
    )
