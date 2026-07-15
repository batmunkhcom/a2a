"""ProjectionPairDataset — builds (src_hidden, tgt_hidden) training pairs.

Each text line in a shared corpus is passed through both source and target
models to extract hidden states. Positive pairs are same-text; negative
pairs are different-text combinations.
"""

from __future__ import annotations

from typing import Any

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ProjectionPairDataset:
    """Dataset of (source_hidden, target_hidden) pairs for projection training.

    Created from a shared text corpus processed through two different models.
    """

    def __init__(
        self,
        src_hiddens: Any,
        tgt_hiddens: Any,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("ProjectionPairDataset requires PyTorch")

        self._src = src_hiddens
        self._tgt = tgt_hiddens
        if len(src_hiddens) != len(tgt_hiddens):
            raise ValueError(
                f"Mismatched lengths: src={len(src_hiddens)}, tgt={len(tgt_hiddens)}"
            )

    def __len__(self) -> int:
        return len(self._src)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self._src[idx], self._tgt[idx]

    def get_negative_pair(self, idx: int) -> tuple[Any, Any]:
        """Return a negative pair (src at idx, tgt at different idx)."""
        neg_idx = (idx + 1) % len(self._tgt)
        return self._src[idx], self._tgt[neg_idx]

    @classmethod
    def from_lists(
        cls,
        src_list: list[Any],
        tgt_list: list[Any],
    ) -> ProjectionPairDataset:
        """Create dataset from lists of hidden states.

        Args:
            src_list: List of source hidden state tensors.
            tgt_list: List of target hidden state tensors.

        Returns:
            ProjectionPairDataset.
        """
        return cls(
            torch.stack([t.detach().cpu() for t in src_list]),
            torch.stack([t.detach().cpu() for t in tgt_list]),
        )

    @classmethod
    def from_random(
        cls,
        num_pairs: int,
        src_dim: int,
        tgt_dim: int,
        seed: int = 42,
    ) -> ProjectionPairDataset:
        """Create a dataset with random synthetic hidden states for testing.

        Args:
            num_pairs: Number of pairs to generate.
            src_dim: Source hidden dimension.
            tgt_dim: Target hidden dimension.
            seed: Random seed for reproducibility.

        Returns:
            ProjectionPairDataset with controlled random tensors.
        """
        torch.manual_seed(seed)
        src = torch.randn(num_pairs, src_dim) * 0.02
        tgt = torch.randn(num_pairs, tgt_dim) * 0.02
        return cls(src, tgt)

    @property
    def src_dim(self) -> int:
        return self._src.shape[-1]

    @property
    def tgt_dim(self) -> int:
        return self._tgt.shape[-1]

    @property
    def num_pairs(self) -> int:
        return len(self._src)
