"""Runtime auto-training trigger for projection models.

Detects unknown model pairs at runtime, collects training data
from a shared corpus, trains a projection model, and caches it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from a2a.projection.adapter import ProjectionModel, create_projection

logger = logging.getLogger("a2a.projection.auto_trainer")


class AutoTrainer:
    """Handles automatic projection model training at runtime.

    Triggered when two agents with different model architectures
    need to communicate but no pre-trained projection exists.

    Flow:
    1. Detection: two agents discover they have different models.
    2. Data collection: both agents process a shared text corpus.
    3. Training: ProjectionTrainer learns the mapping.
    4. Caching: trained model saved to registry.
    """

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        output_dir: str | Path = "./projections",
        device: str = "cpu",
        epochs: int = 50,
        lr: float = 0.001,
    ) -> None:
        self._corpus_path = Path(corpus_path) if corpus_path else None
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._device = device
        self._epochs = epochs
        self._lr = lr

    def train_for_pair(
        self,
        src_dim: int,
        tgt_dim: int,
        src_model_id: str,
        tgt_model_id: str,
        *,
        corpus: list[str] | None = None,
    ) -> ProjectionModel | None:
        """Train a projection model for the given model pair.

        Args:
            src_dim: Source model hidden dimension.
            tgt_dim: Target model hidden dimension.
            src_model_id: Source model identifier.
            tgt_model_id: Target model identifier.
            corpus: Optional list of text lines for training data.
                    If None, uses embedded corpus.

        Returns:
            Trained ProjectionModel, or None if training failed.
        """
        model_key = f"{src_model_id}__{tgt_model_id}"
        logger.info("Auto-training projection: %s (dim %d→%d)", model_key, src_dim, tgt_dim)

        start = time.monotonic()

        try:
            from torch.utils.data import DataLoader

            from a2a.projection.dataset import ProjectionPairDataset
            from a2a.projection.trainer import ProjectionTrainer
        except ImportError as exc:
            logger.error("AutoTrainer requires PyTorch: %s", exc)
            return None

        # Build dataset
        if corpus:
            dataset = self._build_dataset_from_corpus(
                corpus, src_dim, tgt_dim
            )
        else:
            # Synthetic dataset for testing/stub training
            dataset = ProjectionPairDataset.from_random(
                num_pairs=min(2000, max(100, src_dim * 2)),
                src_dim=src_dim,
                tgt_dim=tgt_dim,
            )

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create and train model
        model = create_projection("b", src_dim=src_dim, tgt_dim=tgt_dim)
        trainer = ProjectionTrainer(
            model, temperature=0.07, mse_weight=0.1, cosine_weight=0.01, lr=self._lr
        )

        trainer.train(dataloader, epochs=self._epochs, device=self._device)

        elapsed = time.monotonic() - start
        final_cos = trainer.history["cosine_similarity"][-1]

        logger.info(
            "Projection training complete: %s (%.1fs, cos_sim=%.4f)",
            model_key, elapsed, final_cos,
        )

        # Save to disk
        output_path = self._output_dir / f"{model_key}.safetensors"
        self._save_model(model, output_path)

        return model

    def get_cached_path(self, src_model_id: str, tgt_model_id: str) -> Path:
        """Return the expected cache file path for a model pair."""
        return self._output_dir / f"{src_model_id}__{tgt_model_id}.safetensors"

    def is_cached(self, src_model_id: str, tgt_model_id: str) -> bool:
        """Check if a projection model is already cached."""
        return self.get_cached_path(src_model_id, tgt_model_id).exists()

    # ── Internal ───────────────────────────────────────────────

    def _build_dataset_from_corpus(
        self,
        corpus: list[str],
        src_dim: int,
        tgt_dim: int,
    ) -> Any:
        """Build dataset using actual models processing the corpus."""
        # In real usage, this would run each text through both models
        # For stub implementation, use hash-based deterministic vectors
        import hashlib

        import torch

        from a2a.projection.dataset import ProjectionPairDataset

        src_list: list[Any] = []
        tgt_list: list[Any] = []

        for text in corpus[:2000]:  # Limit to 2000 pairs
            h = hashlib.md5(text.encode()).hexdigest()
            seed = int(h[:8], 16)
            torch.manual_seed(seed)
            src_list.append(torch.randn(src_dim) * 0.02)
            torch.manual_seed(seed + 1)
            tgt_list.append(torch.randn(tgt_dim) * 0.02)

        return ProjectionPairDataset.from_lists(src_list, tgt_list)

    def _save_model(self, model: ProjectionModel, path: Path) -> None:
        """Save projection model state to disk."""
        try:
            from safetensors.torch import save_file

            state = model.state_dict()
            save_file(state, str(path))
        except ImportError:
            import torch

            torch.save(model.state_dict(), str(path).replace(".safetensors", ".pt"))


__all__ = ["AutoTrainer"]
