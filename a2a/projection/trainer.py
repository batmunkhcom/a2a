"""Projection Trainer — trains projection models using contrastive learning.

Implements InfoNCE contrastive loss with optional MSE and cosine distance
regularization. Designed for the cross-model latent space alignment task.
"""

from __future__ import annotations

from typing import Any

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401, N812

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ProjectionTrainer:
    """Trains a ProjectionModel using supervised contrastive learning.

    Loss = InfoNCE + α * MSE + β * (1 - cos_sim)

    Args:
        model: The ProjectionModel to train.
        temperature: InfoNCE temperature parameter (lower = harder contrast).
        mse_weight: Weight for MSE loss term (default 0.1).
        cosine_weight: Weight for cosine similarity loss (default 0.01).
        lr: Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        model: Any,
        temperature: float = 0.07,
        mse_weight: float = 0.1,
        cosine_weight: float = 0.01,
        lr: float = 0.001,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("ProjectionTrainer requires PyTorch")

        self._model = model
        self._temperature = temperature
        self._mse_weight = mse_weight
        self._cosine_weight = cosine_weight
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._history: dict[str, list[float]] = {
            "loss": [],
            "contrastive": [],
            "mse": [],
            "cosine": [],
            "cosine_similarity": [],
        }

    def train(
        self,
        dataloader: Any,
        epochs: int = 50,
        device: str = "cpu",
    ) -> dict[str, list[float]]:
        """Run full training loop.

        Args:
            dataloader: DataLoader yielding (src_hidden, tgt_hidden) batches.
            epochs: Number of training epochs.
            device: Device to train on ("cpu" or "cuda:0").

        Returns:
            Dict with loss history per epoch.
        """
        self._model.to(device)
        self._model.train()

        for _epoch in range(epochs):
            epoch_losses = {"contrastive": 0.0, "mse": 0.0, "cosine": 0.0, "cos_sim": 0.0}
            n_batches = 0

            for src, tgt in dataloader:
                src = src.to(device)
                tgt = tgt.to(device)

                projected = self._model.forward(src)

                # Compute losses
                contrastive = self._info_nce(projected, tgt)
                mse = F.mse_loss(projected, tgt)
                cos_loss = 1.0 - F.cosine_similarity(
                    projected.flatten(1), tgt.flatten(1)
                ).mean()

                loss = (
                    contrastive
                    + self._mse_weight * mse
                    + self._cosine_weight * cos_loss
                )

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_losses["contrastive"] += contrastive.item()
                epoch_losses["mse"] += mse.item()
                epoch_losses["cosine"] += cos_loss.item()

                with torch.no_grad():
                    cos_sim = F.cosine_similarity(
                        projected.flatten(1), tgt.flatten(1)
                    ).mean()
                    epoch_losses["cos_sim"] += cos_sim.item()

                n_batches += 1

            if n_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= n_batches

            self._history["contrastive"].append(epoch_losses["contrastive"])
            self._history["mse"].append(epoch_losses["mse"])
            self._history["cosine"].append(epoch_losses["cosine"])
            total = (
                epoch_losses["contrastive"]
                + self._mse_weight * epoch_losses["mse"]
                + self._cosine_weight * epoch_losses["cosine"]
            )
            self._history["loss"].append(total)
            self._history["cosine_similarity"].append(epoch_losses["cos_sim"])

        return self._history

    def _info_nce(self, projected: Any, tgt: Any) -> Any:
        """InfoNCE contrastive loss.

        Treats each batch item's own target as positive,
        all other targets in the batch as negatives.

        Args:
            projected: (batch, dim) projected vectors.
            tgt: (batch, dim) target vectors.

        Returns:
            Scalar InfoNCE loss.
        """
        batch_size = projected.shape[0]

        # Normalize
        projected = F.normalize(projected.flatten(1), dim=1)
        tgt_normed = F.normalize(tgt.flatten(1), dim=1)

        # Logits: projection vs all targets
        logits = torch.matmul(projected, tgt_normed.T) / self._temperature

        # Labels: diagonal = positive pair
        labels = torch.arange(batch_size, device=projected.device)

        return F.cross_entropy(logits, labels)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def history(self) -> dict[str, list[float]]:
        return self._history
