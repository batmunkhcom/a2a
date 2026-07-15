"""ProjectionRegistry — persistent cache of trained projection models.

Stores projection models by source_model__target_model key.
Supports disk persistence (safetensors or PyTorch checkpoint).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("a2a.projection.registry")


class ProjectionRegistry:
    """Cache and retrieve trained projection models.

    Registration key format: "source_model_id__target_model_id"
    """

    def __init__(self, cache_dir: str | Path = "./projections") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}
        self._dims: dict[str, tuple[int, int]] = {}

    def get(self, src_model: str, tgt_model: str) -> Any | None:
        """Retrieve a cached projection model.

        Args:
            src_model: Source model ID.
            tgt_model: Target model ID.

        Returns:
            ProjectionModel if cached, None otherwise.
        """
        key = f"{src_model}__{tgt_model}"
        if key in self._models:
            return self._models[key]

        # Try loading from disk
        return self._load_from_disk(key)

    def put(
        self,
        src_model: str,
        tgt_model: str,
        model: Any,
        src_dim: int,
        tgt_dim: int,
        save: bool = True,
    ) -> None:
        """Cache a projection model.

        Args:
            src_model: Source model ID.
            tgt_model: Target model ID.
            model: Trained ProjectionModel.
            src_dim: Source hidden dimension.
            tgt_dim: Target hidden dimension.
            save: Whether to persist to disk.
        """
        key = f"{src_model}__{tgt_model}"
        self._models[key] = model
        self._dims[key] = (src_dim, tgt_dim)

        if save:
            self._save_to_disk(key, model)

    def load(self, src_model: str, tgt_model: str, path: str | Path) -> Any | None:
        """Load a projection model from a specific file path.

        Args:
            src_model: Source model ID.
            tgt_model: Target model ID.
            path: Path to the saved model file.

        Returns:
            Loaded ProjectionModel, or None on failure.
        """
        key = f"{src_model}__{tgt_model}"
        path = Path(path)

        if not path.exists():
            logger.warning("Projection file not found: %s", path)
            return None

        try:
            try:
                from safetensors.torch import load_file

                state = load_file(str(path))
            except ImportError:
                import torch

                state = torch.load(str(path), map_location="cpu", weights_only=True)

            config = state.get("config", {})
            src_dim = config.get("src_dim", 0)
            tgt_dim = config.get("tgt_dim", 0)

            from a2a.projection.adapter import ProjectionModel

            model = ProjectionModel(
                src_dim=src_dim or state.get("net.0.weight", {}).shape[0],
                tgt_dim=tgt_dim or state.get("net.-1.weight", {}).shape[0],
                variant=config.get("variant", "b"),
            )
            model.load_state_dict(state)

            self._models[key] = model
            self._dims[key] = (src_dim, tgt_dim)
            return model

        except Exception as exc:
            logger.error("Failed to load projection %s: %s", key, exc)
            return None

    def save(self, src_model: str, tgt_model: str) -> None:
        """Persist a cached model to disk."""
        key = f"{src_model}__{tgt_model}"
        model = self._models.get(key)
        if model:
            self._save_to_disk(key, model)

    def list_cached(self) -> list[str]:
        """Return all cached projection keys."""
        return list(self._models.keys())

    def has(self, src_model: str, tgt_model: str) -> bool:
        """Check if a projection is cached."""
        return f"{src_model}__{tgt_model}" in self._models

    # ── Internal ───────────────────────────────────────────────

    def _save_to_disk(self, key: str, model: Any) -> None:
        """Save model state to disk."""
        path = self._cache_dir / f"{key}.safetensors"
        try:
            from safetensors.torch import save_file

            save_file(model.state_dict(), str(path))
        except ImportError:
            import torch

            torch.save(model.state_dict(), str(path).replace(".safetensors", ".pt"))

    def _load_from_disk(self, key: str) -> Any | None:
        """Try loading a projection from disk."""
        parts = key.split("__")
        return self.load(
            parts[0], parts[1], self._cache_dir / f"{key}.safetensors"
        )
