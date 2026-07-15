"""Normalizer plugin — L2-normalizes input tensors."""

import numpy as np
from a2a.agent.base import BasePlugin


class NormalizerPlugin(BasePlugin):
    """L2-normalizes input tensor for stable cross-model transfer."""

    def __init__(self):
        super().__init__(
            agent_id="normalizer",
            model="model-b",
            labels=["hidden_state"],
        )

    def process_tensor(
        self, data: np.ndarray, label: str, source_agent: str
    ) -> np.ndarray:
        norm = np.linalg.norm(data, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        return data / norm

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
