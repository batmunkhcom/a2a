"""Mixer plugin — linear combination of two input tensors."""

import numpy as np
from a2a.agent.base import BasePlugin


class MixerPlugin(BasePlugin):
    """Combines tensors from two sources with configurable weights."""

    def __init__(self):
        super().__init__(
            agent_id="mixer",
            model="model-b",
            labels=["hidden_state", "mixed_state"],
        )

    def process_tensor(
        self, data: np.ndarray, label: str, source_agent: str
    ) -> np.ndarray:
        if label == "mixed_state":
            alpha = self.config.get("alpha", 0.5)
            # data is already combined externally, just scale
            return data * alpha
        return data

    async def initialize(self) -> None:
        self.config.setdefault("alpha", 0.5)

    async def shutdown(self) -> None:
        pass
