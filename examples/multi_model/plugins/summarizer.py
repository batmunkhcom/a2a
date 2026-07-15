"""Summarizer plugin — averages tensor along batch dimension."""

import numpy as np
from a2a.agent.base import BasePlugin


class SummarizerPlugin(BasePlugin):
    """Reduces tensor to summary representation."""

    def __init__(self):
        super().__init__(
            agent_id="summarizer",
            model="model-a",
            labels=["hidden_state"],
        )

    def process_tensor(
        self, data: np.ndarray, label: str, source_agent: str
    ) -> np.ndarray:
        if data.ndim == 2:
            return data.mean(axis=0, keepdims=True)
        return data

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass
