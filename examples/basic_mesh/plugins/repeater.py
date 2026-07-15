"""Repeater plugin — echoes input tensors back."""

import numpy as np
from a2a.agent.base import BasePlugin


class RepeaterPlugin(BasePlugin):
    """Simple plugin that returns the input tensor as-is."""

    def __init__(self):
        super().__init__(
            agent_id="repeater",
            model="base",
            labels=["hidden_state"],
        )

    def process_tensor(
        self, data: np.ndarray, label: str, source_agent: str
    ) -> np.ndarray:
        return data

    async def initialize(self) -> None:
        self._count = 0

    async def shutdown(self) -> None:
        pass
