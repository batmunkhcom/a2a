"""Inverter plugin — inverts the sign of input tensors."""

import numpy as np
from a2a.agent.base import BasePlugin


class InverterPlugin(BasePlugin):
    """Plugin that inverts (negates) the input tensor."""

    def __init__(self):
        super().__init__(
            agent_id="inverter",
            model="base",
            labels=["hidden_state"],
        )

    def process_tensor(
        self, data: np.ndarray, label: str, source_agent: str
    ) -> np.ndarray:
        return -data

    async def initialize(self) -> None:
        self._count = 0

    async def shutdown(self) -> None:
        pass
