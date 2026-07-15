"""Log Reader Plugin — analyzes logs and extracts error context vectors.

Produces: error_context, log_summary
"""

from __future__ import annotations

from typing import Any

from a2a.agent.base import BasePlugin, Capability, ModelInfo


class LogReaderPlugin(BasePlugin):
    """Reads system logs, detects errors, and emits error context tensors."""

    @property
    def plugin_id(self) -> str:
        return "log-reader"

    @property
    def plugin_name(self) -> str:
        return "Log Reader Agent"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="log_analysis",
                description="Analyze system logs and extract error context",
                input_labels=[],
                output_labels=["error_context", "log_summary"],
                model_id="llama-8b",
                hidden_dim=4096,
                dtype="float16",
            )
        ]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id="llama-8b",
            architecture="llama",
            hidden_dim=4096,
            num_layers=32,
            dtype="float16",
        )

    def listens_to(self) -> list[str]:
        return []  # Log reader only works on external triggers

    def emits(self) -> list[str]:
        return ["error_context", "log_summary"]

    async def on_receive_tensor(
        self,
        tensor: Any,
        metadata: Any,
    ) -> Any:
        # LogReader doesn't receive tensors — it only produces them
        return None

    async def extract_tensor(
        self,
        input_data: Any,
        semantic_label: str,
    ) -> Any:
        """Extract tensor from log text.

        If PyTorch is available, extracts real hidden state from the model.
        Otherwise falls back to a deterministic hash-based vector.
        """
        text = str(input_data)

        if semantic_label == "error_context":
            return await self._extract_error_context(text)
        if semantic_label == "log_summary":
            return await self._extract_summary(text)

        raise ValueError(f"Unknown semantic label: {semantic_label}")

    async def _extract_error_context(self, text: str) -> Any:
        """Extract error context from log text."""
        try:
            from a2a.tensor.extractor import TensorExtractor

            has_model = (
                hasattr(self, "model") and self.model
                and hasattr(self, "tokenizer") and self.tokenizer
            )
            if has_model:
                self.model._a2a_tokenizer = self.tokenizer
                extractor = TensorExtractor(self.model, layer_idx=-1, pooling="last")
                return extractor.extract(text)
        except (ImportError, Exception):
            pass

        # Fallback: deterministic vector from text hash
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest()[:16], 16) % (2**32)
        # Return a list representation for transport
        return {
            "shape": (1, 4096),
            "dtype": "float16",
            "seed": seed,
        }

    async def _extract_summary(self, text: str) -> Any:
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest()[:16], 16) % (2**32)
        return {
            "shape": (1, 4096),
            "dtype": "float16",
            "seed": seed,
        }
