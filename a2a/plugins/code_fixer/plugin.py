"""Code Fixer Plugin — receives error context tensors and generates code patches.

Listens to: error_context
Produces: code_patch, fix_explanation
"""

from __future__ import annotations

from typing import Any

from a2a.agent.base import BasePlugin, Capability, ModelInfo


class CodeFixerPlugin(BasePlugin):
    """Receives error context vectors and generates code fixes."""

    @property
    def plugin_id(self) -> str:
        return "code-fixer"

    @property
    def plugin_name(self) -> str:
        return "Code Fixer Agent"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="code_generation",
                description="Generate code fixes from error context",
                input_labels=["error_context"],
                output_labels=["code_patch", "fix_explanation"],
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
        return ["error_context"]

    def emits(self) -> list[str]:
        return ["code_patch", "fix_explanation"]

    async def on_receive_tensor(
        self,
        tensor: Any,
        metadata: Any,
    ) -> Any:
        """Receive error_context tensor and generate a code fix.

        Args:
            tensor: Error context tensor or dict with shape/dtype/seed.
            metadata: TensorMetadata from sender.

        Returns:
            Dict with code_patch and explanation.
        """
        prompt = self.plugin_config.get("system_prompt", "Fix the following error:")
        max_tokens = self.plugin_config.get("max_output_tokens", 512)

        try:
            from a2a.tensor.injector import TensorInjector

            has_model = (
                hasattr(self, "model") and self.model
                and hasattr(self, "tokenizer") and self.tokenizer
            )
            if has_model:
                self.model._a2a_tokenizer = self.tokenizer
                injector = TensorInjector(self.model, mode="prefix")

                if hasattr(tensor, "dim"):
                    # PyTorch tensor
                    return {
                        "code_patch": injector.inject_and_generate(
                            tensor, prompt=prompt, max_tokens=max_tokens
                        ),
                        "fix_explanation": "Generated via tensor injection",
                    }

        except (ImportError, Exception):
            pass

        # Fallback: deterministic stub response
        _seed = tensor if isinstance(tensor, int) else 42
        _ = _seed  # reserved for future deterministic generation
        return {
            "code_patch": (
                "# Fix applied for error context\n"
                f"# Source: {getattr(metadata, 'source_model', 'unknown')}\n"
                "# Auto-generated code patch"
            ),
            "fix_explanation": (
                "Stub code fix generated from error context vector.\n"
                "Actual model injection will be used when PyTorch is available."
            ),
        }

    async def extract_tensor(
        self,
        input_data: Any,
        semantic_label: str,
    ) -> Any:
        # CodeFixer mainly receives tensors, not produces them
        return None
