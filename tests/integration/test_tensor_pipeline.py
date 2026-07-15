"""End-to-end tensor pipeline test: extract → encode → decode → inject.

Requires PyTorch. Uses FakeModel for testing without real model weights.
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.tensor.dtype import convert_dtype, validate_tensor  # noqa: E402
from a2a.transport.codec import decode_tensor, encode_tensor  # noqa: E402


class FakePipeline:
    """Simulate the full pipeline without real HF models."""

    def __init__(self, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim

    def extract(self, text: str) -> torch.Tensor:
        """Fake extraction — deterministic from text hash."""
        seed = sum(ord(c) for c in text)
        torch.manual_seed(seed)
        return torch.randn(1, self.hidden_dim)

    def inject(self, tensor: torch.Tensor, prompt: str) -> str:
        """Fake injection — returns prompt + tensor stats."""
        return f"Generated from prompt=[{prompt}] with tensor norm={tensor.norm().item():.2f}"


def test_extract_to_encode_to_decode_to_inject_pipeline() -> None:
    pipeline = FakePipeline(hidden_dim=64)

    # Step 1: Extract
    text = "ERROR: NullPointerException at line 45"
    hidden = pipeline.extract(text)
    assert hidden.shape == (1, 64)

    # Step 2: Encode
    raw = hidden.detach().cpu().numpy().tobytes()
    encoded = encode_tensor(raw, hidden.shape, "float32")
    assert len(encoded) > 0

    # Step 3: Decode
    decoded = decode_tensor(encoded)
    assert decoded.shape == (1, 64)
    assert decoded.dtype == "float32"

    # Step 4: Validate
    assert validate_tensor(decoded.values, "float32") is True

    # Step 5: Reconstruct and inject
    restored = torch.frombuffer(bytearray(decoded.values), dtype=torch.float32).reshape(
        decoded.shape
    )
    result = pipeline.inject(restored, "Fix this error:")
    assert "Fix this error" in result
    assert "tensor norm" in result


def test_pipeline_with_dtype_conversion() -> None:
    """Test the pipeline with FP32 → FP16 conversion in transit."""
    pipeline = FakePipeline(hidden_dim=128)

    # Extract
    hidden = pipeline.extract("test log line")
    raw = hidden.detach().cpu().numpy().tobytes()
    encoded_fp32 = encode_tensor(raw, (1, 128), "float32")

    # Simulate network: decode, convert to FP16, re-encode
    decoded = decode_tensor(encoded_fp32)
    fp16_data = convert_dtype(decoded.values, "float32", "float16")
    encoded_fp16 = encode_tensor(fp16_data, (1, 128), "float16")

    # Receive: decode back to FP32
    decoded2 = decode_tensor(encoded_fp16)
    assert decoded2.dtype == "float16"

    fp32_data = convert_dtype(decoded2.values, "float16", "float32")
    restored = torch.frombuffer(bytearray(fp32_data), dtype=torch.float32).reshape(1, 128)

    # Validate
    assert validate_tensor(fp32_data, "float32") is True
    assert restored.shape == (1, 128)


def test_pipeline_nan_detection() -> None:
    """Tensor with NaN should fail validation."""
    raw = b"\x00" * (4 * 16)
    encoded = encode_tensor(raw, (16,), "float32")

    decoded = decode_tensor(encoded)

    # Inject NaN into decoded values
    import struct

    nan_bytes = bytearray(decoded.values)
    nan_bytes[0:4] = struct.pack("<f", float("nan"))
    # Re-encode with NaN
    encoded_nan = encode_tensor(bytes(nan_bytes), (16,), "float32")
    decoded_nan = decode_tensor(encoded_nan)

    assert validate_tensor(decoded_nan.values, "float32") is False
