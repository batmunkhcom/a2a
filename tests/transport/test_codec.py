"""Tests for tensor codec (encode/decode roundtrip)."""

import struct

import pytest

from a2a.transport.codec import decode_tensor, encode_tensor, validate_tensor_values


def _make_data(shape: tuple[int, ...], dtype: str) -> bytes:
    """Build deterministic test data bytes matching shape * dtype size."""
    from a2a.transport.codec import _compute_byte_count

    return b"\x42" * _compute_byte_count(shape, dtype)


# ── Encoding ───────────────────────────────────────────────────


def test_encode_fp32_tensor() -> None:
    data = _make_data((1, 4096), "float32")
    frame = encode_tensor(data, (1, 4096), "float32")
    assert len(frame) > 0


def test_encode_fp16_tensor() -> None:
    data = _make_data((1, 2048), "float16")
    frame = encode_tensor(data, (1, 2048), "float16")
    assert len(frame) > 0


def test_encode_bf16_tensor() -> None:
    data = _make_data((1, 4096), "bfloat16")
    frame = encode_tensor(data, (1, 4096), "bfloat16")
    assert len(frame) > 0


def test_encode_multidim_tensor() -> None:
    data = _make_data((4, 64, 128), "float32")
    frame = encode_tensor(data, (4, 64, 128), "float32")
    assert len(frame) > 0


def test_encode_empty_tensor_raises() -> None:
    with pytest.raises(ValueError):
        encode_tensor(b"", (), "float32")


def test_encode_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        encode_tensor(b"short", (1, 4096), "float32")


def test_encode_unsupported_dtype_raises() -> None:
    with pytest.raises(ValueError):
        encode_tensor(b"\x00" * 4, (1,), "float64")


# ── Decoding ───────────────────────────────────────────────────


def test_decode_roundtrip_fp32() -> None:
    shape = (1, 4096)
    dtype = "float32"
    data = _make_data(shape, dtype)
    encoded = encode_tensor(data, shape, dtype)
    result = decode_tensor(encoded)
    assert result.shape == shape
    assert result.dtype == dtype
    assert result.values == data


def test_decode_roundtrip_fp16() -> None:
    shape = (1, 2048)
    dtype = "float16"
    data = _make_data(shape, dtype)
    encoded = encode_tensor(data, shape, dtype)
    result = decode_tensor(encoded)
    assert result.shape == shape
    assert result.dtype == dtype
    assert result.values == data


def test_decode_roundtrip_bf16() -> None:
    shape = (2, 1024)
    dtype = "bfloat16"
    data = _make_data(shape, dtype)
    encoded = encode_tensor(data, shape, dtype)
    result = decode_tensor(encoded)
    assert result.shape == shape
    assert result.dtype == dtype
    assert result.values == data


def test_decode_roundtrip_multidim() -> None:
    shape = (4, 32, 128)
    dtype = "float32"
    data = _make_data(shape, dtype)
    encoded = encode_tensor(data, shape, dtype)
    result = decode_tensor(encoded)
    assert result.shape == shape
    assert result.values == data


def test_decode_invalid_header_raises() -> None:
    with pytest.raises(ValueError, match="Invalid magic"):
        decode_tensor(b"\x00" * 32)


def test_decode_frame_too_short_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        decode_tensor(b"\x00" * 4)


def test_decode_checksum_mismatch_raises() -> None:
    shape = (1, 128)
    dtype = "float32"
    data = _make_data(shape, dtype)
    encoded = bytearray(encode_tensor(data, shape, dtype))
    encoded[-1] = (encoded[-1] ^ 0xFF) & 0xFF
    with pytest.raises(ValueError, match="Checksum"):
        decode_tensor(bytes(encoded))


def test_decode_unrecognized_msg_type_ok() -> None:
    data = _make_data((1, 64), "float32")
    encoded = encode_tensor(data, (1, 64), "float32", msg_type=0x99)
    result = decode_tensor(encoded)
    assert result.shape == (1, 64)


# ── Validation ─────────────────────────────────────────────────


def test_validate_clean_tensor_accepted() -> None:
    dtype = "float32"
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    assert validate_tensor_values(data, dtype) is True


def test_validate_nan_rejected() -> None:
    dtype = "float32"
    data = struct.pack("<4f", 1.0, float("nan"), 3.0, 4.0)
    assert validate_tensor_values(data, dtype) is False


def test_validate_inf_rejected() -> None:
    dtype = "float32"
    data = struct.pack("<4f", 1.0, float("inf"), 3.0, 4.0)
    assert validate_tensor_values(data, dtype) is False


def test_validate_negative_inf_rejected() -> None:
    dtype = "float32"
    data = struct.pack("<1f", float("-inf"))
    assert validate_tensor_values(data, dtype) is False
