"""Tests for dtype conversion and tensor validation (pure Python, no torch)."""

import math
import struct

import pytest

from a2a.tensor.dtype import (
    _float16_to_float32,
    _float32_to_float16,
    convert_dtype,
    is_finite,
    validate_l2_norm,
    validate_tensor,
)

# ── Roundtrip helpers ──────────────────────────────────────────


def _pack_float32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_float32(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


# ── float16 conversion ─────────────────────────────────────────


def test_fp32_to_fp16_and_back() -> None:
    values = [1.0, -2.5, 0.0, 3.14159, -0.001, 65504.0, -65504.0]
    for v in values:
        h = _float32_to_float16(v)
        restored = _float16_to_float32(h)
        # Half-precision loses some precision
        assert abs(restored - v) / max(abs(v), 0.001) < 0.01, f"v={v}, restored={restored}"


def test_fp16_special_values() -> None:
    # NaN
    nan_h = _float32_to_float16(float("nan"))
    restored_nan = _float16_to_float32(nan_h)
    assert math.isnan(restored_nan)

    # +Inf
    inf_h = _float32_to_float16(float("inf"))
    assert _float16_to_float32(inf_h) == float("inf")

    # -Inf
    neg_h = _float32_to_float16(float("-inf"))
    assert _float16_to_float32(neg_h) == float("-inf")

    # Zero
    zero_h = _float32_to_float16(0.0)
    assert _float16_to_float32(zero_h) == 0.0


# ── dtype conversion ──────────────────────────────────────────


def test_convert_dtype_fp32_to_fp16() -> None:
    data = _pack_float32([1.0, 2.0, 3.0, 4.0])
    result = convert_dtype(data, "float32", "float16")
    assert len(result) == 8  # 4 × 2 bytes


def test_convert_dtype_fp16_to_fp32() -> None:
    # First, convert known values
    orig = _pack_float32([1.0, -2.0, 0.5])
    fp16 = convert_dtype(orig, "float32", "float16")
    fp32 = convert_dtype(fp16, "float16", "float32")

    restored = _unpack_float32(fp32)
    for i, expected in enumerate([1.0, -2.0, 0.5]):
        assert abs(restored[i] - expected) < 0.01


def test_convert_dtype_same_type_identity() -> None:
    data = _pack_float32([1.0, 2.0])
    result = convert_dtype(data, "float32", "float32")
    assert result == data


def test_convert_dtype_bf16() -> None:
    data = _pack_float32([1.0, 2.0, 3.0])
    bf16 = convert_dtype(data, "float32", "bfloat16")
    fp32 = convert_dtype(bf16, "bfloat16", "float32")
    restored = _unpack_float32(fp32)
    # bfloat16 keeps exponent, truncates mantissa → very close to original
    for r, e in zip(restored, [1.0, 2.0, 3.0], strict=False):
        assert abs(r - e) < 1e-6


def test_convert_dtype_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        convert_dtype(b"\x00", "float32", "float64")


# ── Validation ─────────────────────────────────────────────────


def test_validate_nan_rejected() -> None:
    data = struct.pack("<4f", 1.0, float("nan"), 3.0, 4.0)
    assert validate_tensor(data, "float32") is False


def test_validate_inf_rejected() -> None:
    data = struct.pack("<4f", 1.0, float("inf"), 3.0, 4.0)
    assert validate_tensor(data, "float32") is False


def test_validate_negative_inf_rejected() -> None:
    data = struct.pack("<1f", float("-inf"))
    assert validate_tensor(data, "float32") is False


def test_validate_clean_tensor_accepted() -> None:
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    assert validate_tensor(data, "float32") is True


def test_validate_fp16_clean() -> None:
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    fp16 = convert_dtype(data, "float32", "float16")
    assert validate_tensor(fp16, "float16") is True


def test_is_finite_alias() -> None:
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    assert is_finite(data, "float32") is True


# ── L2 norm validation ─────────────────────────────────────────


def test_l2_norm_within_bounds() -> None:
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    # sqrt(1+4+9+16) = sqrt(30) ≈ 5.48
    assert validate_l2_norm(data, "float32", max_norm=10.0, min_norm=0.1) is True


def test_l2_norm_too_large() -> None:
    data = struct.pack("<4f", 100.0, 200.0, 300.0, 400.0)
    assert validate_l2_norm(data, "float32", max_norm=10.0) is False


def test_l2_norm_too_small() -> None:
    data = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
    assert validate_l2_norm(data, "float32", min_norm=0.001) is False


# ── Int dtypes ─────────────────────────────────────────────────


def test_convert_int32_basic() -> None:
    data = struct.pack("<4i", 1, -2, 3, -4)
    result = convert_dtype(data, "int32", "int64")
    assert len(result) == 32  # 4 × 8 bytes


def test_convert_int32_to_fp32() -> None:
    data = struct.pack("<3i", 1, 2, 3)
    result = convert_dtype(data, "int32", "float32")
    values = _unpack_float32(result)
    assert values == [1.0, 2.0, 3.0]
