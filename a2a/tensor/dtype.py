"""Tensor dtype conversion and value validation utilities.

Provides:
- Safe dtype conversion between FP32, FP16, BF16 without PyTorch.
- Tensor value validation (NaN/Inf/L2-norm checks) for sanity filtering.
"""

from __future__ import annotations

import math
import struct
from typing import Any

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── Constants ───────────────────────────────────────────────────

SUPPORTED_DTYPES = frozenset({"float16", "float32", "bfloat16", "int32", "int64"})

_DTYPE_ITEMSIZE: dict[str, int] = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "int32": 4,
    "int64": 8,
}


# ── Pure-Python dtype conversion ───────────────────────────────


def _float32_to_float16(value: float) -> int:
    """Convert a Python float to IEEE 754 half-precision bits (uint16).

    Handles subnormals, NaN, Inf.
    """
    # Pack as float32 first
    f32 = struct.pack(">f", value)
    f32_int = struct.unpack(">I", f32)[0]

    sign = (f32_int >> 16) & 0x8000
    exponent = (f32_int >> 23) & 0xFF
    mantissa = f32_int & 0x007FFFFF

    if exponent == 0xFF:  # NaN or Inf
        if mantissa != 0:
            return sign | 0x7E00  # NaN
        return sign | 0x7C00  # Inf

    if exponent == 0:  # Zero or subnormal
        return sign

    # Normalised number
    new_exp = exponent - 127 + 15
    if new_exp >= 0x1F:  # Overflow → Inf
        return sign | 0x7C00

    if new_exp <= 0:  # Underflow → zero or subnormal
        return sign

    return sign | (new_exp << 10) | ((mantissa >> 13) & 0x03FF)


def _float16_to_float32(h: int) -> float:
    """Convert IEEE 754 half-precision bits to Python float."""
    sign = (h >> 15) & 1
    exponent = (h >> 10) & 0x1F
    mantissa = h & 0x03FF

    if exponent == 0x1F:  # NaN or Inf
        if mantissa != 0:
            return float("nan")
        return float("inf") * (1 if sign == 0 else -1)

    if exponent == 0:
        if mantissa == 0:
            return 0.0 * (1 if sign == 0 else -1)
        exponent = -14  # subnormal
    else:
        mantissa |= 0x0400  # add implicit leading 1
        exponent -= 15

    value = mantissa * (2.0 ** (exponent - 10))
    return -value if sign else value


# ── Public API ─────────────────────────────────────────────────


def convert_dtype(data: bytes, src_dtype: str, tgt_dtype: str) -> bytes:
    """Convert raw tensor bytes between dtypes.

    Args:
        data: Raw tensor byte buffer.
        src_dtype: Source dtype string (float32, float16, bfloat16).
        tgt_dtype: Target dtype string.

    Returns:
        Converted byte buffer.

    Raises:
        ValueError: If source or target dtype is unsupported.
    """
    if src_dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported source dtype: {src_dtype}")
    if tgt_dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported target dtype: {tgt_dtype}")

    if src_dtype == tgt_dtype:
        return data

    src_size = _DTYPE_ITEMSIZE[src_dtype]
    tgt_size = _DTYPE_ITEMSIZE[tgt_dtype]

    num_elements = len(data) // src_size
    result = bytearray(num_elements * tgt_size)

    for i in range(num_elements):
        chunk = data[i * src_size : (i + 1) * src_size]

        # Step 1: decode source → Python float
        value = _decode_bytes(chunk, src_dtype)

        # Step 2: encode Python float → target bytes
        encoded = _encode_bytes(value, tgt_dtype)
        result[i * tgt_size : (i + 1) * tgt_size] = encoded

    return bytes(result)


def convert_dtype_torch(
    tensor: Any,
    target_dtype: str,
) -> Any:
    """Convert a PyTorch tensor to a target dtype string.

    Args:
        tensor: PyTorch tensor.
        target_dtype: "float16", "float32", "bfloat16", "int32", "int64"

    Returns:
        Converted PyTorch tensor.
    """
    if not _HAS_TORCH:
        raise ImportError("convert_dtype_torch requires PyTorch")

    torch_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    if target_dtype not in torch_dtypes:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")

    return tensor.to(torch_dtypes[target_dtype])


def validate_tensor(data: bytes, dtype: str) -> bool:
    """Check that tensor bytes contain no NaN or Inf values.

    Args:
        data: Raw tensor byte buffer.
        dtype: Data type string.

    Returns:
        True if all values are finite.
    """
    itemsize = _DTYPE_ITEMSIZE[dtype]
    num_elements = len(data) // itemsize

    for i in range(num_elements):
        chunk = data[i * itemsize : (i + 1) * itemsize]
        value = _decode_bytes(chunk, dtype)

        if math.isnan(value) or math.isinf(value):
            return False

    return True


def validate_tensor_torch(tensor: Any) -> bool:
    """Check that a PyTorch tensor contains no NaN or Inf values.

    Args:
        tensor: PyTorch tensor.

    Returns:
        True if all values are finite.
    """
    if not _HAS_TORCH:
        raise ImportError("validate_tensor_torch requires PyTorch")

    return bool(torch.isfinite(tensor).all().item())


def validate_l2_norm(
    data: bytes,
    dtype: str,
    max_norm: float = 1000.0,
    min_norm: float = 0.001,
) -> bool:
    """Check that a tensor's L2 norm is within acceptable bounds.

    Args:
        data: Raw tensor byte buffer.
        dtype: Data type string.
        max_norm: Maximum allowed L2 norm.
        min_norm: Minimum allowed L2 norm.

    Returns:
        True if L2 norm is within [min_norm, max_norm].
    """
    itemsize = _DTYPE_ITEMSIZE[dtype]
    num_elements = len(data) // itemsize

    sum_sq = 0.0
    for i in range(num_elements):
        chunk = data[i * itemsize : (i + 1) * itemsize]
        value = _decode_bytes(chunk, dtype)
        sum_sq += value * value

    norm = math.sqrt(sum_sq)
    return min_norm <= norm <= max_norm


# ── Internal helpers ───────────────────────────────────────────


def _decode_bytes(data: bytes, dtype: str) -> float:
    """Decode dtype-specific bytes to a Python float."""
    if dtype == "float32":
        (val,) = struct.unpack("<f", data)
        return val

    if dtype == "float16":
        h = struct.unpack("<H", data)[0]
        return _float16_to_float32(h)

    if dtype == "bfloat16":
        # bfloat16 = top 16 bits of float32
        b16 = struct.unpack("<H", data)[0]
        f32_bits = b16 << 16
        (val,) = struct.unpack("<f", struct.pack("<I", f32_bits))
        return val

    if dtype in ("int32", "int64"):
        val = int.from_bytes(data, "little", signed=True)
        return float(val)

    raise ValueError(f"Unknown dtype: {dtype}")


def _encode_bytes(value: float, dtype: str) -> bytes:
    """Encode a Python float to dtype-specific bytes."""
    if dtype == "float32":
        return struct.pack("<f", float(value))

    if dtype == "float16":
        h = _float32_to_float16(float(value))
        return struct.pack("<H", h)

    if dtype == "bfloat16":
        # bfloat16 = truncate float32 mantissa
        f32_bytes = struct.pack("<f", float(value))
        return f32_bytes[2:4]  # top 16 bits

    if dtype == "int32":
        return int(value).to_bytes(4, "little", signed=True)

    if dtype == "int64":
        return int(value).to_bytes(8, "little", signed=True)

    raise ValueError(f"Unknown dtype: {dtype}")


def is_finite(data: bytes, dtype: str) -> bool:
    """Alias for validate_tensor."""
    return validate_tensor(data, dtype)
