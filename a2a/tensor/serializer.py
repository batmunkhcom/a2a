"""Safetensors-based tensor serialization for A2A protocol.

Serializes/deserializes tensors to/from bytes using the safetensors
format. Falls back to a simple binary format if safetensors is
unavailable.
"""

from __future__ import annotations

import struct
from io import BytesIO
from typing import Any

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from safetensors.torch import load as _safetensors_load  # noqa: F401
    from safetensors.torch import save as _safetensors_save  # noqa: F401

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


# ── Public API ─────────────────────────────────────────────────


def serialize_tensor(tensor: Any, *, name: str = "tensor") -> bytes:
    """Serialize a PyTorch tensor to bytes (safetensors or fallback).

    Args:
        tensor: PyTorch tensor.
        name: Key name for the tensor in the safetensors dict.

    Returns:
        Serialized bytes.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not _HAS_TORCH:
        raise ImportError("serialize_tensor requires PyTorch")

    if _HAS_SAFETENSORS:
        buf = BytesIO()
        _safetensors_save({name: tensor}, buf)  # type: ignore[call-arg]
        return buf.getvalue()

    # Fallback: simple binary format
    return _serialize_fallback(tensor, name)


def deserialize_tensor(data: bytes, *, name: str = "tensor") -> Any:
    """Deserialize bytes back to a PyTorch tensor.

    Args:
        data: Serialized bytes (safetensors or fallback format).
        name: Key name to extract from the dict.

    Returns:
        PyTorch tensor.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If data is corrupt or format unknown.
    """
    if not _HAS_TORCH:
        raise ImportError("deserialize_tensor requires PyTorch")

    if _HAS_SAFETENSORS:
        try:
            with BytesIO(data) as buf:
                tensors = _safetensors_load(buf)  # type: ignore[call-arg]
                if name in tensors:
                    return tensors[name]
                # Try first available key
                first_key = next(iter(tensors))
                return tensors[first_key]
        except (ValueError, RuntimeError, KeyError):
            pass  # Fall through to fallback

    return _deserialize_fallback(data)


def save_tensor_to_file(tensor: Any, path: str, *, name: str = "tensor") -> None:
    """Serialize tensor and write to file."""
    data = serialize_tensor(tensor, name=name)
    with open(path, "wb") as f:
        f.write(data)


def load_tensor_from_file(path: str, *, name: str = "tensor") -> Any:
    """Read serialized tensor from file."""
    with open(path, "rb") as f:
        data = f.read()
    return deserialize_tensor(data, name=name)


def serialize_tensors(tensors: dict[str, Any]) -> bytes:
    """Serialize multiple named tensors at once."""
    if not _HAS_TORCH:
        raise ImportError("serialize_tensors requires PyTorch")

    if _HAS_SAFETENSORS:
        buf = BytesIO()
        _safetensors_save(tensors, buf)  # type: ignore[call-arg]
        return buf.getvalue()

    # Fallback: concatenated binary
    parts: list[bytes] = []
    # Header: number of tensors (uint32)
    parts.append(struct.pack("<I", len(tensors)))
    for name, tensor in tensors.items():
        tensor_bytes = _serialize_fallback(tensor, name)
        parts.append(struct.pack("<I", len(tensor_bytes)))
        parts.append(tensor_bytes)
    return b"".join(parts)


def deserialize_tensors(data: bytes) -> dict[str, Any]:
    """Deserialize multiple named tensors."""
    if not _HAS_TORCH:
        raise ImportError("deserialize_tensors requires PyTorch")

    if _HAS_SAFETENSORS:
        try:
            with BytesIO(data) as buf:
                return dict(_safetensors_load(buf))  # type: ignore[call-arg]
        except (ValueError, RuntimeError):
            pass

    return _deserialize_fallback_multi(data)


# ── Fallback binary format ─────────────────────────────────────

# Simple header: magic(4) + ndim(4) + [dim0(4) dim1(4) ...] + raw_data
_FALLBACK_MAGIC = b"\xA2\xA1\xF8\x02"


def _serialize_fallback(tensor: Any, name: str = "") -> bytes:
    """Serialize a tensor to simple binary format."""
    shape = tuple(tensor.shape)
    ndim = len(shape)

    # Determine dtype byte
    dtype_map = {
        torch.float16: b"\x01",
        torch.float32: b"\x02",
        torch.bfloat16: b"\x03",
        torch.int32: b"\x04",
        torch.int64: b"\x05",
    }
    dtype_byte = dtype_map.get(tensor.dtype, b"\x02")  # type: ignore[arg-type]

    header = _FALLBACK_MAGIC + dtype_byte + struct.pack("<I", ndim)
    for dim in shape:
        header += struct.pack("<I", dim)

    # Add name if present
    name_bytes = name.encode("utf-8")
    header += struct.pack("<I", len(name_bytes)) + name_bytes

    # Raw tensor data (contiguous, row-major)
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()

    return header + raw


def _deserialize_fallback(data: bytes) -> Any:
    """Deserialize fallback format to a tensor."""
    if data[:4] != _FALLBACK_MAGIC:
        raise ValueError("Not a valid fallback tensor format")

    offset = 4
    dtype_byte = data[offset : offset + 1]
    offset += 1

    ndim = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    shape: list[int] = []
    for _ in range(ndim):
        dim = struct.unpack_from("<I", data, offset)[0]
        shape.append(dim)
        offset += 4

    # Skip name
    name_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + name_len

    # Map dtype byte back
    dtype_map_rev: dict[bytes, Any] = {
        b"\x01": torch.float16,
        b"\x02": torch.float32,
        b"\x03": torch.bfloat16,
        b"\x04": torch.int32,
        b"\x05": torch.int64,
    }
    torch_dtype = dtype_map_rev.get(dtype_byte, torch.float32)

    # Calculate expected size
    itemsize = {
        torch.float16: 2,
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }[torch_dtype]

    total = 1
    for dim in shape:
        total *= dim
    expected_bytes = total * itemsize

    raw = data[offset : offset + expected_bytes]
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Data length mismatch: expected {expected_bytes}, got {len(raw)}"
        )

    tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype)  # type: ignore[arg-type]
    return tensor.reshape(shape)


def _deserialize_fallback_multi(data: bytes) -> dict[str, Any]:
    """Deserialize multi-tensor fallback format."""
    offset = 0
    num_tensors = struct.unpack("<I", data[offset : offset + 4])[0]
    offset += 4

    result: dict[str, Any] = {}
    for i in range(num_tensors):
        chunk_len = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        result[f"tensor_{i}"] = _deserialize_fallback(chunk)

    return result
