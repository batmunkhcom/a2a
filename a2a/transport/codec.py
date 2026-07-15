"""A2A tensor codec — binary serialization for tensor data.

Wire format:
  4 bytes  magic       0xA2A0
  2 bytes  version     1
  2 bytes  msg_type    0x01=UNICAST, etc.
  4 bytes  payload_len  total payload bytes (excluding header)

  Payload (FlatBuffers-compatible layout):
    4 bytes  dtype_len   length of dtype string
    N bytes  dtype_str   "float16", "float32", "bfloat16" (UTF-8)
    4 bytes  ndim        number of dimensions
    N*4      shape       one uint32 per dimension
    4 bytes  data_len    raw tensor bytes length
    N bytes  data        raw tensor data (row-major, C-order)
"""

from __future__ import annotations

import struct
import zlib
from enum import IntEnum
from typing import NamedTuple

# ── Constants ───────────────────────────────────────────────────

MAGIC = 0xA2A0
VERSION = 1
HEADER_SIZE = 12  # magic(4) + version(2) + msg_type(2) + payload_len(4)


# ── Message types ──────────────────────────────────────────────

class MessageType(IntEnum):
    UNICAST = 0x01
    MULTICAST = 0x02
    DISCOVER = 0x03
    CAPABILITY = 0x04
    PROJECT_REQ = 0x05
    PROJECT_RESP = 0x06
    PROJECT_AUTO = 0x07
    PROJECT_READY = 0x08
    KEEPALIVE = 0x09
    ACK = 0x0A
    ERROR = 0x0B
    BACKPRESSURE = 0x0C


# ── Supported dtypes ───────────────────────────────────────────

SUPPORTED_DTYPES = frozenset({"float16", "float32", "bfloat16", "int32", "int64"})

# Map dtype string to struct format character and byte size
_DTYPE_MAP: dict[str, tuple[str, int]] = {
    "float16": ("e", 2),  # IEEE 754 half-precision
    "float32": ("f", 4),
    "bfloat16": ("e", 2),
    "int32": ("i", 4),
    "int64": ("q", 8),
}


class TensorData(NamedTuple):
    """Decoded tensor data ready for injection into models."""

    values: bytes  # raw tensor bytes
    shape: tuple[int, ...]
    dtype: str


# ── Public API ─────────────────────────────────────────────────

def encode_tensor(
    data: bytes,
    shape: tuple[int, ...],
    dtype: str = "float32",
    msg_type: int = MessageType.UNICAST,
) -> bytes:
    """Encode raw tensor bytes into a wire-format binary message.

    Args:
        data: Raw tensor bytes (row-major, C-order).
        shape: Tensor dimensions, e.g. (1, 4096).
        dtype: Data type string (float16, float32, bfloat16).
        msg_type: A2A message type from MessageType enum.

    Returns:
        Complete wire-format bytes ready for gRPC transport.

    Raises:
        ValueError: If dtype unsupported, shape empty, or data length mismatch.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported: {sorted(SUPPORTED_DTYPES)}"
        )

    if not shape:
        raise ValueError("Shape must not be empty")

    ndim = len(shape)
    expected_len = _compute_byte_count(shape, dtype)
    if len(data) != expected_len:
        raise ValueError(
            f"Data length {len(data)} does not match shape {shape} * "
            f"dtype {dtype} = {expected_len} bytes"
        )

    # Build payload
    payload_parts: list[bytes] = []

    # dtype string
    dtype_encoded = dtype.encode("utf-8")
    payload_parts.append(struct.pack("<I", len(dtype_encoded)))
    payload_parts.append(dtype_encoded)

    # ndim + shape
    payload_parts.append(struct.pack("<I", ndim))
    for dim in shape:
        payload_parts.append(struct.pack("<I", dim))

    # raw data
    payload_parts.append(struct.pack("<I", len(data)))
    payload_parts.append(data)

    payload = b"".join(payload_parts)

    # Build header
    header = struct.pack(
        "<IHHI", MAGIC, VERSION, msg_type, len(payload)
    )

    # Checksum (CRC32 of header + payload)
    checksum = struct.pack("<I", zlib.crc32(header + payload) & 0xFFFFFFFF)

    return header + payload + checksum


def decode_tensor(frame: bytes) -> TensorData:
    """Decode a wire-format frame back into tensor data.

    Args:
        frame: Complete wire-format frame (header + payload + checksum).

    Returns:
        TensorData named tuple with values, shape, and dtype.

    Raises:
        ValueError: If magic mismatch, version unsupported, or corruption detected.
    """
    if len(frame) < HEADER_SIZE + 4 + 4:  # header + min payload + checksum
        raise ValueError(f"Frame too short: {len(frame)} bytes (min {HEADER_SIZE + 4 + 4})")

    # Parse header
    magic, version, msg_type, payload_len = struct.unpack_from(
        "<IHHI", frame, 0
    )

    if magic != MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:04X} (expected 0x{MAGIC:04X})")

    if version != VERSION:
        raise ValueError(f"Unsupported version: {version} (expected {VERSION})")

    # Verify checksum
    expected_checksum = struct.unpack("<I", frame[-4:])[0]
    computed = zlib.crc32(frame[:-4]) & 0xFFFFFFFF
    if computed != expected_checksum:
        raise ValueError(
            f"Checksum mismatch: computed 0x{computed:08X}, "
            f"expected 0x{expected_checksum:08X}"
        )

    # Parse payload
    payload = frame[HEADER_SIZE : HEADER_SIZE + payload_len]
    offset = 0

    # dtype string
    dtype_len = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    dtype = payload[offset : offset + dtype_len].decode("utf-8")
    offset += dtype_len

    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype in payload: {dtype}")

    # ndim + shape
    ndim = struct.unpack_from("<I", payload, offset)[0]
    offset += 4

    shape: list[int] = []
    for _ in range(ndim):
        dim = struct.unpack_from("<I", payload, offset)[0]
        shape.append(dim)
        offset += 4

    if not shape:
        raise ValueError("Decoded shape is empty")

    # raw data
    data_len = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    values = payload[offset : offset + data_len]

    expected_len = _compute_byte_count(tuple(shape), dtype)
    if data_len != expected_len:
        raise ValueError(
            f"Payload data length {data_len} does not match "
            f"shape {tuple(shape)} * dtype {dtype} = {expected_len}"
        )

    return TensorData(values=values, shape=tuple(shape), dtype=dtype)


def validate_tensor_values(data: bytes, dtype: str) -> bool:
    """Validate tensor bytes for NaN/Inf.

    Returns True if all values are finite (no NaN, no Inf).
    Returns False if any NaN or Inf detected.
    """
    fmt, itemsize = _DTYPE_MAP[dtype]
    count = len(data) // itemsize

    for i in range(count):
        chunk = data[i * itemsize : (i + 1) * itemsize]
        value = struct.unpack(f"<{fmt}", chunk)[0]

        import math

        if math.isnan(value) or math.isinf(value):
            return False

    return True


# ── Helpers ────────────────────────────────────────────────────

def _compute_byte_count(shape: tuple[int, ...], dtype: str) -> int:
    """Calculate expected byte count for given shape and dtype."""
    _, itemsize = _DTYPE_MAP[dtype]
    count = 1
    for dim in shape:
        count *= dim
    return count * itemsize
