"""Protocol message helper functions and re-exports.

Convenience module for constructing A2A protocol messages.
"""

from __future__ import annotations

from a2a.protocol.generated.a2a_pb2 import (
    AgentInfo,
    BackpressureSignal,
    BackpressureStatus,
    CapabilityInfo,
    DiscoverRequest,
    DiscoverResponse,
    HealthRequest,
    HealthResponse,
    ProjectionRequest,
    ProjectionResponse,
    TensorMetadata,
    TensorRequest,
    TensorResponse,
)

__all__ = [
    "TensorMetadata",
    "TensorRequest",
    "TensorResponse",
    "DiscoverRequest",
    "DiscoverResponse",
    "HealthRequest",
    "HealthResponse",
    "ProjectionRequest",
    "ProjectionResponse",
    "AgentInfo",
    "CapabilityInfo",
    "BackpressureSignal",
    "BackpressureStatus",
    "make_metadata",
]


def make_metadata(
    source_model: str = "",
    target_model: str = "",
    semantic_label: str = "",
    tensor_dtype: str = "float32",
    tensor_shape: list[int] | None = None,
    session_id: str = "",
) -> TensorMetadata:
    """Create a TensorMetadata message with sensible defaults."""
    from datetime import datetime, timezone

    return TensorMetadata(
        source_model=source_model,
        target_model=target_model,
        semantic_label=semantic_label,
        tensor_dtype=tensor_dtype,
        tensor_shape=tensor_shape or [],
        session_id=session_id,
        timestamp=int(datetime.now(timezone.utc).timestamp() * 1_000_000),
        confidence=1.0,
        source_layer=4294967295,  # max uint32 = -1 (last layer)
        target_layer=0,
    )
