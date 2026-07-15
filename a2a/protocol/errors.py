"""A2A Protocol error codes and error-to-gRPC-response conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

from a2a.protocol.generated import a2a_pb2  # noqa: I001

# ── Error Codes ────────────────────────────────────────────────

TENSOR_SHAPE_MISMATCH = 100
TENSOR_DTYPE_MISMATCH = 101
TENSOR_VALUE_INVALID = 102

PROJECTION_NOT_FOUND = 200
PROJECTION_TRAINING_FAILED = 201

MODEL_NOT_FOUND = 300
MODEL_OVERLOADED = 301

AUTH_FAILED = 400
PERMISSION_DENIED = 401
MESH_MISMATCH = 402

INTERNAL_ERROR = 500
PLUGIN_CRASHED = 501
GENERATION_FAILED = 502

VERSION_MISMATCH = 600


# ── Error dataclass ────────────────────────────────────────────


@dataclass
class A2AError(Exception):
    """A2A protocol error with code and optional details."""

    code: int
    message: str
    source_agent: str = ""
    details: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"[{self.code}] {self.message}")


# ── gRPC response builders ─────────────────────────────────────


def error_to_response(error: A2AError) -> a2a_pb2.TensorResponse:  # type: ignore[name-defined]
    """Convert an A2AError to a gRPC TensorResponse with error fields."""
    return a2a_pb2.TensorResponse(  # type: ignore[attr-defined]
        accepted=False,
        error_message=error.message,
        error_code=error.code,
    )


def is_retryable(code: int) -> bool:
    """Check whether an error code indicates a retryable condition."""
    return code in (INTERNAL_ERROR, MODEL_OVERLOADED, 500, 301)


def is_non_retryable(code: int) -> bool:
    """Check whether an error code is definitively non-retryable."""
    return code in (AUTH_FAILED, PERMISSION_DENIED, MESH_MISMATCH, 400, 401, 402)
