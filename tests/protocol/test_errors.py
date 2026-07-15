"""Tests for A2A error codes and error handling."""

from a2a.protocol.errors import (
    AUTH_FAILED,
    INTERNAL_ERROR,
    MESH_MISMATCH,
    MODEL_OVERLOADED,
    PERMISSION_DENIED,
    TENSOR_SHAPE_MISMATCH,
    TENSOR_VALUE_INVALID,
    A2AError,
    error_to_response,
    is_non_retryable,
    is_retryable,
)


def test_error_creation() -> None:
    err = A2AError(code=TENSOR_SHAPE_MISMATCH, message="Shape mismatch")
    assert err.code == 100
    assert "Shape mismatch" in err.message


def test_error_to_response() -> None:
    err = A2AError(code=TENSOR_VALUE_INVALID, message="NaN detected")
    resp = error_to_response(err)
    assert resp.accepted is False
    assert resp.error_code == 102
    assert "NaN" in resp.error_message


def test_retryable_codes() -> None:
    assert is_retryable(INTERNAL_ERROR) is True
    assert is_retryable(MODEL_OVERLOADED) is True
    assert is_retryable(500) is True
    assert is_retryable(TENSOR_SHAPE_MISMATCH) is False
    assert is_retryable(AUTH_FAILED) is False


def test_non_retryable_codes() -> None:
    assert is_non_retryable(AUTH_FAILED) is True
    assert is_non_retryable(PERMISSION_DENIED) is True
    assert is_non_retryable(MESH_MISMATCH) is True
    assert is_non_retryable(TENSOR_VALUE_INVALID) is False
    assert is_non_retryable(INTERNAL_ERROR) is False
