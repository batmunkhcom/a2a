"""Tests for protocol message helpers."""

from a2a.protocol.messages import make_metadata


def test_protobuf_metadata_serialization() -> None:
    meta = make_metadata(
        source_model="llama-8b",
        target_model="mistral-7b",
        semantic_label="error_context",
        tensor_dtype="float16",
        tensor_shape=[1, 4096],
        session_id="sess-abc123",
    )

    assert meta.source_model == "llama-8b"
    assert meta.target_model == "mistral-7b"
    assert meta.semantic_label == "error_context"
    assert meta.tensor_dtype == "float16"
    assert list(meta.tensor_shape) == [1, 4096]
    assert meta.session_id == "sess-abc123"
    assert meta.confidence == 1.0
