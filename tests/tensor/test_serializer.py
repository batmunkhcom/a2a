"""Tests for Safetensors/fallback serialization — requires torch."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from a2a.tensor.serializer import (  # noqa: E402
    deserialize_tensor,
    deserialize_tensors,
    serialize_tensor,
    serialize_tensors,
)


def test_safetensors_save_load_roundtrip() -> None:
    original = torch.randn(1, 4096, dtype=torch.float32)
    data = serialize_tensor(original, name="hidden")
    restored = deserialize_tensor(data, name="hidden")

    assert torch.allclose(original, restored, atol=1e-6)
    assert original.shape == restored.shape
    assert original.dtype == restored.dtype


def test_safetensors_multidim_tensor() -> None:
    original = torch.randn(4, 32, 128, dtype=torch.float32)
    data = serialize_tensor(original, name="multi")
    restored = deserialize_tensor(data, name="multi")

    assert torch.allclose(original, restored, atol=1e-6)
    assert restored.shape == (4, 32, 128)


def test_safetensors_fp16() -> None:
    original = torch.randn(1, 2048, dtype=torch.float16)
    data = serialize_tensor(original, name="fp16")
    restored = deserialize_tensor(data, name="fp16")

    assert original.shape == restored.shape
    assert original.dtype == restored.dtype
    assert torch.allclose(original.float(), restored.float(), atol=1e-3)


def test_safetensors_multi_tensor() -> None:
    tensors = {
        "a": torch.randn(1, 64),
        "b": torch.randn(2, 32),
    }
    data = serialize_tensors(tensors)
    restored = deserialize_tensors(data)

    assert set(restored.keys()) == {"a", "b"}
    assert torch.allclose(tensors["a"], restored["a"])
    assert torch.allclose(tensors["b"], restored["b"])


def test_safetensors_file_roundtrip(tmp_path: object) -> None:
    from a2a.tensor.serializer import load_tensor_from_file, save_tensor_to_file

    original = torch.randn(1, 128)
    path = str(tmp_path / "test.safetensors")  # type: ignore[union-attr]

    save_tensor_to_file(original, path, name="t")
    restored = load_tensor_from_file(path, name="t")

    assert torch.allclose(original, restored)
