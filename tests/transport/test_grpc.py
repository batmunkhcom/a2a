"""Integration tests for gRPC transport (server + client in one process)."""

import struct

import pytest

from a2a.protocol.errors import A2AError
from a2a.protocol.messages import make_metadata
from a2a.transport.client import A2AClient
from a2a.transport.codec import encode_tensor
from a2a.transport.server import A2AServer


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def server_address() -> str:
    port = _find_free_port()
    return f"localhost:{port}"


@pytest.fixture
def grpc_server(server_address: str) -> A2AServer:
    host, port = server_address.split(":")
    server = A2AServer(host=host, port=int(port))
    server.start()
    yield server
    server.stop()


@pytest.fixture
def client(server_address: str) -> A2AClient:
    return A2AClient(server_address)


def test_grpc_health_check(grpc_server: object, client: A2AClient) -> None:
    response = client.health_check()
    assert response.status == "ok"
    assert response.version
    assert response.uptime_seconds >= 0


def test_grpc_send_tensor_small(grpc_server: object, client: A2AClient) -> None:
    data = b"\x00" * (4 * 16)  # 16 float32 values
    encoded = encode_tensor(data, (1, 16), "float32")
    meta = make_metadata(semantic_label="test_label", tensor_dtype="float32")

    response = client.send_tensor(encoded, meta)
    assert response.accepted is True


def test_grpc_send_tensor_large(grpc_server: object, client: A2AClient) -> None:
    shape = (1, 4096)
    dtype = "float32"
    data = struct.pack(f"<{4096}f", *([0.5] * 4096))
    encoded = encode_tensor(data, shape, dtype)
    meta = make_metadata(
        source_model="llama-8b",
        target_model="llama-8b",
        semantic_label="error_context",
        tensor_dtype=dtype,
    )

    response = client.send_tensor(encoded, meta)
    assert response.accepted is True


def test_grpc_send_tensor_error_on_empty(grpc_server: object, client: A2AClient) -> None:
    with pytest.raises(A2AError):
        client.send_tensor(b"", make_metadata())


def test_grpc_send_tensor_error_on_nan(grpc_server: object, client: A2AClient) -> None:
    data = struct.pack("<4f", 1.0, float("nan"), 3.0, 4.0)
    encoded = encode_tensor(data, (4,), "float32")
    meta = make_metadata(tensor_dtype="float32")

    with pytest.raises(A2AError) as exc_info:
        client.send_tensor(encoded, meta)

    assert exc_info.value.code == 102


def test_grpc_client_reconnect(grpc_server: object, server_address: str) -> None:
    c1 = A2AClient(server_address)
    h1 = c1.health_check()
    assert h1.status == "ok"
    c1.close()

    c2 = A2AClient(server_address)
    h2 = c2.health_check()
    assert h2.status == "ok"
    c2.close()


def test_grpc_discover(grpc_server: object, client: A2AClient) -> None:
    response = client.discover(agent_id="test-agent")
    assert response is not None
    assert isinstance(response.agents, list) or hasattr(response, "agents")


def test_grpc_stream_tensors(grpc_server: object, server_address: str) -> None:
    client = A2AClient(server_address)
    for _ in range(5):
        data = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        encoded = encode_tensor(data, (8,), "float32")
        meta = make_metadata(tensor_dtype="float32")
        response = client.send_tensor(encoded, meta)
        assert response.accepted is True
    client.close()
