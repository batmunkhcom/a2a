"""Integration tests for Go-Python Unix socket bridge protocol."""

import json
import os
import socket
import struct
import tempfile
import time
from collections.abc import Generator

import numpy as np
import pytest

from a2a.transport.bridge_server import BridgeServer


@pytest.fixture
def bridge_server() -> Generator[BridgeServer, None, None]:
    """Start a bridge server on a temp socket."""
    sock_path = os.path.join(tempfile.gettempdir(), f"a2a-test-bridge-{os.getpid()}.sock")
    server = BridgeServer(sock_path)
    server.start()
    time.sleep(0.05)
    yield server
    server.stop()


def _send_cmd(sock_path: str, cmd: dict, data: bytes = b"") -> tuple[dict, bytes]:
    """Send a command to the bridge server and return the response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(sock_path)
        cmd_bytes = json.dumps(cmd).encode()

        # Wire format: [4B cmd_len][cmd JSON][4B data_len][data bytes]
        sock.sendall(struct.pack(">I", len(cmd_bytes)))
        sock.sendall(cmd_bytes)
        sock.sendall(struct.pack(">I", len(data)))
        if data:
            sock.sendall(data)

        # Read response length
        raw_len = _recv_exact(sock, 4)
        resp_len = struct.unpack(">I", raw_len)[0]
        resp_bytes = _recv_exact(sock, resp_len)
        resp = json.loads(resp_bytes.decode())

        # Read response data length
        raw_data_len = _recv_exact(sock, 4)
        data_len = struct.unpack(">I", raw_data_len)[0]
        resp_data = _recv_exact(sock, data_len) if data_len > 0 else b""

        return resp, resp_data
    finally:
        sock.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


class TestBridgeHealth:
    def test_health_check(self, bridge_server: BridgeServer) -> None:
        resp, data = _send_cmd(bridge_server.socket_path, {"op": "HEALTH"})
        assert resp["status"] == "ok"
        assert resp["version"] == "0.1.0"

    def test_health_has_latency(self, bridge_server: BridgeServer) -> None:
        resp, _ = _send_cmd(bridge_server.socket_path, {"op": "HEALTH"})
        assert "latency_us" in resp
        assert resp["latency_us"] >= 0


class TestBridgeProcess:
    def test_process_echo(self, bridge_server: BridgeServer) -> None:
        tensor = np.random.randn(768).astype(np.float32)
        cmd = {
            "op": "PROCESS",
            "agent_id": "test",
            "label": "hidden_state",
            "dim": 768,
            "dtype": "fp32",
        }
        resp, data = _send_cmd(bridge_server.socket_path, cmd, tensor.tobytes())

        assert resp["status"] == "ok"
        assert resp["dim"] == 768
        assert len(data) == len(tensor.tobytes())

        result = np.frombuffer(data, dtype=np.float32)
        assert np.array_equal(result, tensor)

    def test_process_returns_correct_shape(self, bridge_server: BridgeServer) -> None:
        tensor = np.random.randn(128).astype(np.float32)
        cmd = {"op": "PROCESS", "dim": 128, "dtype": "fp32"}
        resp, data = _send_cmd(bridge_server.socket_path, cmd, tensor.tobytes())

        assert resp["status"] == "ok"
        result = np.frombuffer(data, dtype=np.float32)
        assert result.shape == (128,)

    def test_process_empty_data(self, bridge_server: BridgeServer) -> None:
        cmd = {"op": "PROCESS", "dim": 0}
        resp, data = _send_cmd(bridge_server.socket_path, cmd, b"")

        assert resp["status"] == "ok"
        assert data == b""


class TestBridgeErrors:
    def test_unknown_op(self, bridge_server: BridgeServer) -> None:
        resp, _ = _send_cmd(bridge_server.socket_path, {"op": "UNKNOWN_OP"})
        assert resp["status"] == "error"

    def test_extract_placeholder(self, bridge_server: BridgeServer) -> None:
        resp, data = _send_cmd(bridge_server.socket_path, {"op": "EXTRACT", "dim": 768}, b"data")
        assert resp["status"] == "ok"


class TestBridgeConcurrency:
    def test_parallel_requests(self, bridge_server: BridgeServer) -> None:
        """Ensure multiple parallel requests work correctly."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def send_one(i: int) -> bool:
            tensor = np.array([float(i)], dtype=np.float32)
            resp, data = _send_cmd(
                bridge_server.socket_path,
                {"op": "PROCESS", "dim": 1},
                tensor.tobytes(),
            )
            result = np.frombuffer(data, dtype=np.float32)
            return resp["status"] == "ok" and result[0] == float(i)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(send_one, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert all(results)


class TestBridgeWireFormat:
    def test_large_tensor(self, bridge_server: BridgeServer) -> None:
        """Test with a large tensor (16KB = 4096 float32)."""
        tensor = np.random.randn(4096).astype(np.float32)
        cmd = {"op": "PROCESS", "dim": 4096}
        resp, data = _send_cmd(bridge_server.socket_path, cmd, tensor.tobytes())

        assert resp["status"] == "ok"
        result = np.frombuffer(data, dtype=np.float32)
        assert result.shape == (4096,)
        assert np.array_equal(result, tensor)
