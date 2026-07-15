"""Unix domain socket bridge server for A2A Python ML Core.

Listens on a Unix socket for commands from the Go transport layer.
Protocol: [4B cmd_len][JSON command][4B data_len][raw data bytes]
Response: [4B resp_len][JSON response][4B data_len][raw data bytes]

Commands: PROCESS, EXTRACT, INJECT, PROJECT, HEALTH
"""

import json
import os
import socket
import struct
import time
from collections.abc import Callable
from contextlib import suppress
from threading import Thread

import numpy as np


class BridgeServer:
    """Unix socket server bridging Go transport to Python ML Core."""

    def __init__(self, socket_path: str = "/tmp/a2a-ml.sock") -> None:
        self._socket_path = socket_path
        self._sock: socket.socket | None = None
        self._running = False
        self._handlers: dict[str, Callable] = {}
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        self._handlers["PROCESS"] = self._handle_process
        self._handlers["EXTRACT"] = self._handle_extract
        self._handlers["INJECT"] = self._handle_inject
        self._handlers["PROJECT"] = self._handle_project
        self._handlers["HEALTH"] = self._handle_health

    def register_handler(self, op: str, handler: Callable) -> None:
        self._handlers[op] = handler

    def start(self) -> None:
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self._socket_path)
        self._sock.listen(5)
        self._running = True

        Thread(target=self._accept_loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self._sock:
            self._sock.close()
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                conn, _ = self._sock.accept()
                Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            while True:
                # Read command length
                raw_len = self._recv_exact(conn, 4)
                if not raw_len:
                    break
                cmd_len = struct.unpack(">I", raw_len)[0]

                # Read command JSON
                cmd_json = self._recv_exact(conn, cmd_len)
                if not cmd_json:
                    break
                cmd = json.loads(cmd_json.decode())

                # Read data length
                raw_data_len = self._recv_exact(conn, 4)
                if not raw_data_len:
                    break
                data_len = struct.unpack(">I", raw_data_len)[0]

                # Read data
                data = self._recv_exact(conn, data_len) if data_len > 0 else b""

                # Handle command
                start = time.perf_counter()
                resp, resp_data = self._dispatch(cmd, data)
                elapsed_us = int((time.perf_counter() - start) * 1_000_000)

                resp["latency_us"] = elapsed_us

                # Send response: [4B resp_len][resp JSON][4B data_len][resp data]
                resp_bytes = json.dumps(resp).encode()
                conn.sendall(struct.pack(">I", len(resp_bytes)))
                conn.sendall(resp_bytes)
                conn.sendall(struct.pack(">I", len(resp_data)))
                conn.sendall(resp_data)
        except Exception as exc:
            error_resp = json.dumps({"status": "error", "error": str(exc)}).encode()
            with suppress(OSError):
                conn.sendall(struct.pack(">I", len(error_resp)))
                conn.sendall(error_resp)
                conn.sendall(struct.pack(">I", 0))
        finally:
            conn.close()

    def _dispatch(self, cmd: dict, data: bytes) -> tuple[dict, bytes]:
        op = cmd.get("op", "PROCESS")
        handler = self._handlers.get(op)
        if handler is None:
            return {"status": "error", "error": f"unknown op: {op}"}, b""
        return handler(cmd, data)

    def _handle_process(self, cmd: dict, data: bytes) -> tuple[dict, bytes]:
        """PROCESS: Default handler — echo data back with same shape."""
        dim = cmd.get("dim", 0)
        if data:
            tensor = np.frombuffer(data, dtype=np.float32)
            return {
                "status": "ok",
                "dim": dim,
                "dtype": cmd.get("dtype", "fp32"),
                "label": cmd.get("label", "hidden_state"),
            }, tensor.tobytes()
        return {
            "status": "ok",
            "dim": dim,
            "dtype": cmd.get("dtype", "fp32"),
            "label": cmd.get("label", "hidden_state"),
        }, b""

    def _handle_extract(self, cmd: dict, data: bytes) -> tuple[dict, bytes]:
        """EXTRACT: Extract hidden states from model."""
        # Placeholder — requires torch model
        return {"status": "ok", "dim": cmd.get("dim", 768), "dtype": "fp32"}, data

    def _handle_inject(self, cmd: dict, data: bytes) -> tuple[dict, bytes]:
        """INJECT: Inject tensor into model."""
        return {"status": "ok", "dim": cmd.get("dim", 768), "dtype": "fp32"}, data

    def _handle_project(self, cmd: dict, data: bytes) -> tuple[dict, bytes]:
        """PROJECT: Cross-model projection."""
        return {"status": "ok", "dim": cmd.get("dim", 768), "dtype": "fp32"}, data

    def _handle_health(self, cmd: dict, data: bytes) -> tuple[dict, bytes]:
        """HEALTH: Health check."""
        return {"status": "ok", "version": "0.1.0"}, b""

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    @property
    def socket_path(self) -> str:
        return self._socket_path

    @property
    def running(self) -> bool:
        return self._running
