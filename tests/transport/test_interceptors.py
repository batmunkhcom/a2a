"""Tests for JWT auth, rate limit interceptors, and health HTTP endpoints."""

import time
from unittest.mock import MagicMock

import pytest

from a2a.monitoring.rate_limiter import RateLimiter
from a2a.security.auth import create_token
from a2a.transport.client import A2AClient
from a2a.transport.server import (
    A2AServer,
    AuthInterceptor,
    RateLimitInterceptor,
)


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def port() -> int:
    return _find_free_port()


@pytest.fixture
def address(port: int) -> str:
    return f"localhost:{port}"


@pytest.fixture
def secret() -> str:
    return "test-secret-42"


@pytest.fixture
def mesh_id() -> str:
    return "test-mesh"


class TestAuthInterceptorUnit:
    def test_exempt_healthcheck(self, secret: str, mesh_id: str) -> None:
        interceptor = AuthInterceptor(secret, mesh_id)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/HealthCheck"
        mock.invocation_metadata = {}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None

    def test_exempt_discover(self, secret: str, mesh_id: str) -> None:
        interceptor = AuthInterceptor(secret, mesh_id)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/Discover"
        mock.invocation_metadata = {}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None

    def test_missing_token_rejected(self, secret: str, mesh_id: str) -> None:
        interceptor = AuthInterceptor(secret, mesh_id)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/SendTensor"
        mock.invocation_metadata = {}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None

    def test_invalid_token_rejected(self, secret: str, mesh_id: str) -> None:
        interceptor = AuthInterceptor(secret, mesh_id)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/SendTensor"
        mock.invocation_metadata = {"authorization": "Bearer invalid.token.here"}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None

    def test_valid_token_accepted(self, secret: str, mesh_id: str) -> None:
        interceptor = AuthInterceptor(secret, mesh_id)
        token = create_token("agent-1", mesh_id, secret)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/SendTensor"
        mock.invocation_metadata = {"authorization": f"Bearer {token}"}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None

    def test_disabled_bypasses(self, secret: str, mesh_id: str) -> None:
        interceptor = AuthInterceptor(secret, mesh_id, enabled=False)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/SendTensor"
        mock.invocation_metadata = {}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None


class TestRateLimitInterceptorUnit:
    def test_allows_under_limit(self) -> None:
        limiter = RateLimiter(default_bucket_size=5, default_refill_rate=100.0)
        interceptor = RateLimitInterceptor(limiter)

        def cont(details):
            return ("handler", details)

        for _ in range(5):
            mock = MagicMock()
            mock.method = "/a2a.A2AService/SendTensor"
            mock.invocation_metadata = {"x-agent-id": "agent-1"}
            result = interceptor.intercept_service(cont, mock)
            assert result is not None

    def test_blocks_when_exhausted(self) -> None:
        limiter = RateLimiter(default_bucket_size=1, default_refill_rate=0.0)
        interceptor = RateLimitInterceptor(limiter)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/SendTensor"
        mock.invocation_metadata = {"x-agent-id": "agent-1"}

        assert interceptor.intercept_service(cont, mock) is not None
        # Second call blocks
        assert interceptor.intercept_service(cont, mock) is not None

    def test_disabled_bypasses(self) -> None:
        limiter = RateLimiter(default_bucket_size=0, default_refill_rate=0.0)
        interceptor = RateLimitInterceptor(limiter, enabled=False)

        def cont(details):
            return ("handler", details)

        mock = MagicMock()
        mock.method = "/a2a.A2AService/SendTensor"
        mock.invocation_metadata = {}
        result = interceptor.intercept_service(cont, mock)
        assert result is not None


class TestServerWithInterceptors:
    """Integration tests: start a real server with interceptors enabled."""

    def test_server_starts_with_auth_disabled(
        self, port: int, address: str
    ) -> None:
        server = A2AServer(host="localhost", port=port)
        server.start()
        client = A2AClient(address)
        assert client.health_check().status == "ok"
        client.close()
        server.stop()

    def test_server_starts_with_auth_enabled_no_config_works(
        self, port: int, address: str
    ) -> None:
        """Without a config, auth is off by default — server works fine."""
        server = A2AServer(host="localhost", port=port)
        server.start()
        client = A2AClient(address)
        assert client.health_check().status == "ok"
        client.close()
        server.stop()

    def test_server_health_endpoint_responds(self, port: int) -> None:
        import urllib.request

        server = A2AServer(host="localhost", port=port)
        server.start()
        time.sleep(0.1)

        health_port = server.health_port
        resp = urllib.request.urlopen(f"http://localhost:{health_port}/health")
        assert resp.status == 200

        resp = urllib.request.urlopen(f"http://localhost:{health_port}/health/live")
        assert resp.status == 200

        resp = urllib.request.urlopen(f"http://localhost:{health_port}/health/ready")
        assert resp.status == 200

        server.stop()
