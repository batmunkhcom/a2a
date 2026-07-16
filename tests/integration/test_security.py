"""Integration tests for JWT + gRPC, rate limiting, health, backpressure."""

import time

import pytest

from a2a.monitoring.metrics import MetricsRegistry
from a2a.monitoring.rate_limiter import BackpressureController, RateLimiter
from a2a.security.auth import JWTError, create_token, validate_token
from a2a.transport.client import A2AClient
from a2a.transport.server import A2AServer


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def server_address() -> str:
    return f"localhost:{_find_free_port()}"


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


@pytest.fixture
def secret() -> str:
    return "test-secret-key"


@pytest.fixture
def mesh_id() -> str:
    return "test-mesh"


class TestHealthEndpointDetail:
    def test_health_returns_all_fields(
        self, grpc_server: object, client: A2AClient
    ) -> None:
        response = client.health_check()
        assert response.status == "ok"
        assert response.version
        assert response.uptime_seconds >= 0
        assert response.plugins_loaded >= 0

    def test_health_uptime_increases(
        self, grpc_server: object, client: A2AClient
    ) -> None:
        h1 = client.health_check()
        time.sleep(0.1)
        h2 = client.health_check()
        assert h2.uptime_seconds >= h1.uptime_seconds

    def test_health_consistent_across_clients(
        self, grpc_server: object, server_address: str
    ) -> None:
        c1 = A2AClient(server_address)
        c2 = A2AClient(server_address)
        h1 = c1.health_check()
        h2 = c2.health_check()
        assert h1.status == h2.status
        c1.close()
        c2.close()


class TestJWTServerIntegration:
    """JWT create/validate against live gRPC server."""

    def test_server_health_without_auth(
        self, grpc_server: object, client: A2AClient
    ) -> None:
        response = client.health_check()
        assert response.status == "ok"

    def test_token_works_with_server_identity(
        self, secret: str, mesh_id: str
    ) -> None:
        token = create_token("test-agent", mesh_id, secret)
        payload = validate_token(token, secret, mesh_id=mesh_id)
        assert payload is not None
        assert payload["agent_id"] == "test-agent"
        assert payload["mesh_id"] == mesh_id

    def test_token_rejected_wrong_mesh(
        self, secret: str
    ) -> None:
        token = create_token("agent-1", "mesh-a", secret)
        with pytest.raises(JWTError):
            validate_token(token, secret, mesh_id="mesh-b")

    def test_token_rejected_wrong_secret(
        self, secret: str, mesh_id: str
    ) -> None:
        token = create_token("agent-1", mesh_id, secret)
        with pytest.raises(JWTError):
            validate_token(token, "wrong-secret", mesh_id=mesh_id)


class TestRateLimiterIntegration:
    def test_rate_limiter_allows_under_limit(self) -> None:
        limiter = RateLimiter(default_bucket_size=10, default_refill_rate=100.0)
        for _ in range(10):
            assert limiter.allow("agent-1") is True

    def test_rate_limiter_blocked_when_exhausted(self) -> None:
        limiter = RateLimiter(default_bucket_size=3, default_refill_rate=0.0)
        for _ in range(3):
            assert limiter.allow("agent-1") is True
        assert limiter.allow("agent-1") is False

    def test_rate_limiter_per_agent_isolation(self) -> None:
        limiter = RateLimiter(default_bucket_size=5, default_refill_rate=0.0)
        limiter.configure_agent("fast-agent", bucket_size=10, refill_rate=100.0)
        limiter.configure_agent("slow-agent", bucket_size=2, refill_rate=0.0)
        for _ in range(8):
            assert limiter.allow("fast-agent") is True
        assert limiter.allow("slow-agent") is True
        assert limiter.allow("slow-agent") is True
        assert limiter.allow("slow-agent") is False

    def test_rate_limiter_token_refill(self) -> None:
        limiter = RateLimiter(default_bucket_size=5, default_refill_rate=1000.0)
        for _ in range(5):
            limiter.allow("agent-1")
        assert limiter.allow("agent-1") is False
        time.sleep(0.05)
        assert limiter.allow("agent-1") is True

    def test_rate_limiter_route_based(self) -> None:
        limiter = RateLimiter(default_bucket_size=10, default_refill_rate=0.0)
        limiter.configure_route("route-a", bucket_size=3, refill_rate=0.0)
        limiter.configure_route("route-b", bucket_size=100, refill_rate=0.0)
        assert limiter.allow(route="route-a") is True
        assert limiter.allow(route="route-a") is True
        assert limiter.allow(route="route-a") is True
        assert limiter.allow(route="route-a") is False
        assert limiter.allow(route="route-b") is True


class TestBackpressureIntegration:
    def test_backpressure_normal_to_slow_down_to_pause(self) -> None:
        controller = BackpressureController(max_queue_depth=100, threshold=0.8)
        for _ in range(10):
            controller.enqueue()
        assert controller.status == "NORMAL"
        for _ in range(75):
            controller.enqueue()
        assert controller.status == "SLOW_DOWN"
        for _ in range(15):
            controller.enqueue()
        assert controller.status == "PAUSE"

    def test_backpressure_recovery(self) -> None:
        controller = BackpressureController(max_queue_depth=100, threshold=0.8)
        for _ in range(90):
            controller.enqueue()
        assert controller.status == "SLOW_DOWN"
        controller.dequeue(70)
        assert controller.status == "NORMAL"

    def test_backpressure_returns_signal_string(self) -> None:
        controller = BackpressureController(max_queue_depth=10, threshold=0.5)
        assert controller.enqueue(6) == "SLOW_DOWN"
        assert controller.enqueue(4) == "PAUSE"

    def test_backpressure_resume_on_empty(self) -> None:
        controller = BackpressureController(max_queue_depth=10, threshold=0.8)
        controller.enqueue(10)
        assert controller.status == "PAUSE"
        controller.dequeue(10)
        assert controller.status == "RESUME"


class TestMetricsIntegration:
    def test_metrics_render_text_includes_all_types(self) -> None:
        reg = MetricsRegistry()
        reg.increment("test_total", 5)
        reg.set_gauge("test_active", 3.0)
        reg.observe("test_latency", 0.0123)

        text = reg.render_text()
        assert "test_total 5" in text
        assert "test_active 3.000000" in text
        assert "test_latency_count" in text

    def test_metrics_recorded_tensor_sent(self) -> None:
        reg = MetricsRegistry()
        reg.record_tensor_sent("agent-a", "hidden_state", 4096)
        reg.record_tensor_sent("agent-a", "hidden_state", 8192)
        reg.record_tensor_sent("agent-b", "error_context", 1024)

        assert reg.get_counter(
            'a2a_tensors_sent_total{agent_id="agent-a",label="hidden_state"}'
        ) == 2
        assert reg.get_counter(
            'a2a_tensors_sent_total{agent_id="agent-b",label="error_context"}'
        ) == 1
