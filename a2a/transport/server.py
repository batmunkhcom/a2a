"""A2A gRPC server implementing the A2AService with JWT auth, rate limiting, and mTLS."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent import futures
from http.server import BaseHTTPRequestHandler, HTTPServer

import grpc

from a2a.config.schema import A2AConfig
from a2a.monitoring.rate_limiter import RateLimiter
from a2a.protocol.errors import A2AError, error_to_response
from a2a.protocol.generated import a2a_pb2, a2a_pb2_grpc
from a2a.security.auth import JWTError, validate_token
from a2a.transport.codec import decode_tensor, validate_tensor_values

logger = logging.getLogger(__name__)


class AuthInterceptor(grpc.ServerInterceptor):
    """gRPC interceptor that validates JWT tokens on incoming requests."""

    _EXEMPT_METHODS = frozenset([
        "/a2a.A2AService/HealthCheck",
        "/a2a.A2AService/Discover",
    ])

    def __init__(
        self, secret: str, mesh_id: str | None = None, enabled: bool = True
    ) -> None:
        self._secret = secret
        self._mesh_id = mesh_id
        self._enabled = enabled

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | grpc.unary_unary_rpc_method_handler | None:
        if not self._enabled:
            return continuation(handler_call_details)

        method = handler_call_details.method
        if method in self._EXEMPT_METHODS:
            return continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or {})

        token = metadata.get("authorization", "")
        if token.startswith("Bearer "):
            token = token[7:]

        if not token:
            return _abort_method(grpc.StatusCode.UNAUTHENTICATED, "Missing JWT token")

        try:
            payload = validate_token(token, self._secret, mesh_id=self._mesh_id)
        except JWTError as exc:
            return _abort_method(grpc.StatusCode.UNAUTHENTICATED, str(exc))

        if payload is None:
            return _abort_method(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")

        return continuation(handler_call_details)


class RateLimitInterceptor(grpc.ServerInterceptor):
    """gRPC interceptor that enforces token-bucket rate limiting per agent."""

    def __init__(self, rate_limiter: RateLimiter, enabled: bool = True) -> None:
        self._limiter = rate_limiter
        self._enabled = enabled

    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | grpc.unary_unary_rpc_method_handler | None:
        if not self._enabled:
            return continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or {})

        agent_id = metadata.get("x-agent-id", metadata.get("agent-id", "unknown"))
        route = metadata.get("x-route", handler_call_details.method)

        if not self._limiter.allow(agent_id=agent_id, route=route):
            return _abort_method(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Rate limit exceeded for agent {agent_id}",
            )

        return continuation(handler_call_details)


def _abort_method(
    code: grpc.StatusCode, message: str
) -> grpc.unary_unary_rpc_method_handler:
    """Return an interceptor handler that immediately aborts with an error."""

    def abort_handler(request: object, context: grpc.ServicerContext) -> None:
        context.abort(code, message)

    return grpc.unary_unary_rpc_method_handler(abort_handler)


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        uptime = int(time.time() - self.server.start_time)  # type: ignore[attr-defined]

        if self.path == "/health":
            info = {
                "status": "ok",
                "uptime": uptime,
                "plugins_loaded": self.server.plugins_loaded,  # type: ignore[attr-defined]
            }
            self._json(200, info)
        elif self.path == "/health/live":
            self._json(200, {"alive": True})
        elif self.path == "/health/ready":
            self._json(200, {"ready": True})
        elif self.path == "/metrics":
            self._json(200, {"metrics": "prometheus_text_format"})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:  # type: ignore[override]
        logger.debug(fmt % args)


class A2AServicer(a2a_pb2_grpc.A2AServiceServicer):
    """gRPC servicer for A2A Protocol."""

    def __init__(self, plugin_manager: object | None = None) -> None:
        self._plugin_manager = plugin_manager
        self._start_time = time.time()

    def SendTensor(
        self, request: a2a_pb2.TensorRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.TensorResponse:
        """Handle a single tensor send request."""
        try:
            if not request.tensor_data:
                raise A2AError(code=101, message="Missing tensor_data in request")

            tensor = decode_tensor(request.tensor_data)
            if not validate_tensor_values(tensor.values, tensor.dtype):
                raise A2AError(code=102, message="Tensor contains NaN or Inf values")

            if self._plugin_manager:
                metadata = request.metadata
                _ = metadata, tensor

            return a2a_pb2.TensorResponse(accepted=True, metadata=request.metadata)
        except A2AError as exc:
            return error_to_response(exc)
        except Exception as exc:
            return error_to_response(A2AError(code=500, message=str(exc)))

    def StreamTensors(
        self, request_iterator: grpc.ServicerContext, context: grpc.ServicerContext
    ) -> object:
        """Handle a bidirectional tensor stream."""
        for request in request_iterator:
            try:
                if not request.tensor_data:
                    yield error_to_response(A2AError(code=101, message="Missing tensor_data"))
                    continue
                tensor = decode_tensor(request.tensor_data)
                if not validate_tensor_values(tensor.values, tensor.dtype):
                    yield error_to_response(A2AError(code=102, message="Tensor contains NaN/Inf"))
                    continue
                yield a2a_pb2.TensorResponse(accepted=True, metadata=request.metadata)
            except Exception as exc:
                yield error_to_response(A2AError(code=500, message=str(exc)))

    def Discover(
        self, request: a2a_pb2.DiscoverRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.DiscoverResponse:
        return a2a_pb2.DiscoverResponse(agents=[])

    def RequestProjection(
        self, request: a2a_pb2.ProjectionRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.ProjectionResponse:
        return a2a_pb2.ProjectionResponse(accepted=False, error_message="not implemented")

    def HealthCheck(
        self, request: a2a_pb2.HealthRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.HealthResponse:
        uptime = int(time.time() - self._start_time)
        return a2a_pb2.HealthResponse(
            status="ok", uptime_seconds=uptime, version="0.1.0", plugins_loaded=0,
        )


class A2AServer:
    """Wrapper around gRPC server with lifecycle management, JWT auth, rate limiting,
    mTLS, and HTTP health endpoints.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        plugin_manager: object | None = None,
        max_workers: int = 10,
        config: A2AConfig | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._config = config
        self._start_time = time.time()

        interceptors: list[grpc.ServerInterceptor] = []

        if config:
            # JWT Auth interceptor
            if config.security.auth_mechanism != "none":
                secret = self._resolve_jwt_secret(config)
                mesh_id = config.security.mesh_id
                interceptors.append(AuthInterceptor(
                    secret=secret,
                    mesh_id=mesh_id,
                    enabled=(config.security.auth_mechanism == "jwt"),
                ))

            # Rate limit interceptor
            if config.rate_limit.enabled:
                rl_config = config.rate_limit
                limiter = RateLimiter(
                    default_bucket_size=rl_config.default.bucket_size,
                    default_refill_rate=rl_config.default.refill_rate,
                )
                for agent_id, agent_cfg in rl_config.agents.items():
                    limiter.configure_agent(
                        agent_id, agent_cfg.bucket_size, agent_cfg.refill_rate
                    )
                for route, route_cfg in rl_config.routes.items():
                    limiter.configure_route(
                        route, route_cfg.bucket_size, route_cfg.refill_rate
                    )
                self._rate_limiter = limiter
                interceptors.append(RateLimitInterceptor(limiter, enabled=True))

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            interceptors=interceptors,
        )
        self._servicer = A2AServicer(plugin_manager=plugin_manager)
        a2a_pb2_grpc.add_A2AServiceServicer_to_server(self._servicer, self._server)

        # mTLS / TLS
        if config and config.security.tls_enabled:
            self._add_secure_port(host, port, config)
        else:
            self._server.add_insecure_port(f"{host}:{port}")

        # HTTP health server
        self._health_server: HTTPServer | None = None
        self._health_port = 0  # 0 = OS-assigned free port
        if config and config.runtime.metrics_port:
            self._health_port = config.runtime.metrics_port

    def _add_secure_port(self, host: str, port: int, config: A2AConfig) -> None:
        cert_file = config.security.cert_file or ""
        key_file = config.security.key_file or ""
        ca_file = config.security.ca_file

        if not cert_file or not key_file:
            logger.warning("TLS enabled but cert/key files missing, using insecure")
            self._server.add_insecure_port(f"{host}:{port}")
            return

        try:
            with open(key_file, "rb") as f:
                private_key = f.read()
            with open(cert_file, "rb") as f:
                certificate_chain = f.read()

            if ca_file:
                with open(ca_file, "rb") as f:
                    root_certificates = f.read()
                credentials = grpc.ssl_server_credentials(
                    [(private_key, certificate_chain)],
                    root_certificates=root_certificates,
                    require_client_auth=True,
                )
            else:
                credentials = grpc.ssl_server_credentials(
                    [(private_key, certificate_chain)]
                )

            self._server.add_secure_port(f"{host}:{port}", credentials)
            logger.info("mTLS enabled on %s:%s", host, port)
        except FileNotFoundError as exc:
            logger.warning("TLS cert/key not found (%s), using insecure", exc)
            self._server.add_insecure_port(f"{host}:{port}")

    def start(self) -> None:
        self._server.start()
        self._start_health_server()
        logger.info("A2A gRPC server on %s:%s, health on port %s",
                     self._host, self._port, self._health_port)

    def stop(self, grace: float | None = 2.0) -> None:
        self._stop_health_server()
        self._server.stop(grace)

    def wait_for_termination(self) -> None:
        self._server.wait_for_termination()

    def _start_health_server(self) -> None:
        port = self._health_port if self._health_port > 0 else 8080
        for _attempt in range(5):
            try:
                self._health_server = HTTPServer(("0.0.0.0", port), HealthHandler)
                self._health_server.start_time = self._start_time  # type: ignore[attr-defined]
                self._health_server.plugins_loaded = 0  # type: ignore[attr-defined]
                self._health_port = port
                thread = threading.Thread(target=self._health_server.serve_forever, daemon=True)
                thread.start()
                return
            except OSError:
                port = 0  # OS picks next free port

    def _stop_health_server(self) -> None:
        if self._health_server:
            self._health_server.shutdown()

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"

    @property
    def health_port(self) -> int:
        return self._health_port

    @property
    def servicer(self) -> A2AServicer:
        return self._servicer

    @staticmethod
    def _resolve_jwt_secret(config: A2AConfig) -> str:
        if config.security.jwt_secret_env:
            import os
            return os.environ.get(config.security.jwt_secret_env, "")
        return ""
