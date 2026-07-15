"""A2A gRPC server implementing the A2AService."""

from __future__ import annotations

import time
from concurrent import futures

import grpc

from a2a.protocol.errors import A2AError, error_to_response
from a2a.protocol.generated import a2a_pb2, a2a_pb2_grpc
from a2a.transport.codec import decode_tensor, validate_tensor_values


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
                raise A2AError(
                    code=101,
                    message="Missing tensor_data in request",
                )

            # Decode and validate
            tensor = decode_tensor(request.tensor_data)
            if not validate_tensor_values(tensor.values, tensor.dtype):
                raise A2AError(
                    code=102,
                    message="Tensor contains NaN or Inf values",
                )

            # Route to plugin if manager is available
            if self._plugin_manager:
                metadata = request.metadata
                # TODO: route via PluginManager.route_tensor() — Sprint 3
                _ = metadata, tensor

            return a2a_pb2.TensorResponse(
                accepted=True,
                metadata=request.metadata,
            )

        except A2AError as exc:
            return error_to_response(exc)
        except Exception as exc:
            return error_to_response(
                A2AError(code=500, message=str(exc))
            )

    def StreamTensors(
        self,
        request_iterator: grpc.ServicerContext,
        context: grpc.ServicerContext,
    ):
        """Handle a bidirectional tensor stream."""
        for request in request_iterator:
            try:
                if not request.tensor_data:
                    yield error_to_response(
                        A2AError(code=101, message="Missing tensor_data")
                    )
                    continue

                tensor = decode_tensor(request.tensor_data)
                if not validate_tensor_values(tensor.values, tensor.dtype):
                    yield error_to_response(
                        A2AError(code=102, message="Tensor contains NaN/Inf")
                    )
                    continue

                yield a2a_pb2.TensorResponse(
                    accepted=True,
                    metadata=request.metadata,
                )

            except Exception as exc:
                yield error_to_response(
                    A2AError(code=500, message=str(exc))
                )

    def Discover(
        self, request: a2a_pb2.DiscoverRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.DiscoverResponse:
        """Return list of known agents."""
        # Stub — full discovery in Sprint 5
        return a2a_pb2.DiscoverResponse(agents=[])

    def RequestProjection(
        self, request: a2a_pb2.ProjectionRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.ProjectionResponse:
        """Handle projection training request."""
        # Stub — full projection in Sprint 4
        return a2a_pb2.ProjectionResponse(accepted=False, error_message="not implemented")

    def HealthCheck(
        self, request: a2a_pb2.HealthRequest, context: grpc.ServicerContext
    ) -> a2a_pb2.HealthResponse:
        """Return health status."""
        from a2a._version import __version__

        uptime = int(time.time() - self._start_time)
        return a2a_pb2.HealthResponse(
            status="ok",
            uptime_seconds=uptime,
            version=__version__,
            plugins_loaded=0,
        )


class A2AServer:
    """Wrapper around gRPC server with lifecycle management."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        plugin_manager: object | None = None,
        max_workers: int = 10,
    ) -> None:
        self._host = host
        self._port = port
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers)
        )
        self._servicer = A2AServicer(plugin_manager=plugin_manager)
        a2a_pb2_grpc.add_A2AServiceServicer_to_server(
            self._servicer, self._server
        )
        self._server.add_insecure_port(f"{host}:{port}")

    def start(self) -> None:
        """Start the gRPC server (non-blocking)."""
        self._server.start()

    def stop(self, grace: float | None = 2.0) -> None:
        """Gracefully stop the gRPC server."""
        self._server.stop(grace)

    def wait_for_termination(self) -> None:
        """Block until the server terminates."""
        self._server.wait_for_termination()

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"
