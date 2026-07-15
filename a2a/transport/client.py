"""A2A gRPC client for sending tensors and discovering agents."""

from __future__ import annotations

import grpc

from a2a.protocol.errors import A2AError
from a2a.protocol.generated import a2a_pb2, a2a_pb2_grpc
from a2a.transport.codec import TensorData, decode_tensor


class A2AClient:
    """Synchronous gRPC client for A2A Protocol."""

    def __init__(
        self,
        address: str,
        credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        if credentials:
            self._channel = grpc.secure_channel(address, credentials)
        else:
            self._channel = grpc.insecure_channel(address)

        self._stub = a2a_pb2_grpc.A2AServiceStub(self._channel)

    def send_tensor(
        self,
        encoded_frame: bytes,
        metadata: a2a_pb2.TensorMetadata | None = None,
        timeout: float | None = 30.0,
    ) -> a2a_pb2.TensorResponse:
        """Send a single encoded tensor via gRPC.

        Args:
            encoded_frame: Full wire-format tensor frame bytes.
            metadata: Optional A2A metadata.
            timeout: gRPC timeout in seconds.

        Returns:
            TensorResponse from server.

        Raises:
            A2AError: If the server rejects the request.
        """
        if metadata is None:
            metadata = a2a_pb2.TensorMetadata()

        request = a2a_pb2.TensorRequest(
            metadata=metadata,
            tensor_data=encoded_frame,
        )

        try:
            response = self._stub.SendTensor(request, timeout=timeout)
        except grpc.RpcError as exc:
            code = exc.code()
            if code == grpc.StatusCode.UNAUTHENTICATED:
                raise A2AError(code=400, message=exc.details() or "Authentication failed") from exc
            if code == grpc.StatusCode.PERMISSION_DENIED:
                raise A2AError(code=401, message=exc.details() or "Permission denied") from exc
            if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise A2AError(code=301, message=exc.details() or "Rate limited") from exc
            raise A2AError(code=500, message=str(exc)) from exc

        if not response.accepted:
            raise A2AError(
                code=response.error_code,
                message=response.error_message or "Unknown error",
            )

        return response

    def health_check(self, timeout: float = 5.0) -> a2a_pb2.HealthResponse:
        """Perform a health check on the server."""
        request = a2a_pb2.HealthRequest()
        return self._stub.HealthCheck(request, timeout=timeout)

    def discover(
        self,
        agent_id: str = "",
        mesh_id: str = "",
        timeout: float = 5.0,
    ) -> a2a_pb2.DiscoverResponse:
        """Discover agents on the network."""
        request = a2a_pb2.DiscoverRequest(agent_id=agent_id, mesh_id=mesh_id)
        return self._stub.Discover(request, timeout=timeout)

    def send_tensor_decoded(
        self,
        encoded_frame: bytes,
        metadata: a2a_pb2.TensorMetadata | None = None,
        timeout: float | None = 30.0,
    ) -> TensorData | None:
        """Send a tensor and decode the response if it contains tensor data.

        Returns:
            Decoded TensorData if response contains tensor data, else None.
        """
        response = self.send_tensor(encoded_frame, metadata, timeout)
        if response.tensor_data:
            return decode_tensor(response.tensor_data)
        return None

    def close(self) -> None:
        """Close the gRPC channel."""
        self._channel.close()
