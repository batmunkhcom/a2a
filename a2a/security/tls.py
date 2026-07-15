"""TLS/mTLS configuration helpers for gRPC transport."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TlsConfig:
    """TLS configuration for gRPC server and client."""

    enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None

    def create_server_credentials(self) -> object | None:
        """Create gRPC server credentials from config.

        Returns:
            grpc.ServerCredentials if TLS is enabled, None to use insecure.
        """
        if not self.enabled:
            return None

        try:
            import grpc

            cert = _read_file(self.cert_file)
            key = _read_file(self.key_file)
            ca = _read_file(self.ca_file) if self.ca_file else None

            if ca:
                # mTLS: require client certificates
                return grpc.ssl_server_credentials(
                    [(key, cert)],
                    root_certificates=ca,
                    require_client_auth=True,
                )

            return grpc.ssl_server_credentials(
                [(key, cert)],
                root_certificates=None,
                require_client_auth=False,
            )
        except ImportError:
            return None

    def create_client_credentials(
        self, server_hostname: str | None = None
    ) -> object | None:
        """Create gRPC client channel credentials.

        Args:
            server_hostname: Expected server hostname for verification.

        Returns:
            grpc.ChannelCredentials or None for insecure.
        """
        if not self.enabled:
            return None

        try:
            import grpc

            cert = _read_file(self.cert_file)
            key = _read_file(self.key_file)
            ca = _read_file(self.ca_file) if self.ca_file else None

            if ca and cert and key:
                # mTLS client
                return grpc.ssl_channel_credentials(
                    root_certificates=ca,
                    private_key=key,
                    certificate_chain=cert,
                )

            return grpc.ssl_channel_credentials()

        except ImportError:
            return None


def _read_file(path: str | None) -> bytes:
    if path is None:
        raise FileNotFoundError("TLS file path not configured")
    with open(path, "rb") as f:
        return f.read()
