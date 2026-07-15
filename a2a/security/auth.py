"""JWT authentication — token creation and validation."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from base64 import urlsafe_b64encode
from typing import Any


class JWTError(Exception):
    """Raised on token validation failure."""


def create_token(
    agent_id: str,
    mesh_id: str,
    secret: str,
    permissions: list[str] | None = None,
    expiry_minutes: int = 60,
    algorithm: str = "HS256",
) -> str:
    """Create a signed JWT token for an A2A agent.

    Args:
        agent_id: Agent identifier (e.g. "log-reader-01").
        mesh_id: Mesh identifier for isolation.
        secret: Shared secret for HMAC signing.
        permissions: List of permission strings.
        expiry_minutes: Token lifetime in minutes.
        algorithm: HMAC algorithm ("HS256", "HS384", "HS512").

    Returns:
        Encoded JWT token string.
    """
    header = {"alg": algorithm, "typ": "JWT"}

    now = int(time.time())
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "mesh_id": mesh_id,
        "permissions": permissions or ["MSG_UNICAST", "MSG_DISCOVER"],
        "iat": now,
        "exp": now + expiry_minutes * 60,
    }

    header_b64 = _b64encode(json.dumps(header, separators=(",", ":")))
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")))

    message = f"{header_b64}.{payload_b64}"

    hash_alg = {"HS256": "sha256", "HS384": "sha384", "HS512": "sha512"}.get(
        algorithm, "sha256"
    )
    signature = hmac.new(
        secret.encode(), message.encode(), getattr(hashlib, hash_alg)
    ).digest()
    sig_b64 = _b64encode_raw(signature)

    return f"{message}.{sig_b64}"


def validate_token(
    token: str,
    secret: str,
    *,
    mesh_id: str | None = None,
) -> dict[str, Any] | None:
    """Validate a JWT token and return its payload.

    Args:
        token: Encoded JWT string.
        secret: Shared secret for HMAC verification.
        mesh_id: Optional mesh ID to verify against.

    Returns:
        Payload dict if valid, None otherwise.

    Raises:
        JWTError: If token is malformed.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise JWTError("Invalid token: expected 3 parts")

    header_b64, payload_b64, sig_b64 = parts

    try:
        header = json.loads(_b64decode(header_b64))
    except (ValueError, UnicodeDecodeError) as exc:
        raise JWTError(f"Invalid header encoding: {exc}") from exc

    algorithm = header.get("alg", "HS256")
    hash_alg = {"HS256": "sha256", "HS384": "sha384", "HS512": "sha512"}.get(
        algorithm, "sha256"
    )

    # Verify signature
    message = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(
        secret.encode(), message.encode(), getattr(hashlib, hash_alg)
    ).digest()
    expected_b64 = _b64encode_raw(expected_sig)

    if not hmac.compare_digest(sig_b64.encode(), expected_b64.encode()):
        raise JWTError("Signature verification failed")

    try:
        payload = json.loads(_b64decode(payload_b64))
    except (ValueError, UnicodeDecodeError) as exc:
        raise JWTError(f"Invalid payload encoding: {exc}") from exc

    # Check expiration
    now = int(time.time())
    if payload.get("exp", 0) < now:
        raise JWTError("Token has expired")

    # Check mesh isolation
    if mesh_id and payload.get("mesh_id") != mesh_id:
        raise JWTError(
            f"Mesh ID mismatch: expected {mesh_id}, got {payload.get('mesh_id')}"
        )

    return payload


# ── Helpers ────────────────────────────────────────────────────


def _b64encode(data: str) -> str:
    return urlsafe_b64encode(data.encode()).decode().rstrip("=")


def _b64decode(data: str) -> str:
    # Add padding
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    import base64

    return base64.urlsafe_b64decode(data).decode()


def _b64encode_raw(data: bytes) -> str:
    return urlsafe_b64encode(data).decode().rstrip("=")
