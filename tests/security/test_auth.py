"""Tests for JWT authentication."""


import pytest

from a2a.security.auth import (
    JWTError,
    create_token,
    validate_token,
)

SECRET = "test-secret-key-12345"
MESH_ID = "production-mesh"


def test_create_generates_valid_token() -> None:
    token = create_token("agent-01", MESH_ID, SECRET)
    assert token is not None
    assert token.count(".") == 2  # header.payload.signature


def test_validate_returns_correct_payload() -> None:
    token = create_token("agent-01", MESH_ID, SECRET, permissions=["MSG_UNICAST"])
    payload = validate_token(token, SECRET)
    assert payload is not None
    assert payload["agent_id"] == "agent-01"
    assert payload["mesh_id"] == MESH_ID
    assert "MSG_UNICAST" in payload["permissions"]


def test_validate_rejects_expired_token() -> None:
    token = create_token("agent-01", MESH_ID, SECRET, expiry_minutes=-1)

    with pytest.raises(JWTError, match="expired"):
        validate_token(token, SECRET)


def test_validate_rejects_wrong_secret() -> None:
    token = create_token("agent-01", MESH_ID, SECRET)

    with pytest.raises(JWTError, match="Signature"):
        validate_token(token, "wrong-secret")


def test_validate_checks_mesh_id() -> None:
    token = create_token("agent-01", "mesh-a", SECRET)

    with pytest.raises(JWTError, match="Mesh ID"):
        validate_token(token, SECRET, mesh_id="mesh-b")


def test_validate_malformed_token() -> None:
    with pytest.raises(JWTError):
        validate_token("not.a.token.extra", SECRET)

    with pytest.raises(JWTError):
        validate_token("just-one-part", SECRET)


def test_create_different_agents_different_tokens() -> None:
    t1 = create_token("agent-a", MESH_ID, SECRET)
    t2 = create_token("agent-b", MESH_ID, SECRET)
    assert t1 != t2


def test_algorithm_variants() -> None:
    for alg in ("HS256", "HS384", "HS512"):
        token = create_token("agent-01", MESH_ID, SECRET, algorithm=alg)
        payload = validate_token(token, SECRET)
        assert payload is not None
