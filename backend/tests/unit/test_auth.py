import pytest

# pyrefly: ignore [missing-import]
from app.auth.keys import KeyStore


@pytest.mark.asyncio
async def test_keystore_create_validate_revoke(db_session):
    store = KeyStore(db_session)

    # Create key
    raw_key, api_key = await store.create_key("test_create")
    assert raw_key.startswith("mf_")
    assert api_key.name == "test_create"
    assert api_key.is_active is True

    # Validate valid key
    validated_key = await store.validate_key(raw_key)
    assert validated_key is not None
    assert validated_key.name == "test_create"
    assert validated_key.last_used_at is not None

    # Validate invalid key
    assert await store.validate_key("mf_invalid_key_123") is None

    # List keys
    keys = await store.list_keys()
    assert len(keys) >= 1
    assert any(k.name == "test_create" for k in keys)

    # Revoke key
    assert await store.revoke_key(api_key.key_prefix) is True

    # Validate revoked key
    assert await store.validate_key(raw_key) is None


@pytest.mark.asyncio
async def test_auth_missing_key(client):
    client.headers.pop("X-API-Key", None)
    response = await client.get("/api/v1/tasks")
    assert response.status_code == 401
    assert "Missing API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_auth_invalid_key(client):
    client.headers["X-API-Key"] = "mf_invalid_123"
    response = await client.get("/api/v1/tasks")
    assert response.status_code == 401
    assert "Invalid or revoked API key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_auth_valid_key(client):
    # client fixture automatically includes a valid test_key
    response = await client.get("/api/v1/tasks")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_auth_exempt_paths(client):
    client.headers.pop("X-API-Key", None)

    resp = await client.get("/health")
    assert resp.status_code == 200

    resp = await client.get("/version")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_admin_api_with_valid_db_key(client):
    # Client has a standard API key. It should still be able to access the admin endpoint
    # since it's a valid key (verify_admin_key falls back to standard check).
    resp = await client.post("/api/v1/admin/keys", json={"name": "admin_test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "admin_test"
    assert data["key"].startswith("mf_")


@pytest.mark.asyncio
async def test_admin_api_with_bootstrap_key(client):
    # pyrefly: ignore [missing-import]
    from app.config import get_settings

    settings = get_settings()
    settings.bootstrap_admin_key = "test_bootstrap_token_123"

    # Use the bootstrap key instead of standard key
    client.headers["X-API-Key"] = "test_bootstrap_token_123"

    resp = await client.post("/api/v1/admin/keys", json={"name": "bootstrap_test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "bootstrap_test"

    # Clean up settings
    settings.bootstrap_admin_key = None
