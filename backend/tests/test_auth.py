import pytest
from app.core.config import settings


def test_successful_registration(client):
    """Test user registration with valid data."""
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Jane Doe",
            "email": "jane@example.com",
            "password": "SecurePassword123!",
            "role": "USER",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Jane Doe"
    assert data["email"] == "jane@example.com"
    assert data["role"] == "USER"
    assert data["is_active"] is True
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_duplicate_registration(client):
    """Test that registering an existing email fails with 409 Conflict."""
    payload = {
        "name": "Jane Doe",
        "email": "jane.dup@example.com",
        "password": "SecurePassword123!",
    }
    first_resp = client.post("/api/auth/register", json=payload)
    assert first_resp.status_code == 201

    duplicate_resp = client.post("/api/auth/register", json=payload)
    assert duplicate_resp.status_code == 409
    assert "already exists" in duplicate_resp.json()["detail"]


def test_weak_password_registration(client):
    """Test that registering with a short/weak password fails with 400 or 422."""
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Short Pass User",
            "email": "short@example.com",
            "password": "123",
        },
    )
    assert response.status_code in [400, 422]


def test_public_developer_registration_restricted(client):
    """Test that attempting to register as DEVELOPER without secret is forbidden."""
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Sneaky Hacker",
            "email": "hacker@example.com",
            "password": "SecurePassword123!",
            "role": "DEVELOPER",
        },
    )
    assert response.status_code == 403
    assert "restricted" in response.json()["detail"].lower()


def test_controlled_developer_registration(client):
    """Test registering as DEVELOPER with valid developer secret."""
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Lead Developer",
            "email": "dev@example.com",
            "password": "SecurePassword123!",
            "role": "DEVELOPER",
            "developer_secret": settings.DEVELOPER_SECRET_KEY,
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == "DEVELOPER"


def test_successful_login(client):
    """Test user login returns user profile and sets session cookie."""
    client.post(
        "/api/auth/register",
        json={
            "name": "Alice Smith",
            "email": "alice@example.com",
            "password": "Password123!",
        },
    )

    response = client.post(
        "/api/auth/login",
        json={
            "email": "alice@example.com",
            "password": "Password123!",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "alice@example.com"
    assert data["name"] == "Alice Smith"
    assert "session_id" in response.cookies


def test_incorrect_password_login(client):
    """Test login with wrong password returns 401."""
    client.post(
        "/api/auth/register",
        json={
            "name": "Bob Smith",
            "email": "bob@example.com",
            "password": "CorrectPassword123!",
        },
    )

    response = client.post(
        "/api/auth/login",
        json={
            "email": "bob@example.com",
            "password": "WrongPassword123!",
        },
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


def test_me_without_authentication(client):
    """Test /api/auth/me returns 401 when unauthenticated."""
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_with_authentication(client):
    """Test /api/auth/me returns user info when authenticated via session cookie."""
    client.post(
        "/api/auth/register",
        json={
            "name": "Charlie Brown",
            "email": "charlie@example.com",
            "password": "Password123!",
        },
    )
    # Login automatically persists cookie in client session
    client.post(
        "/api/auth/login",
        json={
            "email": "charlie@example.com",
            "password": "Password123!",
        },
    )

    me_resp = client.get("/api/auth/me")
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["name"] == "Charlie Brown"
    assert me_data["email"] == "charlie@example.com"
    assert me_data["role"] == "USER"
    assert "id" in me_data


def test_logout(client):
    """Test logout destroys the session and clears session cookie."""
    client.post(
        "/api/auth/register",
        json={
            "name": "David Miller",
            "email": "david@example.com",
            "password": "Password123!",
        },
    )
    client.post(
        "/api/auth/login",
        json={
            "email": "david@example.com",
            "password": "Password123!",
        },
    )

    # Logout
    logout_resp = client.post("/api/auth/logout")
    assert logout_resp.status_code == 200
    assert "logged out" in logout_resp.json()["message"].lower()

    # Subsequent /me call should fail
    subsequent_resp = client.get("/api/auth/me")
    assert subsequent_resp.status_code == 401


def test_user_cannot_access_developer_only_endpoint(client):
    """Test that a standard USER receives 403 on developer-only endpoint."""
    client.post(
        "/api/auth/register",
        json={
            "name": "Regular User",
            "email": "regular@example.com",
            "password": "Password123!",
        },
    )
    client.post(
        "/api/auth/login",
        json={
            "email": "regular@example.com",
            "password": "Password123!",
        },
    )

    dev_route_resp = client.get("/api/auth/developer-only")
    assert dev_route_resp.status_code == 403
    assert "DEVELOPER" in dev_route_resp.json()["detail"]


def test_developer_can_access_developer_only_endpoint(client):
    """Test that a DEVELOPER can access developer-only endpoint."""
    client.post(
        "/api/auth/register",
        json={
            "name": "Super Developer",
            "email": "superdev@example.com",
            "password": "Password123!",
            "role": "DEVELOPER",
            "developer_secret": settings.DEVELOPER_SECRET_KEY,
        },
    )
    client.post(
        "/api/auth/login",
        json={
            "email": "superdev@example.com",
            "password": "Password123!",
        },
    )

    dev_route_resp = client.get("/api/auth/developer-only")
    assert dev_route_resp.status_code == 200
    assert dev_route_resp.json()["status"] == "ok"
    assert "Super Developer" in dev_route_resp.json()["message"]
