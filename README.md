# AgentForge

AgentForge is an AI workflow optimization platform.

## Architecture

- **Frontend:** React + TypeScript + Vite (`http://localhost:5173`)
- **Backend:** Python + FastAPI + Uvicorn (`http://localhost:8000`)
- **Database:** PostgreSQL 17 (via Docker)
- **ORM:** SQLAlchemy 2.0
- **Database Migrations:** Alembic
- **Authentication:** Secure Server-Side Sessions (HttpOnly Cookies, not JWT)
- **Roles:** `USER` and `DEVELOPER`

---

## Getting Started

### 1. Start PostgreSQL with Docker

```bash
docker compose up -d
```

PostgreSQL container `agentforge-postgres` will run on port `5432`.

### 2. Configure Backend Environment

Copy `.env.example` to `backend/.env`:

```bash
cp .env.example backend/.env
```

Default development values in `.env`:
```ini
DATABASE_URL=postgresql+psycopg2://agentforge:agentforge_dev_password@localhost:5432/agentforge
SECRET_KEY=agentforge-development-secret-change-later
DEVELOPER_SECRET_KEY=agentforge_dev_secret_key
ENVIRONMENT=development
FRONTEND_ORIGIN=http://localhost:5173
SESSION_COOKIE_NAME=session_id
SESSION_MAX_AGE_SECONDS=604800
COOKIE_SECURE=false
COOKIE_SAMESITE=lax
```

### 3. Install Backend Dependencies

From the `backend` directory:

```bash
cd backend
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run Database Migrations

Apply database migrations using Alembic:

```bash
cd backend
alembic upgrade head
```

To create a new migration in the future:
```bash
alembic revision --autogenerate -m "migration description"
```

### 5. Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

FastAPI Interactive API Documentation:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Authentication & User Management

AgentForge uses **secure server-side sessions** stored in PostgreSQL and delivered to the browser via an `HttpOnly`, `SameSite` cookie (`session_id`).

### Roles & Security
- **USER:** Default role upon registration.
- **DEVELOPER:** Elevated role for developer features and dashboards. Public registration as `DEVELOPER` is prohibited and requires a `developer_secret` key.
- Passwords are encrypted with `bcrypt` (12 rounds) and never logged or returned in responses.

---

## API Endpoints

### 1. Register a User
`POST /api/auth/register`

**Request Body:**
```json
{
  "name": "Alex Mercer",
  "email": "alex@example.com",
  "password": "SecurePassword123!",
  "role": "USER"
}
```

**Response (201 Created):**
```json
{
  "id": "7b2e1f42-...",
  "name": "Alex Mercer",
  "email": "alex@example.com",
  "role": "USER",
  "is_active": true,
  "created_at": "2026-08-30T00:00:00Z"
}
```

*Note: To register as `DEVELOPER` in development, include `"developer_secret": "agentforge_dev_secret_key"` in the payload.*

---

### 2. Login
`POST /api/auth/login`

**Request Body:**
```json
{
  "email": "alex@example.com",
  "password": "SecurePassword123!"
}
```

**Response (200 OK):**
- Sets `Set-Cookie: session_id=<token>; HttpOnly; Path=/; SameSite=Lax`
```json
{
  "id": "7b2e1f42-...",
  "name": "Alex Mercer",
  "email": "alex@example.com",
  "role": "USER",
  "is_active": true,
  "created_at": "2026-08-30T00:00:00Z"
}
```

---

### 3. Get Current User Profile
`GET /api/auth/me`

*Requires authenticated session cookie (`session_id`).*

**Response (200 OK):**
```json
{
  "id": "7b2e1f42-...",
  "name": "Alex Mercer",
  "email": "alex@example.com",
  "role": "USER"
}
```

---

### 4. Logout
`POST /api/auth/logout`

*Destroys the server-side session from database and clears the `session_id` cookie.*

**Response (200 OK):**
```json
{
  "message": "Successfully logged out."
}
```

---

### 5. Developer-Only Endpoint
`GET /api/auth/developer-only`

*Protected by `require_developer` dependency. Returns HTTP 403 Forbidden for `USER` role and HTTP 200 for `DEVELOPER` role.*

---

## Running Automated Tests

Run the test suite with `pytest`:

```bash
cd backend
pytest -v
```

All 12 authentication and authorization test cases execute in an isolated environment.
