import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app as fastapi_app
from app.models.user import User, UserRole, UserSession

# In-memory SQLite database for isolated test execution
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


import app.core.database
import app.services.document_service


@pytest.fixture(scope="function")
def db_session(monkeypatch):
    """Create a fresh database for each test and override SessionLocal."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    monkeypatch.setattr(app.core.database, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(app.services.document_service, "SessionLocal", TestingSessionLocal)
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app, base_url="http://localhost:8000") as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()

