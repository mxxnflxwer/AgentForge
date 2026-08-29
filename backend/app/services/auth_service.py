from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import (
    generate_session_token,
    hash_password,
    validate_password_strength,
    verify_password,
)
from app.models.user import User, UserRole, UserSession
from app.schemas.user import UserRegisterRequest


def register_user(db: Session, data: UserRegisterRequest) -> User:
    """Validate and register a new user."""
    # 1. Validate password strength
    try:
        validate_password_strength(data.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # 2. Check for duplicate email
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # 3. Controlled role assignment
    assigned_role = UserRole.USER
    if data.role == UserRole.DEVELOPER:
        if not data.developer_secret or data.developer_secret != settings.DEVELOPER_SECRET_KEY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Direct DEVELOPER registration is restricted. A valid developer secret is required.",
            )
        assigned_role = UserRole.DEVELOPER

    # 4. Hash password
    hashed_password = hash_password(data.password)

    # 5. Persist user
    new_user = User(
        name=data.name,
        email=data.email,
        password_hash=hashed_password,
        role=assigned_role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def authenticate_user(db: Session, email: str, password: str) -> User:
    """Authenticate user credentials and return the active user."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is currently inactive.",
        )

    return user


def create_user_session(db: Session, user_id: str) -> UserSession:
    """Create a persistent server-side session."""
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.SESSION_MAX_AGE_SECONDS)

    session = UserSession(
        id=token,
        user_id=user_id,
        expires_at=expires_at,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_user_by_session_id(db: Session, session_id: Optional[str]) -> Optional[User]:
    """Retrieve the user associated with a valid active session."""
    if not session_id:
        return None

    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session:
        return None

    # Handle timezone-aware or naive datetime comparison safely
    now = datetime.now(timezone.utc)
    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at <= now:
        db.delete(session)
        db.commit()
        return None

    if not session.user or not session.user.is_active:
        return None

    return session.user


def destroy_user_session(db: Session, session_id: Optional[str]) -> None:
    """Destroy a server-side session."""
    if not session_id:
        return

    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if session:
        db.delete(session)
        db.commit()
