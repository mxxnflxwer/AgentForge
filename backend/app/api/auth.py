from typing import Optional
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User, UserRole
from app.schemas.user import (
    MessageResponse,
    UserLoginRequest,
    UserMeResponse,
    UserRegisterRequest,
    UserResponse,
)
from app.services.auth_service import (
    authenticate_user,
    create_user_session,
    destroy_user_session,
    get_user_by_session_id,
    register_user,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# --- Authorization Dependencies ---

def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    session_id_cookie: Optional[str] = Cookie(default=None, alias=settings.SESSION_COOKIE_NAME),
) -> User:
    """
    Extract the session identifier from the HttpOnly cookie (or header fallback)
    and validate the current active user.
    """
    session_id = session_id_cookie
    # Fallback support for Authorization / custom header in testing or clients
    if not session_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_id = auth_header[7:].strip()
        elif request.headers.get("X-Session-ID"):
            session_id = request.headers.get("X-Session-ID")

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. No active session found.",
        )

    user = get_user_by_session_id(db, session_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please log in again.",
        )

    return user


def require_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency verifying that the caller is an authenticated user."""
    return current_user


def require_developer(current_user: User = Depends(get_current_user)) -> User:
    """Dependency verifying that the caller has DEVELOPER role access."""
    if current_user.role != UserRole.DEVELOPER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: DEVELOPER role required.",
        )
    return current_user


# --- Endpoints ---

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(
    payload: UserRegisterRequest,
    db: Session = Depends(get_db),
):
    """
    Register a new user with standard USER privileges.
    Public registration as DEVELOPER is restricted and rejected unless
    validated via controlled development credentials.
    """
    user = register_user(db=db, data=payload)
    return user


@router.post(
    "/login",
    response_model=UserResponse,
    summary="Log in and create server-side session",
)
def login(
    payload: UserLoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Authenticate user credentials, establish a server-side session,
    and attach an HttpOnly session cookie to the response.
    """
    user = authenticate_user(db=db, email=payload.email, password=payload.password)
    session = create_user_session(db=db, user_id=user.id)

    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=session.id,
        max_age=settings.SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        path="/",
    )

    return user


@router.get(
    "/me",
    response_model=UserMeResponse,
    summary="Get current authenticated user info",
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    """Return the profile of the currently authenticated user."""
    return current_user


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Log out and destroy session",
)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    session_id_cookie: Optional[str] = Cookie(default=None, alias=settings.SESSION_COOKIE_NAME),
):
    """Destroy server session and clear the session cookie."""
    session_id = session_id_cookie
    if not session_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_id = auth_header[7:].strip()
        elif request.headers.get("X-Session-ID"):
            session_id = request.headers.get("X-Session-ID")

    destroy_user_session(db=db, session_id=session_id)

    response.delete_cookie(
        key=settings.SESSION_COOKIE_NAME,
        path="/",
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
    )

    return MessageResponse(message="Successfully logged out.")


@router.get(
    "/developer-only",
    response_model=dict,
    summary="Developer-only test endpoint",
)
def developer_only_route(
    current_dev: User = Depends(require_developer),
):
    """Protected endpoint accessible exclusively by DEVELOPER role."""
    return {
        "status": "ok",
        "message": f"Welcome Developer {current_dev.name}",
        "developer_id": current_dev.id,
    }
