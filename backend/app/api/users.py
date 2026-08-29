from fastapi import APIRouter, Depends
from app.api.auth import require_developer, require_user
from app.models.user import User
from app.schemas.user import UserResponse

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("/profile", response_model=UserResponse, summary="Get full profile of current user")
def get_user_profile(current_user: User = Depends(require_user)):
    """Accessible by any authenticated USER or DEVELOPER."""
    return current_user


@router.get("/developer-status", summary="Check developer status")
def get_developer_status(current_dev: User = Depends(require_developer)):
    """Accessible exclusively by DEVELOPER role."""
    return {
        "status": "active_developer",
        "developer_email": current_dev.email,
        "developer_name": current_dev.name,
    }
