import re
import secrets
import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password securely using bcrypt."""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    try:
        password_bytes = plain_password.encode("utf-8")
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception:
        return False


def validate_password_strength(password: str) -> None:
    """
    Validate password strength:
    - At least 8 characters
    - At least one letter
    - At least one digit or special character
    """
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password cannot exceed 128 characters.")
    if not re.search(r"[A-Za-z]", password):
        raise ValueError("Password must contain at least one letter.")
    if not re.search(r"[\d\W_]", password):
        raise ValueError("Password must contain at least one number or special character.")


def generate_session_token() -> str:
    """Generate a high-entropy random session token."""
    return secrets.token_urlsafe(32)
