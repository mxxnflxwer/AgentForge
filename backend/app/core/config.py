import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://agentforge:agentforge_dev_password@localhost:5432/agentforge"
    SECRET_KEY: str = "agentforge-development-secret-change-later"
    ENVIRONMENT: str = "development"
    FRONTEND_ORIGIN: str = "http://localhost:5173"
    SESSION_COOKIE_NAME: str = "session_id"
    SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 7  # 7 days
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"
    DEVELOPER_SECRET_KEY: str = "agentforge_dev_secret_key"
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB default
    STORAGE_DIR: str = "storage/documents"
    ALLOWED_EXTENSIONS: list[str] = ["pdf", "docx", "txt"]
    CHROMA_PERSIST_DIRECTORY: str = "storage/chromadb"
    CHROMA_COLLECTION_NAME: str = "agentforge_documents"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    @property
    def cors_origins(self) -> list[str]:
        origins = [self.FRONTEND_ORIGIN.rstrip("/")]
        # Also include 127.0.0.1 variant if localhost
        if "localhost" in self.FRONTEND_ORIGIN:
            origins.append(self.FRONTEND_ORIGIN.replace("localhost", "127.0.0.1").rstrip("/"))
        return origins

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()

