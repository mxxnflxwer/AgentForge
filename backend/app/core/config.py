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
    MIN_RELEVANCE_SCORE: float = 0.35
    DEBUG_RETRIEVAL: bool = True

    # LLM Settings: Gemini 2.5 Flash-Lite
    GEMINI_API_KEY: str | None = None
    GEMINI_API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # LLM Settings: Qwen 3.6 27B (via OpenAI-compatible / OpenRouter / DashScope / Groq API)
    QWEN_API_KEY: str | None = None
    QWEN_API_BASE_URL: str = "https://openrouter.ai/api/v1"
    QWEN_MODEL_NAME: str = "qwen/qwen-2.5-72b-instruct"

    # LLM Settings: GPT-OSS 120B (via OpenAI-compatible API)
    GPT_OSS_API_KEY: str | None = None
    GPT_OSS_API_BASE_URL: str = "https://openrouter.ai/api/v1"
    GPT_OSS_MODEL_NAME: str = "openai/gpt-oss-120b"

    # Common LLM settings
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1

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

