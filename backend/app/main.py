import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.api import auth_router, users_router
from app.core.config import settings
from app.core.database import engine

logger = logging.getLogger("agentforge")

app = FastAPI(
    title="AgentForge API",
    description="AI Workflow Optimization Platform - Authentication & User Management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Prevent unhandled internal database errors from leaking to clients."""
    logger.exception(f"Unhandled error occurred processing request to {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred. Please try again later."},
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "AgentForge API",
    }


@app.get("/health/database", tags=["Health"])
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "PostgreSQL",
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "database": "Database connection unavailable",
            },
        )


# Include API Routers
app.include_router(auth_router)
app.include_router(users_router)