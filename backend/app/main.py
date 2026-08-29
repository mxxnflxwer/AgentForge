from fastapi import FastAPI
from app.core.database import engine, Base

app = FastAPI(
    title="AgentForge API",
    description="AI Workflow Optimization Platform",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AgentForge API",
    }


@app.get("/health/database")
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(__import__("sqlalchemy").text("SELECT 1"))

        return {
            "status": "ok",
            "database": "PostgreSQL",
        }

    except Exception as e:
        return {
            "status": "error",
            "database": str(e),
        }