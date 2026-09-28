import uuid
from sqlalchemy import (
    Column,
    DateTime,
    JSON,
    String,
    Text,
    func,
)
from app.core.database import Base


class BenchmarkRun(Base):
    """
    Persisted benchmark evaluation run comparing multiple workflow versions.
    Maintains a reproducible record of identical query/document evaluation across versions.
    """
    __tablename__ = "benchmark_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    benchmark_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(32), default="completed", nullable=False, index=True)
    preset_name = Column(String(64), nullable=True)
    query = Column(Text, nullable=False)
    document_id = Column(String(64), nullable=True)
    expected_answer = Column(Text, nullable=True)
    selected_versions = Column(JSON, nullable=False)  # List of version_ids / "baseline"
    results = Column(JSON, nullable=False)  # List of normalized BenchmarkItemResult dicts
    pareto_summary = Column(JSON, nullable=False)  # Pareto frontier candidate IDs and summary
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
