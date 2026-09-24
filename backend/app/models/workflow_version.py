import uuid
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from app.core.database import Base


class WorkflowVersion(Base):
    """
    Persisted approved workflow configuration version.
    Maintains an immutable snapshot of the workflow parameters and Phase 6 evaluation metrics.
    """
    __tablename__ = "workflow_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version_id = Column(String(64), unique=True, nullable=False, index=True)
    version_number = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_run_id = Column(String(64), nullable=False, index=True)
    source_candidate_id = Column(String(64), nullable=False, index=True)
    model_id = Column(String(128), nullable=False)
    provider = Column(String(64), nullable=False)
    configuration = Column(JSON, nullable=False)
    evaluation_snapshot = Column(JSON, nullable=False)
    status = Column(String(32), default="approved", nullable=False, index=True)
    approved_by = Column(String(255), nullable=False)
    approved_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
