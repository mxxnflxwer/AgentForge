import uuid
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
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
    Lifecycle states: APPROVED, ACTIVE, INACTIVE.
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
    status = Column(String(32), default="APPROVED", nullable=False, index=True)
    approved_by = Column(String(255), nullable=False)
    approved_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class WorkflowExecution(Base):
    """
    Runtime execution records for queries evaluated or executed against the active workflow version.
    Maintains separation between LLM latency and total execution time.
    """
    __tablename__ = "workflow_executions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    execution_id = Column(String(64), unique=True, nullable=False, index=True)
    workflow_version_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    retrieved_chunks = Column(JSON, nullable=True)
    model = Column(String(128), nullable=False)
    provider = Column(String(64), nullable=False)
    input_tokens = Column(Integer, default=0, nullable=False)
    output_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    cost = Column(Float, default=0.0, nullable=False)
    llm_latency_ms = Column(Float, default=0.0, nullable=False)
    total_execution_time_ms = Column(Float, default=0.0, nullable=False)
    status = Column(String(32), default="success", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class WorkflowFeedback(Base):
    """
    End-user and developer feedback associated with a specific production execution.
    Does NOT automatically alter workflow configurations or trigger optimizations.
    """
    __tablename__ = "workflow_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    feedback_id = Column(String(64), unique=True, nullable=False, index=True)
    execution_id = Column(String(64), nullable=False, index=True)
    workflow_version_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    rating = Column(Integer, nullable=False)
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class WorkflowAuditEvent(Base):
    """
    Audit log of governed lifecycle state changes: APPROVAL, ACTIVATION, DEACTIVATION, ROLLBACK, REOPTIMIZATION.
    """
    __tablename__ = "workflow_audit_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    workflow_version_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(32), nullable=False, index=True)
    performed_by = Column(String(36), nullable=True)
    user_email = Column(String(255), nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
