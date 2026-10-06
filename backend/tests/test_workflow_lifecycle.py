import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models.user import User, UserRole
from app.models.workflow_version import (
    WorkflowAuditEvent,
    WorkflowExecution,
    WorkflowFeedback,
    WorkflowVersion,
)
from app.schemas.rag import RAGSearchResponse, RetrievedChunk
from app.services.agent_evo.crossover_service import CrossoverService
from app.services.agent_evo.optimizer import AgentEvoOptimizer
from app.services.agent_evo.workflow import get_baseline_workflow
from app.services.agent_evo.workflow_lifecycle_service import WorkflowLifecycleService
from app.services.llm.base import LLMResponse


@pytest.fixture
def mock_evaluation_env():
    """Mock RAG retrieval and LLM router to ensure ultra-fast, deterministic testing."""
    mock_retrieval = RAGSearchResponse(
        query="What is the diagnosis and prescribed medication?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="c_test_1",
                document_id="doc_test_1",
                chunk_index=0,
                content="Assessment: Primary clinical diagnosis is Type 2 Diabetes with Metformin 500mg daily.",
                similarity_score=0.95,
                distance=0.05,
                relevance_score=0.98,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="The primary diagnosis is Type 2 Diabetes and prescribed Metformin 500mg daily.",
        latency_ms=120.0,
        success=True,
    )

    mock_adapter = MagicMock()
    mock_adapter.name = "Gemini 3.5 Flash-Lite"
    mock_adapter.generate = AsyncMock(return_value=mock_llm_response)
    mock_router = MagicMock()
    mock_router.get_adapter.return_value = mock_adapter

    return mock_retrieval, mock_router


@pytest.fixture
def test_users(db_session: Session):
    """Create test users for role-based governance testing."""
    dev_user = User(
        id="user_dev_123",
        name="Developer Tester",
        email="developer@agentforge.ai",
        password_hash="hashedpassword",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    normal_user = User(
        id="user_clinician_123",
        name="Doctor User",
        email="doctor@agentforge.ai",
        password_hash="hashedpassword",
        role=UserRole.USER,
        is_active=True,
    )
    db_session.merge(dev_user)
    db_session.merge(normal_user)
    db_session.commit()
    return dev_user, normal_user


@pytest.fixture
def parent_workflows(db_session: Session):
    """Seed distinct approved workflow versions for crossover and lifecycle testing."""
    base_wf = get_baseline_workflow()

    # Create v1 (Parent A)
    v1_cfg = base_wf.to_dict()
    v1_cfg["retrieval"]["top_k"] = 3
    v1_cfg["model"]["temperature"] = 0.0
    v1 = WorkflowVersion(
        version_id="wf_v1_lifecycle_test",
        version_number=101,
        name="Version 1: Top-K 3 Deterministic",
        description="Parent A with k=3",
        source_run_id="run_v1",
        source_candidate_id="cand_v1",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=v1_cfg,
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.000030, "llm_latency_ms": 1200.0},
        status="APPROVED",
        approved_by="developer@agentforge.ai",
    )

    # Create v2 (Parent B)
    v2_cfg = base_wf.to_dict()
    v2_cfg["retrieval"]["top_k"] = 7
    v2_cfg["execution"]["context_compression"] = True
    v2 = WorkflowVersion(
        version_id="wf_v2_lifecycle_test",
        version_number=102,
        name="Version 2: Context Compression",
        description="Parent B with compression",
        source_run_id="run_v2",
        source_candidate_id="cand_v2",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=v2_cfg,
        evaluation_snapshot={"groundedness": 0.95, "cost": 0.000025, "llm_latency_ms": 1100.0},
        status="APPROVED",
        approved_by="developer@agentforge.ai",
    )

    db_session.merge(v1)
    db_session.merge(v2)
    db_session.commit()
    return v1, v2


# ===========================================================================
# 1. Crossover Integration in AgentEvo Optimizer
# ===========================================================================

@pytest.mark.asyncio
async def test_crossover_strategy_via_agent_evo_optimizer(db_session, parent_workflows, mock_evaluation_env):
    """Test optimizer natively running candidate generation strategy='crossover'."""
    v1, v2 = parent_workflows
    mock_retrieval, mock_router = mock_evaluation_env
    optimizer = AgentEvoOptimizer()

    with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

        report = await optimizer.run_optimization(
            query="What is the diagnosis and prescribed medication?",
            user_id="user_dev_123",
            db=db_session,
            strategy="crossover",
            parent_workflow_id=v1.version_id,
            parent_b_workflow_id=v2.version_id,
            crossover_map={
                "retrieval": "parent_a",
                "prompt": "parent_b",
                "model": "parent_a",
                "generation": "parent_a",
                "operators": "parent_b",
            },
            post_crossover_mutation="none",
        )

        assert report is not None
        assert report.total_candidates_evaluated >= 1
        crossover_cands = [
            c for c in report.all_candidates
            if (c.mutation_metadata and c.mutation_metadata.mutation_type in ["crossover", "CROSSOVER"])
            or (c.workflow.mutation_metadata and c.workflow.mutation_metadata.mutation_type in ["crossover", "CROSSOVER"])
        ]
        assert len(crossover_cands) >= 1
        child = crossover_cands[0]
        # Verify inheritance: retrieval from Parent A (top_k=3), operators from Parent B (compression=True)
        assert child.workflow.retrieval.top_k == 3
        assert child.workflow.execution.context_compression is True
        # Verify child participated in Pareto evaluation
        assert child.metrics.groundedness is not None
        assert child.metrics.cost > 0


@pytest.mark.asyncio
async def test_crossover_identical_parents_rejected(db_session, parent_workflows):
    """Test optimizer rejects crossover with identical Parent A and Parent B."""
    v1, _ = parent_workflows
    optimizer = AgentEvoOptimizer()

    with pytest.raises(HTTPException) as exc:
        await optimizer.run_optimization(
            query="Test query",
            user_id="user_dev_123",
            db=db_session,
            strategy="crossover",
            parent_workflow_id=v1.version_id,
            parent_b_workflow_id=v1.version_id,
        )
    assert exc.value.status_code == 400
    assert "distinct" in str(exc.value.detail).lower() or "cannot be the same" in str(exc.value.detail).lower()


# ===========================================================================
# 2. Candidate Approval → Immutable Workflow Version
# ===========================================================================

@pytest.mark.asyncio
async def test_candidate_approval_creates_immutable_approved_version(db_session, parent_workflows, mock_evaluation_env):
    """Test developer approval creates an immutable version with status APPROVED and preserves configuration."""
    v1, v2 = parent_workflows
    mock_retrieval, mock_router = mock_evaluation_env
    optimizer = AgentEvoOptimizer()

    with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

        report = await optimizer.run_optimization(
            query="Evaluate antibiotic dosing",
            user_id="user_dev_123",
            db=db_session,
            strategy="crossover",
            parent_workflow_id=v1.version_id,
            parent_b_workflow_id=v2.version_id,
        )

        child_cand = report.all_candidates[0]
        child_cand.is_pareto_optimal = True
        _, approved_ver = optimizer.approve_workflow(
            run_id=report.run_id,
            candidate_id=child_cand.candidate_id,
            approved_by="developer@agentforge.ai",
            db=db_session,
        )

        assert approved_ver.version_id.startswith("wf_v")
        assert approved_ver.status in ["APPROVED", "approved"]
        assert approved_ver.approved_by == "developer@agentforge.ai"
        assert approved_ver.approved_at is not None
        assert approved_ver.source_candidate_id == child_cand.candidate_id
        assert approved_ver.configuration["retrieval"]["top_k"] == child_cand.workflow.retrieval.top_k

        # Invariant: Approval does NOT automatically activate
        active_wf = WorkflowLifecycleService.get_active_workflow(db=db_session)
        assert active_wf.active_version_id != approved_ver.version_id


# ===========================================================================
# 3. Production Activation & Rollback Governance
# ===========================================================================

def test_production_activation_and_deactivation_lifecycle(db_session, parent_workflows):
    """Test explicit activation activates target version and marks previous active as INACTIVE."""
    v1, v2 = parent_workflows

    # 1. Activate v1
    activated_ver, prev_id = WorkflowLifecycleService.activate_version(
        version_id=v1.version_id,
        user_id="user_dev_123",
        user_email="developer@agentforge.ai",
        db=db_session,
    )
    assert activated_ver.status == "ACTIVE"
    assert activated_ver.version_id == v1.version_id

    # Verify DB status
    v1_db = db_session.query(WorkflowVersion).filter_by(version_id=v1.version_id).first()
    assert v1_db.status == "ACTIVE"
    assert v1_db.activated_at is not None

    # Verify Active Workflow resolver returns v1
    active = WorkflowLifecycleService.get_active_workflow(db=db_session)
    assert active.active_version_id == v1.version_id
    assert active.configuration["retrieval"]["top_k"] == 3

    # 2. Now activate v2
    activated_ver2, prev_id2 = WorkflowLifecycleService.activate_version(
        version_id=v2.version_id,
        user_id="user_dev_123",
        user_email="developer@agentforge.ai",
        db=db_session,
    )
    assert activated_ver2.version_id == v2.version_id
    assert prev_id2 == v1.version_id

    # Invariant: Previous version becomes INACTIVE and is never deleted
    v1_db = db_session.query(WorkflowVersion).filter_by(version_id=v1.version_id).first()
    v2_db = db_session.query(WorkflowVersion).filter_by(version_id=v2.version_id).first()
    assert v1_db.status == "INACTIVE"
    assert v2_db.status == "ACTIVE"


def test_rollback_governed_reactivation(db_session, parent_workflows):
    """Test rollback reactivates an earlier approved/inactive version and records audit trail."""
    v1, v2 = parent_workflows

    # Activate v1 then v2
    WorkflowLifecycleService.activate_version(v1.version_id, "user_dev_123", "developer@agentforge.ai", db=db_session)
    WorkflowLifecycleService.activate_version(v2.version_id, "user_dev_123", "developer@agentforge.ai", db=db_session)

    # Rollback to v1
    rolled_ver, prev_active_id = WorkflowLifecycleService.rollback_version(
        target_version_id=v1.version_id,
        user_id="user_dev_123",
        user_email="developer@agentforge.ai",
        reason="Rollback due to testing requirement",
        db=db_session,
    )

    assert rolled_ver.status == "ACTIVE"
    assert rolled_ver.version_id == v1.version_id
    assert prev_active_id == v2.version_id

    v1_db = db_session.query(WorkflowVersion).filter_by(version_id=v1.version_id).first()
    v2_db = db_session.query(WorkflowVersion).filter_by(version_id=v2.version_id).first()
    assert v1_db.status == "ACTIVE"
    assert v2_db.status == "INACTIVE"

    # Verify audit event was logged
    audit_events = WorkflowLifecycleService.list_audit_events(db=db_session)
    rollback_events = [e for e in audit_events if e.event_type == "ROLLBACK"]
    assert len(rollback_events) >= 1
    assert rollback_events[0].workflow_version_id == v1.version_id
    assert rollback_events[0].details.get("previous_active_version_id") == v2.version_id


# ===========================================================================
# 4. Production User Execution & Metrics Logging
# ===========================================================================

@pytest.mark.asyncio
async def test_user_execution_uses_active_workflow_and_logs_telemetry(db_session, parent_workflows, mock_evaluation_env):
    """Test execution loads exact active immutable config and logs separate latency & execution time."""
    v1, _ = parent_workflows
    mock_retrieval, mock_router = mock_evaluation_env
    WorkflowLifecycleService.activate_version(v1.version_id, "user_dev_123", "developer@agentforge.ai", db=db_session)

    with patch("app.services.agent_evo.workflow_lifecycle_service.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.workflow_lifecycle_service.get_llm_router", return_value=mock_router):

        exec_result = await WorkflowLifecycleService.execute_active_workflow(
            query="What is the patient diagnosis?",
            user_id="user_doctor_1",
            db=db_session,
        )

        assert exec_result.workflow_version_id == v1.version_id
        assert exec_result.execution_id is not None
        assert exec_result.total_tokens > 0
        assert exec_result.cost > 0
        assert exec_result.llm_latency_ms > 0
        assert exec_result.total_execution_time_ms >= exec_result.llm_latency_ms

        # Check persistence in workflow_executions
        persisted = db_session.query(WorkflowExecution).filter_by(execution_id=exec_result.execution_id).first()
        assert persisted is not None
        assert persisted.workflow_version_id == v1.version_id
        assert persisted.llm_latency_ms == exec_result.llm_latency_ms
        assert persisted.total_execution_time_ms == exec_result.total_execution_time_ms


# ===========================================================================
# 5. Feedback Logging (No Auto-Optimization)
# ===========================================================================

@pytest.mark.asyncio
async def test_feedback_recorded_without_modifying_workflow_or_auto_triggering(db_session, parent_workflows, mock_evaluation_env):
    """Test feedback records against execution without modifying workflow configuration."""
    v1, _ = parent_workflows
    mock_retrieval, mock_router = mock_evaluation_env
    WorkflowLifecycleService.activate_version(v1.version_id, "user_dev_123", "developer@agentforge.ai", db=db_session)

    with patch("app.services.agent_evo.workflow_lifecycle_service.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.workflow_lifecycle_service.get_llm_router", return_value=mock_router):

        exec_result = await WorkflowLifecycleService.execute_active_workflow(
            query="Verify medication contraindications",
            user_id="user_doctor_1",
            db=db_session,
        )

        cfg_before = dict(v1.configuration)

        fb_result = WorkflowLifecycleService.record_feedback(
            execution_id=exec_result.execution_id,
            user_id="user_doctor_1",
            rating=5,
            feedback_text="Accurate diagnosis and grounded answers.",
            db=db_session,
        )

        assert fb_result.feedback_id is not None
        assert fb_result.execution_id == exec_result.execution_id
        assert fb_result.workflow_version_id == v1.version_id
        assert fb_result.rating == 5

        # Verify workflow configuration remains strictly immutable
        v1_db = db_session.query(WorkflowVersion).filter_by(version_id=v1.version_id).first()
        assert v1_db.configuration == cfg_before
        assert v1_db.status == "ACTIVE"


# ===========================================================================
# 6. API Endpoints & Role Governance
# ===========================================================================

def test_api_role_governance_on_activation(db_session, parent_workflows, test_users):
    """Test that only developers can activate/rollback, and unauthorized requests are rejected."""
    v1, _ = parent_workflows
    dev_user, normal_user = test_users
    client = TestClient(app)

    from app.api.auth import get_current_user

    # Clinician user attempts activation -> 403 Forbidden
    app.dependency_overrides[get_current_user] = lambda: normal_user
    try:
        res = client.post(f"/api/agent-evo/versions/{v1.version_id}/activate", json={"reason": "Clinician attempt"})
        assert res.status_code == 403
    finally:
        # Developer user attempts activation -> 200 OK
        app.dependency_overrides[get_current_user] = lambda: dev_user
        res_dev = client.post(f"/api/agent-evo/versions/{v1.version_id}/activate", json={"reason": "Dev activation"})
        assert res_dev.status_code == 200
        app.dependency_overrides.clear()
