import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.main import app
from app.api.auth import get_current_user
from app.models.user import User, UserRole
from app.models.workflow_version import (
    WorkflowAuditEvent,
    WorkflowExecution,
    WorkflowVersion,
)
from app.schemas.rag import RAGSearchResponse, RetrievedChunk
from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector
from app.services.agent_evo.workflow_lifecycle_service import WorkflowLifecycleService
from app.services.evaluation.groundedness import (
    ClaimStatus,
    calculate_groundedness,
    evaluate_claim_support,
)
from app.services.llm.base import LLMResponse


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_user(db_session: Session):
    user = User(
        id="dev_user_validation_audit",
        name="Audit Evaluator",
        email="audit_eval@agentforge.dev",
        password_hash="mock_hash",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.merge(user)
    db_session.commit()
    return user


def test_groundedness_empty_evidence_guardrail():
    """Verify missing evidence guardrail returns INSUFFICIENT_EVIDENCE and not hallucination."""
    claim = "Patient has severe hypertension"
    eval_claim = evaluate_claim_support(claim, context="")
    assert eval_claim.status == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert "empty or missing evidence" in eval_claim.reason.lower()


def test_groundedness_zero_claims_returns_none():
    """Verify calculate_groundedness returns None when zero factual claims are extracted."""
    result = calculate_groundedness(
        generated_answer="Thank you for your question.",
        retrieved_context="Clinical notes here.",
    )
    assert result.groundedness is None
    assert result.total_claims == 0
    assert result.supported_claims == 0
    assert result.unsupported_claims == 0


def test_active_v10_runtime_enforcement(db_session: Session):
    """Verify RuntimeWorkflowSelector resolves active v10 with top_k=3 and slices correctly."""
    v10 = WorkflowVersion(
        id="wf_v10_test_enforce",
        version_id="wf_v10_test_enforce",
        version_number=10,
        name="Retrieval Reduction v10",
        source_run_id="run_test",
        source_candidate_id="cand_test",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        status="ACTIVE",
        approved_by="admin@agentforge.dev",
        configuration={
            "retrieval": {"top_k": 3, "similarity_threshold": 0.40},
            "model": {"model_id": "gemini-3.5-flash-lite"},
            "prompt": {"template_name": "concise"},
        },
        evaluation_snapshot={"groundedness": 0.95, "total_cost": 0.0001},
    )
    db_session.merge(v10)
    db_session.commit()

    selection = RuntimeWorkflowSelector.select_workflow_for_query(
        query="What is the clinical diagnosis?",
        db=db_session,
    )
    assert selection.version_id == "wf_v10_test_enforce"
    assert selection.configuration["retrieval"]["top_k"] == 3

    # Test baseline isolation override
    baseline_selection = RuntimeWorkflowSelector.select_workflow_for_query(
        query="What is the clinical diagnosis?",
        db=db_session,
        candidate_version_id_override="baseline",
    )
    assert baseline_selection.version_id == "baseline"
    assert baseline_selection.configuration["retrieval"]["top_k"] == 5


@pytest.mark.asyncio
async def test_workflow_lifecycle_null_metrics_on_failure(db_session: Session):
    """Verify execute_active_workflow stores None (NULL) metrics when LLM fails."""
    v10 = WorkflowVersion(
        id="wf_v10_test_failure",
        version_id="wf_v10_test_failure",
        version_number=10,
        name="Active v10 Failure Test",
        source_run_id="run_test",
        source_candidate_id="cand_test",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        status="ACTIVE",
        approved_by="admin@agentforge.dev",
        configuration={
            "retrieval": {"top_k": 3},
            "model": {"model_id": "gemini-3.5-flash-lite"},
        },
        evaluation_snapshot={"groundedness": 0.95, "total_cost": 0.0001},
    )
    db_session.merge(v10)
    db_session.commit()

    failed_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="API rate limit exceeded.",
        latency_ms=0.0,
        success=False,
        error="429 Rate Limit Exceeded",
        raw_usage=None,
    )

    with patch("app.services.agent_evo.workflow_lifecycle_service.execute_rag_retrieval") as mock_rag, \
         patch("app.services.agent_evo.workflow_lifecycle_service.get_llm_router") as mock_router_getter:
        mock_rag.return_value = RAGSearchResponse(query="Test query", status="success", total_results=0, results=[])
        mock_router = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.name = "Gemini 3.5 Flash-Lite"
        mock_adapter.provider = "Google"
        mock_adapter.model_id = "gemini-3.5-flash-lite"
        mock_adapter.generate = AsyncMock(return_value=failed_response)
        mock_router.get_adapter.return_value = mock_adapter
        mock_router_getter.return_value = mock_router

        exec_record = await WorkflowLifecycleService.execute_active_workflow(
            user_id="dev_user_validation_audit",
            query="Test failing query",
            db=db_session,
        )

        assert exec_record.status in ("error", "FAILED")
        assert exec_record.input_tokens is None
        assert exec_record.output_tokens is None
        assert exec_record.total_tokens is None
        assert exec_record.cost is None
        assert exec_record.llm_latency_ms is None


def test_audit_events_retrieval_and_serialization(db_session: Session):
    """Verify list_audit_events returns triggered_by, previous_version_id, and schema alias."""
    event = WorkflowAuditEvent(
        id="audit_event_test_123",
        event_id="evt_audit_test_123",
        event_type="ACTIVATION",
        workflow_version_id="wf_v10_test_enforce",
        performed_by="dev_user_validation_audit",
        user_email="audit_eval@agentforge.dev",
        details={"reason": "Activated v10", "previous_active_version_id": "wf_v9_prev"},
    )
    db_session.merge(event)
    db_session.commit()

    events = WorkflowLifecycleService.list_audit_events(db=db_session, limit=10)
    assert len(events) >= 1
    found = next((e for e in events if e.id == "audit_event_test_123"), None)
    assert found is not None
    assert found.triggered_by == "audit_eval@agentforge.dev"
    assert found.previous_version_id == "wf_v9_prev"


def test_query_answer_enforces_v10_and_honest_metrics(client: TestClient, db_session: Session, test_user: User):
    """Verify POST /api/query/answer selects active v10, slices to top_k=3, and labels provenance."""
    app.dependency_overrides[get_current_user] = lambda: test_user
    try:
        v10 = WorkflowVersion(
            id="wf_v10_qanswer_active",
            version_id="wf_v10_qanswer_active",
            version_number=10,
            name="Production Retrieval Reduction",
            source_run_id="run_qa",
            source_candidate_id="cand_qa",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            status="ACTIVE",
            approved_by="admin@agentforge.dev",
            configuration={
                "retrieval": {"top_k": 3, "similarity_threshold": 0.35},
                "model": {"model_id": "gemini-3.5-flash-lite"},
                "prompt": {"template_name": "default"},
            },
            evaluation_snapshot={"groundedness": 0.95, "total_cost": 0.0001},
        )
        db_session.merge(v10)
        db_session.commit()

        # 5 retrieval chunks returned by RAG
        mock_chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{i}",
                document_id="doc_audit_1",
                chunk_index=i,
                content=f"Medical finding chunk {i} discussing diabetes.",
                section_title=f"Section {i}",
                similarity_score=0.9 - (i * 0.05),
                relevance_score=0.95 - (i * 0.05),
            )
            for i in range(5)
        ]
        mock_rag = RAGSearchResponse(
            query="What is the diagnosis?",
            status="success",
            total_results=5,
            results=mock_chunks,
        )

        mock_llm = LLMResponse(
            model="Gemini 3.5 Flash-Lite",
            provider="Google",
            answer="The diagnosis is diabetes based on findings.",
            latency_ms=120.0,
            success=True,
            raw_usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
        )

        with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag), \
             patch("app.api.query.get_llm_router") as mock_router_getter:
            mock_router = MagicMock()
            mock_adapter = MagicMock()
            mock_adapter.name = "Gemini 3.5 Flash-Lite"
            mock_adapter.provider = "Google"
            mock_adapter.model_id = "gemini-3.5-flash-lite"
            mock_adapter.is_configured = MagicMock(return_value=True)
            mock_adapter.generate = AsyncMock(return_value=mock_llm)
            mock_router.get_adapter.return_value = mock_adapter
            mock_router_getter.return_value = mock_router

            resp = client.post(
                "/api/query/answer",
                json={"query": "What is the diagnosis?", "top_k": 5},
            )
            assert resp.status_code == 200
            data = resp.json()

            # Enforced active v10 top_k=3: sources sliced to 3 despite request having top_k=5
            assert len(data["sources"]) == 3
            assert data["workflow_version_id"] == "wf_v10_qanswer_active"
            assert data["pipeline_type"] == "production"
            assert data["workflow_name"] == "Production Retrieval Reduction"
            assert data["execution_trace"]["stages"]["retrieval"]["top_k"] == 3
            assert data["execution_trace"]["stages"]["retrieval"]["configured_top_k"] == 3
            assert data["execution_trace"]["stages"]["retrieval"]["chunks_retrieved"] == 3
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_query_compare_provenance_and_shared_context(client: TestClient, db_session: Session, test_user: User):
    """Verify POST /api/query/compare labels pipeline as comparative baseline and provides per-model trace."""
    app.dependency_overrides[get_current_user] = lambda: test_user
    try:
        mock_rag = RAGSearchResponse(
            query="What are the symptoms?",
            status="success",
            total_results=2,
            results=[
                RetrievedChunk(
                    chunk_id="chunk_cmp_1",
                    document_id="doc_cmp_1",
                    chunk_index=0,
                    content="Patient reports headache and fever.",
                    section_title="Symptoms",
                    similarity_score=0.88,
                    relevance_score=0.91,
                )
            ],
        )

        mock_resp = LLMResponse(
            model="Gemini 3.5 Flash-Lite",
            provider="Google",
            answer="Symptoms are headache and fever.",
            latency_ms=110.0,
            success=True,
            raw_usage={"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
        )

        with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag), \
             patch("app.api.query.get_llm_router") as mock_router_getter:
            mock_router = MagicMock()
            mock_router.compare_all = AsyncMock(return_value=[mock_resp, mock_resp, mock_resp])
            mock_router_getter.return_value = mock_router

            resp = client.post(
                "/api/query/compare",
                json={"query": "What are the symptoms?", "top_k": 2},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) == 3
            for item in data["results"]:
                assert item["pipeline_type"] == "comparison_shared_retrieval"
                assert item["workflow_version_id"] is None
                assert item["workflow_name"] == "Multi-LLM Comparative Baseline"
                assert item["context_type"] == "shared"
                assert "execution_trace" in item
                assert item["execution_trace"]["pipeline_type"] == "comparison_shared_retrieval"
    finally:
        app.dependency_overrides.pop(get_current_user, None)
