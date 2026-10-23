import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.main import app
from app.models.user import User, UserRole
from app.models.workflow_version import (
    WorkflowAuditEvent,
    WorkflowExecution,
    WorkflowVersion,
)
from app.schemas.rag import RAGSearchResponse, RetrievedChunk
from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector
from app.services.agent_evo.workflow_lifecycle_service import WorkflowLifecycleService
from app.services.llm.base import LLMResponse


@pytest.fixture
def mock_retrieval_two_chunks():
    """Mock RAG retrieval returning 2 chunks, matching the observed execution trace."""
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_assessment_01",
            document_id="doc_patient_1",
            chunk_index=5,
            content="Assessment: findings are consistent with stable angina, likely related to underlying coronary artery disease.",
            section_title="Assessment",
            similarity_score=0.4105,
            relevance_score=0.7596,
        ),
        RetrievedChunk(
            chunk_id="chunk_hpi_02",
            document_id="doc_patient_1",
            chunk_index=2,
            content="History of Present Illness: chest discomfort with exertion for 3 months.",
            section_title="History of Present Illness",
            similarity_score=0.3999,
            relevance_score=0.4763,
        ),
    ]
    return RAGSearchResponse(
        query="What is the disease",
        status="success",
        total_results=2,
        results=chunks,
    )


@pytest.fixture
def mock_gemini_adapter():
    """Mock Gemini 3.5 Flash-Lite LLM adapter."""
    mock_resp = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        answer="The findings are consistent with stable angina, likely related to underlying coronary artery disease.",
        latency_ms=1840.0,
        success=True,
        raw_usage={"prompt_tokens": 412, "completion_tokens": 27, "total_tokens": 439},
    )
    adapter = MagicMock()
    adapter.name = "Gemini 3.5 Flash-Lite"
    adapter.model_id = "gemini-3.5-flash-lite"
    adapter.provider = "Google"
    adapter.is_configured = MagicMock(return_value=True)
    adapter.generate = AsyncMock(return_value=mock_resp)

    router = MagicMock()
    router.get_adapter.return_value = adapter
    return router, adapter


@pytest.fixture
def test_developer_user(db_session: Session):
    user = User(
        id="user_dev_investigator",
        name="Lead Investigator",
        email="lead.investigator@agentforge.ai",
        password_hash="secure_hash",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.merge(user)
    db_session.commit()
    return user


@pytest.fixture
def seeded_versions(db_session: Session):
    """
    Seed v12 and v13 with the exact parameters from the incident:
    Both have task_category 'diagnosis', groundedness=1.0, cost=0.000048.
    v12 is APPROVED.
    v13 is initially APPROVED (to test explicit activation).
    Also seed a historical execution record exec_1792736702035 with v12.
    """
    v12_config = {
        "workflow_id": "cand_prompt_detailed_162",
        "name": "Detailed Clinical Synthesis Prompt",
        "retrieval": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 5,
            "similarity_threshold": 0.35,
            "reranking": True,
        },
        "model": {
            "provider": "Google",
            "model_id": "gemini-3.5-flash-lite",
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        "prompt": {
            "template_name": "detailed_clinical",
            "custom_instructions": None,
        },
        "execution": {
            "timeout_seconds": 30.0,
            "context_compression": False,
        },
        "task_categories": ["diagnosis"],
    }
    v12_eval = {
        "groundedness": 1.0,
        "cost": 0.000048,
        "llm_latency_ms": 2868.19,
        "evaluated_task_categories": ["diagnosis"],
    }

    v13_config = {
        "workflow_id": "cand_prompt_detailed_134",
        "name": "Detailed Clinical Synthesis Prompt",
        "retrieval": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 5,
            "similarity_threshold": 0.35,
            "reranking": True,
        },
        "model": {
            "provider": "Google",
            "model_id": "gemini-3.5-flash-lite",
            "temperature": 0.1,
            "max_tokens": 1024,
        },
        "prompt": {
            "template_name": "detailed_clinical",
            "custom_instructions": None,
        },
        "execution": {
            "timeout_seconds": 30.0,
            "context_compression": False,
        },
        "task_categories": ["diagnosis"],
    }
    v13_eval = {
        "groundedness": 1.0,
        "cost": 0.000048,
        "llm_latency_ms": 1848.92,
        "evaluated_task_categories": ["diagnosis"],
    }

    v12 = WorkflowVersion(
        id="v12_uuid",
        version_id="wf_v12_b58eec",
        version_number=12,
        name="Detailed Clinical Synthesis Prompt",
        source_run_id="evo_b1d4a187e7",
        source_candidate_id="cand_prompt_detailed_162",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=v12_config,
        evaluation_snapshot=v12_eval,
        status="APPROVED",
        approved_by="developer@agentforge.ai",
    )

    v13 = WorkflowVersion(
        id="v13_uuid",
        version_id="wf_v13_49bdc8",
        version_number=13,
        name="Detailed Clinical Synthesis Prompt",
        source_run_id="evo_c456d32aae",
        source_candidate_id="cand_prompt_detailed_134",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=v13_config,
        evaluation_snapshot=v13_eval,
        status="APPROVED",
        approved_by="developer@agentforge.ai",
    )

    hist_exec = WorkflowExecution(
        execution_id="exec_1792736702035",
        workflow_version_id="wf_v12_b58eec",
        user_id="user_dev_investigator",
        query="What is the disease",
        response="Angina",
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        input_tokens=412,
        output_tokens=27,
        total_tokens=439,
        cost=0.000048,
        llm_latency_ms=2311.09,
        total_execution_time_ms=3811.56,
        groundedness=1.0,
        execution_trace={"wf_version_id": "wf_v12_b58eec", "stages": {}},
        status="success",
    )

    v12 = db_session.merge(v12)
    v13 = db_session.merge(v13)
    hist_exec = db_session.merge(hist_exec)
    db_session.commit()
    return v12, v13, hist_exec


@pytest.mark.asyncio
async def test_full_investigation_sequence(
    client: TestClient,
    db_session: Session,
    test_developer_user: User,
    seeded_versions,
    mock_retrieval_two_chunks,
    mock_gemini_adapter,
):
    """
    Test the complete sequence required by Task 10:
    1. Activate v13 explicitly.
    2. Read the active workflow back from the backend.
    3. Execute a single-model production query.
    4. Verify that execution trace records v13 and that its actual configuration was used.
    5. Refresh/reload the frontend endpoint and verify that v13 remains active.
    6. Verify that historical executions retain their original workflow IDs.
    """
    v12, v13, hist_exec = seeded_versions
    mock_router, mock_adapter = mock_gemini_adapter
    app.dependency_overrides[get_current_user] = lambda: test_developer_user

    try:
        # -------------------------------------------------------------------
        # Step 1: Explicitly activate v13
        # -------------------------------------------------------------------
        act_resp = client.post(f"/api/agent-evo/versions/{v13.version_id}/activate")
        assert act_resp.status_code == 200, act_resp.text
        act_data = act_resp.json()
        assert act_data["status"] == "success"
        assert act_data["version_id"] == "wf_v13_49bdc8"
        assert act_data["version"]["status"] == "ACTIVE"
        assert act_data["version"]["version_number"] == 13

        # Confirm database persistence of activation
        v13_db = db_session.query(WorkflowVersion).filter_by(version_id=v13.version_id).first()
        assert v13_db.status == "ACTIVE"
        assert v13_db.activated_at is not None

        # Confirm audit event was recorded
        audit_events = db_session.query(WorkflowAuditEvent).filter_by(
            workflow_version_id=v13.version_id,
            event_type="ACTIVATION",
        ).all()
        assert len(audit_events) == 1
        assert audit_events[0].user_email == "lead.investigator@agentforge.ai"

        # -------------------------------------------------------------------
        # Step 2: Read active workflow back from the backend
        # -------------------------------------------------------------------
        active_resp = client.get("/api/agent-evo/active")
        assert active_resp.status_code == 200
        active_data = active_resp.json()
        assert active_data["is_active_set"] is True
        assert active_data["active_version_id"] == "wf_v13_49bdc8"
        assert active_data["version_number"] == 13
        assert active_data["status"] == "ACTIVE"
        assert active_data["configuration"]["retrieval"]["top_k"] == 5
        assert active_data["configuration"]["retrieval"]["similarity_threshold"] == 0.35
        assert active_data["configuration"]["retrieval"]["reranking"] is True
        assert active_data["configuration"]["model"]["model_id"] == "gemini-3.5-flash-lite"
        assert active_data["configuration"]["prompt"]["template_name"] == "detailed_clinical"

        # -------------------------------------------------------------------
        # Step 3: Execute a single-model production query
        # -------------------------------------------------------------------
        with patch("app.api.query.execute_rag_retrieval", return_value=mock_retrieval_two_chunks), \
             patch("app.api.query.get_llm_router", return_value=mock_router), \
             patch("app.services.agent_evo.workflow_lifecycle_service.execute_rag_retrieval", return_value=mock_retrieval_two_chunks), \
             patch("app.services.agent_evo.workflow_lifecycle_service.get_llm_router", return_value=mock_router):

            query_resp = client.post(
                "/api/query/answer",
                json={
                    "query": "What is the disease",
                    "top_k": 5,
                },
            )
            assert query_resp.status_code == 200, query_resp.text
            query_data = query_resp.json()

            # -------------------------------------------------------------------
            # Step 4: Verify execution trace records v13 & actual config was used
            # -------------------------------------------------------------------
            assert query_data["workflow_version_id"] == "wf_v13_49bdc8"
            assert query_data["pipeline_type"] == "production"
            assert query_data["task_category"] == "diagnosis"
            assert "wf_v13_49bdc8" in query_data["selection_reason"]

            trace = query_data["execution_trace"]
            assert trace is not None
            retrieval_stage = trace["stages"]["retrieval"]
            assert retrieval_stage["top_k"] == 5
            assert retrieval_stage["similarity_threshold"] == 0.35
            assert retrieval_stage["reranking_enabled"] is True
            assert retrieval_stage["chunks_retrieved"] == 2

            prompt_stage = trace["stages"]["prompt"]
            assert prompt_stage["template"] == "detailed_clinical"

            llm_stage = trace["stages"]["llm_generation"]
            assert llm_stage["model_id"] == "gemini-3.5-flash-lite"
            assert llm_stage["provider"] == "Google"
            assert llm_stage["temperature"] == 0.1

            effective_config = trace["effective_configuration"]
            assert effective_config["retrieval"]["top_k"] == 5
            assert effective_config["retrieval"]["similarity_threshold"] == 0.35
            assert effective_config["retrieval"]["reranking"] is True
            assert effective_config["prompt"]["template_name"] == "detailed_clinical"
            assert effective_config["model"]["model_id"] == "gemini-3.5-flash-lite"

            # Check DB record for this new execution
            new_exec = db_session.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id != hist_exec.execution_id
            ).order_by(WorkflowExecution.created_at.desc()).first()
            assert new_exec is not None
            assert new_exec.workflow_version_id == "wf_v13_49bdc8"
            assert new_exec.query == "What is the disease"
            assert len(new_exec.retrieved_chunks) == 2

        # -------------------------------------------------------------------
        # Step 5: Refresh/reload the frontend and verify v13 remains active
        # -------------------------------------------------------------------
        refresh_resp = client.get("/api/agent-evo/active")
        assert refresh_resp.status_code == 200
        refreshed_data = refresh_resp.json()
        assert refreshed_data["active_version_id"] == "wf_v13_49bdc8"
        assert refreshed_data["version_number"] == 13
        assert refreshed_data["status"] == "ACTIVE"

        # List executions and verify history is intact
        exec_list_resp = client.get("/api/agent-evo/executions?limit=10")
        assert exec_list_resp.status_code == 200
        exec_list = exec_list_resp.json()["executions"]
        assert len(exec_list) >= 2

        # -------------------------------------------------------------------
        # Step 6: Verify historical executions retain their original workflow IDs
        # -------------------------------------------------------------------
        db_hist_exec = db_session.query(WorkflowExecution).filter_by(
            execution_id=hist_exec.execution_id
        ).first()
        assert db_hist_exec is not None
        assert db_hist_exec.workflow_version_id == "wf_v12_b58eec"
        assert db_hist_exec.query == "What is the disease"

        # Verify in executions endpoint list as well
        hist_in_api = next((e for e in exec_list if e["execution_id"] == hist_exec.execution_id), None)
        assert hist_in_api is not None
        assert hist_in_api["workflow_version_id"] == "wf_v12_b58eec"

    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_runtime_selector_tie_breaker_logic(db_session: Session, seeded_versions):
    """
    Test direct selector tie-breaker:
    When v12 (v12, approved) and v13 (v13, active) both match category 'diagnosis'
    with identical Pareto scores (groundedness=1.0, cost=0.000048):
    1. Active status must break the tie in favor of v13.
    2. If both were approved (neither active), higher version number (13 > 12) breaks tie.
    """
    v12, v13, _ = seeded_versions

    # Case A: v13 is ACTIVE, v12 is approved
    v13.status = "ACTIVE"
    v12.status = "APPROVED"
    db_session.commit()

    sel = RuntimeWorkflowSelector.select_workflow_for_query("What is the disease", db=db_session)
    assert sel.version_id == "wf_v13_49bdc8"
    assert sel.version_number == 13
    assert sel.status == "ACTIVE"

    # Case B: Both are APPROVED (neither ACTIVE)
    v13.status = "APPROVED"
    v12.status = "APPROVED"
    db_session.commit()

    sel2 = RuntimeWorkflowSelector.select_workflow_for_query("What is the disease", db=db_session)
    assert sel2.version_id == "wf_v13_49bdc8"  # higher version number wins
    assert sel2.version_number == 13

    # Case C: v12 is explicitly ACTIVE
    v12.status = "ACTIVE"
    v13.status = "INACTIVE"
    db_session.commit()

    sel3 = RuntimeWorkflowSelector.select_workflow_for_query("What is the disease", db=db_session)
    assert sel3.version_id == "wf_v12_b58eec"
    assert sel3.version_number == 12
    assert sel3.status == "ACTIVE"
