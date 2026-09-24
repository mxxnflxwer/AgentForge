import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.models.user import User, UserRole
from app.models.workflow_version import WorkflowVersion
from app.services.agent_evo import (
    AgentWorkflow,
    CandidateGenerator,
    CandidateMetrics,
    EvaluatedCandidate,
    EvaluatorAdapter,
    ModelConfig,
    OptimizationReport,
    ParetoArchive,
    PromptConfig,
    RetrievalConfig,
    WorkflowValidator,
    dominates,
    generate_candidates,
    generate_optimization_report,
    get_agent_evo_optimizer,
    get_baseline_workflow,
    validate_workflow,
)
from app.core.security import hash_password
from app.services.evaluation import EvaluationResult
from app.services.llm.base import LLMResponse
from app.schemas.rag import RAGSearchResponse, RetrievedChunk


@pytest.fixture
def evo_test_user(db_session):
    user = User(
        email="agent_evo_tester@agentforge.dev",
        password_hash=hash_password("testpassword123"),
        name="AgentEvo Tester",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


# --- 1. Workflow Representation Tests ---

def test_workflow_representation_defaults():
    wf = get_baseline_workflow()
    assert wf.workflow_id == "baseline"
    assert wf.retrieval.top_k == 5
    assert wf.retrieval.similarity_threshold == 0.35
    assert wf.retrieval.chunk_size == 1000
    assert wf.retrieval.chunk_overlap == 200
    assert wf.model.model_id == "gemini-3.5-flash-lite"
    assert wf.model.provider == "Google"
    assert wf.prompt.template_name == "default_medical"


def test_workflow_serialization_deserialization():
    wf = get_baseline_workflow()
    data = wf.to_dict()
    assert isinstance(data, dict)
    assert data["workflow_id"] == "baseline"
    assert data["retrieval"]["top_k"] == 5

    reconstructed = AgentWorkflow.from_dict(data)
    assert reconstructed.workflow_id == wf.workflow_id
    assert reconstructed.retrieval.top_k == wf.retrieval.top_k
    assert reconstructed.model.model_id == wf.model.model_id


def test_workflow_clone():
    wf = get_baseline_workflow()
    cloned = wf.clone(
        new_id="mutant_01",
        new_name="Mutant 01",
        new_description="High top_k variation",
    )
    cloned.retrieval.top_k = 10

    assert cloned.workflow_id == "mutant_01"
    assert cloned.retrieval.top_k == 10
    # Original remains untouched
    assert wf.workflow_id == "baseline"
    assert wf.retrieval.top_k == 5


# --- 2. Candidate Generator Tests ---

def test_candidate_generator_diversity_and_bounds():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(baseline=baseline, count=5)

    assert len(candidates) == 5
    ids = [c.workflow_id for c in candidates]
    assert len(set(ids)) == 5  # Unique IDs

    # All generated candidates must pass validation
    for cand in candidates:
        res = validate_workflow(cand)
        assert res.is_valid is True, f"Candidate '{cand.workflow_id}' failed validation: {res.errors}"


def test_candidate_generator_respects_count():
    candidates = generate_candidates(count=3)
    assert len(candidates) == 3


# --- 3. Workflow Validator Tests ---

def test_workflow_validator_valid_baseline():
    wf = get_baseline_workflow()
    res = validate_workflow(wf)
    assert res.is_valid is True
    assert res.errors == []


def test_workflow_validator_invalid_top_k():
    wf = get_baseline_workflow()
    wf.retrieval.top_k = 25  # Max is 20
    res = validate_workflow(wf)
    assert res.is_valid is False
    assert any("top_k" in err for err in res.errors)


def test_workflow_validator_invalid_threshold():
    wf = get_baseline_workflow()
    wf.retrieval.similarity_threshold = 1.5  # Max is 1.0
    res = validate_workflow(wf)
    assert res.is_valid is False
    assert any("similarity_threshold" in err for err in res.errors)


def test_workflow_validator_invalid_model():
    wf = get_baseline_workflow()
    wf.model.model_id = "unsupported-gpt-5-future"
    res = validate_workflow(wf)
    assert res.is_valid is False
    assert any("Model" in err for err in res.errors)


def test_workflow_validator_invalid_prompt_template():
    wf = get_baseline_workflow()
    wf.prompt.template_name = "non_existent_template"
    res = validate_workflow(wf)
    assert res.is_valid is False
    assert any("template" in err for err in res.errors)


def test_workflow_validator_invalid_chunk_overlap():
    wf = get_baseline_workflow()
    wf.retrieval.chunk_overlap = 1200  # chunk_size is 1000
    res = validate_workflow(wf)
    assert res.is_valid is False
    assert any("chunk_overlap" in err for err in res.errors)


# --- 4. Pareto Dominance Tests ---

def test_pareto_dominance_higher_perf_same_cost():
    a = CandidateMetrics(workflow_id="a", workflow_name="A", performance=0.90, cost=0.00010)
    b = CandidateMetrics(workflow_id="b", workflow_name="B", performance=0.80, cost=0.00010)
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_pareto_dominance_same_perf_lower_cost():
    a = CandidateMetrics(workflow_id="a", workflow_name="A", performance=0.85, cost=0.00005)
    b = CandidateMetrics(workflow_id="b", workflow_name="B", performance=0.85, cost=0.00010)
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_pareto_dominance_higher_perf_lower_cost():
    a = CandidateMetrics(workflow_id="a", workflow_name="A", performance=0.95, cost=0.00004)
    b = CandidateMetrics(workflow_id="b", workflow_name="B", performance=0.75, cost=0.00012)
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_pareto_dominance_trade_off_neither_dominates():
    a = CandidateMetrics(workflow_id="a", workflow_name="A", performance=0.95, cost=0.00020)
    b = CandidateMetrics(workflow_id="b", workflow_name="B", performance=0.80, cost=0.00005)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_pareto_dominance_equal_metrics():
    a = CandidateMetrics(workflow_id="a", workflow_name="A", performance=0.85, cost=0.00010)
    b = CandidateMetrics(workflow_id="b", workflow_name="B", performance=0.85, cost=0.00010)
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_pareto_dominance_failed_candidate_dominated_by_success():
    a = CandidateMetrics(workflow_id="a", workflow_name="A", performance=0.80, cost=0.00010, status="success")
    b = CandidateMetrics(workflow_id="b", workflow_name="B", performance=0.0, cost=0.0, status="model_error")
    assert dominates(a, b) is True
    assert dominates(b, a) is False


# --- 5. Pareto Archive Tests ---

def test_pareto_archive_incremental_addition_and_pruning():
    archive = ParetoArchive()

    wf_base = get_baseline_workflow()
    c1 = EvaluatedCandidate(
        candidate_id="c1",
        workflow=wf_base,
        metrics=CandidateMetrics(workflow_id="c1", workflow_name="C1", performance=0.80, cost=0.00010, status="success"),
    )
    archive.add_candidate(c1)
    assert len(archive.pareto_candidates) == 1
    assert c1.is_pareto_optimal is True

    # c2 strictly dominates c1 (higher perf, lower cost)
    c2 = EvaluatedCandidate(
        candidate_id="c2",
        workflow=wf_base,
        metrics=CandidateMetrics(workflow_id="c2", workflow_name="C2", performance=0.90, cost=0.00008, status="success"),
    )
    archive.add_candidate(c2)
    assert len(archive.pareto_candidates) == 1
    assert archive.pareto_candidates[0].candidate_id == "c2"
    assert c1.is_pareto_optimal is False
    assert "c2" in c1.dominated_by

    # c3 is a trade-off (higher perf, higher cost) -> should coexist with c2
    c3 = EvaluatedCandidate(
        candidate_id="c3",
        workflow=wf_base,
        metrics=CandidateMetrics(workflow_id="c3", workflow_name="C3", performance=0.98, cost=0.00020, status="success"),
    )
    archive.add_candidate(c3)
    assert len(archive.pareto_candidates) == 2
    assert {c.candidate_id for c in archive.pareto_candidates} == {"c2", "c3"}


# --- 6. Optimization Report Tests ---

def test_optimization_report_generation():
    wf_base = get_baseline_workflow()
    base_cand = EvaluatedCandidate(
        candidate_id="baseline",
        workflow=wf_base,
        metrics=CandidateMetrics(
            workflow_id="baseline",
            workflow_name="Baseline",
            performance=0.85,
            cost=0.00010,
            llm_latency_ms=100.0,
            status="success",
        ),
        is_pareto_optimal=True,
    )

    c1 = EvaluatedCandidate(
        candidate_id="c1",
        workflow=wf_base.clone("c1", "Mutant 1"),
        metrics=CandidateMetrics(
            workflow_id="c1",
            workflow_name="Mutant 1",
            performance=0.90,
            cost=0.00005,
            llm_latency_ms=80.0,
            status="success",
        ),
        is_pareto_optimal=True,
    )

    report = generate_optimization_report(
        run_id="test_run_01",
        query="What is the treatment plan?",
        baseline=base_cand,
        all_candidates=[base_cand, c1],
        pareto_candidates=[c1],
    )

    assert report.run_id == "test_run_01"
    assert report.total_candidates_evaluated == 2
    assert report.pareto_frontier_count == 1
    assert len(report.trade_off_analyses) == 1
    trade_off = report.trade_off_analyses[0]
    assert trade_off.performance_delta_pct > 0
    assert trade_off.cost_delta_pct < 0
    assert "Strict improvement" in trade_off.trade_off_summary


# --- 7. Full AgentEvo Optimizer Pipeline Tests ---

@pytest.mark.asyncio
async def test_agent_evo_optimizer_mocked(evo_test_user, db_session):
    optimizer = get_agent_evo_optimizer()

    mock_retrieval = RAGSearchResponse(
        query="What is the clinical diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="doc_1",
                chunk_index=0,
                content="Assessment: Stable angina pectoris with CAD history.",
                similarity_score=0.92,
                distance=0.08,
                relevance_score=0.95,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="The patient is diagnosed with stable angina pectoris.",
        latency_ms=150.0,
        success=True,
    )

    mock_adapter = MagicMock()
    mock_adapter.name = "Gemini 3.5 Flash-Lite"
    mock_adapter.generate = AsyncMock(return_value=mock_llm_response)
    mock_router = MagicMock()
    mock_router.get_adapter.return_value = mock_adapter

    with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

        report = await optimizer.run_optimization(
            query="What is the clinical diagnosis?",
            user_id=evo_test_user.id,
            document_id="doc_1",
            candidate_count=3,
            db=db_session,
        )

        assert report.run_id.startswith("evo_")
        assert report.total_candidates_evaluated >= 3
        assert report.baseline is not None
        assert len(report.pareto_frontier) >= 1
        assert report.human_approval_required is True

        # Test developer approval on a non-baseline Pareto candidate
        pareto_cand = next((c for c in report.pareto_frontier if c.candidate_id != "baseline"), report.all_candidates[1])
        pareto_cand.is_pareto_optimal = True  # Ensure eligible
        approved_report, version = optimizer.approve_workflow(
            run_id=report.run_id,
            candidate_id=pareto_cand.candidate_id,
            approved_by=evo_test_user.email,
            db=db_session,
        )
        assert approved_report is not None
        assert approved_report.approved_workflow_id == pareto_cand.candidate_id
        assert version is not None
        assert version.source_candidate_id == pareto_cand.candidate_id
        assert version.source_run_id == report.run_id
        assert version.status == "approved"


# --- 8. Phase 7.2 Approval & Versioning API Tests ---

def test_agent_evo_api_endpoints_and_governance(client: TestClient, evo_test_user, db_session):
    # 1. Login user session
    login_res = client.post(
        "/api/auth/login",
        json={"email": evo_test_user.email, "password": "testpassword123"},
    )
    assert login_res.status_code == 200

    mock_retrieval = RAGSearchResponse(
        query="What is the clinical diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                chunk_index=0,
                content="Assessment: Stable angina.",
                similarity_score=0.9,
                distance=0.1,
                relevance_score=0.9,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="The diagnosis is stable angina.",
        latency_ms=120.0,
        success=True,
    )

    mock_adapter = MagicMock()
    mock_adapter.name = "Gemini 3.5 Flash-Lite"
    mock_adapter.generate = AsyncMock(return_value=mock_llm_response)
    mock_router = MagicMock()
    mock_router.get_adapter.return_value = mock_adapter

    with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

        # Trigger optimize endpoint
        res = client.post(
            "/api/agent-evo/optimize",
            json={
                "query": "What is the clinical diagnosis?",
                "candidate_count": 3,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "run_id" in data
        assert "pareto_frontier" in data
        run_id = data["run_id"]

        # List runs
        res_list = client.get("/api/agent-evo/runs")
        assert res_list.status_code == 200
        assert res_list.json()["total_runs"] >= 1

        # Get specific run
        res_run = client.get(f"/api/agent-evo/runs/{run_id}")
        assert res_run.status_code == 200
        assert res_run.json()["run_id"] == run_id

        # Get pareto frontier
        res_pareto = client.get(f"/api/agent-evo/runs/{run_id}/pareto")
        assert res_pareto.status_code == 200
        assert len(res_pareto.json()) >= 1

        # Find eligible non-baseline Pareto candidate
        pareto_cands = [c for c in data["all_candidates"] if c["is_pareto_optimal"] and c["candidate_id"] != "baseline"]
        if not pareto_cands:
            # Force one candidate to be pareto optimal for test coverage
            cand_to_approve = data["all_candidates"][1]["candidate_id"]
            optimizer = get_agent_evo_optimizer()
            run_obj = optimizer.get_run(run_id)
            for c in run_obj.all_candidates:
                if c.candidate_id == cand_to_approve:
                    c.is_pareto_optimal = True
        else:
            cand_to_approve = pareto_cands[0]["candidate_id"]

        # 1. Test Approval of Pareto Candidate
        res_approve = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": run_id,
                "candidate_id": cand_to_approve,
            },
        )
        assert res_approve.status_code == 200
        approve_data = res_approve.json()
        assert approve_data["status"] == "approved"
        assert approve_data["approved_workflow_id"] == cand_to_approve
        assert "version" in approve_data
        version_id = approve_data["version"]["version_id"]
        assert version_id.startswith("wf_v")
        assert approve_data["version"]["source_candidate_id"] == cand_to_approve
        assert approve_data["version"]["source_run_id"] == run_id

        # 2. Test Idempotency: Approving same candidate again returns existing version without duplication
        res_approve_again = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": run_id,
                "candidate_id": cand_to_approve,
            },
        )
        assert res_approve_again.status_code == 200
        assert res_approve_again.json()["version"]["version_id"] == version_id

        # 3. Test List Approved Versions
        res_versions = client.get("/api/agent-evo/versions")
        assert res_versions.status_code == 200
        versions_data = res_versions.json()
        assert versions_data["total_versions"] >= 1
        assert any(v["version_id"] == version_id for v in versions_data["versions"])

        # Test Workflows alias
        res_wf_alias = client.get("/api/workflows/versions")
        assert res_wf_alias.status_code == 200
        assert res_wf_alias.json()["total_versions"] == versions_data["total_versions"]

        # 4. Test Get Specific Version
        res_v_detail = client.get(f"/api/agent-evo/versions/{version_id}")
        assert res_v_detail.status_code == 200
        assert res_v_detail.json()["version_id"] == version_id
        assert res_v_detail.json()["evaluation_snapshot"] is not None

        # 5. Test Baseline Rejection
        res_base_reject = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": run_id,
                "candidate_id": "baseline",
            },
        )
        assert res_base_reject.status_code == 400
        assert "Baseline" in res_base_reject.json()["detail"]

        # 6. Test Unknown Candidate Rejection
        res_unknown_cand = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": run_id,
                "candidate_id": "non_existent_cand_999",
            },
        )
        assert res_unknown_cand.status_code == 404

        # 7. Test Unknown Run Rejection
        res_unknown_run = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": "evo_unknown_run_000",
                "candidate_id": cand_to_approve,
            },
        )
        assert res_unknown_run.status_code == 404

        # 8. Test Dominated Candidate Rejection
        optimizer = get_agent_evo_optimizer()
        run_obj = optimizer.get_run(run_id)
        # Find or create a dominated candidate
        dom_cand = next((c for c in run_obj.all_candidates if not c.is_pareto_optimal and c.candidate_id != "baseline"), None)
        if not dom_cand:
            dom_cand = run_obj.all_candidates[-1]
            dom_cand.is_pareto_optimal = False

        res_dom_reject = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": run_id,
                "candidate_id": dom_cand.candidate_id,
            },
        )
        assert res_dom_reject.status_code == 400
        assert "dominated" in res_dom_reject.json()["detail"].lower()
