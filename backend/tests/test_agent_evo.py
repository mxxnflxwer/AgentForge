import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.models.user import User, UserRole
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
    wf = get_baseline_workflow()

    # Add baseline
    c_base = EvaluatedCandidate(
        candidate_id="base",
        workflow=wf,
        metrics=CandidateMetrics(workflow_id="base", workflow_name="Base", performance=0.80, cost=0.00010),
    )
    assert archive.add_candidate(c_base) is True
    assert len(archive.pareto_candidates) == 1

    # Add strictly better candidate (should prune c_base)
    c_better = EvaluatedCandidate(
        candidate_id="better",
        workflow=wf,
        metrics=CandidateMetrics(workflow_id="better", workflow_name="Better", performance=0.90, cost=0.00008),
    )
    assert archive.add_candidate(c_better) is True
    frontier = archive.pareto_candidates
    assert len(frontier) == 1
    assert frontier[0].candidate_id == "better"

    # Add trade-off candidate (cheaper, lower perf -> non-dominated, both survive)
    c_cheaper = EvaluatedCandidate(
        candidate_id="cheaper",
        workflow=wf,
        metrics=CandidateMetrics(workflow_id="cheaper", workflow_name="Cheaper", performance=0.75, cost=0.00002),
    )
    assert archive.add_candidate(c_cheaper) is True
    frontier = archive.pareto_candidates
    assert len(frontier) == 2
    frontier_ids = [c.candidate_id for c in frontier]
    assert "better" in frontier_ids
    assert "cheaper" in frontier_ids

    # Add dominated candidate (worse perf, higher cost -> rejected)
    c_worse = EvaluatedCandidate(
        candidate_id="worse",
        workflow=wf,
        metrics=CandidateMetrics(workflow_id="worse", workflow_name="Worse", performance=0.60, cost=0.00050),
    )
    assert archive.add_candidate(c_worse) is False
    assert len(archive.pareto_candidates) == 2


# --- 6. Optimization Report Tests ---

def test_optimization_report_generation():
    wf = get_baseline_workflow()
    c_base = EvaluatedCandidate(
        candidate_id="base",
        workflow=wf,
        metrics=CandidateMetrics(workflow_id="base", workflow_name="Baseline", performance=0.80, cost=0.00010, groundedness=0.80),
    )
    c_opt = EvaluatedCandidate(
        candidate_id="opt_1",
        workflow=wf.clone("opt_1", "Optimized 1"),
        metrics=CandidateMetrics(workflow_id="opt_1", workflow_name="Optimized 1", performance=1.0, cost=0.00005, groundedness=1.0),
    )

    report = generate_optimization_report(
        run_id="run_test_01",
        query="What is the clinical diagnosis?",
        baseline=c_base,
        all_candidates=[c_base, c_opt],
        pareto_candidates=[c_opt],
    )

    assert report.run_id == "run_test_01"
    assert report.total_candidates_evaluated == 2
    assert report.pareto_frontier_count == 1
    assert report.human_approval_required is True
    assert len(report.trade_off_analyses) == 1
    assert report.trade_off_analyses[0].performance_delta_pct > 0
    assert report.trade_off_analyses[0].cost_delta_pct < 0


# --- 7. End-to-End Optimizer Execution Tests ---

@pytest.mark.asyncio
async def test_agent_evo_optimizer_mocked(evo_test_user, db_session):
    optimizer = get_agent_evo_optimizer()

    mock_retrieval = RAGSearchResponse(
        query="What is the clinical diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                chunk_index=0,
                content="Assessment: Stable angina pectoris.",
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

        # Test developer approval
        cand_id = report.pareto_frontier[0].candidate_id
        approved_report = optimizer.approve_workflow(run_id=report.run_id, candidate_id=cand_id)
        assert approved_report is not None
        assert approved_report.approved_workflow_id == cand_id


# --- 8. API Endpoint Tests ---

def test_agent_evo_api_endpoints(client: TestClient, evo_test_user):
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

        # Approve workflow
        cand_to_approve = data["pareto_frontier"][0]["candidate_id"]
        res_approve = client.post(
            "/api/agent-evo/approve",
            json={
                "run_id": run_id,
                "candidate_id": cand_to_approve,
            },
        )
        assert res_approve.status_code == 200
        assert res_approve.json()["status"] == "approved"
        assert res_approve.json()["approved_workflow_id"] == cand_to_approve
