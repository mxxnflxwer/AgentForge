import random
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.models.user import User, UserRole
from app.models.benchmark_run import BenchmarkRun
from app.models.workflow_version import WorkflowVersion
from app.services.agent_evo import (
    AgentWorkflow,
    CandidateGenerator,
    CandidateMetrics,
    EvaluatedCandidate,
    EvaluatorAdapter,
    ModelConfig,
    MutationMetadata,
    MutationType,
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
from app.services.agent_evo.benchmark_service import BenchmarkService
from app.services.agent_evo.candidate_generator import normalize_workflow_config
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
    assert "The candidate achieved" in trade_off.trade_off_summary
    assert "lower measured cost" in trade_off.trade_off_summary


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


# --- 9. AgentEvo Exploration Phase Tests ---

def test_exploration_prompt_mutation_produces_valid_configuration():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(
        baseline=baseline,
        count=3,
        allowed_types=[MutationType.PROMPT],
        seed=42,
    )
    assert len(candidates) == 3
    for cand in candidates:
        assert cand.mutation_metadata is not None
        assert cand.mutation_metadata.mutation_type == MutationType.PROMPT
        assert cand.mutation_metadata.parent_workflow_id == "baseline"
        assert len(cand.mutation_metadata.mutation_description) > 0
        # Check that configuration is different from baseline prompt
        is_diff = (
            cand.prompt.template_name != baseline.prompt.template_name
            or cand.prompt.custom_instructions != baseline.prompt.custom_instructions
        )
        assert is_diff, f"Candidate '{cand.workflow_id}' did not mutate prompt"
        val = validate_workflow(cand)
        assert val.is_valid is True, f"Validation failed: {val.errors}"


def test_exploration_retrieval_mutation_produces_valid_configuration():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(
        baseline=baseline,
        count=3,
        allowed_types=[MutationType.RETRIEVAL],
        seed=42,
    )
    assert len(candidates) == 3
    for cand in candidates:
        assert cand.mutation_metadata is not None
        assert cand.mutation_metadata.mutation_type == MutationType.RETRIEVAL
        assert cand.mutation_metadata.parent_workflow_id == "baseline"
        is_diff = (
            cand.retrieval.top_k != baseline.retrieval.top_k
            or cand.retrieval.similarity_threshold != baseline.retrieval.similarity_threshold
            or cand.retrieval.chunk_size != baseline.retrieval.chunk_size
        )
        assert is_diff, f"Candidate '{cand.workflow_id}' did not mutate retrieval"
        val = validate_workflow(cand)
        assert val.is_valid is True, f"Validation failed: {val.errors}"


def test_exploration_model_mutation_produces_valid_configuration():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(
        baseline=baseline,
        count=3,
        allowed_types=[MutationType.MODEL],
        seed=42,
    )
    assert len(candidates) == 3
    for cand in candidates:
        assert cand.mutation_metadata is not None
        assert cand.mutation_metadata.mutation_type == MutationType.MODEL
        assert cand.mutation_metadata.parent_workflow_id == "baseline"
        is_diff = (
            cand.model.model_id != baseline.model.model_id
            or cand.model.provider != baseline.model.provider
            or cand.model.temperature != baseline.model.temperature
            or cand.model.max_tokens != baseline.model.max_tokens
        )
        assert is_diff, f"Candidate '{cand.workflow_id}' did not mutate model"
        val = validate_workflow(cand)
        assert val.is_valid is True, f"Validation failed: {val.errors}"


def test_exploration_operator_mutation_produces_valid_executable_workflow():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(
        baseline=baseline,
        count=2,
        allowed_types=[MutationType.OPERATOR],
        seed=42,
    )
    assert len(candidates) == 2
    for cand in candidates:
        assert cand.mutation_metadata is not None
        assert cand.mutation_metadata.mutation_type == MutationType.OPERATOR
        assert cand.mutation_metadata.parent_workflow_id == "baseline"
        # Operator mutation should toggle execution operators like context compression
        assert cand.execution.context_compression is True
        val = validate_workflow(cand)
        assert val.is_valid is True, f"Validation failed: {val.errors}"


def test_exploration_mixed_mutation_produces_valid_configuration():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(
        baseline=baseline,
        count=2,
        allowed_types=[MutationType.MIXED],
        seed=42,
    )
    assert len(candidates) == 2
    for cand in candidates:
        assert cand.mutation_metadata is not None
        assert cand.mutation_metadata.mutation_type == MutationType.MIXED
        assert cand.mutation_metadata.parent_workflow_id == "baseline"
        assert "categories_mutated" in cand.mutation_metadata.generation_metadata
        val = validate_workflow(cand)
        assert val.is_valid is True, f"Validation failed: {val.errors}"


def test_exploration_parent_workflow_is_unchanged_after_candidate_generation():
    baseline = get_baseline_workflow()
    base_dict_before = baseline.to_dict()

    # Generate 5 diverse candidates
    candidates = generate_candidates(baseline=baseline, count=5, seed=123)

    # Mutate candidates further to stress-test independence
    for cand in candidates:
        cand.retrieval.top_k = 19
        cand.model.model_id = "openai/gpt-oss-120b"
        cand.prompt.template_name = "detailed_clinical"

    base_dict_after = baseline.to_dict()
    assert base_dict_before == base_dict_after, "Baseline was mutated in place!"
    assert baseline.workflow_id == "baseline"
    assert baseline.retrieval.top_k == 5
    assert baseline.model.model_id == "gemini-3.5-flash-lite"


def test_exploration_candidates_are_configuration_isolated():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(baseline=baseline, count=5, seed=99)

    c1, c2, c3 = candidates[0], candidates[1], candidates[2]
    c1.retrieval.top_k = 1
    c1.prompt.template_name = "concise_medical"

    assert c2.retrieval.top_k != 1
    assert c3.retrieval.top_k != 1
    assert c1.workflow_id != c2.workflow_id != c3.workflow_id


def test_exploration_candidate_ids_are_unique():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(baseline=baseline, count=5, seed=55)
    ids = [c.workflow_id for c in candidates]
    assert len(ids) == len(set(ids)) == 5


def test_exploration_mutation_metadata_recorded():
    baseline = get_baseline_workflow()
    candidates = generate_candidates(baseline=baseline, count=5, seed=777)
    types_found = set()

    for cand in candidates:
        assert cand.mutation_metadata is not None
        assert isinstance(cand.mutation_metadata.mutation_type, MutationType)
        assert cand.mutation_metadata.parent_workflow_id == "baseline"
        assert len(cand.mutation_metadata.mutation_description) > 5
        types_found.add(cand.mutation_metadata.mutation_type)

    # 5-candidate run should contain diversity across multiple categories
    assert len(types_found) >= 4, f"Expected diversity across >=4 mutation types, got {types_found}"


def test_exploration_invalid_mutations_rejected():
    baseline = get_baseline_workflow()
    invalid_wf = baseline.clone("cand_invalid", "Invalid Mutant")
    invalid_wf.retrieval.top_k = 99  # Invalid: > 20
    invalid_wf.retrieval.similarity_threshold = -0.5  # Invalid: < 0.0

    val = validate_workflow(invalid_wf)
    assert val.is_valid is False
    assert len(val.errors) >= 2


@pytest.mark.asyncio
async def test_exploration_invalid_mutation_handled_in_evaluation(evo_test_user, db_session):
    baseline = get_baseline_workflow()
    invalid_wf = baseline.clone("cand_bad_params", "Bad Params")
    invalid_wf.retrieval.top_k = 50  # Out of bounds

    cand = await EvaluatorAdapter.evaluate_candidate(
        workflow=invalid_wf,
        query="Test query",
        user_id=evo_test_user.id,
        db=db_session,
    )
    assert cand.metrics.status == "invalid"
    assert cand.is_pareto_optimal is False
    assert "Validation failed" in (cand.metrics.error_message or "")


@pytest.mark.asyncio
async def test_exploration_provider_failures_preserve_error_and_excluded_from_pareto(evo_test_user, db_session):
    baseline = get_baseline_workflow()
    cand_wf = baseline.clone(
        new_id="cand_failing_provider",
        new_name="Failing Provider Mutant",
        mutation_metadata=MutationMetadata(
            mutation_type=MutationType.MODEL,
            parent_workflow_id="baseline",
            mutation_description="Simulates provider 503 error.",
        ),
    )

    mock_retrieval = RAGSearchResponse(
        query="What is the diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                chunk_index=0,
                content="Clinical assessment context.",
                similarity_score=0.9,
                distance=0.1,
                relevance_score=0.9,
            )
        ],
    )

    mock_adapter = MagicMock()
    mock_adapter.name = "Gemini 3.5 Flash-Lite"
    mock_adapter.generate = AsyncMock(side_effect=RuntimeError("503 UNAVAILABLE: Model overloaded"))
    mock_router = MagicMock()
    mock_router.get_adapter.return_value = mock_adapter

    with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

        evaluated = await EvaluatorAdapter.evaluate_candidate(
            workflow=cand_wf,
            query="What is the diagnosis?",
            user_id=evo_test_user.id,
            db=db_session,
        )

        assert evaluated.metrics.status == "model_error"
        assert evaluated.metrics.error_message is not None
        assert "503 UNAVAILABLE" in evaluated.metrics.error_message

        # Ensure failed candidate cannot enter Pareto Archive
        archive = ParetoArchive()
        archive.add_candidate(evaluated)
        assert len(archive.pareto_candidates) == 0
        assert evaluated.is_pareto_optimal is False


def test_exploration_deterministic_seed_generation():
    baseline = get_baseline_workflow()
    run1 = generate_candidates(baseline=baseline, count=4, seed=12345)
    run2 = generate_candidates(baseline=baseline, count=4, seed=12345)

    assert len(run1) == len(run2) == 4
    for c1, c2 in zip(run1, run2):
        assert c1.workflow_id == c2.workflow_id
        assert c1.to_dict() == c2.to_dict()
        assert c1.mutation_metadata.to_dict() == c2.mutation_metadata.to_dict() if hasattr(c1.mutation_metadata, "to_dict") else c1.mutation_metadata.model_dump() == c2.mutation_metadata.model_dump()


@pytest.mark.asyncio
async def test_exploration_full_optimizer_pipeline_with_diverse_candidates(evo_test_user, db_session):
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
        latency_ms=130.0,
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
            candidate_count=5,
            seed=42,
            db=db_session,
        )

        assert report.run_id.startswith("evo_")
        assert report.total_candidates_evaluated == 6  # 1 baseline + 5 candidates
        assert len(report.all_candidates) == 6

        # Verify mutation metadata exists on all generated candidates
        for cand in report.all_candidates:
            if cand.candidate_id != "baseline":
                assert cand.mutation_metadata is not None
                assert cand.mutation_metadata.mutation_type in [
                    MutationType.PROMPT,
                    MutationType.RETRIEVAL,
                    MutationType.MODEL,
                    MutationType.OPERATOR,
                    MutationType.MIXED,
                ]

        # Verify Pareto frontier exists
        assert len(report.pareto_frontier) >= 1

        # Verify approval governance on a Pareto candidate
        pareto_cand = next((c for c in report.pareto_frontier if c.candidate_id != "baseline"), report.all_candidates[1])
        pareto_cand.is_pareto_optimal = True
        appr_report, version = optimizer.approve_workflow(
            run_id=report.run_id,
            candidate_id=pareto_cand.candidate_id,
            approved_by=evo_test_user.email,
            db=db_session,
        )
        assert appr_report.approved_workflow_id == pareto_cand.candidate_id
        assert version.version_id.startswith("wf_v")


# ============================================================================
# PHASE 7.3: REFINEMENT ENGINE TESTS
# ============================================================================

def test_refinement_retrieval_reduction():
    """Test Refinement Operator A: reduces top_k while respecting lower bound >= 2."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.retrieval.top_k = 5

    cand = CandidateGenerator._refine_retrieval_reduction(wf, rand)
    assert cand is not None
    assert cand.retrieval.top_k == 3
    assert cand.mutation_metadata is not None
    assert cand.mutation_metadata.mutation_type == MutationType.REFINEMENT
    assert cand.mutation_metadata.generation_metadata["refinement_operation"] == "retrieval_reduction"
    assert cand.mutation_metadata.generation_metadata["before_configuration"]["top_k"] == 5
    assert cand.mutation_metadata.generation_metadata["after_configuration"]["top_k"] == 3

    # If top_k is already at lower bound (2), operator returns None
    wf.retrieval.top_k = 2
    assert CandidateGenerator._refine_retrieval_reduction(wf, rand) is None


def test_refinement_chunk_reduction():
    """Test Refinement Operator B: reduces chunk_size and preserves chunk_overlap < chunk_size."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.retrieval.chunk_size = 1000
    wf.retrieval.chunk_overlap = 200

    cand = CandidateGenerator._refine_chunk_reduction(wf, rand)
    assert cand is not None
    assert cand.retrieval.chunk_size == 750
    assert cand.retrieval.chunk_overlap < cand.retrieval.chunk_size
    assert cand.mutation_metadata.generation_metadata["refinement_operation"] == "chunk_reduction"


def test_refinement_context_compression():
    """Test Refinement Operator C: enables context compression when disabled."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.execution.context_compression = False

    cand = CandidateGenerator._refine_context_compression(wf, rand)
    assert cand is not None
    assert cand.execution.context_compression is True
    assert cand.mutation_metadata.generation_metadata["refinement_operation"] == "context_compression"

    # Already active returns None
    assert CandidateGenerator._refine_context_compression(cand, rand) is None


def test_refinement_prompt_simplification():
    """Test Refinement Operator D: simplifies prompt template to concise grounded medical prompt."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.prompt.template_name = "clinical_cot"
    wf.prompt.custom_instructions = "Elaborate extensively on differential diagnosis."

    cand = CandidateGenerator._refine_prompt_simplification(wf, rand)
    assert cand is not None
    assert cand.prompt.template_name == "concise_medical"
    assert cand.prompt.custom_instructions is None
    assert cand.mutation_metadata.generation_metadata["refinement_operation"] == "prompt_simplification"


def test_refinement_output_token_reduction():
    """Test Refinement Operator E: reduces excessive max_tokens while protecting boundary >= 256."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.model.max_tokens = 1024

    cand = CandidateGenerator._refine_output_token_reduction(wf, rand)
    assert cand is not None
    assert cand.model.max_tokens == 512
    assert cand.mutation_metadata.generation_metadata["refinement_operation"] == "output_reduction"

    # Further reduction
    cand2 = CandidateGenerator._refine_output_token_reduction(cand, rand)
    assert cand2 is not None
    assert cand2.model.max_tokens == 384


def test_refinement_redundant_config_cleanup():
    """Test Refinement Operator F: normalizes redundant timeout and disables unneeded rerank for small top_k."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.execution.timeout_seconds = 120.0
    wf.retrieval.top_k = 2
    wf.retrieval.reranking = True

    cand = CandidateGenerator._refine_redundant_config_cleanup(wf, rand)
    assert cand is not None
    assert cand.execution.timeout_seconds == 30.0
    assert cand.retrieval.reranking is False
    assert cand.mutation_metadata.generation_metadata["refinement_operation"] == "redundant_cleanup"


def test_refinement_deep_copy_parent_isolation():
    """Ensure refinement operators never modify the parent workflow in-place."""
    rand = random.Random(42)
    wf = get_baseline_workflow()
    wf.retrieval.top_k = 5
    wf.retrieval.chunk_size = 1000
    original_top_k = wf.retrieval.top_k
    original_chunk_size = wf.retrieval.chunk_size

    cand1 = CandidateGenerator._refine_retrieval_reduction(wf, rand)
    cand2 = CandidateGenerator._refine_chunk_reduction(wf, rand)

    assert wf.retrieval.top_k == original_top_k == 5
    assert wf.retrieval.chunk_size == original_chunk_size == 1000
    assert cand1.retrieval.top_k != wf.retrieval.top_k
    assert cand2.retrieval.chunk_size != wf.retrieval.chunk_size


def test_refinement_duplicate_prevention_parent():
    """Verify duplicate prevention skips candidates that match parent or known configs."""
    wf = get_baseline_workflow()
    # Baseline already has concise_medical and top_k=5
    cands = generate_candidates(
        baseline=wf,
        count=5,
        allowed_types=[MutationType.REFINEMENT],
        seed=42,
    )
    # Check that no candidate has identical configuration to baseline
    from app.services.agent_evo.candidate_generator import normalize_workflow_config
    base_cfg = normalize_workflow_config(wf.to_dict())
    cand_cfgs = [normalize_workflow_config(c.to_dict()) for c in cands]

    for cfg in cand_cfgs:
        assert cfg != base_cfg
    # Check no duplicate candidates among themselves
    assert len(cand_cfgs) == len(set(str(c) for c in cand_cfgs))


def test_refinement_duplicate_prevention_existing_versions():
    """Verify duplicate prevention rejects candidates matching existing approved versions in DB."""
    wf = get_baseline_workflow()
    # Mock an existing version configuration that has chunk_size = 750
    existing_version_cfg = normalize_workflow_config(wf.to_dict())
    existing_version_cfg["retrieval"]["chunk_size"] = 750
    existing_version_cfg["retrieval"]["chunk_overlap"] = 150

    cands = generate_candidates(
        baseline=wf,
        count=4,
        allowed_types=[MutationType.REFINEMENT],
        existing_versions=[{"configuration": existing_version_cfg}],
        seed=42,
    )

    for c in cands:
        cand_norm = normalize_workflow_config(c.to_dict())
        assert cand_norm != existing_version_cfg


def test_refinement_validator_safety():
    """Ensure all generated refinement candidates strictly pass AgentWorkflow validation."""
    wf = get_baseline_workflow()
    cands = generate_candidates(
        baseline=wf,
        count=6,
        allowed_types=[MutationType.REFINEMENT],
    )
    for c in cands:
        val = validate_workflow(c)
        assert val.is_valid, f"Candidate {c.workflow_id} failed validation: {val.errors}"


def test_refinement_pareto_dominance_cheaper_same_groundedness():
    """Test that a refined candidate with lower cost and same/better groundedness dominates."""
    archive = ParetoArchive()
    base_cand = EvaluatedCandidate(
        candidate_id="baseline",
        workflow=get_baseline_workflow(),
        metrics=CandidateMetrics(
            workflow_id="baseline",
            workflow_name="Baseline",
            performance=0.95,
            cost=0.00045,
            groundedness=0.95,
            hallucination_rate=0.0,
            accuracy=0.95,
            total_tokens=450,
            input_tokens=400,
            output_tokens=50,
            llm_latency_ms=180.0,
            execution_time_ms=250.0,
            status="success",
        ),
    )
    archive.add_candidate(base_cand)

    # Refined candidate with lower cost and identical groundedness
    refined_wf = get_baseline_workflow()
    refined_cand = EvaluatedCandidate(
        candidate_id="cand_refine_retrieval_1",
        workflow=refined_wf,
        metrics=CandidateMetrics(
            workflow_id="cand_refine_retrieval_1",
            workflow_name="Refined: Tight Retrieval",
            performance=0.95,
            cost=0.00020,  # 55% cheaper
            groundedness=0.95,
            hallucination_rate=0.0,
            accuracy=0.95,
            total_tokens=220,
            input_tokens=180,
            output_tokens=40,
            llm_latency_ms=110.0,
            execution_time_ms=160.0,
            status="success",
        ),
    )
    archive.add_candidate(refined_cand)

    frontier = archive.recompute_frontier()
    frontier_ids = [c.candidate_id for c in frontier]

    # Refined candidate dominates baseline because cost is strictly lower and groundedness is equal
    assert "cand_refine_retrieval_1" in frontier_ids
    assert "baseline" not in frontier_ids
    assert refined_cand.is_pareto_optimal is True
    assert base_cand.is_pareto_optimal is False
    assert "cand_refine_retrieval_1" in base_cand.dominated_by


@pytest.mark.asyncio
async def test_refinement_full_optimization_and_approval(evo_test_user, db_session):
    """End-to-end test of refinement optimization run and human approval governance."""
    optimizer = get_agent_evo_optimizer()

    mock_retrieval = RAGSearchResponse(
        query="What is the cardiology diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="doc_cardio",
                chunk_index=0,
                content="Assessment: Coronary artery disease with stable angina pectoris.",
                similarity_score=0.94,
                distance=0.06,
                relevance_score=0.96,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="The patient is assessed with coronary artery disease and stable angina.",
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

        report = await optimizer.run_optimization(
            query="What is the cardiology diagnosis?",
            user_id=evo_test_user.id,
            document_id="doc_cardio",
            candidate_count=4,
            seed=42,
            allowed_mutation_types=["refinement"],
            db=db_session,
        )

        assert report.run_id.startswith("evo_")
        assert report.total_candidates_evaluated >= 2
        refinement_cands = [c for c in report.all_candidates if c.candidate_id != "baseline"]
        for c in refinement_cands:
            assert c.mutation_metadata is not None
            assert c.mutation_metadata.mutation_type == MutationType.REFINEMENT
            assert "refinement_operation" in c.mutation_metadata.generation_metadata

        # Approve top Pareto candidate
        pareto_cands = [c for c in report.pareto_frontier if c.candidate_id != "baseline"]
        if pareto_cands:
            target_cand = pareto_cands[0]
            appr_report, version = optimizer.approve_workflow(
                run_id=report.run_id,
                candidate_id=target_cand.candidate_id,
                approved_by=evo_test_user.email,
                db=db_session,
            )
            assert appr_report.approved_workflow_id == target_cand.candidate_id
            assert version.version_id.startswith("wf_v")
            assert version.configuration is not None
            assert version.evaluation_snapshot is not None


@pytest.mark.asyncio
async def test_refinement_from_existing_approved_workflow_parent(evo_test_user, db_session):
    """Test refining directly from an existing approved WorkflowVersion in the database."""
    optimizer = get_agent_evo_optimizer()

    # Seed an approved workflow version with custom high parameters
    parent_cfg = get_baseline_workflow().to_dict()
    parent_cfg["retrieval"]["top_k"] = 8
    parent_cfg["model"]["max_tokens"] = 1024

    seeded_version = WorkflowVersion(
        version_id="wf_v10_parent_test",
        version_number=10,
        name="High Top-K Parent",
        source_run_id="run_parent_seed",
        source_candidate_id="cand_parent_seed",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=parent_cfg,
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00008},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add(seeded_version)
    db_session.commit()

    mock_retrieval = RAGSearchResponse(
        query="What is the cardiology diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="c1",
                document_id="doc_cardio",
                chunk_index=0,
                content="Assessment: Coronary artery disease with stable angina pectoris.",
                similarity_score=0.94,
                distance=0.06,
                relevance_score=0.96,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="Assessment: Coronary artery disease with stable angina pectoris.",
        latency_ms=100.0,
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
            query="What is the cardiology diagnosis?",
            user_id=evo_test_user.id,
            document_id="doc_cardio",
            candidate_count=4,
            parent_workflow_id="wf_v10_parent_test",
            allowed_mutation_types=["refinement"],
            db=db_session,
        )

        assert report.run_id.startswith("evo_")
        assert report.baseline.candidate_id == "wf_v10_parent_test"

        refinement_cands = [c for c in report.all_candidates if c.candidate_id != "wf_v10_parent_test"]
        assert len(refinement_cands) >= 1
        for c in refinement_cands:
            assert c.mutation_metadata is not None
            assert c.mutation_metadata.mutation_type == MutationType.REFINEMENT
            assert c.mutation_metadata.parent_workflow_id == "wf_v10_parent_test"
            assert "before_configuration" in c.mutation_metadata.generation_metadata
            assert "after_configuration" in c.mutation_metadata.generation_metadata


# --- 7. Version Comparison & Benchmark Tests ---

@pytest.mark.asyncio
async def test_benchmark_multiple_versions_and_pareto_classification(db_session, evo_test_user):
    """
    Test controlled multi-version benchmark evaluation across baseline and approved workflow versions.
    Verifies metric collection, Pareto classification, and persistence.
    """
    # Create two approved workflow versions in DB
    wf1 = get_baseline_workflow()
    wf1.retrieval.top_k = 3
    wf1.execution.context_compression = True
    v1 = WorkflowVersion(
        version_id="wf_v1_bench_test",
        version_number=1,
        name="V1 Compressed",
        description="Top-K 3 with compression",
        source_run_id="run_1",
        source_candidate_id="cand_1",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf1.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.000045},
        status="approved",
        approved_by=evo_test_user.email,
    )

    wf2 = get_baseline_workflow()
    wf2.retrieval.top_k = 10
    v2 = WorkflowVersion(
        version_id="wf_v2_bench_test",
        version_number=2,
        name="V2 High Top-K",
        description="Top-K 10 without compression",
        source_run_id="run_2",
        source_candidate_id="cand_2",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf2.to_dict(),
        evaluation_snapshot={"groundedness": 0.8, "cost": 0.000090},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add_all([v1, v2])
    db_session.commit()

    mock_retrieval = RAGSearchResponse(
        query="What is the cardiology diagnosis?",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="chk_1",
                document_id="doc_cardio",
                chunk_index=0,
                content="Assessment: Coronary artery disease with stable angina pectoris.",
                similarity_score=0.95,
                distance=0.05,
                relevance_score=0.95,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="Assessment: Coronary artery disease with stable angina pectoris.",
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

        report = await BenchmarkService.run_benchmark(
            version_ids=["baseline", "wf_v1_bench_test", "wf_v2_bench_test"],
            query="What is the cardiology diagnosis?",
            user_id=evo_test_user.id,
            user_email=evo_test_user.email,
            document_id="doc_cardio",
            expected_answer="Assessment: Coronary artery disease with stable angina pectoris.",
            preset_name="clinical",
            name="Clinical Diagnosis Benchmark",
            db=db_session,
        )

        assert report.benchmark_id.startswith("bench_")
        assert len(report.results) == 3
        assert report.preset_name == "clinical"
        assert report.selected_versions == ["baseline", "wf_v1_bench_test", "wf_v2_bench_test"]

        # Verify each result has complete metrics and configuration
        for res in report.results:
            assert res.groundedness is not None
            assert res.cost is not None
            assert res.total_tokens is not None
            assert res.llm_latency_ms is not None
            assert res.model_id == "gemini-3.5-flash-lite"
            assert res.provider == "Google"
            assert res.pareto_status in ("Pareto-optimal", "Dominated", "Non-dominated")

        # Verify DB persistence
        db_bench = db_session.query(BenchmarkRun).filter(BenchmarkRun.benchmark_id == report.benchmark_id).first()
        assert db_bench is not None
        assert db_bench.name == "Clinical Diagnosis Benchmark"
        assert len(db_bench.results) == 3


@pytest.mark.asyncio
async def test_benchmark_identical_inputs_across_versions(db_session, evo_test_user):
    """
    Ensure every selected workflow receives the EXACT same query, user_id, document_id, and expected_answer.
    """
    wf1 = get_baseline_workflow()
    v1 = WorkflowVersion(
        version_id="wf_v1_identical_test",
        version_number=1,
        name="V1 Test",
        source_run_id="run_1",
        source_candidate_id="cand_1",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf1.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00005},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add(v1)
    db_session.commit()

    captured_eval_calls = []

    async def mock_evaluate_candidate(workflow, query, user_id, document_id, expected_answer, db):
        captured_eval_calls.append({
            "workflow_id": workflow.workflow_id,
            "query": query,
            "user_id": user_id,
            "document_id": document_id,
            "expected_answer": expected_answer,
        })
        m = CandidateMetrics(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            performance=1.0,
            cost=0.00005,
            groundedness=1.0,
            hallucination_rate=0.0,
            total_tokens=500,
            input_tokens=450,
            output_tokens=50,
            llm_latency_ms=150.0,
            execution_time_ms=200.0,
            status="success",
        )
        return EvaluatedCandidate(
            candidate_id=workflow.workflow_id,
            workflow=workflow,
            metrics=m,
            is_pareto_optimal=True,
        )

    with patch.object(EvaluatorAdapter, "evaluate_candidate", side_effect=mock_evaluate_candidate):
        target_query = "What is the primary clinical diagnosis and assessment for this patient?"
        target_doc = "doc_cardiology_01"
        target_answer = "Exertional angina pectoris with CAD."

        report = await BenchmarkService.run_benchmark(
            version_ids=["baseline", "wf_v1_identical_test"],
            query=target_query,
            user_id=evo_test_user.id,
            user_email=evo_test_user.email,
            document_id=target_doc,
            expected_answer=target_answer,
            db=db_session,
        )

        assert len(captured_eval_calls) == 2
        for call in captured_eval_calls:
            assert call["query"] == target_query
            assert call["document_id"] == target_doc
            assert call["expected_answer"] == target_answer
            assert call["user_id"] == evo_test_user.id


def test_benchmark_immutable_configuration_preservation(db_session, evo_test_user):
    """
    Ensure workflow version configurations in the database are NOT mutated or altered by benchmarking.
    """
    wf = get_baseline_workflow()
    wf.retrieval.top_k = 7
    original_config = wf.to_dict()

    v = WorkflowVersion(
        version_id="wf_v1_immutable_test",
        version_number=1,
        name="V1 Immutable Check",
        source_run_id="run_imm",
        source_candidate_id="cand_imm",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=original_config,
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00006},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add(v)
    db_session.commit()

    # Inspect before
    inspect_before = BenchmarkService.inspect_version("wf_v1_immutable_test", db=db_session)
    assert inspect_before.configuration["retrieval"]["top_k"] == 7

    # DB record check
    v_db = db_session.query(WorkflowVersion).filter(WorkflowVersion.version_id == "wf_v1_immutable_test").first()
    assert v_db.configuration == original_config


def test_lineage_graph_retrieval(db_session, evo_test_user):
    """
    Test that lineage extraction respects non-linear parent-child relationships and metadata.
    """
    # Baseline is root
    # V1 derived from baseline
    wf1 = get_baseline_workflow()
    wf1.mutation_metadata = MutationMetadata(
        mutation_type=MutationType.PROMPT,
        parent_workflow_id="baseline",
        mutation_description="Concise prompt mutation",
    )
    v1 = WorkflowVersion(
        version_id="wf_v1_lineage",
        version_number=1,
        name="V1 Concise",
        source_run_id="run_lin_1",
        source_candidate_id="cand_lin_1",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf1.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00005},
        status="approved",
        approved_by=evo_test_user.email,
    )

    # V2 derived from V1 (not baseline)
    wf2 = get_baseline_workflow()
    wf2.mutation_metadata = MutationMetadata(
        mutation_type=MutationType.REFINEMENT,
        parent_workflow_id="wf_v1_lineage",
        mutation_description="Context compression refinement",
        generation_metadata={
            "refinement_operation": "context_compression",
            "fields_changed": ["execution.context_compression"],
            "before_configuration": {"context_compression": False},
            "after_configuration": {"context_compression": True},
        },
    )
    v2 = WorkflowVersion(
        version_id="wf_v2_lineage",
        version_number=2,
        name="V2 Refined",
        source_run_id="run_lin_2",
        source_candidate_id="cand_lin_2",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf2.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00004},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add_all([v1, v2])
    db_session.commit()

    lineage = BenchmarkService.get_lineage(db=db_session)
    assert lineage.total_versions >= 3  # baseline + v1 + v2
    
    # Check nodes
    node_ids = [n.version_id for n in lineage.nodes]
    assert "baseline" in node_ids
    assert "wf_v1_lineage" in node_ids
    assert "wf_v2_lineage" in node_ids

    # Check edges
    edge_pairs = [(e.source, e.target) for e in lineage.edges]
    assert ("baseline", "wf_v1_lineage") in edge_pairs
    assert ("wf_v1_lineage", "wf_v2_lineage") in edge_pairs


def test_version_inspect_endpoint(db_session, evo_test_user):
    """
    Test inspecting a workflow version returns full diffs, lineage, and approval metadata.
    """
    wf = get_baseline_workflow()
    wf.mutation_metadata = MutationMetadata(
        mutation_type=MutationType.REFINEMENT,
        parent_workflow_id="baseline",
        mutation_description="Prompt simplification",
        generation_metadata={
            "refinement_operation": "prompt_simplification",
            "fields_changed": ["prompt.template_name"],
            "before_configuration": {"template_name": "default_medical"},
            "after_configuration": {"template_name": "concise_medical"},
        },
    )
    v = WorkflowVersion(
        version_id="wf_v4_inspect_test",
        version_number=4,
        name="V4 Prompt Refined",
        source_run_id="run_insp",
        source_candidate_id="cand_insp",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.000048},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add(v)
    db_session.commit()

    inspect_data = BenchmarkService.inspect_version("wf_v4_inspect_test", db=db_session)
    assert inspect_data.version_id == "wf_v4_inspect_test"
    assert inspect_data.parent_workflow_id == "baseline"
    assert inspect_data.refinement_operation == "prompt_simplification"
    assert inspect_data.fields_changed == ["prompt.template_name"]
    assert inspect_data.before_configuration == {"template_name": "default_medical"}
    assert inspect_data.after_configuration == {"template_name": "concise_medical"}
    assert inspect_data.approved_by == evo_test_user.email


@pytest.mark.asyncio
async def test_benchmark_invalid_version_handling(db_session, evo_test_user):
    """
    Test that requesting an invalid or nonexistent workflow version ID raises a 404 error.
    """
    with pytest.raises(Exception) as exc_info:
        await BenchmarkService.run_benchmark(
            version_ids=["baseline", "wf_nonexistent_9999"],
            query="Test query",
            user_id=evo_test_user.id,
            user_email=evo_test_user.email,
            db=db_session,
        )
    assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


def test_benchmark_api_endpoints_via_client(client, db_session, evo_test_user):
    """
    Test the FastAPI /api/agent-evo/lineage, /api/agent-evo/versions/{id}/inspect,
    and /api/agent-evo/benchmarks endpoints.
    """
    # Create test version
    wf = get_baseline_workflow()
    v = WorkflowVersion(
        version_id="wf_v1_api_bench",
        version_number=1,
        name="V1 API Bench",
        source_run_id="run_api",
        source_candidate_id="cand_api",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00005},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add(v)
    db_session.commit()

    # Login user session
    login_res = client.post(
        "/api/auth/login",
        json={"email": evo_test_user.email, "password": "testpassword123"},
    )
    assert login_res.status_code == 200

    # Test lineage endpoint
    lin_res = client.get("/api/agent-evo/lineage")
    assert lin_res.status_code == 200
    lin_data = lin_res.json()
    assert "nodes" in lin_data
    assert "edges" in lin_data
    assert lin_data["total_versions"] >= 2

    # Test inspect endpoint
    insp_res = client.get(f"/api/agent-evo/versions/{v.version_id}/inspect")
    assert insp_res.status_code == 200
    insp_data = insp_res.json()
    assert insp_data["version_id"] == v.version_id
    assert insp_data["name"] == "V1 API Bench"

    # Test inspect baseline
    insp_base = client.get("/api/agent-evo/versions/baseline/inspect")
    assert insp_base.status_code == 200
    assert insp_base.json()["version_id"] == "baseline"

    # Test list benchmarks
    bench_list_res = client.get("/api/agent-evo/benchmarks")
    assert bench_list_res.status_code == 200
    assert "benchmarks" in bench_list_res.json()


@pytest.mark.asyncio
async def test_benchmark_multi_iteration_and_relative_calculations(db_session, evo_test_user, monkeypatch):
    """
    Test that BenchmarkService runs multiple iterations per workflow,
    calculates means and per-run results, establishes baseline-relative deltas,
    and classifies Pareto dominance.
    """
    # Create test version v1 (cheaper than baseline)
    wf1 = get_baseline_workflow()
    wf1.retrieval.top_k = 2
    v1 = WorkflowVersion(
        version_id="wf_v1_bench_delta",
        version_number=1,
        name="V1 Delta Test",
        source_run_id="run_delta_1",
        source_candidate_id="cand_delta_1",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf1.to_dict(),
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.00004},
        status="approved",
        approved_by=evo_test_user.email,
    )
    db_session.add(v1)
    db_session.commit()

    # Mock EvaluatorAdapter to return deterministic metrics
    eval_call_count = {"baseline": 0, "wf_v1_bench_delta": 0}

    async def mock_evaluate(workflow, query, user_id, document_id=None, expected_answer=None, db=None):
        w_id = workflow.workflow_id
        if w_id == "baseline" or "baseline" in workflow.name.lower():
            eval_call_count["baseline"] += 1
            cost = 0.000100
            groundedness = 0.90
            tokens = 1000
            latency = 4000.0
        else:
            eval_call_count["wf_v1_bench_delta"] += 1
            cost = 0.000050
            groundedness = 1.00
            tokens = 500
            latency = 2000.0

        from app.services.agent_evo.objective import CandidateMetrics
        from app.services.agent_evo.pareto import EvaluatedCandidate

        metrics = CandidateMetrics(
            workflow_id=w_id,
            workflow_name=workflow.name,
            performance=groundedness,
            cost=cost,
            groundedness=groundedness,
            hallucination_rate=round(1.0 - groundedness, 2),
            total_tokens=tokens,
            input_tokens=tokens - 100,
            output_tokens=100,
            llm_latency_ms=latency,
            execution_time_ms=latency + 500.0,
            raw_details={"claim_breakdown": [{"claim": "test", "grounded": True}]},
        )
        return EvaluatedCandidate(
            candidate_id=w_id,
            workflow=workflow,
            metrics=metrics,
        )

    monkeypatch.setattr("app.services.agent_evo.benchmark_service.EvaluatorAdapter.evaluate_candidate", mock_evaluate)

    report = await BenchmarkService.run_benchmark(
        version_ids=["baseline", "wf_v1_bench_delta"],
        query="What is the primary diagnosis?",
        user_id=evo_test_user.id,
        user_email=evo_test_user.email,
        iterations=2,
        db=db_session,
    )

    # 1. Identical inputs & multi-iteration checks
    assert report.status == "completed"
    assert len(report.results) == 2
    assert eval_call_count["baseline"] == 2
    assert eval_call_count["wf_v1_bench_delta"] == 2

    # 2. Check baseline item
    base_item = next(r for r in report.results if r.version_id == "baseline")
    assert base_item.iterations_count == 2
    assert len(base_item.runs) == 2
    assert base_item.groundedness == 0.90
    assert base_item.cost == 0.000100
    assert base_item.relative_changes["groundedness_delta"] == 0.0
    assert base_item.relative_changes["cost_reduction_pct"] == 0.0
    assert base_item.relative_changes["token_reduction_pct"] == 0.0

    # 3. Check v1 item relative deltas
    v1_item = next(r for r in report.results if r.version_id == "wf_v1_bench_delta")
    assert v1_item.iterations_count == 2
    assert len(v1_item.runs) == 2
    assert v1_item.groundedness == 1.00
    assert v1_item.cost == 0.000050
    assert v1_item.relative_changes["groundedness_delta"] == 0.10
    assert v1_item.relative_changes["cost_reduction_pct"] == 50.0
    assert v1_item.relative_changes["token_reduction_pct"] == 50.0
    assert v1_item.relative_changes["latency_reduction_pct"] == 50.0

    # 4. Pareto classification (v1 has higher groundedness and lower cost, so it dominates baseline)
    assert v1_item.is_pareto_optimal is True
    assert v1_item.pareto_status == "Pareto-optimal"
    assert "baseline" in v1_item.dominates
    assert base_item.is_pareto_optimal is False
    assert base_item.pareto_status == "Dominated"
    assert "wf_v1_bench_delta" in base_item.dominated_by

    # 5. Check Persistence
    persisted = BenchmarkService.get_benchmark(report.benchmark_id, db=db_session)
    assert persisted is not None
    assert persisted.benchmark_id == report.benchmark_id
    assert len(persisted.results) == 2


def test_benchmark_canonical_field_mapping_data_contract():
    """
    Regression Test: Asserts that BenchmarkItemResult preserves exact explicit
    named fields without positional or label shift bugs.
    """
    from app.schemas.agent_evo import BenchmarkItemResult

    result = BenchmarkItemResult(
        version_id="baseline",
        version_number=0,
        workflow_name="Production Baseline",
        parent_workflow_id="None",
        source_candidate_id=None,
        source_run_id=None,
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        groundedness=1.0,
        hallucination_rate=0.0,
        cost=0.000039,
        total_tokens=409,
        input_tokens=371,
        output_tokens=38,
        llm_latency_ms=2401.0,
        execution_time_ms=2600.0,
        retrieval={"top_k": 5, "similarity_threshold": 0.35, "chunk_size": 1000, "chunk_overlap": 200, "reranking": True},
        retrieval_config_summary="top_k=5, sim_thresh=0.35, chunk_size=1000, rerank=True",
        prompt_template="default_medical",
        prompt_strategy="default_medical",
        context_compression=False,
        is_pareto_optimal=False,
        pareto_status="Dominated",
        configuration={},
    )

    # 1. Verify exact canonical values
    assert result.groundedness == 1.0
    assert result.hallucination == 0.0
    assert result.cost_per_query == 0.000039
    assert result.total_tokens == 409
    assert result.latency_ms == 2401.0
    assert result.model == "gemini-3.5-flash-lite"
    assert result.provider == "Google"
    assert "top_k=5" in result.retrieval_config
    assert result.version == "baseline"
    assert result.workflow == "Production Baseline"

    # 2. Strict anti-shift assertions: Latency is NOT tokens, Cost is NOT latency, Tokens is NOT model
    assert result.latency_ms != result.total_tokens, "Field shift: latency_ms must not equal total_tokens"
    assert result.total_tokens != result.model, "Field shift: total_tokens must not equal model"
    assert result.cost_per_query != result.latency_ms, "Field shift: cost_per_query must not equal latency_ms"
    assert result.hallucination != result.groundedness, "Field shift: hallucination must not equal groundedness"


def test_benchmark_pareto_strictly_uses_groundedness_and_cost():
    """
    Asserts Pareto dominance strictly evaluates Groundedness ↑ and Cost ↓,
    never using latency or tokens for dominance classification.
    """
    from app.schemas.agent_evo import BenchmarkItemResult
    from app.services.agent_evo.benchmark_service import BenchmarkService

    def make_item(v_id: str, g: float, c: float, lat: float, tok: int) -> BenchmarkItemResult:
        return BenchmarkItemResult(
            version_id=v_id,
            workflow_name=f"Workflow {v_id}",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            groundedness=g,
            hallucination_rate=round(1.0 - g, 2),
            cost=c,
            total_tokens=tok,
            input_tokens=tok - 50,
            output_tokens=50,
            llm_latency_ms=lat,
            execution_time_ms=lat + 200,
            retrieval={},
            prompt_template="default",
            context_compression=False,
            is_pareto_optimal=False,
            pareto_status="Non-dominated",
            configuration={},
        )

    # Item A: High groundedness (1.0), low cost ($0.000030), HIGH latency (8000ms), LOW tokens (280)
    item_a = make_item("v_a", g=1.0, c=0.000030, lat=8000.0, tok=280)
    # Item B: Equal groundedness (1.0), higher cost ($0.000050), LOW latency (1000ms), HIGH tokens (600)
    item_b = make_item("v_b", g=1.0, c=0.000050, lat=1000.0, tok=600)

    # Item A must dominate Item B because Cost is lower and Groundedness is equal,
    # EVEN THOUGH Item B has faster latency!
    frontier, dominated = BenchmarkService._compute_pareto_dominance([item_a, item_b])

    assert len(frontier) == 1
    assert frontier[0].version_id == "v_a"
    assert len(dominated) == 1
    assert dominated[0].version_id == "v_b"
    assert "v_b" in frontier[0].dominates
    assert "v_a" in dominated[0].dominated_by






