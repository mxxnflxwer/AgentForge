import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.models.workflow_version import WorkflowVersion
from app.schemas.agent_evo import (
    CrossoverCandidateRequest,
    CrossoverPreviewRequest,
)
from app.services.agent_evo.crossover_service import CrossoverService
from app.services.agent_evo.validator import WorkflowValidator
from app.services.agent_evo.workflow import (
    AgentWorkflow,
    ExecutionConfig,
    ModelConfig,
    PromptConfig,
    RetrievalConfig,
    get_baseline_workflow,
)


@pytest.fixture
def sample_parents(db_session):
    """Ensure database has baseline and two distinct approved workflow versions for crossover tests."""
    # Ensure baseline workflow is known
    base_wf = get_baseline_workflow()

    # Create v_parent_a (High-Precision Retrieval)
    wf_a_cfg = base_wf.to_dict()
    wf_a_cfg["retrieval"]["top_k"] = 3
    wf_a_cfg["retrieval"]["similarity_threshold"] = 0.45
    wf_a_cfg["model"]["temperature"] = 0.0
    wf_a = WorkflowVersion(
        version_id="wf_test_parent_a",
        version_number=901,
        name="Parent A: High-Precision Retrieval",
        description="Parent A with high precision retrieval and deterministic temperature",
        source_run_id="run_test_a",
        source_candidate_id="cand_test_a",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf_a_cfg,
        evaluation_snapshot={
            "groundedness": 1.0,
            "cost": 0.000030,
            "total_tokens": 280,
            "llm_latency_ms": 1500.0,
            "hallucination_rate": 0.0,
        },
        status="approved",
        approved_by="test@agentforge.ai",
    )

    # Create v_parent_b (Context Compression)
    wf_b_cfg = base_wf.to_dict()
    wf_b_cfg["retrieval"]["top_k"] = 5
    wf_b_cfg["prompt"]["template_name"] = "concise_medical"
    wf_b_cfg["execution"]["context_compression"] = True
    wf_b = WorkflowVersion(
        version_id="wf_test_parent_b",
        version_number=902,
        name="Parent B: Context Compression",
        description="Parent B with concise prompt and context compression enabled",
        source_run_id="run_test_b",
        source_candidate_id="cand_test_b",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=wf_b_cfg,
        evaluation_snapshot={
            "groundedness": 1.0,
            "cost": 0.000035,
            "total_tokens": 350,
            "llm_latency_ms": 1800.0,
            "hallucination_rate": 0.0,
        },
        status="approved",
        approved_by="test@agentforge.ai",
    )

    db_session.merge(wf_a)
    db_session.merge(wf_b)
    db_session.commit()

    return "wf_test_parent_a", "wf_test_parent_b"


# ===========================================================================
# 1. Parent Selection Tests
# ===========================================================================

def test_parent_selection_valid(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, a_id, _, a_name, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, b_id, _, b_name, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    assert a_id == p_a_id
    assert b_id == p_b_id
    assert wf_a.retrieval.top_k == 3
    assert wf_b.retrieval.top_k == 5
    assert wf_b.execution.context_compression is True


def test_parent_selection_baseline_resolution(db_session):
    wf_base, b_id, num, name, _, _, snap = CrossoverService.resolve_parent("baseline", db=db_session)
    assert b_id == "baseline"
    assert num == 0
    assert name == "Production Baseline"
    assert snap["groundedness"] == 1.0


def test_parent_selection_same_parent_rejected(sample_parents, db_session):
    p_a_id, _ = sample_parents
    req = CrossoverPreviewRequest(
        parent_a_version_id=p_a_id,
        parent_b_version_id=p_a_id,
        crossover_map={"retrieval": "parent_a", "prompt": "parent_b"},
    )
    with pytest.raises(HTTPException) as exc:
        CrossoverService.preview_crossover(req, db=db_session)
    assert exc.value.status_code == 400
    assert "cannot be the same" in exc.value.detail


def test_parent_selection_nonexistent_parent_rejected(db_session):
    req = CrossoverPreviewRequest(
        parent_a_version_id="wf_non_existent_9999",
        parent_b_version_id="baseline",
        crossover_map={"retrieval": "parent_a", "prompt": "parent_b"},
    )
    with pytest.raises(HTTPException) as exc:
        CrossoverService.preview_crossover(req, db=db_session)
    assert exc.value.status_code == 404
    assert "not found" in exc.value.detail


# ===========================================================================
# 2. Crossover Recombination & Genome Inheritance Tests
# ===========================================================================

def test_deterministic_component_inheritance(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    crossover_map = {
        "model": "parent_a",
        "prompt": "parent_b",
        "retrieval": "parent_a",
        "generation": "parent_a",
        "operators": "parent_b",
    }

    child_wf, items = CrossoverService.assemble_child_genome(
        parent_a_wf=wf_a,
        parent_b_wf=wf_b,
        parent_a_id=p_a_id,
        parent_b_id=p_b_id,
        crossover_map=crossover_map,
        post_mutation="none",
    )

    # Retrieval strictly from Parent A
    assert child_wf.retrieval.top_k == 3
    assert child_wf.retrieval.similarity_threshold == 0.45

    # Prompt strictly from Parent B
    assert child_wf.prompt.template_name == "concise_medical"

    # Model & Generation strictly from Parent A
    assert child_wf.model.provider == "Google"
    assert child_wf.model.model_id == "gemini-3.5-flash-lite"
    assert child_wf.model.temperature == 0.0

    # Operators strictly from Parent B
    assert child_wf.execution.context_compression is True

    # Inheritance items check
    assert len(items) == 5
    ret_item = next(i for i in items if i.component == "retrieval")
    assert ret_item.source_parent == "parent_a"
    assert ret_item.source_version_id == p_a_id

    prompt_item = next(i for i in items if i.component == "prompt")
    assert prompt_item.source_parent == "parent_b"
    assert prompt_item.source_version_id == p_b_id


def test_invalid_crossover_map_source_rejected(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    with pytest.raises(HTTPException) as exc:
        CrossoverService.assemble_child_genome(
            parent_a_wf=wf_a,
            parent_b_wf=wf_b,
            parent_a_id=p_a_id,
            parent_b_id=p_b_id,
            crossover_map={"retrieval": "parent_c"},
        )
    assert exc.value.status_code == 400
    assert "Invalid crossover source" in exc.value.detail


# ===========================================================================
# 3. Validation & Safety Checks
# ===========================================================================

def test_crossover_validation_pass(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    req = CrossoverPreviewRequest(
        parent_a_version_id=p_a_id,
        parent_b_version_id=p_b_id,
        crossover_map={
            "model": "parent_a",
            "prompt": "parent_b",
            "retrieval": "parent_a",
            "generation": "parent_a",
            "operators": "parent_b",
        },
    )
    preview = CrossoverService.preview_crossover(req, db=db_session)
    assert preview.is_valid is True
    assert len(preview.validation_errors) == 0
    assert preview.resulting_configuration["retrieval"]["top_k"] == 3
    assert preview.resulting_configuration["execution"]["context_compression"] is True


def test_crossover_validation_fails_on_invalid_parameters():
    invalid_wf = get_baseline_workflow()
    # Mutate parameters to be out of valid ranges
    invalid_wf.retrieval.top_k = 25
    invalid_wf.retrieval.chunk_size = 5000
    invalid_wf.retrieval.similarity_threshold = 1.5
    invalid_wf.model.model_id = "unsupported-model-xyz"
    invalid_wf.model.temperature = 2.5
    invalid_wf.prompt.template_name = "unsupported_template"
    invalid_wf.execution.timeout_seconds = 200.0

    val = WorkflowValidator.validate(invalid_wf)
    assert val.is_valid is False
    assert any("top_k" in err for err in val.errors)
    assert any("chunk_size" in err for err in val.errors)
    assert any("similarity_threshold" in err for err in val.errors)
    assert any("unsupported" in err.lower() for err in val.errors)
    assert any("temperature" in err for err in val.errors)
    assert any("timeout_seconds" in err for err in val.errors)


# ===========================================================================
# 4. Post-Crossover Mutation Tests
# ===========================================================================

def test_post_crossover_prompt_mutation(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    # Prompt from parent_a is default_medical; prompt mutation should switch it to concise_medical
    child_wf, _ = CrossoverService.assemble_child_genome(
        parent_a_wf=wf_a,
        parent_b_wf=wf_b,
        parent_a_id=p_a_id,
        parent_b_id=p_b_id,
        crossover_map={"prompt": "parent_a"},
        post_mutation="prompt",
    )
    assert child_wf.prompt.template_name == "concise_medical"


def test_post_crossover_retrieval_mutation(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    # Retrieval from parent_b has top_k=5; post-mutation reduces top_k
    child_wf, _ = CrossoverService.assemble_child_genome(
        parent_a_wf=wf_a,
        parent_b_wf=wf_b,
        parent_a_id=p_a_id,
        parent_b_id=p_b_id,
        crossover_map={"retrieval": "parent_b"},
        post_mutation="retrieval",
    )
    assert child_wf.retrieval.top_k == 4
    assert child_wf.retrieval.similarity_threshold > 0.35


def test_post_crossover_model_mutation(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    child_wf, _ = CrossoverService.assemble_child_genome(
        parent_a_wf=wf_a,
        parent_b_wf=wf_b,
        parent_a_id=p_a_id,
        parent_b_id=p_b_id,
        crossover_map={"model": "parent_b"},
        post_mutation="model",
    )
    assert child_wf.model.temperature == 0.0


def test_post_crossover_operator_mutation(sample_parents, db_session):
    p_a_id, p_b_id = sample_parents
    wf_a, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_a_id, db=db_session)
    wf_b, _, _, _, _, _, _ = CrossoverService.resolve_parent(p_b_id, db=db_session)

    # Operator from parent_a has compression=False; toggling makes it True
    child_wf, _ = CrossoverService.assemble_child_genome(
        parent_a_wf=wf_a,
        parent_b_wf=wf_b,
        parent_a_id=p_a_id,
        parent_b_id=p_b_id,
        crossover_map={"operators": "parent_a"},
        post_mutation="operator",
    )
    assert child_wf.execution.context_compression is True


# ===========================================================================
# 5. Lineage & Dual-Parent Edge Tests
# ===========================================================================

def test_lineage_dual_parent_edges(sample_parents, db_session):
    from app.services.agent_evo.benchmark_service import BenchmarkService

    p_a_id, p_b_id = sample_parents
    base_wf = get_baseline_workflow()

    # Create an approved version resulting from crossover
    cross_cfg = base_wf.to_dict()
    cross_cfg["retrieval"]["top_k"] = 3
    cross_cfg["execution"]["context_compression"] = True
    cross_cfg["mutation_metadata"] = {
        "mutation_type": "mixed",
        "parent_workflow_id": p_a_id,
        "mutation_description": "Crossover of A and B",
        "generation_metadata": {
            "candidate_type": "CROSSOVER",
            "parent_a_version_id": p_a_id,
            "parent_b_version_id": p_b_id,
            "crossover_strategy": "component_level",
            "crossover_map": {"retrieval": "parent_a", "operators": "parent_b"},
        },
    }

    cross_version = WorkflowVersion(
        version_id="wf_test_crossover_child",
        version_number=903,
        name="Crossover Child Approved",
        description="Approved crossover version",
        source_run_id="run_cross_1",
        source_candidate_id="cand_cross_1",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        configuration=cross_cfg,
        evaluation_snapshot={"groundedness": 1.0, "cost": 0.000030},
        status="approved",
        approved_by="admin@agentforge.ai",
    )
    db_session.merge(cross_version)
    db_session.commit()

    lineage = BenchmarkService.get_lineage(db=db_session)
    child_node = next((n for n in lineage.nodes if n.version_id == "wf_test_crossover_child"), None)
    assert child_node is not None
    assert child_node.candidate_type == "CROSSOVER"
    assert child_node.parent_a_version_id == p_a_id
    assert child_node.parent_b_version_id == p_b_id

    # Check incoming edges
    incoming = [e for e in lineage.edges if e.target == "wf_test_crossover_child"]
    assert len(incoming) == 2
    sources = {e.source for e in incoming}
    assert p_a_id in sources
    assert p_b_id in sources


@pytest.fixture
def evo_test_user(db_session):
    from app.core.security import hash_password
    from app.models.user import User, UserRole
    user = User(
        email="crossover_tester@agentforge.dev",
        password_hash=hash_password("testpassword123"),
        name="Crossover Tester",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


# ===========================================================================
# 6. REST API Endpoints Tests
# ===========================================================================

def test_api_crossover_preview_endpoint(client, evo_test_user, sample_parents):
    # 1. Login user session
    login_res = client.post(
        "/api/auth/login",
        json={"email": evo_test_user.email, "password": "testpassword123"},
    )
    assert login_res.status_code == 200

    p_a_id, p_b_id = sample_parents
    payload = {
        "parent_a_version_id": p_a_id,
        "parent_b_version_id": p_b_id,
        "crossover_map": {
            "model": "parent_a",
            "prompt": "parent_b",
            "retrieval": "parent_a",
            "generation": "parent_a",
            "operators": "parent_b",
        },
        "post_crossover_mutation": "none",
    }
    response = client.post("/api/agent-evo/crossover/preview", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert len(data["inheritance_breakdown"]) == 5
    assert data["parent_a"]["version_id"] == p_a_id
    assert data["parent_b"]["version_id"] == p_b_id


def test_api_crossover_candidate_execution(client, evo_test_user, sample_parents):
    from unittest.mock import AsyncMock, patch, MagicMock
    from app.schemas.rag import RAGSearchResponse, RetrievedChunk
    from app.services.llm.base import LLMResponse

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
                content="Assessment: Stable angina pectoris.",
                similarity_score=0.9,
                distance=0.1,
                relevance_score=0.9,
            )
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="The primary clinical diagnosis is stable angina pectoris.",
        latency_ms=150.0,
        success=True,
    )

    mock_adapter = MagicMock()
    mock_adapter.name = "Gemini 3.5 Flash-Lite"
    mock_adapter.generate = AsyncMock(return_value=mock_llm_response)
    mock_router = MagicMock()
    mock_router.get_adapter.return_value = mock_adapter

    p_a_id, p_b_id = sample_parents
    payload = {
        "parent_a_version_id": p_a_id,
        "parent_b_version_id": p_b_id,
        "crossover_map": {
            "model": "parent_a",
            "prompt": "parent_b",
            "retrieval": "parent_a",
            "generation": "parent_a",
            "operators": "parent_b",
        },
        "post_crossover_mutation": "none",
        "query": "What is the primary clinical diagnosis?",
    }

    with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
         patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

        response = client.post("/api/agent-evo/crossover/candidates", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["candidate_type"] == "CROSSOVER"
        assert data["parent_a_version_id"] == p_a_id
        assert data["parent_b_version_id"] == p_b_id
        assert "candidate_id" in data
        assert "groundedness" in data
        assert "cost" in data
        assert "pareto_status" in data

        # Test GET candidate endpoint
        cand_id = data["candidate_id"]
        get_res = client.get(f"/api/agent-evo/crossover/candidates/{cand_id}")
        assert get_res.status_code == 200
        assert get_res.json()["candidate_id"] == cand_id

        # Test list candidates endpoint
        list_res = client.get("/api/agent-evo/crossover/candidates")
        assert list_res.status_code == 200
        assert len(list_res.json()) >= 1

