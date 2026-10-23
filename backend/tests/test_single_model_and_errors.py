import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.models.user import User, UserRole
from app.services.llm.base import LLMResponse
from app.services.llm.gemini import GeminiAdapter
from app.services.llm.gemma import GemmaAdapter
from app.services.llm.gpt_oss import GPTOSSAdapter
from app.services.llm.qwen import QwenAdapter
from app.services.llm.router import LLMRouter, get_llm_router
from app.services.evaluation.evaluator import EvaluationEngine, EvaluationInput
from app.schemas.rag import RAGSearchResponse, RetrievedChunk
from app.api.auth import get_current_user
from app.main import app


@pytest.fixture
def mock_auth_user(db_session):
    user = User(
        email="routing_test_user@agentforge.dev",
        password_hash="hashed_pw_test",
        name="Routing Test User",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def mock_rag_retrieval():
    return RAGSearchResponse(
        query="What is the diagnosis?",
        intent="CLINICAL_QUERY",
        status="success",
        total_results=1,
        results=[
            RetrievedChunk(
                chunk_id="chk-123",
                document_id="doc-456",
                chunk_index=0,
                content="Assessment: Stable angina pectoris. Prescribed aspirin 81mg.",
                section_title="Assessment and Plan",
                similarity_score=0.92,
                relevance_score=0.92,
                distance=0.08,
            )
        ],
    )


def test_unsupported_model_returns_400(client: TestClient, mock_auth_user):
    """Unsupported model ID raises HTTP 400 instead of silently falling back to Gemini."""
    resp = client.post(
        "/api/query/answer",
        json={
            "query": "What is the diagnosis?",
            "model": "non-existent-model-xyz",
            "top_k": 3,
        },
    )
    assert resp.status_code == 400
    assert "not supported or unavailable" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_single_model_gemini_routing(client: TestClient, mock_auth_user, mock_rag_retrieval):
    """Selecting Gemini routes to Gemini adapter and returns proper metadata."""
    mock_gemini_resp = LLMResponse(
        answer="Patient has stable angina pectoris.",
        model="Gemini 3.5 Flash-Lite",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        latency_ms=120.5,
        success=True,
        raw_usage={"input_tokens": 40, "output_tokens": 15, "total_tokens": 55},
    )

    with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag_retrieval), \
         patch.object(GeminiAdapter, "generate", new_callable=AsyncMock) as mock_gemini_gen, \
         patch.object(QwenAdapter, "generate", new_callable=AsyncMock) as mock_qwen_gen, \
         patch.object(GPTOSSAdapter, "generate", new_callable=AsyncMock) as mock_gpt_gen:

        mock_gemini_gen.return_value = mock_gemini_resp

        resp = client.post(
            "/api/query/answer",
            json={
                "query": "What is the diagnosis?",
                "model": "gemini",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Gemini 3.5 Flash-Lite"
        assert data["model_id"] == "gemini-3.5-flash-lite"
        assert data["provider"] == "Google"
        assert data["success"] is True
        assert data["status"] == "success"
        assert data["answer"] == "Patient has stable angina pectoris."

        # Verify ONLY Gemini was called
        mock_gemini_gen.assert_called_once()
        mock_qwen_gen.assert_not_called()
        mock_gpt_gen.assert_not_called()


@pytest.mark.asyncio
async def test_single_model_qwen_routing(client: TestClient, mock_auth_user, mock_rag_retrieval):
    """Selecting Qwen routes to Qwen adapter only without Gemini fallback."""
    mock_qwen_resp = LLMResponse(
        answer="Diagnosis confirmed: Stable angina pectoris.",
        model="Qwen 3.6 27B",
        model_id="qwen/qwen3.6-27b",
        provider="OpenRouter",
        latency_ms=250.0,
        success=True,
        raw_usage={"input_tokens": 45, "output_tokens": 18, "total_tokens": 63},
    )

    with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag_retrieval), \
         patch.object(GeminiAdapter, "generate", new_callable=AsyncMock) as mock_gemini_gen, \
         patch.object(QwenAdapter, "generate", new_callable=AsyncMock) as mock_qwen_gen, \
         patch.object(GPTOSSAdapter, "generate", new_callable=AsyncMock) as mock_gpt_gen:

        mock_qwen_gen.return_value = mock_qwen_resp

        resp = client.post(
            "/api/query/answer",
            json={
                "query": "What is the diagnosis?",
                "model": "qwen",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Qwen 3.6 27B"
        assert data["model_id"] == "qwen/qwen3.6-27b"
        assert data["provider"] == "OpenRouter"
        assert data["success"] is True
        assert data["status"] == "success"
        assert "Stable angina" in data["answer"]

        # Verify ONLY Qwen was called
        mock_qwen_gen.assert_called_once()
        mock_gemini_gen.assert_not_called()
        mock_gpt_gen.assert_not_called()


@pytest.mark.asyncio
async def test_single_model_gemma_routing(client: TestClient, mock_auth_user, mock_rag_retrieval):
    """Selecting Gemma routes to Gemma adapter only without Gemini or Qwen fallback."""
    mock_gemma_resp = LLMResponse(
        answer="The clinical diagnosis is stable angina pectoris.",
        model="Gemma 4 26B A4B (OpenRouter)",
        model_id="google/gemma-4-26b-a4b-it:free",
        provider="OpenRouter",
        latency_ms=350.0,
        success=True,
        raw_usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
    )

    with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag_retrieval), \
         patch.object(GeminiAdapter, "generate", new_callable=AsyncMock) as mock_gemini_gen, \
         patch.object(QwenAdapter, "generate", new_callable=AsyncMock) as mock_qwen_gen, \
         patch.object(GemmaAdapter, "generate", new_callable=AsyncMock) as mock_gemma_gen:

        mock_gemma_gen.return_value = mock_gemma_resp

        resp = client.post(
            "/api/query/answer",
            json={
                "query": "What is the diagnosis?",
                "model": "gemma",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Gemma 4 26B A4B (OpenRouter)"
        assert data["model_id"] == "google/gemma-4-26b-a4b-it:free"
        assert data["provider"] == "OpenRouter"
        assert data["success"] is True
        assert data["status"] == "success"
        assert "stable angina" in data["answer"]

        # Verify ONLY Gemma was called
        mock_gemma_gen.assert_called_once()
        mock_gemini_gen.assert_not_called()
        mock_qwen_gen.assert_not_called()


@pytest.mark.asyncio
async def test_single_model_legacy_gpt_oss_alias_routing(client: TestClient, mock_auth_user, mock_rag_retrieval):
    """Selecting legacy 'gpt_oss' model alias routes seamlessly to Gemma adapter."""
    mock_gemma_resp = LLMResponse(
        answer="The clinical diagnosis is stable angina pectoris.",
        model="Gemma 4 26B A4B (OpenRouter)",
        model_id="google/gemma-4-26b-a4b-it:free",
        provider="OpenRouter",
        latency_ms=350.0,
        success=True,
        raw_usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
    )

    with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag_retrieval), \
         patch.object(GemmaAdapter, "generate", new_callable=AsyncMock) as mock_gemma_gen:

        mock_gemma_gen.return_value = mock_gemma_resp

        resp = client.post(
            "/api/query/answer",
            json={
                "query": "What is the diagnosis?",
                "model": "gpt_oss",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Gemma 4 26B A4B (OpenRouter)"
        assert data["provider"] == "OpenRouter"
        assert data["success"] is True


@pytest.mark.asyncio
async def test_gemma_insufficient_credits_handling(client: TestClient, mock_auth_user, mock_rag_retrieval):
    """HTTP 402 or credit error produces insufficient_credits status and N/A (None) tokens and cost."""
    mock_failed_resp = LLMResponse(
        answer="Insufficient provider credits: OpenRouter requires credits or free-tier quota is exhausted.",
        model="Gemma 4 26B A4B (OpenRouter)",
        model_id="google/gemma-4-26b-a4b-it:free",
        provider="OpenRouter",
        latency_ms=180.0,
        success=False,
        error="Insufficient provider credits: HTTP 402 Payment Required",
        raw_usage=None,
    )

    with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag_retrieval), \
         patch.object(GemmaAdapter, "generate", new_callable=AsyncMock) as mock_gemma_gen:

        mock_gemma_gen.return_value = mock_failed_resp

        resp = client.post(
            "/api/query/answer",
            json={
                "query": "What is the diagnosis?",
                "model": "gemma",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Gemma 4 26B A4B (OpenRouter)"
        assert data["model_id"] == "google/gemma-4-26b-a4b-it:free"
        assert data["provider"] == "OpenRouter"
        assert data["success"] is False
        assert data["status"] == "insufficient_credits"
        assert "Insufficient provider credits" in data["answer"]
        assert data["error"] is not None

        # Telemetry verification: failure MUST NOT fabricate tokens, costs, or 100% groundedness
        eval_metrics = data["evaluation"]
        assert eval_metrics is not None
        assert eval_metrics["groundedness"] is None
        assert eval_metrics["total_tokens"] is None
        assert eval_metrics["input_tokens"] is None
        assert eval_metrics["output_tokens"] is None
        assert eval_metrics["total_cost"] is None
        assert eval_metrics["input_cost"] is None
        assert eval_metrics["output_cost"] is None
        assert eval_metrics["details"]["status"] == "insufficient_credits"


@pytest.mark.asyncio
async def test_multi_model_comparison_with_one_failed_provider(client: TestClient, mock_auth_user, mock_rag_retrieval):
    """In 3-model comparison, if Gemma fails with 402/429, Gemini and Qwen succeed and Gemma shows failure."""
    gemini_resp = LLMResponse(
        answer="Gemini answer: Stable angina pectoris.",
        model="Gemini 3.5 Flash-Lite",
        model_id="gemini-3.5-flash-lite",
        provider="Google",
        latency_ms=110.0,
        success=True,
        raw_usage={"input_tokens": 40, "output_tokens": 15, "total_tokens": 55},
    )
    qwen_resp = LLMResponse(
        answer="Qwen answer: Stable angina pectoris.",
        model="Qwen 3.6 27B",
        model_id="qwen/qwen3.6-27b",
        provider="OpenRouter",
        latency_ms=210.0,
        success=True,
        raw_usage={"input_tokens": 42, "output_tokens": 16, "total_tokens": 58},
    )
    gemma_resp = LLMResponse(
        answer="Insufficient provider credits: OpenRouter free tier limit reached.",
        model="Gemma 4 26B A4B (OpenRouter)",
        model_id="google/gemma-4-26b-a4b-it:free",
        provider="OpenRouter",
        latency_ms=85.0,
        success=False,
        error="Insufficient provider credits: HTTP 402",
        raw_usage=None,
    )

    with patch("app.api.query.execute_rag_retrieval", return_value=mock_rag_retrieval), \
         patch.object(GeminiAdapter, "generate", new_callable=AsyncMock, return_value=gemini_resp), \
         patch.object(QwenAdapter, "generate", new_callable=AsyncMock, return_value=qwen_resp), \
         patch.object(GemmaAdapter, "generate", new_callable=AsyncMock, return_value=gemma_resp):

        resp = client.post(
            "/api/query/compare",
            json={
                "query": "What is the diagnosis?",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        results = data["results"]
        assert len(results) == 3

        gemini_res = next(r for r in results if r["model_id"] == "gemini-3.5-flash-lite")
        qwen_res = next(r for r in results if r["model_id"] == "qwen/qwen3.6-27b")
        gemma_res = next(r for r in results if r["model_id"] == "google/gemma-4-26b-a4b-it:free")

        # Gemini and Qwen must be successful
        assert gemini_res["success"] is True
        assert gemini_res["evaluation"]["groundedness"] is not None
        assert gemini_res["evaluation"]["total_cost"] is not None

        assert qwen_res["success"] is True
        assert qwen_res["evaluation"]["groundedness"] is not None
        assert qwen_res["evaluation"]["total_cost"] is not None

        # Gemma must be failed with credit notice and None metrics
        assert gemma_res["success"] is False
        assert "Insufficient provider credits" in gemma_res["answer"]
        assert gemma_res["evaluation"]["groundedness"] is None
        assert gemma_res["evaluation"]["total_tokens"] is None
        assert gemma_res["evaluation"]["total_cost"] is None
        assert gemma_res["evaluation"]["details"]["status"] == "insufficient_credits"


def test_evaluator_engine_handles_all_error_types():
    """EvaluationEngine distinguishes provider_error, insufficient_credits, and empty_response."""
    engine = EvaluationEngine()

    # 1. Insufficient credits
    res_credits = engine.evaluate(
        EvaluationInput(
            question="What is the diagnosis?",
            generated_answer="Insufficient provider credits",
            retrieved_context="Stable angina",
            model_name="GPT-OSS 120B",
            success=False,
            error="HTTP 402: You have no remaining credits",
        )
    )
    assert res_credits.groundedness is None
    assert res_credits.total_tokens is None
    assert res_credits.total_cost is None
    assert res_credits.details["status"] == "insufficient_credits"

    # 2. Provider error
    res_err = engine.evaluate(
        EvaluationInput(
            question="What is the diagnosis?",
            generated_answer="Connection timeout",
            retrieved_context="Stable angina",
            model_name="Qwen 3.6 27B",
            success=False,
            error="Connection timed out after 30 seconds",
        )
    )
    assert res_err.groundedness is None
    assert res_err.total_tokens is None
    assert res_err.total_cost is None
    assert res_err.details["status"] == "provider_error"

    # 3. Empty response
    res_empty = engine.evaluate(
        EvaluationInput(
            question="What is the diagnosis?",
            generated_answer="",
            retrieved_context="Stable angina",
            model_name="Gemini 3.5 Flash-Lite",
            success=True,
            error=None,
        )
    )
    assert res_empty.groundedness is None
    assert res_empty.details["status"] == "empty_response"


def test_pareto_archive_excludes_failed_candidates():
    """Failed candidates (status != 'success' or groundedness is None) are strictly excluded from Pareto frontier."""
    from app.services.agent_evo import ParetoArchive, EvaluatedCandidate, CandidateMetrics, get_baseline_workflow, dominates

    wf = get_baseline_workflow()
    archive = ParetoArchive()

    cand_success = EvaluatedCandidate(
        candidate_id="cand_success",
        workflow=wf.clone("cand_success", "Successful Mutant"),
        metrics=CandidateMetrics(
            workflow_id="cand_success",
            workflow_name="Successful Mutant",
            performance=0.90,
            groundedness=0.90,
            cost=0.0001,
            llm_latency_ms=100.0,
            status="success",
        ),
    )

    cand_credits = EvaluatedCandidate(
        candidate_id="cand_failed_credits",
        workflow=wf.clone("cand_failed_credits", "Credits Exhausted Mutant"),
        metrics=CandidateMetrics(
            workflow_id="cand_failed_credits",
            workflow_name="Credits Exhausted Mutant",
            performance=0.0,
            groundedness=None,
            cost=0.0,
            llm_latency_ms=0.0,
            status="insufficient_credits",
            error_message="HTTP 402 No credits remaining",
        ),
    )

    cand_provider = EvaluatedCandidate(
        candidate_id="cand_failed_provider",
        workflow=wf.clone("cand_failed_provider", "Provider Error Mutant"),
        metrics=CandidateMetrics(
            workflow_id="cand_failed_provider",
            workflow_name="Provider Error Mutant",
            performance=0.0,
            groundedness=None,
            cost=0.0,
            llm_latency_ms=0.0,
            status="failed",
            error_message="HTTP 500 Server Error",
        ),
    )

    # Adding failed candidates must return False and not enter Pareto archive
    assert archive.add_candidate(cand_credits) is False
    assert archive.add_candidate(cand_provider) is False
    assert archive.add_candidate(cand_success) is True

    pareto_ids = [c.candidate_id for c in archive.pareto_candidates]
    assert "cand_success" in pareto_ids
    assert "cand_failed_credits" not in pareto_ids
    assert "cand_failed_provider" not in pareto_ids

    # Dominance checks
    assert dominates(cand_success.metrics, cand_credits.metrics) is True
    assert dominates(cand_credits.metrics, cand_success.metrics) is False
    assert dominates(cand_credits.metrics, cand_provider.metrics) is False
