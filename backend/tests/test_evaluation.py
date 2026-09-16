import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.models.user import User, UserRole
from app.services.evaluation import (
    EvaluationEngine,
    EvaluationInput,
    EvaluationResult,
    calculate_accuracy,
    calculate_cost,
    calculate_groundedness,
    calculate_hallucination_rate,
    evaluate_workflow,
    extract_token_usage,
    get_evaluation_engine,
    get_model_pricing,
)
from app.schemas.rag import RAGSearchResponse, RetrievedChunk
from app.services.llm.base import LLMResponse


@pytest.fixture
def mock_eval_user(db_session):
    user = User(
        email="eval_test_user@agentforge.dev",
        password_hash="hashed_pw_test",
        name="Evaluation Test User",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


# --- 1. Accuracy Tests ---

def test_accuracy_exact_match():
    acc = calculate_accuracy(
        generated_answer="Stable angina pectoris.",
        expected_answer="Stable angina pectoris.",
    )
    assert acc == 1.0


def test_accuracy_high_factual_overlap():
    acc = calculate_accuracy(
        generated_answer="The patient was diagnosed with stable angina pectoris and mild coronary artery disease.",
        expected_answer="Diagnosis is stable angina pectoris due to coronary artery disease.",
    )
    assert acc is not None
    assert acc >= 0.85


def test_accuracy_low_for_incorrect_answer():
    acc = calculate_accuracy(
        generated_answer="Patient is diagnosed with acute appendicitis and pneumonia.",
        expected_answer="Diagnosis is stable angina pectoris.",
    )
    assert acc is not None
    assert acc <= 0.30


def test_accuracy_none_when_no_reference():
    acc = calculate_accuracy(
        generated_answer="Stable angina pectoris.",
        expected_answer=None,
    )
    assert acc is None


# --- 2. Groundedness Tests ---

def test_groundedness_fully_supported():
    context = (
        "Assessment: Patient presents with stable angina pectoris. "
        "Plan: Prescribed Atorvastatin 40mg daily and Aspirin 81mg."
    )
    answer = "The diagnosis is stable angina pectoris. Prescribed Atorvastatin 40mg and Aspirin 81mg."

    result = calculate_groundedness(generated_answer=answer, retrieved_context=context)
    assert result.groundedness >= 0.90
    assert result.supported_claims >= 1
    assert result.unsupported_claims == 0


def test_groundedness_partially_supported():
    context = "Assessment: Patient presents with stable angina pectoris."
    answer = "The diagnosis is stable angina pectoris. Patient was scheduled for emergency brain surgery."

    result = calculate_groundedness(generated_answer=answer, retrieved_context=context)
    assert 0.0 < result.groundedness < 1.0
    assert result.supported_claims >= 1
    assert result.unsupported_claims >= 1


def test_groundedness_safe_not_found_response():
    context = "Assessment: Stable angina pectoris."
    answer = "The requested information was not found in the uploaded document."

    result = calculate_groundedness(generated_answer=answer, retrieved_context=context)
    assert result.groundedness == 1.0
    assert result.total_claims == 0


# --- 3. Hallucination Tests ---

def test_hallucination_rate_zero_when_grounded():
    context = "Assessment: Stable angina pectoris. Blood pressure: 130/80 mmHg."
    answer = "The patient has stable angina pectoris with blood pressure 130/80."

    h_res = calculate_hallucination_rate(generated_answer=answer, retrieved_context=context)
    assert h_res.hallucination_rate == 0.0
    assert h_res.hallucinated_claims == 0
    assert len(h_res.unsupported_claims) == 0


def test_hallucination_rate_detects_unsupported_claims():
    context = "Assessment: Stable angina pectoris."
    answer = "The patient has stable angina pectoris. Patient's phone number is 555-0199 and SSN is 123-45-6789."

    h_res = calculate_hallucination_rate(generated_answer=answer, retrieved_context=context)
    assert h_res.hallucination_rate > 0.0
    assert h_res.hallucinated_claims >= 1
    assert len(h_res.unsupported_claims) >= 1
    # Verify hallucinated claim is preserved for the optimization report
    unsupported_str = " ".join(h_res.unsupported_claims)
    assert "555-0199" in unsupported_str or "phone" in unsupported_str.lower() or "ssn" in unsupported_str.lower()


# --- 4. Token Usage Tests ---

def test_token_usage_raw_usage_gemini():
    raw_usage = {
        "promptTokenCount": 150,
        "candidatesTokenCount": 45,
        "totalTokenCount": 195,
    }
    tokens = extract_token_usage(
        raw_usage=raw_usage,
        generated_answer="Answer text",
        retrieved_context="Context text",
    )
    assert tokens.input_tokens == 150
    assert tokens.output_tokens == 45
    assert tokens.total_tokens == 195
    assert tokens.input_tokens + tokens.output_tokens == tokens.total_tokens


def test_token_usage_raw_usage_openai():
    raw_usage = {
        "prompt_tokens": 200,
        "completion_tokens": 60,
        "total_tokens": 260,
    }
    tokens = extract_token_usage(
        raw_usage=raw_usage,
        generated_answer="Answer text",
        retrieved_context="Context text",
    )
    assert tokens.input_tokens == 200
    assert tokens.output_tokens == 60
    assert tokens.total_tokens == 260
    assert tokens.input_tokens + tokens.output_tokens == tokens.total_tokens


def test_token_usage_fallback_estimation():
    tokens = extract_token_usage(
        raw_usage=None,
        generated_answer="Stable angina pectoris diagnosed.",
        retrieved_context="Assessment: Patient presents with stable angina pectoris.",
        query="What is the diagnosis?",
    )
    assert tokens.input_tokens > 0
    assert tokens.output_tokens > 0
    assert tokens.total_tokens == tokens.input_tokens + tokens.output_tokens


# --- 5. Cost Computation Tests ---

def test_cost_calculation():
    # Model: Gemini 3.5 Flash-Lite (input: $0.075 / 1M, output: $0.30 / 1M)
    cost = calculate_cost(
        model_name="gemini-3.5-flash-lite",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost.input_cost == 0.075
    assert cost.output_cost == 0.30
    assert round(cost.total_cost, 4) == 0.375


def test_cost_calculation_fractional():
    cost = calculate_cost(
        model_name="openai/gpt-oss-120b",
        input_tokens=1000,
        output_tokens=200,
    )
    # 1000 / 1M * 0.50 = 0.0005
    # 200 / 1M * 0.50 = 0.0001
    assert round(cost.input_cost, 6) == 0.0005
    assert round(cost.output_cost, 6) == 0.0001
    assert round(cost.total_cost, 6) == 0.0006


# --- 6. Complete Evaluation Engine Tests ---

def test_evaluation_engine_complete():
    engine = EvaluationEngine()
    eval_input = EvaluationInput(
        question="What is the clinical diagnosis?",
        generated_answer="The diagnosis is stable angina pectoris.",
        retrieved_context="Assessment: Patient presents with stable angina pectoris.",
        expected_answer="Stable angina pectoris.",
        model_name="gemini-3.5-flash-lite",
        input_tokens=150,
        output_tokens=30,
        latency_ms=850.0,
        execution_time_ms=1150.0,
    )

    result = engine.evaluate(eval_input)

    assert result.model == "gemini-3.5-flash-lite"
    assert result.accuracy == 1.0
    assert result.groundedness == 1.0
    assert result.hallucination_rate == 0.0
    assert result.supported_claims >= 1
    assert result.unsupported_claims == []
    assert result.input_tokens == 150
    assert result.output_tokens == 30
    assert result.total_tokens == 180
    assert result.latency_ms == 850.0
    assert result.execution_time_ms == 1150.0
    assert result.total_cost > 0.0
    assert "claim_breakdown" in result.details


def test_evaluate_workflow_interface():
    workflow_data = {
        "question": "What is the treatment plan?",
        "generated_answer": "Prescribed Aspirin 81mg and Atorvastatin 40mg daily.",
        "retrieved_context": "Plan: Prescribe Aspirin 81mg and Atorvastatin 40mg daily.",
        "model_name": "qwen-3.6-27b",
        "latency_ms": 920.0,
    }

    result = evaluate_workflow(workflow_data)
    assert isinstance(result, EvaluationResult)
    assert result.model == "qwen-3.6-27b"
    assert result.groundedness >= 0.90
    assert result.latency_ms == 920.0


# --- 7. API Endpoint Tests ---

def test_api_evaluate_endpoint(client: TestClient):
    payload = {
        "question": "What is the diagnosis?",
        "generated_answer": "Patient diagnosed with stable angina.",
        "retrieved_context": "Assessment: Patient diagnosed with stable angina.",
        "expected_answer": "Stable angina",
        "model_name": "gemini-3.5-flash-lite",
        "input_tokens": 120,
        "output_tokens": 25,
        "latency_ms": 780.0,
    }

    res = client.post("/api/evaluation/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "gemini-3.5-flash-lite"
    assert data["accuracy"] == 1.0
    assert data["groundedness"] == 1.0
    assert data["hallucination_rate"] == 0.0
    assert data["total_tokens"] == 145
    assert data["total_cost"] > 0


def test_api_evaluate_query_endpoint(client: TestClient, mock_eval_user):
    from app.api.auth import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: mock_eval_user

    try:
        mock_retrieval = RAGSearchResponse(
            query="What is the clinical diagnosis?",
            status="success",
            intent="diagnosis",
            total_results=1,
            results=[
                RetrievedChunk(
                    chunk_id="chk_eval_1",
                    document_id="doc_eval_1",
                    chunk_index=1,
                    content="Assessment: Stable angina pectoris.",
                    section_title="Assessment",
                    similarity_score=0.95,
                    relevance_score=0.98,
                )
            ],
        )

        with patch("app.api.evaluation.execute_rag_retrieval", return_value=mock_retrieval), \
             patch("app.api.evaluation.get_llm_router") as mock_router_getter:
            
            mock_router = MagicMock()
            mock_adapter = MagicMock()
            mock_adapter.name = "Gemini 3.5 Flash-Lite"
            mock_router.get_adapter.return_value = mock_adapter

            mock_llm_resp = LLMResponse(
                model="Gemini 3.5 Flash-Lite",
                provider="Google",
                answer="The patient is diagnosed with stable angina pectoris.",
                latency_ms=110.0,
                success=True,
                raw_usage={"promptTokenCount": 90, "candidatesTokenCount": 20},
            )
            mock_router.generate_single = AsyncMock(return_value=mock_llm_resp)
            mock_router_getter.return_value = mock_router

            payload = {
                "query": "What is the clinical diagnosis?",
                "model": "gemini",
                "expected_answer": "Stable angina pectoris",
            }

            res = client.post("/api/evaluation/evaluate-query", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["query"] == "What is the clinical diagnosis?"
            assert "stable angina pectoris" in data["answer"]
            assert data["evaluation"]["groundedness"] == 1.0
            assert data["evaluation"]["accuracy"] == 1.0
            assert data["evaluation"]["hallucination_rate"] == 0.0
            assert data["evaluation"]["total_tokens"] == 110
            assert data["evaluation"]["latency_ms"] == 110.0
            assert data["evaluation"]["total_cost"] > 0
    finally:
        app.dependency_overrides.pop(get_current_user, None)
