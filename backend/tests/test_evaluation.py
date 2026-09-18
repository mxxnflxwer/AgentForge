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
    extract_factual_claims,
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
    assert len(result.supported_claims) >= 1
    assert result.supported_claim_count >= 1
    assert result.unsupported_claims == []
    assert result.unsupported_claim_count == 0
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


# --- 8. Morphological & Spelling Variant Tests (Specialty vs Speciality) ---

def test_groundedness_morphological_variants_speciality():
    """Verify that UK/US spelling variants (speciality vs specialty) are recognized as grounded."""
    context = (
        "### Section: Document Overview (Chunk #0)\n"
        "Document Overview: Patient 58 yo female. Specialty: Endocrinology. Chief Complaint: Thyroid nodule evaluation."
    )
    # Both US 'specialty' and UK 'speciality' must achieve 100% groundedness and 0% hallucination
    ans_us = "The specialty is Endocrinology."
    ans_uk = "The speciality is Endocrinology."

    res_us = calculate_groundedness(ans_us, context)
    res_uk = calculate_groundedness(ans_uk, context)

    assert res_us.groundedness == 1.0
    assert res_us.unsupported_claims == 0
    assert res_uk.groundedness == 1.0
    assert res_uk.unsupported_claims == 0

    h_us = calculate_hallucination_rate(ans_us, context, res_us)
    h_uk = calculate_hallucination_rate(ans_uk, context, res_uk)

    assert h_us.hallucination_rate == 0.0
    assert h_uk.hallucination_rate == 0.0


def test_groundedness_inflections():
    """Verify inflectional variants (diagnosed vs diagnosis, prescribed vs prescription) are grounded."""
    context = (
        "Assessment: Confirmed diagnosis of Type 2 Diabetes Mellitus. "
        "Plan: Prescription for Metformin 500mg daily."
    )
    answer = "The patient was diagnosed with Type 2 Diabetes Mellitus and prescribed Metformin 500mg."

    res = calculate_groundedness(answer, context)
    assert res.groundedness >= 0.90
    assert res.unsupported_claims == 0


def test_compare_endpoint_model_specific_timing(client: TestClient, mock_eval_user):
    """Verify that /api/query/compare returns model-specific latency and execution time."""
    from app.api.auth import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: mock_eval_user

    try:
        mock_retrieval = RAGSearchResponse(
            query="What is the specialty?",
            status="success",
            intent="specific_fact",
            total_results=1,
            results=[
                RetrievedChunk(
                    chunk_id="chk_time_1",
                    document_id="doc_time_1",
                    chunk_index=0,
                    content="Specialty: Endocrinology.",
                    section_title="Document Overview",
                    similarity_score=0.90,
                    relevance_score=0.92,
                )
            ],
        )

        mock_responses = [
            LLMResponse(
                model="Gemini 3.5 Flash-Lite",
                provider="Google",
                answer="The specialty is Endocrinology.",
                latency_ms=1100.0,
                success=True,
            ),
            LLMResponse(
                model="Qwen 3.6 27B",
                provider="OpenRouter",
                answer="The specialty is Endocrinology.",
                latency_ms=1450.0,
                success=True,
            ),
            LLMResponse(
                model="GPT-OSS 120B",
                provider="Hugging Face",
                answer="The speciality is Endocrinology.",
                latency_ms=4200.0,
                success=True,
            ),
        ]

        with patch("app.api.query.execute_rag_retrieval", return_value=mock_retrieval), \
             patch("app.api.query.get_llm_router") as mock_router_getter:

            mock_router = MagicMock()
            mock_router.compare_all = AsyncMock(return_value=mock_responses)
            mock_router_getter.return_value = mock_router

            payload = {
                "query": "What is the specialty?",
                "expected_answer": "Endocrinology",
            }

            res = client.post("/api/query/compare", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["context"] is not None
            assert len(data["results"]) == 3

            gemini_res = data["results"][0]
            qwen_res = data["results"][1]
            gpt_res = data["results"][2]

            # All 3 models must now have Groundedness = 1.0 (including GPT-OSS with 'speciality')
            assert gemini_res["evaluation"]["groundedness"] == 1.0
            assert qwen_res["evaluation"]["groundedness"] == 1.0
            assert gpt_res["evaluation"]["groundedness"] == 1.0

            # Latencies must be model-specific
            assert gemini_res["latency_ms"] == 1100.0
            assert qwen_res["latency_ms"] == 1450.0
            assert gpt_res["latency_ms"] == 4200.0

            # Execution times must be model-specific (not all identical 4.47s)
            assert gemini_res["evaluation"]["execution_time_ms"] < qwen_res["evaluation"]["execution_time_ms"]
            assert qwen_res["evaluation"]["execution_time_ms"] < gpt_res["evaluation"]["execution_time_ms"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)


# --- 9. Comprehensive Validation Cases (A - H) ---

def test_validation_case_a_exact_factual_answer():
    """Test A — Exact factual answer."""
    context = "Blood pressure: 118/76."
    answer = "The patient's blood pressure was 118/76."

    res = calculate_groundedness(answer, context)
    assert res.groundedness == 1.0
    assert res.supported_claims == 1
    assert res.unsupported_claims == 0

    h_res = calculate_hallucination_rate(answer, context, res)
    assert h_res.hallucination_rate == 0.0
    assert h_res.unsupported_claims == []


def test_validation_case_b_paraphrased_factual_answer():
    """Test B — Paraphrased factual answer."""
    context = "Blood pressure: 118/76."
    answer = "The recorded blood pressure was 118/76."

    res = calculate_groundedness(answer, context)
    assert res.groundedness == 1.0
    assert res.supported_claims == 1
    assert res.unsupported_claims == 0

    h_res = calculate_hallucination_rate(answer, context, res)
    assert h_res.hallucination_rate == 0.0
    assert h_res.unsupported_claims == []


def test_validation_case_c_unsupported_fact():
    """Test C — Unsupported fact."""
    context = "Blood pressure: 118/76."
    answer = "The patient's blood pressure was 118/76 and the patient has diabetes."

    res = calculate_groundedness(answer, context)
    assert res.groundedness == 0.50
    assert res.supported_claims == 1
    assert res.unsupported_claims == 1

    h_res = calculate_hallucination_rate(answer, context, res)
    assert h_res.hallucination_rate == 0.50
    assert len(h_res.unsupported_claims) == 1
    assert "patient has diabetes" in h_res.unsupported_claims[0].lower()


def test_validation_case_d_out_of_domain():
    """Test D — Out-of-domain."""
    context = "Blood pressure: 118/76. Specialty: Endocrinology."
    answer = "The requested information was not found in the uploaded document."

    res = calculate_groundedness(answer, context)
    assert res.groundedness == 1.0
    assert res.supported_claims == 0
    assert res.unsupported_claims == 0

    h_res = calculate_hallucination_rate(answer, context, res)
    assert h_res.hallucination_rate == 0.0
    assert h_res.unsupported_claims == []


def test_validation_case_e_empty_response():
    """Test E — Empty response and provider API failure (metrics N/A)."""
    engine = EvaluationEngine()
    context = "Blood pressure: 118/76."

    # 1. Provider API Failure
    error_input = EvaluationInput(
        question="What is the blood pressure?",
        generated_answer="No answer returned by model.",
        retrieved_context=context,
        model_name="qwen-3.6-27b",
        success=False,
        error="Empty choices list in response",
    )
    res_err = engine.evaluate(error_input)
    assert res_err.groundedness is None  # N/A
    assert res_err.hallucination_rate is None  # N/A
    assert res_err.accuracy is None  # N/A
    assert res_err.total_tokens == 0
    assert res_err.unsupported_claims == []
    assert res_err.details["status"] == "model_error"
    assert res_err.details["error_type"] == "provider_error"

    # 2. Empty model response
    empty_input = EvaluationInput(
        question="What is the blood pressure?",
        generated_answer="",
        retrieved_context=context,
        model_name="gemini-3.5-flash-lite",
        success=True,
    )
    res_empty = engine.evaluate(empty_input)
    assert res_empty.groundedness is None  # N/A
    assert res_empty.hallucination_rate is None  # N/A
    assert res_empty.accuracy is None  # N/A
    assert res_empty.total_tokens == 0
    assert res_empty.unsupported_claims == []
    assert res_empty.details["status"] == "empty_response"
    assert res_empty.details["error_type"] == "empty_model_response"


def test_validation_case_f_mixed_claims():
    """Test F — Mixed claims."""
    context = "Blood pressure: 118/76.\nMedication: atorvastatin 40 mg daily."
    answer = "The patient's blood pressure was 118/76 and they were prescribed atorvastatin 40 mg daily. The patient also has diabetes."

    res = calculate_groundedness(answer, context)
    assert round(res.groundedness, 3) == 0.667
    assert res.supported_claims == 2
    assert res.unsupported_claims == 1

    h_res = calculate_hallucination_rate(answer, context, res)
    assert round(h_res.hallucination_rate, 3) == 0.333
    assert len(h_res.unsupported_claims) == 1
    assert "patient also has diabetes" in h_res.unsupported_claims[0].lower()


def test_validation_case_g_markdown_summary():
    """Test G — Markdown summary (headings/formatting not treated as claims)."""
    context = (
        "### Section: Physical Examination\nBlood pressure: 118/76.\n"
        "### Section: Assessment\nPatient presents with stable thyroid nodule.\n"
        "### Section: Plan\nPrescribe levothyroxine 50mcg daily."
    )
    answer = (
        "# Summary:\n"
        "## Assessment:\n"
        "Patient presents with stable thyroid nodule.\n"
        "## Plan:\n"
        "Prescribe levothyroxine 50mcg daily.\n"
        "## Physical Examination:\n"
        "Blood pressure: 118/76."
    )

    claims = extract_factual_claims(answer)
    assert "Summary:" not in claims
    assert "Assessment:" not in claims
    assert "Plan:" not in claims
    assert "Physical Examination:" not in claims
    assert len(claims) == 3

    res = calculate_groundedness(answer, context)
    assert res.groundedness == 1.0
    assert res.supported_claims == 3
    assert res.unsupported_claims == 0


def test_validation_case_h_diagnosis():
    """Test H — Diagnosis."""
    context = "Findings are consistent with primary hypothyroidism, most likely due to autoimmune (Hashimoto's) thyroiditis."
    answer = "Based on the provided document, the clinical diagnosis is primary hypothyroidism, most likely due to autoimmune (Hashimoto's) thyroiditis."

    res = calculate_groundedness(answer, context)
    assert res.groundedness == 1.0
    assert res.supported_claims == 1
    assert res.unsupported_claims == 0

    h_res = calculate_hallucination_rate(answer, context, res)
    assert h_res.hallucination_rate == 0.0
    assert h_res.unsupported_claims == []


