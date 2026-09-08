import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.models.user import User, UserRole
from app.services.llm.base import (
    DEFAULT_MEDICAL_SYSTEM_PROMPT,
    MEDICAL_DISCLAIMER,
    LLMResponse,
    build_medical_prompt,
)
from app.services.llm.gemini import GeminiAdapter
from app.services.llm.gpt_oss import GPTOSSAdapter
from app.services.llm.qwen import QwenAdapter
from app.services.llm.router import LLMRouter, get_llm_router
from app.schemas.rag import RAGSearchResponse, RetrievedChunk


@pytest.fixture
def mock_auth_user(db_session):
    user = User(
        email="test_llm_user@agentforge.dev",
        password_hash="hashed_pw_test",
        name="LLM Test User",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_prompt_formatting():
    prompt = build_medical_prompt(
        query="What is the clinical diagnosis?",
        context="Assessment: Stable angina pectoris.",
    )
    assert "[DOCUMENT CONTEXT]" in prompt
    assert "Assessment: Stable angina pectoris." in prompt
    assert "[USER QUESTION]" in prompt
    assert "What is the clinical diagnosis?" in prompt
    assert "[INSTRUCTIONS]" in prompt


def test_offline_fallback():
    adapter = GeminiAdapter()
    resp = adapter.generate_offline_fallback(
        query="What is the diagnosis?",
        context="Assessment: Patient diagnosed with Type 2 Diabetes.",
        reason="No API key provided",
    )
    assert resp.model == "Gemini 2.5 Flash-Lite"
    assert resp.success is False
    assert "No API key provided" in (resp.error or "")
    assert "Type 2 Diabetes" in resp.answer


@pytest.mark.asyncio
async def test_gemini_adapter_mock_success():
    adapter = GeminiAdapter(api_key="mock_gemini_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Patient has stable angina according to assessment."}]
                }
            }
        ],
        "usageMetadata": {"totalTokenCount": 85},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await adapter.generate(
            query="What is the diagnosis?",
            context="Patient has stable angina.",
        )
        assert result.success is True
        assert result.model == "Gemini 2.5 Flash-Lite"
        assert result.provider == "Google"
        assert "stable angina" in result.answer
        assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_qwen_adapter_mock_success():
    adapter = QwenAdapter(api_key="mock_qwen_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Coronary artery disease is indicated in the report."
                }
            }
        ],
        "usage": {"total_tokens": 92},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await adapter.generate(
            query="What condition is indicated?",
            context="Findings show coronary artery disease.",
        )
        assert result.success is True
        assert result.model == "Qwen 3.6 27B"
        assert result.provider == "Qwen"
        assert "Coronary artery disease" in result.answer


@pytest.mark.asyncio
async def test_gpt_oss_adapter_mock_success():
    adapter = GPTOSSAdapter(api_key="mock_gpt_oss_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Blood pressure was recorded as 130/85 mmHg."
                }
            }
        ],
        "usage": {"total_tokens": 64},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await adapter.generate(
            query="What is the blood pressure?",
            context="Physical Exam: BP 130/85 mmHg.",
        )
        assert result.success is True
        assert result.model == "GPT-OSS 120B"
        assert result.provider == "OpenAI-Compatible"
        assert "130/85" in result.answer


@pytest.mark.asyncio
async def test_llm_router_compare_all():
    router = LLMRouter()
    
    # Mock all generate calls
    with patch.object(GeminiAdapter, "generate", new_callable=AsyncMock) as mock_gemini, \
         patch.object(QwenAdapter, "generate", new_callable=AsyncMock) as mock_qwen, \
         patch.object(GPTOSSAdapter, "generate", new_callable=AsyncMock) as mock_gpt:
        
        mock_gemini.return_value = LLMResponse(
            model="Gemini 2.5 Flash-Lite",
            provider="Google",
            answer="Gemini answer.",
            latency_ms=105.2,
            success=True,
        )
        mock_qwen.return_value = LLMResponse(
            model="Qwen 3.6 27B",
            provider="Qwen",
            answer="Qwen answer.",
            latency_ms=150.8,
            success=True,
        )
        mock_gpt.return_value = LLMResponse(
            model="GPT-OSS 120B",
            provider="OpenAI-Compatible",
            answer="GPT-OSS answer.",
            latency_ms=210.4,
            success=True,
        )

        results = await router.compare_all(
            query="What is the diagnosis?",
            context="Assessment: Hypertension.",
        )

        assert len(results) == 3
        model_names = [r.model for r in results]
        assert "Gemini 2.5 Flash-Lite" in model_names
        assert "Qwen 3.6 27B" in model_names
        assert "GPT-OSS 120B" in model_names
        assert all(r.success for r in results)


def test_api_models_endpoint(client: TestClient):
    response = client.get("/api/query/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    names = [m["name"] for m in data["models"]]
    assert "Gemini 2.5 Flash-Lite" in names
    assert "Qwen 3.6 27B" in names
    assert "GPT-OSS 120B" in names


def test_api_query_answer_not_found_safety(client: TestClient, mock_auth_user):
    from app.api.auth import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: mock_auth_user

    try:
        # Mock retrieval returning not_found
        with patch("app.api.query.execute_rag_retrieval") as mock_rag:
            mock_rag.return_value = RAGSearchResponse(
                query="What is the patient phone number?",
                status="not_found",
                intent="unsupported_query",
                total_results=0,
                results=[],
                message="The requested information was not found in the uploaded document.",
            )

            res = client.post(
                "/api/query/answer",
                json={
                    "query": "What is the patient phone number?",
                    "model": "gemini",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "not_found"
            assert "The requested information was not found in the uploaded document." in data["answer"]
            assert data["sources"] == []
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_api_query_answer_success(client: TestClient, mock_auth_user):
    from app.api.auth import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: mock_auth_user

    try:
        mock_retrieval = RAGSearchResponse(
            query="What is the clinical diagnosis?",
            status="success",
            intent="diagnosis",
            total_results=1,
            results=[
                RetrievedChunk(
                    chunk_id="chk_123",
                    document_id="doc_abc",
                    chunk_index=2,
                    content="Assessment: Stable angina pectoris.",
                    section_title="Assessment",
                    similarity_score=0.92,
                    relevance_score=0.95,
                )
            ],
        )

        with patch("app.api.query.execute_rag_retrieval", return_value=mock_retrieval), \
             patch.object(GeminiAdapter, "generate", new_callable=AsyncMock) as mock_gemini:
            
            mock_gemini.return_value = LLMResponse(
                model="Gemini 2.5 Flash-Lite",
                provider="Google",
                answer="The patient is diagnosed with stable angina pectoris.",
                latency_ms=115.0,
                success=True,
            )

            res = client.post(
                "/api/query/answer",
                json={
                    "query": "What is the clinical diagnosis?",
                    "model": "gemini",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert "stable angina pectoris" in data["answer"]
            assert len(data["sources"]) == 1
            assert data["sources"][0]["chunk_id"] == "chk_123"
            assert data["sources"][0]["section_title"] == "Assessment"
            assert data["disclaimer"] == MEDICAL_DISCLAIMER
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_api_query_compare_success(client: TestClient, mock_auth_user):
    from app.api.auth import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: mock_auth_user

    try:
        mock_retrieval = RAGSearchResponse(
            query="What is the clinical diagnosis?",
            status="success",
            intent="diagnosis",
            total_results=1,
            results=[
                RetrievedChunk(
                    chunk_id="chk_123",
                    document_id="doc_abc",
                    chunk_index=2,
                    content="Assessment: Stable angina pectoris.",
                    section_title="Assessment",
                    similarity_score=0.92,
                    relevance_score=0.95,
                )
            ],
        )

        with patch("app.api.query.execute_rag_retrieval", return_value=mock_retrieval), \
             patch.object(LLMRouter, "compare_all", new_callable=AsyncMock) as mock_compare:
            
            mock_compare.return_value = [
                LLMResponse(
                    model="Gemini 2.5 Flash-Lite",
                    provider="Google",
                    answer="Diagnosis: Stable angina.",
                    latency_ms=98.0,
                    success=True,
                ),
                LLMResponse(
                    model="Qwen 3.6 27B",
                    provider="Qwen",
                    answer="Clinical diagnosis is stable angina pectoris.",
                    latency_ms=160.0,
                    success=True,
                ),
                LLMResponse(
                    model="GPT-OSS 120B",
                    provider="OpenAI-Compatible",
                    answer="Findings indicate stable angina.",
                    latency_ms=210.0,
                    success=True,
                ),
            ]

            res = client.post(
                "/api/query/compare",
                json={
                    "query": "What is the clinical diagnosis?",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert len(data["results"]) == 3
            assert data["results"][0]["model"] == "Gemini 2.5 Flash-Lite"
            assert data["results"][1]["model"] == "Qwen 3.6 27B"
            assert data["results"][2]["model"] == "GPT-OSS 120B"
            assert len(data["sources"]) == 1
    finally:
        app.dependency_overrides.pop(get_current_user, None)
