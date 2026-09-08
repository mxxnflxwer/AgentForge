import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.query import (
    AvailableModelsResponse,
    ModelComparisonResult,
    ModelMetadata,
    QueryAnswerRequest,
    QueryAnswerResponse,
    QueryAnswerSource,
    QueryCompareRequest,
    QueryCompareResponse,
)
from app.schemas.rag import RAGSearchResponse
from app.services.llm import MEDICAL_DISCLAIMER, get_llm_router
from app.services.retrieval_service import execute_rag_retrieval

logger = logging.getLogger("agentforge.api.query")

router = APIRouter(prefix="/api/query", tags=["LLM Query & Comparison"])


def build_context_from_retrieval(retrieval: RAGSearchResponse) -> str:
    """Build unified context string from retrieved RAG results."""
    if retrieval.summary_context and retrieval.summary_context.strip():
        return retrieval.summary_context.strip()

    context_parts: List[str] = []
    for chunk in retrieval.results:
        sec = chunk.section_title or "General"
        context_parts.append(f"### Section: {sec} (Chunk #{chunk.chunk_index})\n{chunk.content}")

    return "\n\n".join(context_parts).strip()


def extract_sources(retrieval: RAGSearchResponse) -> List[QueryAnswerSource]:
    """Map retrieved chunks to standardized source list."""
    return [
        QueryAnswerSource(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            section_title=r.section_title,
            similarity_score=r.similarity_score,
            relevance_score=r.relevance_score,
        )
        for r in retrieval.results
    ]


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    status_code=status.HTTP_200_OK,
    summary="List supported LLM models and provider configuration status",
)
def get_supported_models() -> AvailableModelsResponse:
    router_instance = get_llm_router()
    models_data = [
        ModelMetadata(
            name=m["name"],
            provider=m["provider"],
            model_id=m["model_id"],
            is_configured=m["is_configured"],
        )
        for m in router_instance.list_models()
    ]
    return AvailableModelsResponse(models=models_data)


@router.post(
    "/answer",
    response_model=QueryAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Query document using selected LLM with intent-aware RAG context",
)
async def query_answer(
    request: QueryAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QueryAnswerResponse:
    """
    End-to-end grounded document answering:
    1. Runs query-aware RAG retrieval pipeline (intent detection + semantic search + reranking).
    2. Validates retrieved relevance. If no information is found, returns safe not_found response.
    3. If context is found, sends context to selected LLM adapter.
    4. Returns grounded answer, retrieved sources, latency, and medical safety disclaimer.
    """
    retrieval = execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )

    llm_router = get_llm_router()
    adapter = llm_router.get_adapter(request.model) or llm_router.get_adapter("gemini")
    model_display_name = adapter.name if adapter else request.model

    # Anti-hallucination check: if RAG found nothing relevant, return immediate safe response
    if retrieval.status == "not_found" or not retrieval.results:
        logger.info(f"Query '{request.query[:30]}' returned no relevant chunks. Returning safe not_found.")
        return QueryAnswerResponse(
            status="not_found",
            query=request.query,
            intent=retrieval.intent,
            model=model_display_name,
            answer="The requested information was not found in the uploaded document.",
            sources=[],
            latency_ms=0.0,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    context_str = build_context_from_retrieval(retrieval)
    llm_resp = await llm_router.generate_single(
        model_key=request.model,
        query=request.query,
        context=context_str,
    )

    sources = extract_sources(retrieval)

    return QueryAnswerResponse(
        status="success" if llm_resp.success or llm_resp.answer else "error",
        query=request.query,
        intent=retrieval.intent,
        model=llm_resp.model,
        answer=llm_resp.answer,
        sources=sources,
        latency_ms=llm_resp.latency_ms,
        disclaimer=MEDICAL_DISCLAIMER,
    )


@router.post(
    "/compare",
    response_model=QueryCompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare all 3 LLMs (Gemini 2.5 Flash-Lite, Qwen 3.6 27B, GPT-OSS 120B) concurrently",
)
async def query_compare(
    request: QueryCompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QueryCompareResponse:
    """
    Fair multi-model comparative evaluation:
    1. Executes RAG retrieval ONCE to obtain the ground-truth document context.
    2. Sends the exact same query, context, and system prompt concurrently to all 3 LLMs.
    3. Measures individual model latencies and returns structured side-by-side results.
    """
    retrieval = execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )

    llm_router = get_llm_router()

    # Anti-hallucination guardrail for comparison mode
    if retrieval.status == "not_found" or not retrieval.results:
        logger.info(f"Comparison query '{request.query[:30]}' returned no relevant chunks.")
        not_found_results = [
            ModelComparisonResult(
                model=m["name"],
                provider=m["provider"],
                answer="The requested information was not found in the uploaded document.",
                latency_ms=0.0,
                success=True,
                error=None,
            )
            for m in llm_router.list_models()
        ]
        return QueryCompareResponse(
            status="not_found",
            query=request.query,
            intent=retrieval.intent,
            context=None,
            sources=[],
            results=not_found_results,
            disclaimer=MEDICAL_DISCLAIMER,
        )

    context_str = build_context_from_retrieval(retrieval)
    llm_responses = await llm_router.compare_all(
        query=request.query,
        context=context_str,
    )

    comparison_results = [
        ModelComparisonResult(
            model=r.model,
            provider=r.provider,
            answer=r.answer,
            latency_ms=r.latency_ms,
            success=r.success,
            error=r.error,
        )
        for r in llm_responses
    ]

    sources = extract_sources(retrieval)

    return QueryCompareResponse(
        status="success",
        query=request.query,
        intent=retrieval.intent,
        context=context_str,
        sources=sources,
        results=comparison_results,
        disclaimer=MEDICAL_DISCLAIMER,
    )
