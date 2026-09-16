import logging
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.evaluation import EvaluationResponse
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
from app.services.evaluation import EvaluationInput, EvaluationResult, get_evaluation_engine
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


def convert_eval_result_to_response(result: EvaluationResult) -> EvaluationResponse:
    """Helper to convert EvaluationResult to EvaluationResponse schema."""
    return EvaluationResponse(
        model=result.model,
        accuracy=result.accuracy,
        groundedness=result.groundedness,
        hallucination_rate=result.hallucination_rate,
        supported_claims=result.supported_claims,
        unsupported_claims=result.unsupported_claims,
        total_claims=result.total_claims,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        latency_ms=result.latency_ms,
        execution_time_ms=result.execution_time_ms,
        input_cost=result.input_cost,
        output_cost=result.output_cost,
        total_cost=result.total_cost,
        details=result.details,
    )


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
    summary="Query document using selected LLM with intent-aware RAG context & AgentEvo evaluation",
)
async def query_answer(
    request: QueryAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> QueryAnswerResponse:
    """
    End-to-end grounded document answering and evaluation:
    1. Runs query-aware RAG retrieval pipeline (intent detection + semantic search + reranking).
    2. Validates retrieved relevance. If no information is found, returns safe not_found response.
    3. If context is found, sends context to selected LLM adapter.
    4. Evaluates groundedness, hallucination rate, token usage, cost, latency, and accuracy.
    5. Returns grounded answer, retrieved sources, latency, evaluation metrics, and medical disclaimer.
    """
    overall_start = time.perf_counter()

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

    eval_engine = get_evaluation_engine()

    # Anti-hallucination check: if RAG found nothing relevant, return immediate safe response
    if retrieval.status == "not_found" or not retrieval.results:
        logger.info(f"Query '{request.query[:30]}' returned no relevant chunks. Returning safe not_found.")
        total_exec_ms = round((time.perf_counter() - overall_start) * 1000, 2)
        
        eval_input = EvaluationInput(
            question=request.query,
            generated_answer="The requested information was not found in the uploaded document.",
            retrieved_context="",
            expected_answer=request.expected_answer,
            model_name=model_display_name,
            latency_ms=0.0,
            execution_time_ms=total_exec_ms,
        )
        eval_result = eval_engine.evaluate(eval_input)

        return QueryAnswerResponse(
            status="not_found",
            query=request.query,
            intent=retrieval.intent,
            model=model_display_name,
            answer="The requested information was not found in the uploaded document.",
            sources=[],
            latency_ms=0.0,
            disclaimer=MEDICAL_DISCLAIMER,
            evaluation=convert_eval_result_to_response(eval_result),
        )

    context_str = build_context_from_retrieval(retrieval)
    llm_resp = await llm_router.generate_single(
        model_key=request.model,
        query=request.query,
        context=context_str,
    )

    total_exec_ms = round((time.perf_counter() - overall_start) * 1000, 2)

    # Run AgentEvo Evaluation Engine on the response
    eval_input = EvaluationInput(
        question=request.query,
        generated_answer=llm_resp.answer,
        retrieved_context=context_str,
        expected_answer=request.expected_answer,
        model_name=llm_resp.model,
        latency_ms=llm_resp.latency_ms,
        execution_time_ms=total_exec_ms,
        raw_usage=llm_resp.raw_usage,
    )
    eval_result = eval_engine.evaluate(eval_input)

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
        evaluation=convert_eval_result_to_response(eval_result),
    )


@router.post(
    "/compare",
    response_model=QueryCompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare all 3 LLMs concurrently and evaluate metrics for each model",
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
    3. Evaluates groundedness, hallucination rate, tokens, cost, latency, and accuracy per model.
    4. Measures individual model latencies and returns structured side-by-side results.
    """
    overall_start = time.perf_counter()

    retrieval = execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )

    llm_router = get_llm_router()
    eval_engine = get_evaluation_engine()

    # Anti-hallucination guardrail for comparison mode
    if retrieval.status == "not_found" or not retrieval.results:
        logger.info(f"Comparison query '{request.query[:30]}' returned no relevant chunks.")
        total_exec_ms = round((time.perf_counter() - overall_start) * 1000, 2)
        
        not_found_results = []
        for m in llm_router.list_models():
            eval_input = EvaluationInput(
                question=request.query,
                generated_answer="The requested information was not found in the uploaded document.",
                retrieved_context="",
                expected_answer=request.expected_answer,
                model_name=m["name"],
                latency_ms=0.0,
                execution_time_ms=total_exec_ms,
            )
            e_res = eval_engine.evaluate(eval_input)

            not_found_results.append(
                ModelComparisonResult(
                    model=m["name"],
                    provider=m["provider"],
                    answer="The requested information was not found in the uploaded document.",
                    latency_ms=0.0,
                    success=True,
                    error=None,
                    evaluation=convert_eval_result_to_response(e_res),
                )
            )

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

    total_exec_ms = round((time.perf_counter() - overall_start) * 1000, 2)

    comparison_results: List[ModelComparisonResult] = []
    for r in llm_responses:
        eval_input = EvaluationInput(
            question=request.query,
            generated_answer=r.answer,
            retrieved_context=context_str,
            expected_answer=request.expected_answer,
            model_name=r.model,
            latency_ms=r.latency_ms,
            execution_time_ms=total_exec_ms,
            raw_usage=r.raw_usage,
        )
        e_res = eval_engine.evaluate(eval_input)

        comparison_results.append(
            ModelComparisonResult(
                model=r.model,
                provider=r.provider,
                answer=r.answer,
                latency_ms=r.latency_ms,
                success=r.success,
                error=r.error,
                evaluation=convert_eval_result_to_response(e_res),
            )
        )

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
