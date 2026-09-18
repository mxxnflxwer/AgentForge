import logging
import time
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    WorkflowEvaluationQueryRequest,
    WorkflowEvaluationQueryResponse,
)
from app.services.evaluation import EvaluationInput, EvaluationResult, get_evaluation_engine
from app.services.llm import MEDICAL_DISCLAIMER, get_llm_router
from app.services.retrieval_service import execute_rag_retrieval

logger = logging.getLogger("agentforge.api.evaluation")

router = APIRouter(prefix="/api/evaluation", tags=["AgentEvo Evaluation Engine"])


def convert_result_to_response(result: EvaluationResult) -> EvaluationResponse:
    """Helper to convert EvaluationResult to API response schema."""
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


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate an LLM generation result against context with AgentEvo metrics",
)
def evaluate_generation(request: EvaluationRequest) -> EvaluationResponse:
    """
    Direct metric evaluation on provided question, answer, context, and telemetry:
    - Calculates Groundedness (% of factual claims supported by context)
    - Calculates Hallucination Rate (% of unsupported/fabricated claims)
    - Calculates Accuracy (when optional reference answer is provided)
    - Computes Token Counts (input, output, total)
    - Computes Token Cost (input cost, output cost, total cost)
    - Evaluates Latency & Pipeline Execution Time
    """
    engine = get_evaluation_engine()
    eval_input = EvaluationInput(
        question=request.question,
        generated_answer=request.generated_answer,
        retrieved_context=request.retrieved_context,
        expected_answer=request.expected_answer,
        model_name=request.model_name,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        latency_ms=request.latency_ms,
        execution_time_ms=request.execution_time_ms,
    )

    result = engine.evaluate(eval_input)
    return convert_result_to_response(result)


@router.post(
    "/evaluate-query",
    response_model=WorkflowEvaluationQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute end-to-end RAG + LLM workflow and evaluate metrics with AgentEvo engine",
)
async def evaluate_rag_workflow(
    request: WorkflowEvaluationQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowEvaluationQueryResponse:
    """
    Execute full RAG retrieval + LLM answering + AgentEvo Evaluation:
    1. Runs intent-aware RAG retrieval.
    2. Generates grounded answer via selected LLM adapter.
    3. Runs AgentEvo Evaluation Engine across all metrics.
    4. Returns grounded answer, sources, and full evaluation score card.
    """
    overall_start = time.perf_counter()

    # 1. RAG Retrieval
    retrieval = execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )

    context_str = ""
    if retrieval.summary_context and retrieval.summary_context.strip():
        context_str = retrieval.summary_context.strip()
    elif retrieval.results:
        parts = [f"Chunk #{r.chunk_index} — {r.section_title or 'General'}\n{r.content}" for r in retrieval.results]
        context_str = "\n\n".join(parts)

    sources_list = [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "chunk_index": r.chunk_index,
            "section_title": r.section_title,
            "relevance_score": r.relevance_score,
        }
        for r in retrieval.results
    ]

    llm_router = get_llm_router()
    adapter = llm_router.get_adapter(request.model) or llm_router.get_adapter("gemini")
    model_name = adapter.name if adapter else request.model

    # Anti-hallucination check if no context was found
    if retrieval.status == "not_found" or not retrieval.results:
        answer_text = "The requested information was not found in the uploaded document."
        latency_ms = 0.0
        raw_usage = None
    else:
        llm_resp = await llm_router.generate_single(
            model_key=request.model,
            query=request.query,
            context=context_str,
        )
        answer_text = llm_resp.answer
        latency_ms = llm_resp.latency_ms
        raw_usage = llm_resp.raw_usage

    total_execution_ms = round((time.perf_counter() - overall_start) * 1000, 2)

    # 2. AgentEvo Evaluation Engine
    engine = get_evaluation_engine()
    eval_input = EvaluationInput(
        question=request.query,
        generated_answer=answer_text,
        retrieved_context=context_str,
        expected_answer=request.expected_answer,
        model_name=model_name,
        latency_ms=latency_ms,
        execution_time_ms=total_execution_ms,
        raw_usage=raw_usage,
    )

    eval_result = engine.evaluate(eval_input)

    return WorkflowEvaluationQueryResponse(
        query=request.query,
        model=model_name,
        answer=answer_text,
        context=context_str,
        sources=sources_list,
        evaluation=convert_result_to_response(eval_result),
        disclaimer=MEDICAL_DISCLAIMER,
    )
