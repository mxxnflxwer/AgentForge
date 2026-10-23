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
        context_parts.append(f"Chunk #{chunk.chunk_index} — {sec}\n{chunk.content}")

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
        supported_claim_count=result.supported_claim_count,
        unsupported_claim_count=result.unsupported_claim_count,
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
    retrieval_start = time.perf_counter()

    requested_model = (request.model or "").strip()
    version_override = request.workflow_version_id
    if requested_model.lower() == "baseline":
        version_override = "baseline"

    from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector
    selection = RuntimeWorkflowSelector.select_workflow_for_query(
        query=request.query,
        db=db,
        candidate_version_id_override=version_override,
    )

    active_version_id = selection.version_id
    task_category = selection.task_category
    selection_reason = selection.selection_reason
    cfg = selection.configuration or {}
    pipeline_type = "baseline" if active_version_id == "baseline" else "production"

    effective_top_k = cfg.get("retrieval", {}).get("top_k", request.top_k)
    sim_threshold = cfg.get("retrieval", {}).get("similarity_threshold", 0.35)
    enable_reranking = bool(cfg.get("retrieval", {}).get("reranking", True))
    context_compression = cfg.get("execution", {}).get("context_compression", False)
    
    if not requested_model or requested_model.lower() in ("default", "baseline"):
        model_to_use = cfg.get("model", {}).get("model_id", "gemini")
    else:
        model_to_use = requested_model

    llm_router = get_llm_router()
    adapter = llm_router.get_adapter(model_to_use)
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model adapter for '{model_to_use}' is not supported or unavailable. Supported models: gemini, qwen, gemma (or legacy gpt_oss)",
        )
    model_display_name = adapter.name
    model_provider = adapter.provider
    model_id = adapter.model_id
    prompt_template_name = cfg.get("prompt", {}).get("template_name", "default")
    custom_instructions = cfg.get("prompt", {}).get("custom_instructions")
    wf_temperature = cfg.get("model", {}).get("temperature")
    wf_max_tokens = cfg.get("model", {}).get("max_tokens")

    retrieval = execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=effective_top_k,
        relevance_threshold=sim_threshold,
        db=db,
        enable_reranking=enable_reranking,
    )
    # Strictly enforce configured top_k limit into final context and source list
    if retrieval.results and len(retrieval.results) > effective_top_k:
        retrieval.results = retrieval.results[:effective_top_k]
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

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

        exec_id = f"exec_{int(time.time()*1000)}"
        trace_not_found = {
            "pipeline_type": pipeline_type,
            "stages": {
                "retrieval": {
                    "latency_ms": retrieval_ms,
                    "chunks_retrieved": 0,
                    "reranking_enabled": enable_reranking,
                    "similarity_threshold": sim_threshold,
                    "top_k": effective_top_k,
                },
                "compression": {"enabled": context_compression, "effective_context_length": 0},
                "prompt": {"template": prompt_template_name, "custom_instructions_applied": bool(custom_instructions)},
                "llm_generation": {"model": model_display_name, "latency_ms": 0.0, "status": "skipped_not_found"},
                "evidence_verification": {
                    "groundedness": eval_result.groundedness,
                    "total_claims": eval_result.total_claims,
                    "supported_claims": eval_result.supported_claim_count,
                    "unsupported_claims": eval_result.unsupported_claim_count,
                    "claim_breakdown": eval_result.details.get("claim_breakdown", []),
                },
            },
            "effective_configuration": {
                "model": {"model_id": model_id, "provider": model_provider},
                "retrieval": {"top_k": effective_top_k, "similarity_threshold": sim_threshold, "reranking": enable_reranking},
                "prompt": {"template_name": prompt_template_name, "custom_instructions": custom_instructions},
                "execution": {"context_compression": context_compression},
            },
            "task_category": task_category,
            "selection_reason": selection_reason,
        }
        logger.info(
            f"Query classification: '{task_category}' | Selected workflow version: '{active_version_id}' | "
            f"Selection reason: '{selection_reason}' | Execution metrics: groundedness={eval_result.groundedness}, "
            f"cost={eval_result.total_cost}, latency={eval_result.latency_ms}ms"
        )
        return QueryAnswerResponse(
            status="not_found",
            query=request.query,
            intent=retrieval.intent,
            model=model_display_name,
            model_id=model_id,
            provider=model_provider,
            success=True,
            error=None,
            answer="The requested information was not found in the uploaded document.",
            context=None,
            sources=[],
            latency_ms=0.0,
            disclaimer=MEDICAL_DISCLAIMER,
            pipeline_type=pipeline_type,
            evaluation=convert_eval_result_to_response(eval_result),
            execution_id=exec_id,
            workflow_version_id=active_version_id,
            workflow_name=selection.workflow_name,
            task_category=task_category,
            selection_reason=selection_reason,
            groundedness=eval_result.groundedness,
            execution_trace=trace_not_found,
        )

    raw_context_str = build_context_from_retrieval(retrieval)
    context_str = raw_context_str
    if context_compression:
        from app.services.agent_evo.evaluator_adapter import apply_context_compression_operator
        context_str = apply_context_compression_operator(raw_context_str)

    from app.services.agent_evo.evaluator_adapter import build_candidate_prompt
    sys_prompt = build_candidate_prompt(
        template_name=prompt_template_name,
        query=request.query,
        context=context_str,
        custom_instructions=custom_instructions,
    )

    llm_resp = await adapter.generate(
        query=request.query,
        context=context_str,
        system_prompt=sys_prompt,
        temperature=wf_temperature,
        max_tokens=wf_max_tokens,
    )

    # Individual pipeline execution time: retrieval + LLM generation
    pipeline_exec_ms = round(retrieval_ms + (llm_resp.latency_ms or 0.0), 2)
    total_exec_ms = round((time.perf_counter() - overall_start) * 1000, 2)

    # Run AgentEvo Evaluation Engine on the response
    eval_input = EvaluationInput(
        question=request.query,
        generated_answer=llm_resp.answer,
        retrieved_context=context_str,
        expected_answer=request.expected_answer,
        model_name=llm_resp.model,
        latency_ms=llm_resp.latency_ms,
        execution_time_ms=pipeline_exec_ms,
        raw_usage=llm_resp.raw_usage,
        success=llm_resp.success,
        error=llm_resp.error,
    )
    eval_result = eval_engine.evaluate(eval_input)

    sources = extract_sources(retrieval)
    exec_id = f"exec_{int(time.time()*1000)}"

    eval_status = eval_result.details.get("status", "success")
    if not llm_resp.success:
        response_status = "insufficient_credits" if eval_status == "insufficient_credits" else "error"
    else:
        response_status = "success"

    # Construct structured runtime execution trace
    execution_trace = {
        "pipeline_type": pipeline_type,
        "stages": {
            "retrieval": {
                "latency_ms": retrieval_ms,
                "chunks_retrieved": len(sources),
                "reranking_enabled": enable_reranking,
                "similarity_threshold": sim_threshold,
                "top_k": effective_top_k,
                "configured_top_k": effective_top_k,
            },
            "compression": {
                "enabled": context_compression,
                "original_context_length": len(raw_context_str),
                "effective_context_length": len(context_str),
            },
            "prompt": {
                "template": prompt_template_name,
                "custom_instructions_applied": bool(custom_instructions),
            },
            "llm_generation": {
                "model": llm_resp.model,
                "model_id": llm_resp.model_id or model_id,
                "provider": llm_resp.provider or adapter.provider,
                "latency_ms": llm_resp.latency_ms or 0.0,
                "input_tokens": eval_result.input_tokens if llm_resp.success else None,
                "output_tokens": eval_result.output_tokens if llm_resp.success else None,
                "temperature": wf_temperature,
                "max_tokens": wf_max_tokens,
                "success": llm_resp.success,
            },
            "evidence_verification": {
                "groundedness": eval_result.groundedness,
                "hallucination_rate": eval_result.hallucination_rate,
                "total_claims": eval_result.total_claims,
                "supported_claims": eval_result.supported_claim_count,
                "unsupported_claims": eval_result.unsupported_claim_count,
                "claim_breakdown": eval_result.details.get("claim_breakdown", []),
            },
        },
        "effective_configuration": {
            "model": {
                "model_id": model_id,
                "provider": adapter.provider,
                "temperature": wf_temperature,
                "max_tokens": wf_max_tokens,
            },
            "retrieval": {
                "top_k": effective_top_k,
                "similarity_threshold": sim_threshold,
                "reranking": enable_reranking,
            },
            "prompt": {
                "template_name": prompt_template_name,
                "custom_instructions": custom_instructions,
            },
            "execution": {
                "context_compression": context_compression,
            },
        },
        "task_category": task_category,
        "selection_reason": selection_reason,
    }

    # Persist runtime execution record in database
    if db:
        try:
            from app.models.workflow_version import WorkflowExecution
            import datetime
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            db_exec = WorkflowExecution(
                execution_id=exec_id,
                workflow_version_id=active_version_id,
                user_id=current_user.id,
                query=request.query,
                response=llm_resp.answer,
                retrieved_chunks=[s.model_dump() for s in sources],
                model=llm_resp.model,
                provider=llm_resp.provider or adapter.provider,
                input_tokens=eval_result.input_tokens if llm_resp.success else None,
                output_tokens=eval_result.output_tokens if llm_resp.success else None,
                total_tokens=eval_result.total_tokens if llm_resp.success else None,
                cost=eval_result.total_cost if llm_resp.success else None,
                llm_latency_ms=llm_resp.latency_ms if (llm_resp.success and llm_resp.latency_ms is not None) else None,
                total_execution_time_ms=total_exec_ms,
                groundedness=eval_result.groundedness,
                execution_trace=execution_trace,
                status=response_status,
                created_at=now_utc,
            )
            db.add(db_exec)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to record runtime execution metrics: {e}")

    logger.info(
        f"Query classification: '{task_category}' | Selected workflow version: '{active_version_id}' | "
        f"Model: '{llm_resp.model}' ({llm_resp.provider}) | Status: '{response_status}' | "
        f"Selection reason: '{selection_reason}' | Execution metrics: groundedness={eval_result.groundedness}, "
        f"cost={eval_result.total_cost}, latency={eval_result.latency_ms}ms"
    )

    return QueryAnswerResponse(
        status=response_status,
        query=request.query,
        intent=retrieval.intent,
        model=llm_resp.model,
        model_id=llm_resp.model_id or model_id,
        provider=llm_resp.provider or adapter.provider,
        success=llm_resp.success,
        error=llm_resp.error,
        answer=llm_resp.answer,
        context=context_str,
        sources=sources,
        latency_ms=llm_resp.latency_ms or 0.0,
        disclaimer=MEDICAL_DISCLAIMER,
        pipeline_type=pipeline_type,
        evaluation=convert_eval_result_to_response(eval_result),
        execution_id=exec_id,
        workflow_version_id=active_version_id,
        workflow_name=selection.workflow_name,
        task_category=task_category,
        selection_reason=selection_reason,
        groundedness=eval_result.groundedness,
        execution_trace=execution_trace,
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
    retrieval_start = time.perf_counter()

    retrieval = execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

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
                success=True,
                error=None,
            )
            e_res = eval_engine.evaluate(eval_input)

            nf_trace = {
                "pipeline_type": "comparison_shared_retrieval",
                "workflow_name": "Multi-LLM Comparative Baseline",
                "workflow_version_id": None,
                "context_type": "shared",
                "stages": {
                    "retrieval": {
                        "type": "shared_context",
                        "chunks_retrieved": 0,
                        "latency_ms": retrieval_ms,
                        "top_k": request.top_k,
                    },
                    "llm_generation": {
                        "model": m["name"],
                        "model_id": m["model_id"],
                        "provider": m["provider"],
                        "latency_ms": 0.0,
                        "status": "skipped_not_found",
                    },
                    "evidence_verification": {
                        "groundedness": e_res.groundedness,
                        "total_claims": 0,
                        "supported_claims": 0,
                        "unsupported_claims": 0,
                        "claim_breakdown": [],
                    },
                },
                "effective_configuration": {
                    "retrieval": {"top_k": request.top_k, "mode": "shared_retrieval"},
                    "model": {"model": m["name"], "provider": m["provider"]},
                },
            }

            not_found_results.append(
                ModelComparisonResult(
                    model=m["name"],
                    model_id=m["model_id"],
                    provider=m["provider"],
                    answer="The requested information was not found in the uploaded document.",
                    latency_ms=0.0,
                    success=True,
                    error=None,
                    pipeline_type="comparison_shared_retrieval",
                    workflow_version_id=None,
                    workflow_name="Multi-LLM Comparative Baseline",
                    context_type="shared",
                    execution_trace=nf_trace,
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

    total_batch_wall_ms = round((time.perf_counter() - overall_start) * 1000, 2)
    sources = extract_sources(retrieval)

    comparison_results: List[ModelComparisonResult] = []
    for r in llm_responses:
        # Calculate dedicated per-model execution time: retrieval time + model latency
        model_exec_time_ms = round(retrieval_ms + (r.latency_ms or 0.0), 2)

        eval_input = EvaluationInput(
            question=request.query,
            generated_answer=r.answer,
            retrieved_context=context_str,
            expected_answer=request.expected_answer,
            model_name=r.model,
            latency_ms=r.latency_ms,
            execution_time_ms=model_exec_time_ms,
            raw_usage=r.raw_usage,
            success=r.success,
            error=r.error,
        )
        e_res = eval_engine.evaluate(eval_input)
        e_res.details["batch_comparison_wall_time_ms"] = total_batch_wall_ms
        e_res.details["retrieval_latency_ms"] = retrieval_ms

        model_trace = {
            "pipeline_type": "comparison_shared_retrieval",
            "workflow_name": "Multi-LLM Comparative Baseline",
            "workflow_version_id": None,
            "context_type": "shared",
            "stages": {
                "retrieval": {
                    "type": "shared_context",
                    "chunks_retrieved": len(sources),
                    "latency_ms": retrieval_ms,
                    "top_k": request.top_k,
                },
                "llm_generation": {
                    "model": r.model,
                    "model_id": r.model_id,
                    "provider": r.provider,
                    "latency_ms": r.latency_ms if r.success else None,
                    "input_tokens": e_res.input_tokens if r.success else None,
                    "output_tokens": e_res.output_tokens if r.success else None,
                    "cost": e_res.total_cost if r.success else None,
                    "success": r.success,
                    "error": r.error,
                },
                "evidence_verification": {
                    "groundedness": e_res.groundedness,
                    "total_claims": e_res.total_claims,
                    "supported_claims": e_res.supported_claim_count,
                    "unsupported_claims": e_res.unsupported_claim_count,
                    "claim_breakdown": e_res.details.get("claim_breakdown", []),
                },
            },
            "effective_configuration": {
                "retrieval": {"top_k": request.top_k, "mode": "shared_retrieval"},
                "model": {"model": r.model, "provider": r.provider},
            },
        }

        comparison_results.append(
            ModelComparisonResult(
                model=r.model,
                model_id=r.model_id,
                provider=r.provider,
                answer=r.answer,
                latency_ms=r.latency_ms or 0.0,
                success=r.success,
                error=r.error,
                pipeline_type="comparison_shared_retrieval",
                workflow_version_id=None,
                workflow_name="Multi-LLM Comparative Baseline",
                context_type="shared",
                execution_trace=model_trace,
                evaluation=convert_eval_result_to_response(e_res),
            )
        )

    return QueryCompareResponse(
        status="success",
        query=request.query,
        intent=retrieval.intent,
        context=context_str,
        sources=sources,
        results=comparison_results,
        disclaimer=MEDICAL_DISCLAIMER,
    )
