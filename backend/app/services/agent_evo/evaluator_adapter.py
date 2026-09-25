import logging
import time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.services.agent_evo.objective import CandidateMetrics
from app.services.agent_evo.pareto import EvaluatedCandidate
from app.services.agent_evo.validator import validate_workflow
from app.services.agent_evo.workflow import AgentWorkflow
from app.services.evaluation import EvaluationInput, EvaluationResult, get_evaluation_engine
from app.services.llm import MEDICAL_DISCLAIMER, get_llm_router
from app.services.llm.base import build_medical_prompt
from app.services.retrieval_service import build_extractive_summary_context, execute_rag_retrieval

logger = logging.getLogger("agentforge.services.agent_evo.evaluator_adapter")


def build_candidate_prompt(
    template_name: str,
    query: str,
    context: str,
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Format prompt based on the workflow's configured template.
    """
    cleaned_context = (context or "").strip()
    cleaned_query = (query or "").strip()

    if custom_instructions and custom_instructions.strip():
        return f"""[DOCUMENT CONTEXT]
{cleaned_context}

[USER QUESTION]
{cleaned_query}

[INSTRUCTIONS]
{custom_instructions.strip()}"""

    if template_name == "concise_medical":
        return f"""[DOCUMENT CONTEXT]
{cleaned_context}

[USER QUESTION]
{cleaned_query}

[INSTRUCTIONS]
Provide a concise, direct answer strictly using the provided document context.
Avoid unnecessary conversational filler. Keep the response factual and minimal.
If the information is missing, state: "The requested information was not found in the uploaded document.\""""

    if template_name == "detailed_clinical":
        return f"""[DOCUMENT CONTEXT]
{cleaned_context}

[USER QUESTION]
{cleaned_query}

[INSTRUCTIONS]
Provide a comprehensive, structured clinical synthesis based on the provided document context.
Include relevant findings, medications, and clinical recommendations where mentioned.
If the information is missing, state: "The requested information was not found in the uploaded document.\""""

    # Default medical prompt
    return build_medical_prompt(query=cleaned_query, context=cleaned_context)


def apply_context_compression_operator(context_str: str) -> str:
    """
    Context compression operator: Removes redundant blank lines and limits excessive chunk noise
    while preserving clinical entities, numbers, and structured sections.
    """
    if not context_str or not context_str.strip():
        return ""
    lines = context_str.strip().split("\n")
    compressed_lines = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Deduplicate identical consecutive lines
        if stripped in seen and len(stripped) > 20:
            continue
        seen.add(stripped)
        compressed_lines.append(stripped)
    return "\n".join(compressed_lines)


class EvaluatorAdapter:
    """
    Adapter bridging AgentEvo workflow execution with the Phase 6 Evaluation Engine.
    Executes RAG retrieval, LLM generation, and Phase 6 metrics calculation.
    """

    @classmethod
    async def evaluate_candidate(
        cls,
        workflow: AgentWorkflow,
        query: str,
        user_id: str,
        document_id: Optional[str] = None,
        expected_answer: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> EvaluatedCandidate:
        """
        Execute and evaluate a single candidate workflow end-to-end.
        """
        overall_start = time.perf_counter()
        logger.info(f"Starting evaluation of candidate '{workflow.workflow_id}' ({workflow.name}).")

        # 1. Configuration Validation Guardrail
        val_res = validate_workflow(workflow)
        if not val_res.is_valid:
            err_msg = "; ".join(val_res.errors)
            metrics = CandidateMetrics.from_failure(
                workflow_id=workflow.workflow_id,
                workflow_name=workflow.name,
                status="invalid",
                error_message=f"Validation failed: {err_msg}",
            )
            return EvaluatedCandidate(
                candidate_id=workflow.workflow_id,
                workflow=workflow,
                metrics=metrics,
                is_pareto_optimal=False,
                mutation_metadata=workflow.mutation_metadata,
            )

        # 2. RAG Retrieval Stage with candidate retrieval parameters
        retrieval_start = time.perf_counter()
        try:
            retrieval = execute_rag_retrieval(
                user_id=user_id,
                query=query,
                document_id=document_id,
                top_k=workflow.retrieval.top_k,
                relevance_threshold=workflow.retrieval.similarity_threshold,
                db=db,
            )
            retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
        except Exception as e:
            logger.error(f"Retrieval error in candidate '{workflow.workflow_id}': {e}")
            metrics = CandidateMetrics.from_failure(
                workflow_id=workflow.workflow_id,
                workflow_name=workflow.name,
                status="retrieval_error",
                error_message=str(e),
            )
            return EvaluatedCandidate(
                candidate_id=workflow.workflow_id,
                workflow=workflow,
                metrics=metrics,
                is_pareto_optimal=False,
                mutation_metadata=workflow.mutation_metadata,
            )

        # Build context string
        context_str = ""
        if retrieval.summary_context and retrieval.summary_context.strip():
            context_str = retrieval.summary_context.strip()
        elif retrieval.results:
            parts = [
                f"Chunk #{r.chunk_index} — {r.section_title or 'General'}\n{r.content}"
                for r in retrieval.results
            ]
            context_str = "\n\n".join(parts)

        # Context Compression Operator Application
        if workflow.execution.context_compression and context_str:
            context_str = apply_context_compression_operator(context_str)

        # 3. LLM Generation Stage
        llm_router = get_llm_router()
        adapter = llm_router.get_adapter(workflow.model.model_id) or llm_router.get_adapter(workflow.model.provider) or llm_router.get_adapter("gemini")
        model_display_name = adapter.name if adapter else workflow.model.model_id

        # Anti-hallucination guardrail if no context was retrieved
        if retrieval.status == "not_found" or not retrieval.results or not context_str.strip():
            answer_text = "The requested information was not found in the uploaded document."
            latency_ms = 0.0
            raw_usage = None
            gen_success = True
            gen_error = None
        else:
            try:
                system_prompt = build_candidate_prompt(
                    template_name=workflow.prompt.template_name,
                    query=query,
                    context=context_str,
                    custom_instructions=workflow.prompt.custom_instructions,
                )
                llm_resp = await adapter.generate(
                    query=query,
                    context=context_str,
                    system_prompt=system_prompt,
                )
                answer_text = llm_resp.answer
                latency_ms = llm_resp.latency_ms
                raw_usage = llm_resp.raw_usage
                gen_success = llm_resp.success
                gen_error = llm_resp.error
            except Exception as e:
                logger.error(f"LLM generation failed in candidate '{workflow.workflow_id}': {e}")
                answer_text = ""
                latency_ms = 0.0
                raw_usage = None
                gen_success = False
                gen_error = str(e)

        total_exec_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        # 4. Phase 6 Evaluation Engine Execution
        eval_engine = get_evaluation_engine()
        eval_input = EvaluationInput(
            question=query,
            generated_answer=answer_text,
            retrieved_context=context_str,
            expected_answer=expected_answer,
            model_name=model_display_name,
            latency_ms=latency_ms,
            execution_time_ms=total_exec_ms,
            raw_usage=raw_usage,
            success=gen_success,
            error=gen_error,
        )

        eval_result = eval_engine.evaluate(eval_input)
        metrics = CandidateMetrics.from_eval_result(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            eval_result=eval_result,
            status="success" if gen_success and bool(answer_text) else ("empty_response" if not answer_text and gen_success else "model_error"),
            error_message=gen_error,
        )
        metrics.raw_details["retrieval_ms"] = retrieval_ms
        metrics.raw_details["chunks_retrieved"] = len(retrieval.results)

        return EvaluatedCandidate(
            candidate_id=workflow.workflow_id,
            workflow=workflow,
            metrics=metrics,
            is_pareto_optimal=False,
            mutation_metadata=workflow.mutation_metadata,
        )
