import logging
import time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

import uuid
from app.models.document import Document, DocumentChunk, DocumentProcessingStatus
from app.services.agent_evo.objective import CandidateMetrics
from app.services.agent_evo.pareto import EvaluatedCandidate
from app.services.agent_evo.validator import validate_workflow
from app.services.agent_evo.workflow import AgentWorkflow
from app.services.document.base import DocumentSection
from app.services.document.chunker import DocumentChunker
from app.services.document.pipeline import get_document_pipeline
from app.services.evaluation import EvaluationInput, EvaluationResult, get_evaluation_engine
from app.services.llm import MEDICAL_DISCLAIMER, get_llm_router
from app.services.llm.base import build_medical_prompt
from app.services.retrieval_service import build_extractive_summary_context, execute_rag_retrieval
from app.services.storage import get_storage_backend
from app.services.vector_store import VectorStore

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


def get_or_create_candidate_vector_store(
    user_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_id: Optional[str] = None,
    db: Optional[Session] = None,
) -> Optional[VectorStore]:
    """
    Build or retrieve an isolated candidate-specific vector store collection
    for a workflow candidate with non-baseline chunking settings.
    Guarantees production collection ('agentforge_medical_docs') and database
    records are never modified or overwritten.
    """
    if chunk_size == 1000 and chunk_overlap == 200:
        return None

    coll_name = f"cand_chunks_{chunk_size}_{chunk_overlap}"
    cand_store = VectorStore(collection_name=coll_name)

    if not db:
        return cand_store

    try:
        where_filter = {"user_id": {"$eq": str(user_id)}}
        if document_id:
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": str(user_id)}},
                    {"document_id": {"$eq": str(document_id)}},
                ]
            }
        existing = cand_store._collection.get(where=where_filter, limit=1)
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            return cand_store

        if document_id:
            docs = db.query(Document).filter(Document.id == document_id).all()
        else:
            docs = (
                db.query(Document)
                .filter(
                    Document.owner_id == user_id,
                    Document.processing_status == DocumentProcessingStatus.COMPLETED,
                )
                .all()
            )

        storage = get_storage_backend()
        pipeline = get_document_pipeline()
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for doc in docs:
            sections: List[DocumentSection] = []
            try:
                file_bytes = storage.read(doc.stored_filename)
                processor = pipeline.get_processor_for_type(doc.file_type)
                extracted = processor.extract(file_bytes)
                cleaned = pipeline.cleaner.clean(extracted.raw_text)
                sections = pipeline.section_detector.detect_sections(cleaned)
            except Exception:
                if doc.chunks:
                    sections_map: Dict[str, List[str]] = {}
                    for chk in sorted(doc.chunks, key=lambda c: c.chunk_index):
                        sec = chk.section_title or "General"
                        if sec not in sections_map:
                            sections_map[sec] = []
                        sections_map[sec].append(chk.content)
                    for sec, pieces in sections_map.items():
                        full_sec = "\n\n".join(pieces)
                        sections.append(
                            DocumentSection(
                                title=sec,
                                content=full_sec,
                                start_char=0,
                                end_char=len(full_sec),
                            )
                        )

            if not sections:
                continue

            chunks_data = chunker.chunk_sections(sections)
            transient_chunks: List[DocumentChunk] = []
            for cd in chunks_data:
                transient_chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        document_id=doc.id,
                        chunk_index=cd.chunk_index,
                        content=cd.content,
                        section_title=cd.section_title,
                        token_count=cd.token_count,
                        character_count=cd.character_count,
                    )
                )

            cand_store.upsert_document_chunks(
                user_id=user_id,
                document_id=doc.id,
                chunks=transient_chunks,
            )

        return cand_store
    except Exception as e:
        logger.warning(f"Could not build candidate vector store for {chunk_size}/{chunk_overlap}: {e}")
        return cand_store


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
            cand_store = None
            cand_chunk_size = getattr(workflow.retrieval, "chunk_size", 1000)
            cand_chunk_overlap = getattr(workflow.retrieval, "chunk_overlap", 200)
            if cand_chunk_size != 1000 or cand_chunk_overlap != 200:
                cand_store = get_or_create_candidate_vector_store(
                    user_id=user_id,
                    chunk_size=cand_chunk_size,
                    chunk_overlap=cand_chunk_overlap,
                    document_id=document_id,
                    db=db,
                )

            retrieval = execute_rag_retrieval(
                user_id=user_id,
                query=query,
                document_id=document_id,
                top_k=workflow.retrieval.top_k,
                relevance_threshold=workflow.retrieval.similarity_threshold,
                db=db,
                enable_reranking=getattr(workflow.retrieval, "reranking", True),
                vector_store=cand_store,
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
        adapter = llm_router.get_adapter(workflow.model.model_id) or llm_router.get_adapter(workflow.model.provider)
        if not adapter:
            logger.error(f"Candidate model '{workflow.model.model_id}' is not supported.")
            return EvaluatedCandidate(
                candidate_id=workflow.workflow_id,
                workflow=workflow,
                metrics=CandidateMetrics.from_failure(
                    workflow_id=workflow.workflow_id,
                    workflow_name=workflow.name,
                    status="unsupported_model",
                    error_message=f"Model adapter for '{workflow.model.model_id}' is not supported.",
                ),
                is_pareto_optimal=False,
                mutation_metadata=workflow.mutation_metadata,
            )
        model_display_name = adapter.name

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
                    temperature=workflow.model.temperature,
                    max_tokens=workflow.model.max_tokens,
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

    @classmethod
    async def evaluate_candidate_workload(
        cls,
        workflow: AgentWorkflow,
        queries: List[str],
        user_id: str,
        document_id: Optional[str] = None,
        expected_answers: Optional[List[Optional[str]]] = None,
        db: Optional[Session] = None,
    ) -> EvaluatedCandidate:
        """
        Evaluate a single candidate workflow across an entire evaluation workload (multiple queries).
        Executes every query independently, records per-query metrics & evidence, and computes unrounded aggregate metrics.
        """
        if not queries:
            raise ValueError("Workload evaluation requires at least one query.")

        # Single-query fallback
        if len(queries) == 1:
            exp_ans = expected_answers[0] if expected_answers and len(expected_answers) > 0 else None
            return await cls.evaluate_candidate(
                workflow=workflow,
                query=queries[0],
                user_id=user_id,
                document_id=document_id,
                expected_answer=exp_ans,
                db=db,
            )

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

        per_query_evals: List[Dict[str, Any]] = []
        expected_answers = expected_answers or [None] * len(queries)

        llm_router = get_llm_router()
        adapter = (
            llm_router.get_adapter(workflow.model.model_id)
            or llm_router.get_adapter(workflow.model.provider)
        )
        if not adapter:
            logger.error(f"Candidate model '{workflow.model.model_id}' is not supported.")
            return EvaluatedCandidate(
                candidate_id=workflow.workflow_id,
                workflow=workflow,
                metrics=CandidateMetrics.from_failure(
                    workflow_id=workflow.workflow_id,
                    workflow_name=workflow.name,
                    status="unsupported_model",
                    error_message=f"Model adapter for '{workflow.model.model_id}' is not supported.",
                ),
                is_pareto_optimal=False,
                mutation_metadata=workflow.mutation_metadata,
            )
        model_display_name = adapter.name
        eval_engine = get_evaluation_engine()

        for idx, q in enumerate(queries):
            q_start = time.perf_counter()
            exp_ans = expected_answers[idx] if idx < len(expected_answers) else None

            # RAG Retrieval
            retrieval_start = time.perf_counter()
            try:
                cand_store = None
                cand_chunk_size = getattr(workflow.retrieval, "chunk_size", 1000)
                cand_chunk_overlap = getattr(workflow.retrieval, "chunk_overlap", 200)
                if cand_chunk_size != 1000 or cand_chunk_overlap != 200:
                    cand_store = get_or_create_candidate_vector_store(
                        user_id=user_id,
                        chunk_size=cand_chunk_size,
                        chunk_overlap=cand_chunk_overlap,
                        document_id=document_id,
                        db=db,
                    )

                retrieval = execute_rag_retrieval(
                    user_id=user_id,
                    query=q,
                    document_id=document_id,
                    top_k=workflow.retrieval.top_k,
                    relevance_threshold=workflow.retrieval.similarity_threshold,
                    db=db,
                    enable_reranking=getattr(workflow.retrieval, "reranking", True),
                    vector_store=cand_store,
                )
                retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
            except Exception as e:
                logger.error(f"Retrieval error for query '{q[:30]}' in '{workflow.workflow_id}': {e}")
                per_query_evals.append({
                    "query_index": idx + 1,
                    "query": q,
                    "response": "",
                    "groundedness": 0.0,
                    "hallucination_rate": 1.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "llm_latency_ms": 0.0,
                    "execution_time_ms": 0.0,
                    "status": "retrieval_error",
                    "error_message": str(e),
                })
                continue

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

            if workflow.execution.context_compression and context_str:
                context_str = apply_context_compression_operator(context_str)

            # LLM Generation
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
                        query=q,
                        context=context_str,
                        custom_instructions=workflow.prompt.custom_instructions,
                    )
                    llm_resp = await adapter.generate(
                        query=q,
                        context=context_str,
                        system_prompt=system_prompt,
                        temperature=workflow.model.temperature,
                        max_tokens=workflow.model.max_tokens,
                    )
                    answer_text = llm_resp.answer
                    latency_ms = llm_resp.latency_ms
                    raw_usage = llm_resp.raw_usage
                    gen_success = llm_resp.success
                    gen_error = llm_resp.error
                except Exception as e:
                    logger.error(f"LLM generation failed for query '{q[:30]}' in '{workflow.workflow_id}': {e}")
                    answer_text = ""
                    latency_ms = 0.0
                    raw_usage = None
                    gen_success = False
                    gen_error = str(e)

            q_exec_ms = round((time.perf_counter() - q_start) * 1000, 2)

            # Phase 6 Evaluation
            eval_input = EvaluationInput(
                question=q,
                generated_answer=answer_text,
                retrieved_context=context_str,
                expected_answer=exp_ans,
                model_name=model_display_name,
                latency_ms=latency_ms,
                execution_time_ms=q_exec_ms,
                raw_usage=raw_usage,
                success=gen_success,
                error=gen_error,
            )
            eval_result = eval_engine.evaluate(eval_input)

            per_query_evals.append({
                "query_index": idx + 1,
                "query_id": f"q_{idx + 1}",
                "query": q,
                "workflow_id": workflow.workflow_id,
                "workflow_name": workflow.name,
                "response": answer_text,
                "groundedness": eval_result.groundedness if eval_result.groundedness is not None else 0.0,
                "hallucination_rate": eval_result.hallucination_rate if eval_result.hallucination_rate is not None else 0.0,
                "supported_claims": eval_result.supported_claims,
                "unsupported_claims": eval_result.unsupported_claims,
                "supported_claim_count": eval_result.supported_claim_count,
                "unsupported_claim_count": eval_result.unsupported_claim_count,
                "total_claims": eval_result.total_claims,
                "input_tokens": eval_result.input_tokens,
                "output_tokens": eval_result.output_tokens,
                "total_tokens": eval_result.total_tokens,
                "cost": eval_result.total_cost,
                "llm_latency_ms": eval_result.latency_ms,
                "execution_time_ms": eval_result.execution_time_ms,
                "provider": workflow.model.provider,
                "model_id": model_display_name,
                "status": "success" if gen_success and bool(answer_text) else ("empty_response" if not answer_text and gen_success else "model_error"),
                "chunks_retrieved": len(retrieval.results),
                "error_message": gen_error,
            })

        # Aggregate raw metrics across the entire workload
        total_q = len(per_query_evals)
        groundedness_vals = [e["groundedness"] for e in per_query_evals if e["groundedness"] is not None]
        hallucination_vals = [e["hallucination_rate"] for e in per_query_evals if e["hallucination_rate"] is not None]
        cost_vals = [e["cost"] for e in per_query_evals]
        tokens_vals = [e["total_tokens"] for e in per_query_evals]
        in_tokens_vals = [e["input_tokens"] for e in per_query_evals]
        out_tokens_vals = [e["output_tokens"] for e in per_query_evals]
        lat_vals = [e["llm_latency_ms"] for e in per_query_evals]
        exec_vals = [e["execution_time_ms"] for e in per_query_evals]

        avg_groundedness = sum(groundedness_vals) / max(1, len(groundedness_vals)) if groundedness_vals else 0.0
        avg_hallucination = sum(hallucination_vals) / max(1, len(hallucination_vals)) if hallucination_vals else 0.0
        avg_cost_per_query = sum(cost_vals) / max(1, total_q)
        avg_tokens_per_query = int(round(sum(tokens_vals) / max(1, total_q)))
        avg_in_tokens = int(round(sum(in_tokens_vals) / max(1, total_q)))
        avg_out_tokens = int(round(sum(out_tokens_vals) / max(1, total_q)))
        avg_latency_ms = sum(lat_vals) / max(1, total_q)
        avg_exec_ms = sum(exec_vals) / max(1, total_q)

        all_success = all(e["status"] == "success" for e in per_query_evals)
        cand_status = "success" if all_success else "partial_success" if any(e["status"] == "success" for e in per_query_evals) else "failed"

        # Explicitly distinguish real provider calls from simulation / mocked environments
        is_mocked = any(
            "[Simulation" in str(e.get("response", ""))
            or (adapter and not getattr(adapter, "is_configured", lambda: True)())
            or "mock" in type(adapter).__name__.lower()
            for e in per_query_evals
        )

        metrics = CandidateMetrics(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            performance=round(avg_groundedness, 4),
            cost=round(avg_cost_per_query, 6),
            groundedness=avg_groundedness,
            hallucination_rate=avg_hallucination,
            accuracy=None,
            total_tokens=avg_tokens_per_query,
            input_tokens=avg_in_tokens,
            output_tokens=avg_out_tokens,
            llm_latency_ms=round(avg_latency_ms, 2),
            execution_time_ms=round(avg_exec_ms, 2),
            status=cand_status,
            error_message=None if all_success else "Some queries in workload experienced warnings/errors",
            raw_details={
                "is_workload": True,
                "is_mocked": is_mocked,
                "evaluation_mode": "mocked_or_saved_response" if is_mocked else "live_provider",
                "evaluation_count": total_q,
                "workload_queries": queries,
                "avg_groundedness": avg_groundedness,
                "avg_hallucination": avg_hallucination,
                "avg_cost_per_query": avg_cost_per_query,
                "avg_tokens_per_query": avg_tokens_per_query,
                "avg_latency_ms": avg_latency_ms,
                "per_query_evaluations": per_query_evals,
            },
        )

        return EvaluatedCandidate(
            candidate_id=workflow.workflow_id,
            workflow=workflow,
            metrics=metrics,
            is_pareto_optimal=False,
            mutation_metadata=workflow.mutation_metadata,
        )

