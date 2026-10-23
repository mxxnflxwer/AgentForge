import datetime
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.workflow_version import (
    WorkflowAuditEvent,
    WorkflowExecution,
    WorkflowFeedback,
    WorkflowVersion,
)
from app.schemas.agent_evo import (
    ActiveWorkflowResponse,
    WorkflowAuditEventResponse,
    WorkflowExecutionResponse,
    WorkflowFeedbackResponse,
    WorkflowVersionResponse,
)
from app.services.agent_evo.evaluator_adapter import (
    apply_context_compression_operator,
    build_candidate_prompt,
)
from app.services.agent_evo.workflow import get_baseline_workflow
from app.services.evaluation import (
    calculate_cost,
    extract_token_usage,
)
from app.services.llm import MEDICAL_DISCLAIMER, get_llm_router
from app.services.retrieval_service import execute_rag_retrieval

logger = logging.getLogger("agentforge.services.agent_evo.lifecycle")


class WorkflowLifecycleService:
    """
    Manages governed lifecycle states after human approval:
    APPROVED -> ACTIVE -> INACTIVE / ROLLBACK -> Runtime Execution -> Feedback -> Re-optimization.
    """

    @classmethod
    def activate_version(
        cls,
        version_id: str,
        user_id: str,
        user_email: str,
        db: Session,
    ) -> Tuple[WorkflowVersion, Optional[str]]:
        """
        Promotes an approved workflow version to ACTIVE production status.
        Ensures only ONE workflow version is ACTIVE at any given time.
        Preserves previous versions and records an audit event.
        """
        target_version = db.query(WorkflowVersion).filter(
            (WorkflowVersion.version_id == version_id) | (WorkflowVersion.id == version_id)
        ).first()

        if not target_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow version '{version_id}' not found.",
            )

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        previous_active = db.query(WorkflowVersion).filter(
            WorkflowVersion.status.in_(["ACTIVE", "active"]),
            WorkflowVersion.id != target_version.id,
        ).first()

        prev_version_id = None
        if previous_active:
            prev_version_id = previous_active.version_id
            previous_active.status = "INACTIVE"
            logger.info(f"Deactivated previous active workflow version '{prev_version_id}'.")

            # Record deactivation audit event
            deact_audit = WorkflowAuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                workflow_version_id=prev_version_id,
                event_type="DEACTIVATION",
                performed_by=user_id,
                user_email=user_email,
                details={"superseded_by": target_version.version_id, "timestamp": now_utc.isoformat()},
                created_at=now_utc,
            )
            db.add(deact_audit)

        target_version.status = "ACTIVE"
        target_version.activated_at = now_utc

        # Record activation audit event
        act_audit = WorkflowAuditEvent(
            event_id=f"audit_{uuid.uuid4().hex[:8]}",
            workflow_version_id=target_version.version_id,
            event_type="ACTIVATION",
            performed_by=user_id,
            user_email=user_email,
            details={
                "previous_active_version_id": prev_version_id,
                "version_number": target_version.version_number,
                "name": target_version.name,
                "timestamp": now_utc.isoformat(),
            },
            created_at=now_utc,
        )
        db.add(act_audit)
        db.commit()
        db.refresh(target_version)

        logger.info(f"User '{user_email}' activated workflow version '{target_version.version_id}' (v{target_version.version_number}).")
        return target_version, prev_version_id

    @classmethod
    def rollback_version(
        cls,
        target_version_id: str,
        user_id: str,
        user_email: str,
        reason: Optional[str],
        db: Session,
    ) -> Tuple[WorkflowVersion, Optional[str]]:
        """
        Governed rollback to a previous approved/inactive workflow version.
        Makes target version ACTIVE, deactivates current active version, and logs an auditable rollback event.
        """
        target_version = db.query(WorkflowVersion).filter(
            (WorkflowVersion.version_id == target_version_id) | (WorkflowVersion.id == target_version_id)
        ).first()

        if not target_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target workflow version '{target_version_id}' for rollback not found.",
            )

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        current_active = db.query(WorkflowVersion).filter(
            WorkflowVersion.status.in_(["ACTIVE", "active"])
        ).first()

        prev_active_id = current_active.version_id if current_active else None

        if current_active and current_active.id == target_version.id:
            return target_version, prev_active_id

        if current_active:
            current_active.status = "INACTIVE"
            deact_audit = WorkflowAuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:8]}",
                workflow_version_id=current_active.version_id,
                event_type="DEACTIVATION",
                performed_by=user_id,
                user_email=user_email,
                details={"superseded_by_rollback": target_version.version_id, "reason": reason},
                created_at=now_utc,
            )
            db.add(deact_audit)

        target_version.status = "ACTIVE"
        target_version.activated_at = now_utc

        rollback_audit = WorkflowAuditEvent(
            event_id=f"audit_{uuid.uuid4().hex[:8]}",
            workflow_version_id=target_version.version_id,
            event_type="ROLLBACK",
            performed_by=user_id,
            user_email=user_email,
            details={
                "previous_active_version_id": prev_active_id,
                "reason": reason or "Governed rollback to earlier version",
                "timestamp": now_utc.isoformat(),
            },
            created_at=now_utc,
        )
        db.add(rollback_audit)
        db.commit()
        db.refresh(target_version)

        logger.info(f"User '{user_email}' executed rollback to workflow version '{target_version.version_id}'.")
        return target_version, prev_active_id

    @classmethod
    def get_active_workflow(cls, db: Session) -> ActiveWorkflowResponse:
        """
        Retrieves the currently ACTIVE production workflow version.
        Falls back to Production Baseline if no version has been explicitly activated.
        """
        active_version = db.query(WorkflowVersion).filter(
            WorkflowVersion.status.in_(["ACTIVE", "active"])
        ).first()

        if active_version:
            return ActiveWorkflowResponse(
                is_active_set=True,
                active_version_id=active_version.version_id,
                version_number=active_version.version_number,
                name=active_version.name,
                model_id=active_version.model_id,
                provider=active_version.provider,
                configuration=active_version.configuration,
                status="ACTIVE",
                activated_at=active_version.activated_at.isoformat() if active_version.activated_at else None,
                evaluation_snapshot=active_version.evaluation_snapshot,
            )

        baseline = get_baseline_workflow()
        return ActiveWorkflowResponse(
            is_active_set=False,
            active_version_id="baseline",
            version_number=0,
            name="Production Baseline (Default)",
            model_id=baseline.model.model_id,
            provider=baseline.model.provider,
            configuration=baseline.to_dict(),
            status="ACTIVE (DEFAULT BASELINE)",
            activated_at=None,
            evaluation_snapshot=None,
        )

    @classmethod
    async def execute_active_workflow(
        cls,
        query: str,
        user_id: str,
        document_id: Optional[str] = None,
        expected_answer: Optional[str] = None,
        workflow_version_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> WorkflowExecutionResponse:
        """
        Executes a user query using an eligible approved workflow configuration resolved by RuntimeWorkflowSelector.
        Captures LLM latency and Total execution time as separate metrics.
        Persists runtime execution records for continuous monitoring.
        """
        overall_start = time.perf_counter()
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"

        # 1. Resolve Workflow Configuration via Runtime Selector
        from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector
        selection = RuntimeWorkflowSelector.select_workflow_for_query(
            query=query,
            db=db,
            candidate_version_id_override=workflow_version_id,
        )
        active_version_id = selection.version_id
        wf_config = selection.configuration or {}
        task_category = selection.task_category
        selection_reason = selection.selection_reason

        retrieval_cfg = wf_config.get("retrieval", {})
        model_cfg = wf_config.get("model", {})
        prompt_cfg = wf_config.get("prompt", {})
        exec_cfg = wf_config.get("execution", {})

        top_k = retrieval_cfg.get("top_k", 5)
        sim_threshold = retrieval_cfg.get("similarity_threshold", 0.35)
        enable_reranking = bool(retrieval_cfg.get("reranking", True))
        model_id = model_cfg.get("model_id", "gemini-3.5-flash-lite")
        provider = model_cfg.get("provider", "Google")
        template_name = prompt_cfg.get("template_name", "default_medical")
        custom_instructions = prompt_cfg.get("custom_instructions")
        context_compression = exec_cfg.get("context_compression", False)
        wf_temperature = model_cfg.get("temperature")
        wf_max_tokens = model_cfg.get("max_tokens")

        # 2. Execute RAG Retrieval with exact active configuration
        retrieval_start = time.perf_counter()
        retrieval = execute_rag_retrieval(
            user_id=user_id,
            query=query,
            document_id=document_id,
            top_k=top_k,
            relevance_threshold=sim_threshold,
            db=db,
            enable_reranking=enable_reranking,
        )
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        context_str = ""
        if retrieval.summary_context and retrieval.summary_context.strip():
            context_str = retrieval.summary_context.strip()
        elif retrieval.results:
            parts = []
            for r in retrieval.results:
                sec = r.section_title or "General"
                parts.append(f"[{sec}]\n{r.content}")
            context_str = "\n\n".join(parts)

        raw_context_str = context_str
        # 3. Apply Context Compression Operator if enabled in active workflow
        if context_compression:
            context_str = apply_context_compression_operator(raw_context_str)

        # 4. Build Prompt
        final_prompt = build_candidate_prompt(
            template_name=template_name,
            query=query,
            context=context_str,
            custom_instructions=custom_instructions,
        )

        # 5. LLM Invocation
        llm_router = get_llm_router()
        adapter = llm_router.get_adapter(model_id) or llm_router.get_adapter("gemini")
        model_name = adapter.name if adapter else model_id

        llm_start = time.perf_counter()
        llm_success = True
        llm_error = None
        try:
            if adapter and hasattr(adapter, "generate"):
                llm_resp = await adapter.generate(
                    query=query,
                    context=context_str,
                    system_prompt=final_prompt,
                    temperature=wf_temperature,
                    max_tokens=wf_max_tokens,
                )
            else:
                llm_resp = await llm_router.generate_single(
                    model_key=model_id,
                    query=query,
                    context=context_str,
                    system_prompt=final_prompt,
                    temperature=wf_temperature,
                    max_tokens=wf_max_tokens,
                )
            llm_latency_ms = round(llm_resp.latency_ms if llm_resp.latency_ms else (time.perf_counter() - llm_start) * 1000, 2)
            answer_text = llm_resp.answer if llm_resp.success or llm_resp.answer else "The requested information could not be generated."
            raw_usage = llm_resp.raw_usage or {}
            llm_success = llm_resp.success
            llm_error = llm_resp.error
        except Exception as e:
            logger.error(f"LLM execution failed during active workflow query: {e}")
            llm_latency_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            answer_text = f"Execution error: {str(e)}"
            raw_usage = {}
            llm_success = False
            llm_error = str(e)

        pipeline_exec_ms = round(retrieval_ms + llm_latency_ms, 2)
        total_execution_time_ms = round(max((time.perf_counter() - overall_start) * 1000, pipeline_exec_ms), 2)

        # 6. Evaluation Engine Execution for Groundedness & Claim Verification
        from app.services.evaluation.evaluator import get_evaluation_engine, EvaluationInput
        eval_engine = get_evaluation_engine()
        eval_input = EvaluationInput(
            question=query,
            generated_answer=answer_text,
            retrieved_context=context_str,
            expected_answer=expected_answer,
            model_name=model_name,
            latency_ms=llm_latency_ms,
            execution_time_ms=pipeline_exec_ms,
            raw_usage=raw_usage,
            success=llm_success,
            error=llm_error,
        )
        eval_result = eval_engine.evaluate(eval_input)

        input_tokens = eval_result.input_tokens if llm_success else None
        output_tokens = eval_result.output_tokens if llm_success else None
        total_tokens = eval_result.total_tokens if llm_success else None
        total_cost = eval_result.total_cost if llm_success else None

        # 7. Format Source List
        sources = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "section_title": r.section_title,
                "similarity_score": r.similarity_score,
                "relevance_score": r.relevance_score,
            }
            for r in (retrieval.results or [])
        ]

        # 8. Structured Runtime Stage Execution Trace
        execution_trace = {
            "stages": {
                "retrieval": {
                    "latency_ms": retrieval_ms,
                    "chunks_retrieved": len(retrieval.results or []),
                    "reranking_enabled": enable_reranking,
                    "similarity_threshold": sim_threshold,
                    "top_k": top_k,
                },
                "compression": {
                    "enabled": context_compression,
                    "original_context_length": len(raw_context_str),
                    "effective_context_length": len(context_str),
                },
                "prompt": {
                    "template": template_name,
                    "custom_instructions_applied": bool(custom_instructions),
                },
                "llm_generation": {
                    "model": model_name,
                    "provider": provider,
                    "latency_ms": llm_latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "temperature": wf_temperature,
                    "max_tokens": wf_max_tokens,
                    "success": llm_success,
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
            "effective_configuration": wf_config,
            "task_category": task_category,
            "selection_reason": selection_reason,
        }

        # 9. Persist Runtime Execution Record
        if db:
            exec_record = WorkflowExecution(
                execution_id=execution_id,
                workflow_version_id=active_version_id,
                user_id=user_id,
                query=query,
                response=answer_text,
                retrieved_chunks=sources,
                model=model_name,
                provider=provider,
                input_tokens=eval_result.input_tokens if llm_success else None,
                output_tokens=eval_result.output_tokens if llm_success else None,
                total_tokens=eval_result.total_tokens if llm_success else None,
                cost=eval_result.total_cost if llm_success else None,
                llm_latency_ms=llm_latency_ms if (llm_success and llm_latency_ms is not None) else None,
                total_execution_time_ms=total_execution_time_ms,
                groundedness=eval_result.groundedness,
                execution_trace=execution_trace,
                status="success" if llm_success else "error",
                created_at=now_utc,
            )
            db.add(exec_record)
            db.commit()

        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_version_id=active_version_id,
            query=query,
            response=answer_text,
            sources=sources,
            model=model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=total_cost,
            llm_latency_ms=llm_latency_ms if llm_success else None,
            total_execution_time_ms=total_execution_time_ms,
            status="success" if llm_success else "error",
            created_at=now_utc.isoformat(),
            task_category=task_category,
            selection_reason=selection_reason,
            groundedness=eval_result.groundedness,
            execution_trace=execution_trace,
        )

    @classmethod
    def record_feedback(
        cls,
        execution_id: str,
        rating: int,
        feedback_text: Optional[str],
        user_id: Optional[str],
        db: Session,
    ) -> WorkflowFeedbackResponse:
        """
        Records user/developer feedback against an execution without altering workflow configuration or triggering optimization.
        """
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id
        ).first()

        version_id = execution.workflow_version_id if execution else "unknown"
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        feedback_id = f"fb_{uuid.uuid4().hex[:10]}"

        feedback_rec = WorkflowFeedback(
            feedback_id=feedback_id,
            execution_id=execution_id,
            workflow_version_id=version_id,
            user_id=user_id,
            rating=rating,
            feedback_text=feedback_text,
            created_at=now_utc,
        )
        db.add(feedback_rec)
        db.commit()
        db.refresh(feedback_rec)

        logger.info(f"Recorded feedback '{feedback_id}' (rating: {rating}) for execution '{execution_id}'.")
        return WorkflowFeedbackResponse(
            feedback_id=feedback_rec.feedback_id,
            execution_id=feedback_rec.execution_id,
            workflow_version_id=feedback_rec.workflow_version_id,
            rating=feedback_rec.rating,
            feedback_text=feedback_rec.feedback_text,
            created_at=feedback_rec.created_at.isoformat(),
        )

    @classmethod
    def list_executions(cls, limit: int = 50, db: Optional[Session] = None) -> List[WorkflowExecutionResponse]:
        """Lists recent runtime workflow executions."""
        if not db:
            return []
        records = db.query(WorkflowExecution).order_by(WorkflowExecution.created_at.desc()).limit(limit).all()
        return [
            WorkflowExecutionResponse(
                execution_id=r.execution_id,
                workflow_version_id=r.workflow_version_id,
                query=r.query,
                response=r.response or "",
                sources=r.retrieved_chunks or [],
                model=r.model,
                provider=r.provider,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                total_tokens=r.total_tokens,
                cost=r.cost,
                llm_latency_ms=r.llm_latency_ms,
                total_execution_time_ms=r.total_execution_time_ms,
                status=r.status,
                created_at=r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
                groundedness=r.groundedness,
                execution_trace=r.execution_trace,
            )
            for r in records
        ]

    @classmethod
    def list_feedback(cls, limit: int = 50, db: Optional[Session] = None) -> List[WorkflowFeedbackResponse]:
        """Lists all feedback items."""
        if not db:
            return []
        records = db.query(WorkflowFeedback).order_by(WorkflowFeedback.created_at.desc()).limit(limit).all()
        return [
            WorkflowFeedbackResponse(
                feedback_id=r.feedback_id,
                execution_id=r.execution_id,
                workflow_version_id=r.workflow_version_id,
                rating=r.rating,
                feedback_text=r.feedback_text,
                created_at=r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
            )
            for r in records
        ]

    @classmethod
    def list_audit_events(cls, limit: int = 50, db: Optional[Session] = None) -> List[WorkflowAuditEventResponse]:
        """Lists workflow lifecycle audit events."""
        if not db:
            return []
        records = db.query(WorkflowAuditEvent).order_by(WorkflowAuditEvent.created_at.desc()).limit(limit).all()
        return [
            WorkflowAuditEventResponse(
                id=r.id or r.event_id,
                event_id=r.event_id,
                workflow_version_id=r.workflow_version_id,
                event_type=r.event_type,
                user_email=r.user_email,
                triggered_by=r.user_email,
                previous_version_id=(r.details or {}).get("previous_active_version_id") or (r.details or {}).get("superseded_by") or (r.details or {}).get("superseded_by_rollback"),
                details=r.details,
                created_at=r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
            )
            for r in records
        ]
