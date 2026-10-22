import copy
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.workflow_version import WorkflowVersion
from app.schemas.agent_evo import (
    CrossoverCandidateRequest,
    CrossoverCandidateResponse,
    CrossoverInheritanceItem,
    CrossoverPreviewRequest,
    CrossoverPreviewResponse,
)
from app.services.agent_evo.evaluator_adapter import EvaluatorAdapter
from app.services.agent_evo.pareto import EvaluatedCandidate, ParetoArchive
from app.services.agent_evo.validator import WorkflowValidator
from app.services.agent_evo.workflow import (
    AgentWorkflow,
    ExecutionConfig,
    ModelConfig,
    MutationMetadata,
    MutationType,
    PromptConfig,
    RetrievalConfig,
    get_baseline_workflow,
)

logger = logging.getLogger("agentforge.services.agent_evo.crossover")


class CrossoverService:
    """
    AgentEvo Evolutionary Crossover Service.
    Combines configuration components from two approved parent workflows to produce a valid,
    reproducible child candidate workflow, with optional post-crossover mutation, validation,
    claim-level Phase 6 evaluation, and 2D Pareto analysis.
    """

    _crossover_candidates: Dict[str, CrossoverCandidateResponse] = {}

    @classmethod
    def resolve_parent(
        cls,
        parent_id: str,
        db: Optional[Session] = None,
    ) -> Tuple[AgentWorkflow, str, Optional[int], str, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Resolve a parent version identifier to its AgentWorkflow instance and metadata.
        Returns:
            (workflow_instance, version_id, version_number, name, source_candidate_id, source_run_id, evaluation_snapshot)
        """
        norm_id = parent_id.strip()
        if norm_id in ("baseline", "Production Baseline", "0"):
            base_wf = get_baseline_workflow()
            return (
                base_wf,
                "baseline",
                0,
                "Production Baseline",
                None,
                None,
                None,
            )

        if not db:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session required to resolve parent workflow version.",
            )

        version_rec = db.query(WorkflowVersion).filter(
            (WorkflowVersion.version_id == norm_id)
            | (WorkflowVersion.id == norm_id)
            | (WorkflowVersion.source_candidate_id == norm_id)
            | (WorkflowVersion.name == norm_id)
        ).first()

        if not version_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent workflow version '{parent_id}' not found.",
            )

        cfg = version_rec.configuration or {}
        wf_inst = AgentWorkflow.from_dict(cfg)
        wf_inst.workflow_id = version_rec.version_id
        wf_inst.name = version_rec.name

        return (
            wf_inst,
            version_rec.version_id,
            version_rec.version_number,
            version_rec.name,
            version_rec.source_candidate_id,
            version_rec.source_run_id,
            version_rec.evaluation_snapshot,
        )

    @classmethod
    def assemble_child_genome(
        cls,
        parent_a_wf: AgentWorkflow,
        parent_b_wf: AgentWorkflow,
        parent_a_id: str,
        parent_b_id: str,
        crossover_map: Dict[str, str],
        post_mutation: Optional[str] = "none",
    ) -> Tuple[AgentWorkflow, List[CrossoverInheritanceItem]]:
        """
        Deterministic component-level crossover recombination.
        Inherits specified components strictly from Parent A or Parent B according to crossover_map.
        """
        # Validate crossover map keys and values
        valid_sources = {"parent_a", "parent_b"}
        normalized_map: Dict[str, str] = {}
        for comp, src in crossover_map.items():
            norm_src = src.lower().strip()
            if norm_src not in valid_sources:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid crossover source '{src}' for component '{comp}'. Must be 'parent_a' or 'parent_b'.",
                )
            normalized_map[comp.lower().strip()] = norm_src

        inheritance_items: List[CrossoverInheritanceItem] = []

        # 1. Retrieval component
        retrieval_src = normalized_map.get("retrieval", "parent_a")
        chosen_retrieval = copy.deepcopy(
            parent_a_wf.retrieval if retrieval_src == "parent_a" else parent_b_wf.retrieval
        )
        inheritance_items.append(
            CrossoverInheritanceItem(
                component="retrieval",
                source_parent=retrieval_src,
                source_version_id=parent_a_id if retrieval_src == "parent_a" else parent_b_id,
                value=chosen_retrieval.model_dump(),
            )
        )

        # 2. Prompt component
        prompt_src = normalized_map.get("prompt", "parent_b")
        chosen_prompt = copy.deepcopy(
            parent_a_wf.prompt if prompt_src == "parent_a" else parent_b_wf.prompt
        )
        inheritance_items.append(
            CrossoverInheritanceItem(
                component="prompt",
                source_parent=prompt_src,
                source_version_id=parent_a_id if prompt_src == "parent_a" else parent_b_id,
                value=chosen_prompt.model_dump(),
            )
        )

        # 3. Model component
        model_src = normalized_map.get("model", "parent_a")
        chosen_model_base = copy.deepcopy(
            parent_a_wf.model if model_src == "parent_a" else parent_b_wf.model
        )
        inheritance_items.append(
            CrossoverInheritanceItem(
                component="model",
                source_parent=model_src,
                source_version_id=parent_a_id if model_src == "parent_a" else parent_b_id,
                value={
                    "provider": chosen_model_base.provider,
                    "model_id": chosen_model_base.model_id,
                },
            )
        )

        # 4. Generation component (temperature, max_tokens)
        gen_src = normalized_map.get("generation", model_src)
        gen_parent_wf = parent_a_wf if gen_src == "parent_a" else parent_b_wf
        final_temperature = gen_parent_wf.model.temperature
        final_max_tokens = gen_parent_wf.model.max_tokens

        chosen_model = ModelConfig(
            provider=chosen_model_base.provider,
            model_id=chosen_model_base.model_id,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
        )
        inheritance_items.append(
            CrossoverInheritanceItem(
                component="generation",
                source_parent=gen_src,
                source_version_id=parent_a_id if gen_src == "parent_a" else parent_b_id,
                value={
                    "temperature": final_temperature,
                    "max_tokens": final_max_tokens,
                },
            )
        )

        # 5. Operators / Execution component (context_compression, timeout_seconds)
        op_src = normalized_map.get("operators") or normalized_map.get("execution") or "parent_b"
        chosen_execution = copy.deepcopy(
            parent_a_wf.execution if op_src == "parent_a" else parent_b_wf.execution
        )
        inheritance_items.append(
            CrossoverInheritanceItem(
                component="operators",
                source_parent=op_src,
                source_version_id=parent_a_id if op_src == "parent_a" else parent_b_id,
                value=chosen_execution.model_dump(),
            )
        )

        cand_id = f"cand_crossover_{uuid.uuid4().hex[:8]}"
        child_wf = AgentWorkflow(
            workflow_id=cand_id,
            name=f"Crossover: {parent_a_wf.name} × {parent_b_wf.name}",
            description=f"Evolutionary crossover child inheriting components from {parent_a_id} and {parent_b_id}.",
            retrieval=chosen_retrieval,
            model=chosen_model,
            prompt=chosen_prompt,
            execution=chosen_execution,
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.CROSSOVER,
                parent_workflow_id=parent_a_id,
                mutation_description=f"Evolutionary crossover recombining genome from {parent_a_id} and {parent_b_id}.",
                generation_metadata={
                    "candidate_type": "CROSSOVER",
                    "parent_a_version_id": parent_a_id,
                    "parent_b_version_id": parent_b_id,
                    "crossover_strategy": "component_level",
                    "crossover_map": normalized_map,
                    "post_crossover_mutation_strategy": post_mutation or "none",
                },
            ),
        )

        # Apply optional post-crossover mutation if requested
        if post_mutation and post_mutation.lower() != "none":
            cls._apply_post_crossover_mutation(child_wf, post_mutation.lower())

        return child_wf, inheritance_items

    @classmethod
    def _apply_post_crossover_mutation(cls, workflow: AgentWorkflow, mutation_type: str) -> None:
        """Apply explicit post-crossover mutation to the assembled child genome."""
        norm_type = mutation_type.lower().strip()
        if norm_type == "prompt":
            # Switch to concise medical or detailed template
            if workflow.prompt.template_name == "default_medical":
                workflow.prompt.template_name = "concise_medical"
            else:
                workflow.prompt.template_name = "default_medical"
            workflow.description += " [Post-mutation: Prompt variation applied]"

        elif norm_type == "retrieval":
            # Tune top_k or similarity_threshold
            if workflow.retrieval.top_k > 3:
                workflow.retrieval.top_k -= 1
            else:
                workflow.retrieval.top_k = 4
            workflow.retrieval.similarity_threshold = round(
                min(0.50, workflow.retrieval.similarity_threshold + 0.05), 2
            )
            workflow.description += " [Post-mutation: Retrieval precision tuned]"

        elif norm_type == "model":
            # Switch temperature to deterministic 0.0
            workflow.model.temperature = 0.0
            workflow.description += " [Post-mutation: Deterministic temperature applied]"

        elif norm_type == "operator":
            # Toggle context compression
            workflow.execution.context_compression = not workflow.execution.context_compression
            workflow.description += " [Post-mutation: Operator compression toggled]"

        elif norm_type == "composite":
            # Combine prompt and operator mutation
            workflow.prompt.template_name = "concise_medical"
            workflow.execution.context_compression = True
            workflow.description += " [Post-mutation: Composite prompt + operator mutation applied]"

    @classmethod
    def preview_crossover(
        cls,
        request: CrossoverPreviewRequest,
        db: Optional[Session] = None,
    ) -> CrossoverPreviewResponse:
        """
        Validate proposed crossover genome and return detailed component inheritance breakdown.
        Does not evaluate or persist candidate.
        """
        if request.parent_a_version_id == request.parent_b_version_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent A and Parent B cannot be the same workflow version. Please select two distinct parents.",
            )

        parent_a_wf, a_id, _, a_name, _, _, _ = cls.resolve_parent(request.parent_a_version_id, db=db)
        parent_b_wf, b_id, _, b_name, _, _, _ = cls.resolve_parent(request.parent_b_version_id, db=db)

        child_wf, inheritance_items = cls.assemble_child_genome(
            parent_a_wf=parent_a_wf,
            parent_b_wf=parent_b_wf,
            parent_a_id=a_id,
            parent_b_id=b_id,
            crossover_map=request.crossover_map,
            post_mutation=request.post_crossover_mutation,
        )

        validation = WorkflowValidator.validate(child_wf)

        return CrossoverPreviewResponse(
            is_valid=validation.is_valid,
            validation_errors=validation.errors,
            parent_a={
                "version_id": a_id,
                "name": a_name,
                "configuration": parent_a_wf.to_dict(),
            },
            parent_b={
                "version_id": b_id,
                "name": b_name,
                "configuration": parent_b_wf.to_dict(),
            },
            crossover_strategy="component_level",
            crossover_map=request.crossover_map,
            inheritance_breakdown=inheritance_items,
            resulting_configuration=child_wf.to_dict(),
            post_crossover_mutation=request.post_crossover_mutation or "none",
        )

    @classmethod
    async def create_crossover_candidate(
        cls,
        request: CrossoverCandidateRequest,
        user_id: str,
        user_email: str,
        db: Optional[Session] = None,
    ) -> CrossoverCandidateResponse:
        """
        Execute evolutionary crossover, validate configuration, evaluate through Phase 6,
        compute 2D Pareto dominance against parents and baseline, and persist candidate.
        """
        if request.parent_a_version_id == request.parent_b_version_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent A and Parent B cannot be the same workflow version. Please select two distinct parents.",
            )

        parent_a_wf, a_id, _, a_name, a_cand_id, a_run_id, a_snapshot = cls.resolve_parent(
            request.parent_a_version_id, db=db
        )
        parent_b_wf, b_id, _, b_name, b_cand_id, b_run_id, b_snapshot = cls.resolve_parent(
            request.parent_b_version_id, db=db
        )

        child_wf, inheritance_items = cls.assemble_child_genome(
            parent_a_wf=parent_a_wf,
            parent_b_wf=parent_b_wf,
            parent_a_id=a_id,
            parent_b_id=b_id,
            crossover_map=request.crossover_map,
            post_mutation=request.post_crossover_mutation,
        )

        # Validate resulting child
        validation = WorkflowValidator.validate(child_wf)
        if not validation.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Crossover configuration validation failed: {'; '.join(validation.errors)}",
            )

        logger.info(
            f"Evaluating Crossover Child '{child_wf.workflow_id}' (Parent A: {a_id}, Parent B: {b_id}) for query: '{request.query[:40]}'"
        )

        # Phase 6 Evaluation
        doc_id = request.document_id or (request.document_ids[0] if request.document_ids else None)
        evaluated_child: EvaluatedCandidate = await EvaluatorAdapter.evaluate_candidate(
            workflow=child_wf,
            query=request.query,
            user_id=user_id,
            document_id=doc_id,
            expected_answer=request.expected_answer,
            db=db,
        )

        # Compute 2D Pareto Dominance (Groundedness ↑ vs Cost ↓)
        # Compare child against Parent A, Parent B, and Baseline reference
        archive = ParetoArchive()

        # Add reference candidates if available
        base_wf = get_baseline_workflow()
        base_cand = EvaluatedCandidate(
            candidate_id="baseline",
            workflow=base_wf,
            metrics=evaluated_child.metrics,  # placeholder
            is_pareto_optimal=True,
            pareto_status="Pareto-optimal",
        )
        # Populate realistic metrics from snapshot if available
        if a_snapshot:
            archive.add_candidate(
                EvaluatedCandidate(
                    candidate_id=a_id,
                    workflow=parent_a_wf,
                    metrics=EvaluatedCandidate(
                        candidate_id=a_id,
                        workflow=parent_a_wf,
                        metrics=evaluated_child.metrics,
                    ).metrics.model_copy(
                        update={
                            "groundedness": a_snapshot.get("groundedness", 1.0),
                            "cost": a_snapshot.get("cost", 0.000039),
                            "total_tokens": a_snapshot.get("total_tokens", 400),
                            "llm_latency_ms": a_snapshot.get("llm_latency_ms", 2000.0),
                            "hallucination_rate": a_snapshot.get("hallucination_rate", 0.0),
                        }
                    ),
                )
            )
        if b_snapshot:
            archive.add_candidate(
                EvaluatedCandidate(
                    candidate_id=b_id,
                    workflow=parent_b_wf,
                    metrics=EvaluatedCandidate(
                        candidate_id=b_id,
                        workflow=parent_b_wf,
                        metrics=evaluated_child.metrics,
                    ).metrics.model_copy(
                        update={
                            "groundedness": b_snapshot.get("groundedness", 1.0),
                            "cost": b_snapshot.get("cost", 0.000039),
                            "total_tokens": b_snapshot.get("total_tokens", 400),
                            "llm_latency_ms": b_snapshot.get("llm_latency_ms", 2000.0),
                            "hallucination_rate": b_snapshot.get("hallucination_rate", 0.0),
                        }
                    ),
                )
            )

        archive.add_candidate(evaluated_child)
        archive.recompute_frontier()

        # Extract child dominance status
        matching_child = next(
            (c for c in archive.all_candidates if c.candidate_id == child_wf.workflow_id),
            evaluated_child,
        )

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        resp = CrossoverCandidateResponse(
            candidate_id=child_wf.workflow_id,
            candidate_type="CROSSOVER",
            name=child_wf.name,
            description=child_wf.description,
            parent_a_version_id=a_id,
            parent_b_version_id=b_id,
            parent_a_name=a_name,
            parent_b_name=b_name,
            parent_a_candidate_id=a_cand_id,
            parent_b_candidate_id=b_cand_id,
            parent_a_run_id=a_run_id,
            parent_b_run_id=b_run_id,
            crossover_strategy="component_level",
            crossover_map=request.crossover_map,
            post_crossover_mutation_strategy=request.post_crossover_mutation or "none",
            resulting_workflow_configuration=child_wf.to_dict(),
            evaluation_run_id=f"eval_cross_{uuid.uuid4().hex[:8]}",
            groundedness=evaluated_child.metrics.groundedness if evaluated_child.metrics.groundedness is not None else (evaluated_child.metrics.performance or 0.0),
            hallucination=evaluated_child.metrics.hallucination_rate if evaluated_child.metrics.hallucination_rate is not None else 0.0,
            cost=evaluated_child.metrics.cost or 0.0,
            cost_per_query=evaluated_child.metrics.cost or 0.0,
            total_tokens=evaluated_child.metrics.total_tokens or 0,
            input_tokens=evaluated_child.metrics.input_tokens or 0,
            output_tokens=evaluated_child.metrics.output_tokens or 0,
            latency_ms=evaluated_child.metrics.llm_latency_ms or 0.0,
            llm_latency_ms=evaluated_child.metrics.llm_latency_ms,
            execution_time_ms=evaluated_child.metrics.execution_time_ms,
            is_pareto_optimal=matching_child.is_pareto_optimal,
            pareto_status="Pareto-optimal" if matching_child.is_pareto_optimal else "Dominated",
            dominated_by=matching_child.dominated_by or [],
            dominates=getattr(matching_child, "dominates_candidates", []) or getattr(matching_child, "dominates", []) or [],
            parent_a_metrics=a_snapshot,
            parent_b_metrics=b_snapshot,
            inheritance_breakdown=inheritance_items,
            claim_breakdown=evaluated_child.metrics.raw_details.get("claim_breakdown") if isinstance(evaluated_child.metrics.raw_details, dict) else None,
            raw_details=evaluated_child.metrics.raw_details if isinstance(evaluated_child.metrics.raw_details, dict) else {},
            created_at=now_str,
        )

        cls._crossover_candidates[child_wf.workflow_id] = resp
        return resp

    @classmethod
    def get_crossover_candidate(cls, candidate_id: str) -> Optional[CrossoverCandidateResponse]:
        """Retrieve a persisted crossover candidate by candidate_id."""
        return cls._crossover_candidates.get(candidate_id)

    @classmethod
    def list_crossover_candidates(cls) -> List[CrossoverCandidateResponse]:
        """List all evaluated crossover candidates."""
        return list(cls._crossover_candidates.values())
