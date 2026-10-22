import asyncio
import datetime
import logging
from typing import Dict, List, Optional, Tuple
import uuid
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.workflow_version import WorkflowAuditEvent, WorkflowVersion
from app.services.agent_evo.candidate_generator import generate_candidates
from app.services.agent_evo.evaluator_adapter import EvaluatorAdapter
from app.services.agent_evo.pareto import EvaluatedCandidate, ParetoArchive
from app.services.agent_evo.report import OptimizationReport, generate_optimization_report
from app.services.agent_evo.workflow import AgentWorkflow, get_baseline_workflow

logger = logging.getLogger("agentforge.services.agent_evo.optimizer")


class AgentEvoOptimizer:
    """
    Central AgentEvo Optimization Engine.
    Executes evolutionary multi-candidate optimization over AgentForge workflows:
    1. Evaluates baseline workflow.
    2. Generates diverse mutated candidate workflows.
    3. Evaluates all candidates through Phase 6 evaluator adapter.
    4. Computes Pareto dominance and maintains Pareto archive.
    5. Produces comprehensive optimization trade-off report.
    6. Manages human developer approval state and workflow versioning.
    """

    def __init__(self):
        # In-memory storage for optimization runs
        self._runs: Dict[str, OptimizationReport] = {}

    def get_run(self, run_id: str) -> Optional[OptimizationReport]:
        """Retrieve an optimization run report by run_id."""
        return self._runs.get(run_id)

    def list_runs(self) -> List[OptimizationReport]:
        """List all completed optimization runs, newest first."""
        return sorted(self._runs.values(), key=lambda r: r.timestamp, reverse=True)

    def approve_workflow(
        self,
        run_id: str,
        candidate_id: str,
        approved_by: str,
        db: Optional[Session] = None,
    ) -> Tuple[OptimizationReport, WorkflowVersion]:
        """
        Record human developer approval for a specific candidate workflow and persist a WorkflowVersion.
        Enforces:
        - Optimization run must exist.
        - Candidate must exist in that run.
        - Baseline cannot be approved as an optimization candidate.
        - Only Pareto-optimal candidates can be approved (dominated rejected).
        - Idempotency: multiple approvals for the same candidate do not duplicate workflow versions.
        - Production baseline remains unchanged.
        """
        run = self._runs.get(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization run '{run_id}' not found.",
            )

        # Verify candidate exists in run
        matching_cand = next((c for c in run.all_candidates if c.candidate_id == candidate_id), None)
        if not matching_cand:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Candidate '{candidate_id}' not found in optimization run '{run_id}'.",
            )

        # Validation Rule: Baseline is not an optimization candidate
        if candidate_id == "baseline" or matching_cand.candidate_id == "baseline":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Baseline workflow cannot be approved as an optimization candidate.",
            )

        # Validation Rule: Only Pareto-optimal candidates are eligible for approval
        if not matching_cand.is_pareto_optimal:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Candidate '{candidate_id}' is dominated and not eligible for approval. Only Pareto-optimal workflows can be approved.",
            )

        # Check for existing version (Idempotency)
        existing_version = None
        if db:
            existing_version = db.query(WorkflowVersion).filter(
                WorkflowVersion.source_run_id == run_id,
                WorkflowVersion.source_candidate_id == candidate_id,
            ).first()

        if existing_version:
            run.approved_workflow_id = candidate_id
            logger.info(
                f"Candidate '{candidate_id}' in run '{run_id}' was already approved as version '{existing_version.version_id}'."
            )
            return run, existing_version

        # Sequential version number calculation
        version_number = 1
        if db:
            count = db.query(WorkflowVersion).count()
            version_number = count + 1

        version_id = f"wf_v{version_number}_{uuid.uuid4().hex[:6]}"
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # Determine evaluated task categories from run workload or query
        from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector
        task_cats: List[str] = []
        if getattr(run, "workload", None) and isinstance(run.workload, list):
            task_cats = list(dict.fromkeys(RuntimeWorkflowSelector.classify_query_task(q) for q in run.workload))
        elif getattr(run, "query", None):
            task_cats = [RuntimeWorkflowSelector.classify_query_task(run.query)]
        else:
            task_cats = ["specific_fact"]

        config_dict = matching_cand.workflow.to_dict()
        config_dict["task_categories"] = task_cats

        if getattr(run, "scheduler_decision", None):
            sched_dec = run.scheduler_decision
            config_dict["scheduler_decision"] = sched_dec.model_dump() if hasattr(sched_dec, "model_dump") else sched_dec

        if matching_cand.mutation_metadata:
            mut_meta = matching_cand.mutation_metadata
            config_dict["mutation_metadata"] = mut_meta.model_dump() if hasattr(mut_meta, "model_dump") else mut_meta

        eval_snapshot_dict = matching_cand.metrics.model_dump()
        eval_snapshot_dict["evaluated_task_categories"] = task_cats

        wf_version = WorkflowVersion(
            version_id=version_id,
            version_number=version_number,
            name=matching_cand.workflow.name,
            description=matching_cand.workflow.description or f"Approved optimization candidate from run {run_id}",
            source_run_id=run_id,
            source_candidate_id=candidate_id,
            model_id=matching_cand.workflow.model.model_id,
            provider=matching_cand.workflow.model.provider,
            configuration=config_dict,
            evaluation_snapshot=eval_snapshot_dict,
            status="approved",
            approved_by=approved_by,
            approved_at=now_utc,
            created_at=now_utc,
        )

        if db:
            audit_event = WorkflowAuditEvent(
                event_id=f"audit_{uuid.uuid4().hex[:10]}",
                workflow_version_id=version_id,
                event_type="APPROVAL",
                performed_by=None,
                user_email=approved_by,
                details={
                    "source_run_id": run_id,
                    "source_candidate_id": candidate_id,
                    "version_number": version_number,
                    "scheduler_decision": config_dict.get("scheduler_decision"),
                    "mutation_metadata": config_dict.get("mutation_metadata"),
                },
                created_at=now_utc,
            )
            db.add(wf_version)
            db.add(audit_event)
            db.commit()
            db.refresh(wf_version)

        run.approved_workflow_id = candidate_id
        logger.info(f"Developer '{approved_by}' approved workflow '{candidate_id}' creating version '{version_id}'.")
        return run, wf_version

    def list_versions(self, db: Optional[Session] = None) -> List[WorkflowVersion]:
        """Retrieve all approved workflow versions, newest first."""
        if db:
            return db.query(WorkflowVersion).order_by(WorkflowVersion.version_number.desc()).all()
        return []

    def get_version(self, version_id: str, db: Optional[Session] = None) -> Optional[WorkflowVersion]:
        """Retrieve a specific workflow version by version_id or database UUID."""
        if db:
            return db.query(WorkflowVersion).filter(
                (WorkflowVersion.version_id == version_id) | (WorkflowVersion.id == version_id)
            ).first()
        return None

    async def run_optimization(
        self,
        query: str,
        user_id: str,
        workload: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        expected_answer: Optional[str] = None,
        expected_answers: Optional[List[Optional[str]]] = None,
        candidate_count: int = 5,
        max_iterations: int = 1,
        seed: Optional[int] = None,
        strategy: str = "mutation",
        parent_workflow_id: Optional[str] = None,
        parent_b_workflow_id: Optional[str] = None,
        crossover_map: Optional[Dict[str, str]] = None,
        post_crossover_mutation: Optional[str] = "none",
        refinement_strategy: Optional[str] = None,
        allowed_mutation_types: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ) -> OptimizationReport:
        """
        Execute full AgentEvo workflow optimization end-to-end supporting:
        1. Single-Query or Multi-Query Workload evaluation.
        2. Mutation / Exploration
        3. Targeted Refinement
        4. Crossover / Recombination
        5. Adaptive Evolutionary Scheduling
        """
        run_id = f"evo_{uuid.uuid4().hex[:10]}"
        
        # Determine evaluation workload queries
        queries_to_eval = [q.strip() for q in workload if q.strip()] if workload else [query.strip()]
        is_workload = len(queries_to_eval) > 1
        primary_query = queries_to_eval[0] if queries_to_eval else query

        logger.info(
            f"Starting AgentEvo optimization run '{run_id}' [strategy={strategy}, workload_size={len(queries_to_eval)}] for query: '{primary_query[:40]}'"
        )

        archive = ParetoArchive()
        from app.services.agent_evo.crossover_service import CrossoverService
        from app.services.agent_evo.scheduler import AgentEvoScheduler, SchedulerDecision

        scheduler_decision_dict: Optional[Dict[str, Any]] = None
        effective_strategy = strategy

        # Helper evaluation runner that automatically switches between single-query and workload evaluation
        async def evaluate_wf(wf: AgentWorkflow) -> EvaluatedCandidate:
            if is_workload:
                return await EvaluatorAdapter.evaluate_candidate_workload(
                    workflow=wf,
                    queries=queries_to_eval,
                    user_id=user_id,
                    document_id=document_id,
                    expected_answers=expected_answers,
                    db=db,
                )
            else:
                return await EvaluatorAdapter.evaluate_candidate(
                    workflow=wf,
                    query=primary_query,
                    user_id=user_id,
                    document_id=document_id,
                    expected_answer=expected_answer,
                    db=db,
                )

        # -------------------------------------------------------------------
        # Strategy: Adaptive Evolutionary Scheduling
        # -------------------------------------------------------------------
        if strategy == "adaptive":
            # 1. Resolve and Evaluate Baseline first to inform scheduler
            parent_wf = get_baseline_workflow()
            if parent_workflow_id and parent_workflow_id not in ("baseline", "Production Baseline"):
                if db:
                    found_version = db.query(WorkflowVersion).filter(
                        (WorkflowVersion.version_id == parent_workflow_id)
                        | (WorkflowVersion.id == parent_workflow_id)
                        | (WorkflowVersion.source_candidate_id == parent_workflow_id)
                    ).first()
                    if found_version and isinstance(found_version.configuration, dict):
                        parent_wf = AgentWorkflow.from_dict(found_version.configuration)
                        parent_wf.workflow_id = found_version.version_id
                        parent_wf.name = found_version.name

            baseline_evaluated = await evaluate_wf(parent_wf)
            baseline_evaluated.candidate_id = parent_wf.workflow_id
            archive.add_candidate(baseline_evaluated)

            # 2. Query available approved versions
            available_versions = self.list_versions(db=db)

            # 3. Schedule Strategy
            decision: SchedulerDecision = AgentEvoScheduler.select_strategy(
                iteration=1,
                max_iterations=max_iterations,
                baseline=baseline_evaluated,
                archive_candidates=archive.all_candidates,
                available_versions=available_versions,
                candidate_count=candidate_count,
            )
            scheduler_decision_dict = decision.model_dump()
            effective_strategy = decision.strategy
            logger.info(
                f"Adaptive Scheduler selected stage '{decision.stage_name}' ({decision.strategy}): {decision.reason}"
            )

            # Override parameters with scheduler decisions
            if decision.strategy == "crossover":
                parent_a_id = decision.parameters.get("parent_workflow_id", "baseline")
                parent_b_id = decision.parameters.get("parent_b_workflow_id")
                crossover_map_to_use = decision.parameters.get("crossover_map")
                post_crossover_mutation = decision.parameters.get("post_crossover_mutation", "none")
            elif decision.strategy == "refinement":
                refinement_strategy = decision.parameters.get("refinement_strategy", "all")
                allowed_mutation_types = ["refinement"]
            elif decision.strategy == "mutation":
                allowed_mutation_types = decision.parameters.get("allowed_mutation_types", ["all"])

        # -------------------------------------------------------------------
        # Strategy: Crossover / Recombination
        # -------------------------------------------------------------------
        if effective_strategy == "crossover" or (strategy != "adaptive" and parent_b_workflow_id):
            parent_a_id = parent_workflow_id or "baseline"
            parent_b_id = parent_b_workflow_id

            if not parent_b_id:
                # If crossover was selected but no Parent B was given, choose first available approved version
                avail_vers = self.list_versions(db=db)
                distinct_ver = next((v for v in avail_vers if v.version_id != parent_a_id), None)
                if distinct_ver:
                    parent_b_id = distinct_ver.version_id
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Parent B workflow version must be specified for crossover / recombination.",
                    )

            if parent_a_id == parent_b_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent A and Parent B cannot be the same workflow version. Please select two distinct parents.",
                )

            parent_a_wf, a_id, _, a_name, _, _, _ = CrossoverService.resolve_parent(parent_a_id, db=db)
            parent_b_wf, b_id, _, b_name, _, _, _ = CrossoverService.resolve_parent(parent_b_id, db=db)

            # Step 1: Evaluate Reference Parent A
            logger.info(f"Evaluating reference Parent A '{parent_a_wf.workflow_id}'...")
            baseline_evaluated = await evaluate_wf(parent_a_wf)
            baseline_evaluated.candidate_id = parent_a_wf.workflow_id
            archive.add_candidate(baseline_evaluated)

            # Step 1b: Evaluate Reference Parent B
            logger.info(f"Evaluating reference Parent B '{parent_b_wf.workflow_id}'...")
            parent_b_evaluated = await evaluate_wf(parent_b_wf)
            parent_b_evaluated.candidate_id = parent_b_wf.workflow_id
            archive.add_candidate(parent_b_evaluated)

            # Step 2: Assemble Recombined Child Genome
            crossover_map_to_use = crossover_map or {
                "retrieval": "parent_a",
                "prompt": "parent_b",
                "model": "parent_a",
                "generation": "parent_a",
                "operators": "parent_b",
            }
            child_wf, inheritance_items = CrossoverService.assemble_child_genome(
                parent_a_wf=parent_a_wf,
                parent_b_wf=parent_b_wf,
                parent_a_id=a_id,
                parent_b_id=b_id,
                crossover_map=crossover_map_to_use,
                post_mutation=post_crossover_mutation or "none",
            )

            # Apply Optional Post-Crossover Mutation
            if post_crossover_mutation and post_crossover_mutation != "none":
                child_wf = CrossoverService._apply_post_crossover_mutation(child_wf, post_crossover_mutation)

            candidates_to_eval = [child_wf]

        # -------------------------------------------------------------------
        # Strategy: Mutation / Exploration or Refinement
        # -------------------------------------------------------------------
        else:
            # Step 1: Resolve and Evaluate Reference / Parent Workflow (if not already evaluated in adaptive step)
            parent_wf = get_baseline_workflow()
            if parent_workflow_id and parent_workflow_id not in ("baseline", "Production Baseline"):
                if db:
                    found_version = db.query(WorkflowVersion).filter(
                        (WorkflowVersion.version_id == parent_workflow_id)
                        | (WorkflowVersion.id == parent_workflow_id)
                        | (WorkflowVersion.source_candidate_id == parent_workflow_id)
                    ).first()
                    if found_version and isinstance(found_version.configuration, dict):
                        parent_wf = AgentWorkflow.from_dict(found_version.configuration)
                        parent_wf.workflow_id = found_version.version_id
                        parent_wf.name = found_version.name
                        logger.info(f"Loaded parent workflow '{found_version.version_id}' ({found_version.name}) for run '{run_id}'.")
                    else:
                        logger.warning(f"Parent workflow '{parent_workflow_id}' not found in DB; using baseline.")

            if not archive.all_candidates:
                logger.info(f"Evaluating reference workflow '{parent_wf.workflow_id}'...")
                baseline_evaluated = await evaluate_wf(parent_wf)
                baseline_evaluated.candidate_id = parent_wf.workflow_id
                archive.add_candidate(baseline_evaluated)
            else:
                baseline_evaluated = archive.all_candidates[0]

            # Step 2: Generate Diverse Mutated / Refined Candidate Workflows
            existing_versions_configs = []
            if db:
                try:
                    db_versions = db.query(WorkflowVersion).all()
                    existing_versions_configs = [v.configuration for v in db_versions if v.configuration]
                except Exception as e:
                    logger.warning(f"Failed to query existing workflow versions for duplicate prevention: {e}")

            types_to_use = allowed_mutation_types
            if effective_strategy == "refinement" and not types_to_use:
                types_to_use = ["refinement"]

            candidates_to_eval = generate_candidates(
                baseline=parent_wf,
                count=candidate_count,
                seed=seed,
                allowed_types=types_to_use,
                existing_versions=existing_versions_configs,
            )
            logger.info(f"Generated {len(candidates_to_eval)} candidate workflows.")

        # Step 3: Evaluate Candidates Sequentially across single-query or workload
        for cand_wf in candidates_to_eval:
            try:
                evaluated_cand = await evaluate_wf(cand_wf)
                archive.add_candidate(evaluated_cand)
            except Exception as e:
                logger.error(f"Unexpected evaluation failure for candidate '{cand_wf.workflow_id}': {e}")

        # Step 4: Recompute Final Pareto Frontier based on aggregate Groundedness & Cost
        pareto_frontier = archive.recompute_frontier()
        all_evaluated = archive.all_candidates

        logger.info(
            f"Optimization run '{run_id}' completed: {len(all_evaluated)} evaluated, {len(pareto_frontier)} Pareto-optimal."
        )

        # Step 5: Generate Report
        report = generate_optimization_report(
            run_id=run_id,
            query=primary_query,
            baseline=baseline_evaluated,
            all_candidates=all_evaluated,
            pareto_candidates=pareto_frontier,
            is_workload=is_workload,
            workload_queries=queries_to_eval,
            strategy=strategy,
            scheduler_decision=scheduler_decision_dict,
        )

        # Store in run registry
        self._runs[run_id] = report
        return report


# Global Singleton Optimizer Instance
_optimizer_instance: Optional[AgentEvoOptimizer] = None


def get_agent_evo_optimizer() -> AgentEvoOptimizer:
    """Get singleton AgentEvoOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = AgentEvoOptimizer()
    return _optimizer_instance
