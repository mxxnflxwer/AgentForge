import asyncio
import datetime
import logging
from typing import Dict, List, Optional, Tuple
import uuid
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.workflow_version import WorkflowVersion
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

        wf_version = WorkflowVersion(
            version_id=version_id,
            version_number=version_number,
            name=matching_cand.workflow.name,
            description=matching_cand.workflow.description or f"Approved optimization candidate from run {run_id}",
            source_run_id=run_id,
            source_candidate_id=candidate_id,
            model_id=matching_cand.workflow.model.model_id,
            provider=matching_cand.workflow.model.provider,
            configuration=matching_cand.workflow.to_dict(),
            evaluation_snapshot=matching_cand.metrics.model_dump(),
            status="approved",
            approved_by=approved_by,
            approved_at=now_utc,
            created_at=now_utc,
        )

        if db:
            db.add(wf_version)
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
        document_id: Optional[str] = None,
        expected_answer: Optional[str] = None,
        candidate_count: int = 5,
        seed: Optional[int] = None,
        allowed_mutation_types: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ) -> OptimizationReport:
        """
        Execute full AgentEvo workflow optimization end-to-end.
        """
        run_id = f"evo_{uuid.uuid4().hex[:10]}"
        logger.info(f"Starting AgentEvo optimization run '{run_id}' for query: '{query[:40]}'")

        archive = ParetoArchive()

        # Step 1: Evaluate Baseline Workflow
        baseline_wf = get_baseline_workflow()
        logger.info("Evaluating baseline workflow...")
        baseline_evaluated: EvaluatedCandidate = await EvaluatorAdapter.evaluate_candidate(
            workflow=baseline_wf,
            query=query,
            user_id=user_id,
            document_id=document_id,
            expected_answer=expected_answer,
            db=db,
        )
        archive.add_candidate(baseline_evaluated)

        # Step 2: Generate Diverse Mutated Candidate Workflows
        candidates_to_eval = generate_candidates(
            baseline=baseline_wf,
            count=candidate_count,
            seed=seed,
            allowed_types=allowed_mutation_types,
        )
        logger.info(f"Generated {len(candidates_to_eval)} mutated candidate workflows.")

        # Step 3: Evaluate Candidates Sequentially / Concurrently
        for cand_wf in candidates_to_eval:
            try:
                evaluated_cand = await EvaluatorAdapter.evaluate_candidate(
                    workflow=cand_wf,
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    expected_answer=expected_answer,
                    db=db,
                )
                archive.add_candidate(evaluated_cand)
            except Exception as e:
                logger.error(f"Unexpected evaluation failure for candidate '{cand_wf.workflow_id}': {e}")

        # Step 4: Recompute Final Pareto Frontier
        pareto_frontier = archive.recompute_frontier()
        all_evaluated = archive.all_candidates

        logger.info(
            f"Optimization run '{run_id}' completed: {len(all_evaluated)} evaluated, {len(pareto_frontier)} Pareto-optimal."
        )

        # Step 5: Generate Report
        report = generate_optimization_report(
            run_id=run_id,
            query=query,
            baseline=baseline_evaluated,
            all_candidates=all_evaluated,
            pareto_candidates=pareto_frontier,
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
