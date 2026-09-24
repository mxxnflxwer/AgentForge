import asyncio
import logging
import uuid
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

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
    6. Manages human developer approval state.
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

    def approve_workflow(self, run_id: str, candidate_id: str) -> Optional[OptimizationReport]:
        """
        Record human developer approval for a specific candidate workflow.
        Does NOT automatically deploy or overwrite production without explicit governance.
        """
        run = self._runs.get(run_id)
        if not run:
            return None

        # Verify candidate exists in run
        matching_cand = next((c for c in run.all_candidates if c.candidate_id == candidate_id), None)
        if not matching_cand:
            logger.warning(f"Candidate '{candidate_id}' not found in run '{run_id}'.")
            return None

        run.approved_workflow_id = candidate_id
        logger.info(f"Developer approved workflow '{candidate_id}' for optimization run '{run_id}'.")
        return run

    async def run_optimization(
        self,
        query: str,
        user_id: str,
        document_id: Optional[str] = None,
        expected_answer: Optional[str] = None,
        candidate_count: int = 5,
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

        # Step 2: Generate Mutated Candidate Workflows
        candidates_to_eval = generate_candidates(baseline=baseline_wf, count=candidate_count)
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
