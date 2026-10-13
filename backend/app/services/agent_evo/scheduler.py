import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.models.workflow_version import WorkflowVersion
from app.services.agent_evo.pareto import EvaluatedCandidate

logger = logging.getLogger("agentforge.services.agent_evo.scheduler")


class SchedulerDecision(BaseModel):
    """Deterministic, explainable decision made by the AgentEvo Evolutionary Scheduler."""
    strategy: str = Field(..., description="Selected strategy: 'mutation', 'refinement', 'crossover'")
    stage_name: str = Field(..., description="Human-readable stage title")
    reason: str = Field(..., description="Factual, explainable rationale derived from optimization state")
    iteration: int = Field(1, description="Current optimization loop iteration")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters tailored for this stage")


class AgentEvoScheduler:
    """
    Deterministic Adaptive Evolutionary Scheduler for AgentForge AgentEvo.
    Selects among:
      - Mutation / Exploration (stochastic parameter exploration)
      - Targeted Refinement (deterministic simplification and context compression)
      - Crossover / Recombination (inheriting configuration components from two parent workflows)
    
    Principles:
      - Pure deterministic rule-based scheduler (no LLM, no hallucinated decisions).
      - Transparent and explainable rationale visible to developers.
      - Respects hard bounds on iterations and candidate count.
      - Never modifies Pareto objectives.
    """

    @classmethod
    def select_strategy(
        cls,
        iteration: int = 1,
        max_iterations: int = 1,
        baseline: Optional[EvaluatedCandidate] = None,
        archive_candidates: Optional[List[EvaluatedCandidate]] = None,
        available_versions: Optional[List[WorkflowVersion]] = None,
        candidate_count: int = 4,
    ) -> SchedulerDecision:
        """
        Evaluate optimization state and select the next evolutionary strategy.
        """
        archive_candidates = archive_candidates or []
        available_versions = available_versions or []

        # -------------------------------------------------------------------
        # Rule 1: Initial iteration or empty archive -> Exploration
        # -------------------------------------------------------------------
        if iteration == 1 or not archive_candidates:
            return SchedulerDecision(
                strategy="mutation",
                stage_name="Broad Exploration",
                reason="Initial optimization stage. Exploring diverse parameter space across prompt templates, retrieval top-K, model temperatures, and operators.",
                iteration=iteration,
                parameters={"allowed_mutation_types": ["all"], "candidate_count": candidate_count},
            )

        # Compute archive metrics
        successful_cands = [c for c in archive_candidates if c.metrics.status == "success" and c.candidate_id != "baseline"]
        pareto_cands = [c for c in archive_candidates if c.is_pareto_optimal and c.candidate_id != "baseline"]

        base_perf = baseline.metrics.performance if baseline and baseline.metrics.status == "success" else 1.0
        base_cost = baseline.metrics.cost if baseline and baseline.metrics.status == "success" else 0.00008

        # -------------------------------------------------------------------
        # Rule 2: Groundedness is preserved/high, but cost is not yet reduced -> Targeted Refinement
        # -------------------------------------------------------------------
        high_groundedness_exists = any(
            (c.metrics.groundedness or c.metrics.performance) >= (base_perf - 0.02)
            for c in successful_cands
        )
        cost_reduced = any(
            c.metrics.cost < (base_cost * 0.85)
            for c in successful_cands
        )

        if high_groundedness_exists and not cost_reduced:
            return SchedulerDecision(
                strategy="refinement",
                stage_name="Targeted Refinement",
                reason=f"Candidates maintained high measured groundedness ({base_perf * 100:.1f}%) but showed limited cost reduction. Scheduling Targeted Refinement (context compression and retrieval simplification) to reduce cost.",
                iteration=iteration,
                parameters={
                    "refinement_strategy": "all",
                    "candidate_count": min(candidate_count, 4),
                },
            )

        # -------------------------------------------------------------------
        # Rule 3: Distinct approved versions exist -> Crossover / Recombination
        # -------------------------------------------------------------------
        distinct_approved = [v for v in available_versions if v.version_id != "baseline"]
        if len(distinct_approved) >= 1 and len(pareto_cands) >= 1:
            parent_a = "baseline"
            parent_b = distinct_approved[0].version_id
            return SchedulerDecision(
                strategy="crossover",
                stage_name="Crossover Recombination",
                reason=f"Found high-performing approved workflow versions (Parent A: '{parent_a}', Parent B: '{parent_b}'). Scheduling Crossover to recombine grounded retrieval with token-compressed execution.",
                iteration=iteration,
                parameters={
                    "parent_workflow_id": parent_a,
                    "parent_b_workflow_id": parent_b,
                    "crossover_map": {
                        "retrieval": "parent_a",
                        "prompt": "parent_b",
                        "model": "parent_a",
                        "generation": "parent_a",
                        "operators": "parent_b",
                    },
                    "post_crossover_mutation": "none",
                    "candidate_count": 1,
                },
            )

        # -------------------------------------------------------------------
        # Rule 4: Stagnation / Further Refinement -> Focused Exploration
        # -------------------------------------------------------------------
        return SchedulerDecision(
            strategy="mutation",
            stage_name="Focused Exploration",
            reason=f"Refining Pareto frontier. Exploring stochastic variations across operator DAGs and prompt templates (Iteration {iteration}/{max_iterations}).",
            iteration=iteration,
            parameters={"allowed_mutation_types": ["operator", "prompt", "model"], "candidate_count": candidate_count},
        )
