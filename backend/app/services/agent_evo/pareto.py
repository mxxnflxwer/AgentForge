import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.agent_evo.objective import CandidateMetrics
from app.services.agent_evo.workflow import AgentWorkflow

logger = logging.getLogger("agentforge.services.agent_evo.pareto")


class EvaluatedCandidate(BaseModel):
    """An evaluated workflow candidate with configuration, metrics, and Pareto status."""
    candidate_id: str = Field(..., description="Unique candidate identifier")
    workflow: AgentWorkflow = Field(..., description="Full workflow configuration")
    metrics: CandidateMetrics = Field(..., description="Evaluated performance and cost metrics")
    is_pareto_optimal: bool = Field(False, description="Whether candidate is on the Pareto frontier")
    dominated_by: List[str] = Field(default_factory=list, description="IDs of candidates that dominate this candidate")
    dominates_candidates: List[str] = Field(default_factory=list, description="IDs of candidates this candidate dominates")


def dominates(a: CandidateMetrics, b: CandidateMetrics) -> bool:
    """
    Check if candidate metrics 'a' Pareto-dominates candidate metrics 'b'.

    Pareto Dominance Rule:
        a dominates b iff:
            a.performance >= b.performance AND
            a.cost <= b.cost AND
            (a.performance > b.performance OR a.cost < b.cost)

    Where:
        higher performance is better
        lower cost is better
    """
    # Failed / zero-evaluation candidates cannot dominate successful ones
    if a.status != "success" and b.status == "success":
        return False
    if a.status == "success" and b.status != "success":
        return True

    perf_better_or_equal = a.performance >= b.performance
    cost_better_or_equal = a.cost <= b.cost

    strictly_better_perf = a.performance > b.performance
    strictly_better_cost = a.cost < b.cost

    return perf_better_or_equal and cost_better_or_equal and (strictly_better_perf or strictly_better_cost)


class ParetoArchive:
    """
    Dynamic Pareto Archive maintaining the non-dominated workflow candidates.
    Supports incremental candidate evaluation and continuous frontier maintenance.
    """

    def __init__(self):
        self._archive: List[EvaluatedCandidate] = []
        self._all_evaluated: List[EvaluatedCandidate] = []

    @property
    def pareto_candidates(self) -> List[EvaluatedCandidate]:
        """Return the current set of non-dominated Pareto-optimal candidates."""
        return sorted(
            self._archive,
            key=lambda c: (-c.metrics.performance, c.metrics.cost),
        )

    @property
    def all_candidates(self) -> List[EvaluatedCandidate]:
        """Return all evaluated candidates."""
        return self._all_evaluated

    def add_candidate(self, candidate: EvaluatedCandidate) -> bool:
        """
        Evaluate a candidate against the archive:
        1. Checks if candidate is dominated by any existing archive member.
        2. Removes all archive members dominated by the new candidate.
        3. Adds the candidate if non-dominated.
        4. Updates cross-dominance tracking.

        Returns True if candidate entered the Pareto archive, False otherwise.
        """
        self._all_evaluated.append(candidate)
        new_metrics = candidate.metrics

        # Step 1: Check if new candidate is dominated by existing archive members
        is_dominated = False
        for existing in self._archive:
            if dominates(existing.metrics, new_metrics):
                is_dominated = True
                candidate.dominated_by.append(existing.candidate_id)
                existing.dominates_candidates.append(candidate.candidate_id)

        if is_dominated:
            candidate.is_pareto_optimal = False
            logger.info(
                f"Candidate '{candidate.candidate_id}' is dominated (perf={new_metrics.performance}, cost=${new_metrics.cost:.6f})."
            )
            return False

        # Step 2: Remove archive members dominated by the new candidate
        surviving_archive: List[EvaluatedCandidate] = []
        for existing in self._archive:
            if dominates(new_metrics, existing.metrics):
                existing.is_pareto_optimal = False
                existing.dominated_by.append(candidate.candidate_id)
                candidate.dominates_candidates.append(existing.candidate_id)
                logger.info(
                    f"Candidate '{candidate.candidate_id}' dominates existing archive candidate '{existing.candidate_id}'."
                )
            else:
                surviving_archive.append(existing)

        # Step 3: Add new non-dominated candidate to archive
        candidate.is_pareto_optimal = True
        surviving_archive.append(candidate)
        self._archive = surviving_archive

        logger.info(
            f"Candidate '{candidate.candidate_id}' added to Pareto archive (perf={new_metrics.performance}, cost=${new_metrics.cost:.6f}). Archive size: {len(self._archive)}"
        )
        return True

    def recompute_frontier(self) -> List[EvaluatedCandidate]:
        """
        Recompute full Pareto dominance across all evaluated candidates.
        Guarantees mathematical correctness regardless of arrival order.
        """
        self._archive = []
        for cand in self._all_evaluated:
            cand.dominated_by = []
            cand.dominates_candidates = []
            cand.is_pareto_optimal = False

        for i, c_i in enumerate(self._all_evaluated):
            is_dom = False
            for j, c_j in enumerate(self._all_evaluated):
                if i == j:
                    continue
                if dominates(c_j.metrics, c_i.metrics):
                    is_dom = True
                    c_i.dominated_by.append(c_j.candidate_id)
                    c_j.dominates_candidates.append(c_i.candidate_id)

            if not is_dom and c_i.metrics.status in ("success", "evaluated"):
                c_i.is_pareto_optimal = True
                self._archive.append(c_i)

        return self.pareto_candidates
