import datetime
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.agent_evo.objective import CandidateMetrics
from app.services.agent_evo.pareto import EvaluatedCandidate

logger = logging.getLogger("agentforge.services.agent_evo.report")


class ParetoTradeOff(BaseModel):
    """Analysis of a Pareto candidate's trade-offs compared to baseline."""
    candidate_id: str
    name: str
    performance_delta_pct: float
    cost_delta_pct: float
    latency_delta_pct: float
    trade_off_summary: str
    workflow_differences: Dict[str, Any]


class OptimizationReport(BaseModel):
    """Structured AgentEvo Optimization Report."""
    run_id: str = Field(..., description="Unique optimization run identifier")
    timestamp: str = Field(..., description="ISO 8601 execution timestamp")
    query: str = Field(..., description="Target query optimized")
    baseline: EvaluatedCandidate = Field(..., description="Baseline reference candidate")
    total_candidates_evaluated: int = Field(..., description="Total candidates generated and evaluated")
    successful_candidates: int = Field(..., description="Number of successfully evaluated candidates")
    failed_candidates: int = Field(..., description="Number of failed/invalid candidates")
    pareto_frontier_count: int = Field(..., description="Number of candidates on the Pareto frontier")
    pareto_frontier: List[EvaluatedCandidate] = Field(default_factory=list, description="Non-dominated Pareto candidates")
    all_candidates: List[EvaluatedCandidate] = Field(default_factory=list, description="All evaluated candidates")
    trade_off_analyses: List[ParetoTradeOff] = Field(default_factory=list, description="Comparative trade-off breakdown")
    executive_summary: str = Field(..., description="Summary of optimization results")
    human_approval_required: bool = Field(True, description="Human developer approval is required before deployment")
    approved_workflow_id: Optional[str] = Field(None, description="ID of human-approved candidate if approved")


def compute_workflow_diff(base_wf: Dict[str, Any], cand_wf: Dict[str, Any]) -> Dict[str, Any]:
    """Compute parameter differences between baseline and candidate workflow."""
    diffs = {}
    for section in ["retrieval", "model", "prompt", "execution"]:
        base_sec = base_wf.get(section, {})
        cand_sec = cand_wf.get(section, {})
        sec_diff = {}
        for k, v in cand_sec.items():
            if base_sec.get(k) != v:
                sec_diff[k] = {"baseline": base_sec.get(k), "candidate": v}
        if sec_diff:
            diffs[section] = sec_diff
    return diffs


def generate_optimization_report(
    run_id: str,
    query: str,
    baseline: EvaluatedCandidate,
    all_candidates: List[EvaluatedCandidate],
    pareto_candidates: List[EvaluatedCandidate],
) -> OptimizationReport:
    """
    Generate a comprehensive AgentEvo Optimization Report analyzing Pareto trade-offs.
    """
    successful = [c for c in all_candidates if c.metrics.status == "success"]
    failed = [c for c in all_candidates if c.metrics.status != "success"]

    trade_offs: List[ParetoTradeOff] = []
    base_perf = max(0.0001, baseline.metrics.performance)
    base_cost = max(0.000001, baseline.metrics.cost)
    base_lat = max(1.0, baseline.metrics.llm_latency_ms)

    for p in pareto_candidates:
        perf_delta = round(((p.metrics.performance - base_perf) / base_perf) * 100, 2)
        cost_delta = round(((p.metrics.cost - base_cost) / base_cost) * 100, 2)
        lat_delta = round(((p.metrics.llm_latency_ms - base_lat) / base_lat) * 100, 2)

        summary_parts = []
        if p.candidate_id == baseline.candidate_id:
            summary_parts.append("Baseline configuration is Pareto-optimal.")
        else:
            if perf_delta > 0 and cost_delta <= 0:
                summary_parts.append(f"Strict improvement: +{perf_delta}% groundedness at {abs(cost_delta)}% lower cost.")
            elif perf_delta >= 0 and cost_delta < 0:
                summary_parts.append(f"Cost reduction: {abs(cost_delta)}% cheaper with equal groundedness.")
            elif perf_delta > 0 and cost_delta > 0:
                summary_parts.append(f"Performance trade-off: +{perf_delta}% groundedness with +{cost_delta}% cost increase.")
            else:
                summary_parts.append(f"Alternative trade-off point: Groundedness {p.metrics.performance * 100:.1f}%, Cost ${p.metrics.cost:.6f}.")

        diff = compute_workflow_diff(baseline.workflow.to_dict(), p.workflow.to_dict())

        trade_offs.append(
            ParetoTradeOff(
                candidate_id=p.candidate_id,
                name=p.workflow.name,
                performance_delta_pct=perf_delta,
                cost_delta_pct=cost_delta,
                latency_delta_pct=lat_delta,
                trade_off_summary=" ".join(summary_parts),
                workflow_differences=diff,
            )
        )

    exec_summary = (
        f"Optimization run '{run_id}' evaluated {len(all_candidates)} total candidates. "
        f"{len(pareto_candidates)} candidates form the Pareto-optimal frontier. "
        f"Developer review and approval required before applying workflow changes."
    )

    return OptimizationReport(
        run_id=run_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        query=query,
        baseline=baseline,
        total_candidates_evaluated=len(all_candidates),
        successful_candidates=len(successful),
        failed_candidates=len(failed),
        pareto_frontier_count=len(pareto_candidates),
        pareto_frontier=pareto_candidates,
        all_candidates=all_candidates,
        trade_off_analyses=trade_offs,
        executive_summary=exec_summary,
        human_approval_required=True,
        approved_workflow_id=None,
    )
