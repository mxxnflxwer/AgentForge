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
    base_is_valid = baseline.metrics.status == "success"
    base_perf = baseline.metrics.performance
    base_cost = baseline.metrics.cost
    base_lat = baseline.metrics.llm_latency_ms

    for p in pareto_candidates:
        if base_is_valid:
            perf_denom = max(0.0001, base_perf)
            cost_denom = max(0.000001, base_cost)
            lat_denom = max(1.0, base_lat)
            
            # Groundedness difference in percentage points
            groundedness_diff_pp = round((p.metrics.performance - base_perf) * 100, 2)
            perf_delta = groundedness_diff_pp
            cost_delta = round(((p.metrics.cost - base_cost) / cost_denom) * 100, 2)
            lat_delta = round(((p.metrics.llm_latency_ms - base_lat) / lat_denom) * 100, 2)

            summary_parts = []
            if p.candidate_id == baseline.candidate_id:
                summary_parts.append("Baseline configuration is Pareto-optimal.")
            else:
                # Factual objective-specific descriptions using actual measured values
                if abs(groundedness_diff_pp) < 0.01:
                    perf_desc = "the same measured groundedness as the baseline"
                elif groundedness_diff_pp > 0:
                    perf_desc = f"{groundedness_diff_pp:+.2f} percentage points higher measured groundedness"
                else:
                    perf_desc = f"{abs(groundedness_diff_pp):.2f} percentage points lower measured groundedness"

                if abs(cost_delta) < 0.01:
                    cost_desc = "identical measured cost"
                elif cost_delta < 0:
                    cost_desc = f"{abs(cost_delta):.2f}% lower measured cost"
                else:
                    cost_desc = f"{cost_delta:.2f}% higher measured cost"

                summary_parts.append(f"The candidate achieved {perf_desc} with {cost_desc}.")

                if abs(lat_delta) >= 0.1:
                    lat_dir = "lower" if lat_delta < 0 else "higher"
                    summary_parts.append(f"It recorded {abs(lat_delta):.1f}% {lat_dir} measured latency in this evaluation run.")
        else:
            perf_delta = 0.0
            cost_delta = 0.0
            lat_delta = 0.0
            err_reason = baseline.metrics.error_message or baseline.metrics.status
            summary_parts = [
                f"Baseline evaluation unavailable ({err_reason}). Direct comparison deltas unavailable. "
                f"Candidate achieved Groundedness {p.metrics.performance * 100:.1f}% at ${p.metrics.cost:.6f} cost / query."
            ]

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

    total_workflows = len(all_candidates)
    baseline_count = 1 if baseline else 0
    generated_count = max(0, total_workflows - baseline_count)
    pareto_count = len(pareto_candidates)

    wf_word = "workflow" if total_workflows == 1 else "workflows"
    base_word = "baseline" if baseline_count == 1 else "baselines"
    cand_word = "generated candidate" if generated_count == 1 else "generated candidates"
    pareto_wf_str = "workflow forms" if pareto_count == 1 else "workflows form"

    exec_summary = (
        f"Optimization run {run_id} evaluated {total_workflows} {wf_word}:\n"
        f"{baseline_count} {base_word} + {generated_count} {cand_word}.\n\n"
        f"{pareto_count} {pareto_wf_str} the Pareto-optimal frontier.\n\n"
        f"Developer review and approval are required before applying workflow changes."
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

