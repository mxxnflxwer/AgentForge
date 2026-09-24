import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.services.evaluation.evaluator import EvaluationResult

logger = logging.getLogger("agentforge.services.agent_evo.objective")


class CandidateMetrics(BaseModel):
    """Normalized metrics container for an evaluated candidate workflow."""
    workflow_id: str = Field(..., description="ID of the evaluated workflow")
    workflow_name: str = Field(..., description="Name of the evaluated workflow")
    performance: float = Field(0.0, description="Primary performance objective score (Groundedness) [0.0 - 1.0]")
    cost: float = Field(0.0, description="Primary cost objective (Total Cost in USD)")
    groundedness: Optional[float] = Field(None, description="Groundedness ratio [0.0 - 1.0]")
    hallucination_rate: Optional[float] = Field(None, description="Hallucination rate [0.0 - 1.0]")
    accuracy: Optional[float] = Field(None, description="Accuracy score against reference if available")
    total_tokens: int = Field(0, description="Total tokens used (input + output)")
    input_tokens: int = Field(0, description="Input prompt tokens")
    output_tokens: int = Field(0, description="Output candidate tokens")
    llm_latency_ms: float = Field(0.0, description="LLM generation latency in ms")
    execution_time_ms: float = Field(0.0, description="Total pipeline execution time in ms")
    status: str = Field("success", description="Status: 'success', 'empty_response', 'model_error', 'invalid'")
    error_message: Optional[str] = Field(None, description="Error message if evaluation failed")
    raw_details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic details from evaluator")

    @classmethod
    def from_eval_result(
        cls,
        workflow_id: str,
        workflow_name: str,
        eval_result: EvaluationResult,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> "CandidateMetrics":
        """Extract and normalize objective scores directly from Phase 6 EvaluationResult."""
        # Performance objective = groundedness (default to 0.0 if None due to failure)
        perf_score = eval_result.groundedness if eval_result.groundedness is not None else 0.0
        cost_score = eval_result.total_cost if eval_result.total_cost is not None else 0.0

        eval_status = eval_result.details.get("status", status)
        eval_err = eval_result.details.get("error_message") or error_message

        return cls(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            performance=round(perf_score, 4),
            cost=round(cost_score, 6),
            groundedness=eval_result.groundedness,
            hallucination_rate=eval_result.hallucination_rate,
            accuracy=eval_result.accuracy,
            total_tokens=eval_result.total_tokens,
            input_tokens=eval_result.input_tokens,
            output_tokens=eval_result.output_tokens,
            llm_latency_ms=eval_result.latency_ms,
            execution_time_ms=eval_result.execution_time_ms,
            status=eval_status,
            error_message=eval_err,
            raw_details=eval_result.details,
        )

    @classmethod
    def from_failure(
        cls,
        workflow_id: str,
        workflow_name: str,
        status: str,
        error_message: str,
    ) -> "CandidateMetrics":
        """Create a zero-performance failed candidate record."""
        return cls(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            performance=0.0,
            cost=0.0,
            groundedness=None,
            hallucination_rate=None,
            accuracy=None,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            llm_latency_ms=0.0,
            execution_time_ms=0.0,
            status=status,
            error_message=error_message,
            raw_details={"status": status, "error": error_message},
        )
