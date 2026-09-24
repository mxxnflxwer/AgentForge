from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.agent_evo.pareto import EvaluatedCandidate
from app.services.agent_evo.report import OptimizationReport, ParetoTradeOff


class OptimizationRunRequest(BaseModel):
    """Request payload to initiate an AgentEvo workflow optimization run."""
    query: str = Field(
        ...,
        min_length=1,
        description="The target query to optimize workflows for.",
        examples=["What is the clinical diagnosis?"],
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to scope retrieval to a specific document.",
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Optional ground truth reference answer for accuracy calculation.",
        examples=["Stable angina pectoris, likely related to underlying coronary artery disease."],
    )
    candidate_count: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of candidate workflow mutations to generate and evaluate.",
    )


class OptimizationRunListResponse(BaseModel):
    """List of completed optimization runs."""
    total_runs: int
    runs: List[OptimizationReport]


class ApproveWorkflowRequest(BaseModel):
    """Request to approve a specific candidate workflow from an optimization run."""
    run_id: str = Field(..., description="ID of the optimization run")
    candidate_id: str = Field(..., description="ID of the candidate workflow to approve")


class ApproveWorkflowResponse(BaseModel):
    """Response confirming human developer workflow approval."""
    status: str
    run_id: str
    approved_workflow_id: str
    message: str
    report: OptimizationReport
