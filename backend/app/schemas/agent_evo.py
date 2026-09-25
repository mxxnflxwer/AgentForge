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
        examples=["What is the primary clinical diagnosis and assessment for this patient?"],
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
    seed: Optional[int] = Field(
        None,
        description="Optional seed for deterministic candidate generation.",
    )
    allowed_mutation_types: Optional[List[str]] = Field(
        None,
        description="Optional subset of mutation categories to explore: prompt, retrieval, model, operator, mixed.",
    )


class OptimizationRunListResponse(BaseModel):
    """List of completed optimization runs."""
    total_runs: int
    runs: List[OptimizationReport]


class WorkflowVersionResponse(BaseModel):
    """Schema for a persisted approved workflow version."""
    id: str = Field(..., description="Internal database UUID")
    version_id: str = Field(..., description="Human-readable version identifier e.g. wf_v1_...")
    version_number: int = Field(..., description="Sequential version number")
    name: str = Field(..., description="Workflow display name")
    description: Optional[str] = Field(None, description="Workflow description")
    source_run_id: str = Field(..., description="Optimization run ID where this candidate was evaluated")
    source_candidate_id: str = Field(..., description="Candidate ID in the optimization run")
    model_id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Model provider")
    configuration: Dict[str, Any] = Field(..., description="Full workflow configuration dictionary")
    evaluation_snapshot: Dict[str, Any] = Field(..., description="Snapshot of Phase 6 evaluation metrics at approval")
    status: str = Field("approved", description="Workflow version governance status")
    approved_by: str = Field(..., description="Developer who approved this workflow version")
    approved_at: str = Field(..., description="ISO 8601 approval timestamp")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class WorkflowVersionListResponse(BaseModel):
    """List of all approved workflow versions."""
    total_versions: int
    versions: List[WorkflowVersionResponse]


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
    version: Optional[WorkflowVersionResponse] = Field(None, description="Created or existing workflow version")
