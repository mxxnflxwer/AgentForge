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
    parent_workflow_id: Optional[str] = Field(
        None,
        description="Optional parent workflow version ID e.g. wf_v4_b0844c or candidate ID to refine/mutate from.",
    )
    allowed_mutation_types: Optional[List[str]] = Field(
        None,
        description="Optional subset of mutation categories to explore or refine: prompt, retrieval, model, operator, mixed, refinement.",
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


# ---------------------------------------------------------------------------
# Phase 7.3+ Version Benchmarking & Comparison Schemas
# ---------------------------------------------------------------------------

class BenchmarkRunRequest(BaseModel):
    """Request payload to execute a controlled multi-version benchmark."""
    version_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of workflow version IDs (e.g. 'baseline', 'wf_v1_...', 'wf_v2_...') to benchmark.",
    )
    name: Optional[str] = Field(
        "Version Benchmark Comparison",
        description="Display name for this benchmark run.",
    )
    description: Optional[str] = Field(
        None,
        description="Optional description or note for the benchmark.",
    )
    preset_name: Optional[str] = Field(
        "clinical",
        description="Preset identifier or custom benchmark description.",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="The identical query to evaluate across all selected workflow versions.",
        examples=["What is the primary clinical diagnosis and assessment for this patient?"],
    )
    document_id: Optional[str] = Field(
        None,
        description="Target document ID for retrieval across all versions.",
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Target document IDs for retrieval across all versions.",
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Ground truth reference answer for accuracy computation.",
    )
    iterations: int = Field(
        1,
        ge=1,
        le=5,
        description="Number of evaluation iterations to run per workflow for measuring mean and variation.",
    )
    repetitions: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Number of evaluation repetitions per workflow (alias for iterations).",
    )


class BenchmarkItemResult(BaseModel):
    """Evaluation and Pareto classification metrics for a single version in a benchmark."""
    version_id: str
    version_number: Optional[int] = None
    workflow_name: str
    parent_workflow_id: Optional[str] = None
    source_candidate_id: Optional[str] = None
    source_run_id: Optional[str] = None
    model_id: str
    provider: str
    groundedness: float
    hallucination_rate: float
    cost: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    llm_latency_ms: float
    execution_time_ms: float
    retrieval: Dict[str, Any]
    retrieval_config_summary: Optional[str] = None
    prompt_template: str
    prompt_strategy: Optional[str] = None
    context_compression: bool
    operator_config: Optional[Dict[str, Any]] = None
    is_pareto_optimal: bool
    pareto_status: str  # "Pareto-optimal" | "Dominated" | "Non-dominated"
    dominated_by: List[str] = []
    dominates: List[str] = []
    claim_breakdown: Optional[List[Dict[str, Any]]] = None
    raw_details: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any]
    iterations_count: int = 1
    runs: Optional[List[Dict[str, Any]]] = None
    relative_changes: Optional[Dict[str, float]] = None

    # Canonical named fields for unambiguous rendering and API contract
    version: Optional[str] = None
    workflow: Optional[str] = None
    parent_workflow: Optional[str] = None
    cost_per_query: Optional[float] = None
    hallucination: Optional[float] = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    retrieval_config: Optional[str] = None
    std_dev: Optional[Dict[str, float]] = None

    def model_post_init(self, __context: Any) -> None:
        """Ensure canonical aliases are always populated and synchronized."""
        if self.version is None:
            self.version = self.version_id
        if self.workflow is None:
            self.workflow = self.workflow_name
        if self.parent_workflow is None:
            self.parent_workflow = self.parent_workflow_id or "baseline"
        if self.cost_per_query is None:
            self.cost_per_query = self.cost
        if self.hallucination is None:
            self.hallucination = self.hallucination_rate
        if self.latency_ms is None:
            self.latency_ms = self.llm_latency_ms
        if self.model is None:
            self.model = self.model_id
        if self.retrieval_config is None:
            self.retrieval_config = self.retrieval_config_summary or f"k={self.retrieval.get('top_k', 5)}"



class BenchmarkReportResponse(BaseModel):
    """Complete report for a multi-version benchmark execution."""
    benchmark_id: str
    name: str
    description: Optional[str] = None
    status: str = "completed"
    preset_name: Optional[str] = None
    query: str
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    expected_answer: Optional[str] = None
    repetitions: Optional[int] = None
    selected_versions: List[str]
    results: List[BenchmarkItemResult]
    pareto_frontier: List[BenchmarkItemResult]
    dominated_versions: List[BenchmarkItemResult]
    summary: str
    created_by: str
    created_at: str


class BenchmarkRunListResponse(BaseModel):
    """List of all persisted benchmark runs."""
    total_benchmarks: int
    benchmarks: List[BenchmarkReportResponse]


class VersionLineageNode(BaseModel):
    """Single node in the workflow lineage graph."""
    version_id: str
    version_number: Optional[int] = None
    name: str
    parent_workflow_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_candidate_id: Optional[str] = None
    model_id: str
    provider: str
    mutation_type: Optional[str] = None
    refinement_operation: Optional[str] = None
    fields_changed: Optional[List[str]] = None
    rationale: Optional[str] = None
    before_configuration: Optional[Dict[str, Any]] = None
    after_configuration: Optional[Dict[str, Any]] = None
    evaluation_snapshot: Optional[Dict[str, Any]] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    status: str = "approved"
    configuration: Optional[Dict[str, Any]] = None


class VersionLineageEdge(BaseModel):
    """Directed parent-to-child edge in the lineage graph."""
    source: str
    target: str
    label: Optional[str] = None


class VersionLineageResponse(BaseModel):
    """Complete workflow lineage graph."""
    total_versions: int
    nodes: List[VersionLineageNode]
    edges: List[VersionLineageEdge]


class VersionInspectResponse(BaseModel):
    """Detailed inspection metadata for a single workflow version."""
    version_id: str
    version_number: Optional[int] = None
    name: str
    description: Optional[str] = None
    parent_workflow_id: Optional[str] = None
    parent_version: Optional[Dict[str, Any]] = None
    source_run_id: Optional[str] = None
    source_candidate_id: Optional[str] = None
    model_id: str
    provider: str
    mutation_type: Optional[str] = None
    refinement_operation: Optional[str] = None
    fields_changed: Optional[List[str]] = None
    rationale: Optional[str] = None
    before_configuration: Optional[Dict[str, Any]] = None
    after_configuration: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any]
    evaluation_snapshot: Dict[str, Any]
    status: str
    approved_by: str
    approved_at: str
    created_at: str
