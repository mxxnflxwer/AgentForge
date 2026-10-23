from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.agent_evo.pareto import EvaluatedCandidate
from app.services.agent_evo.report import OptimizationReport, ParetoTradeOff


class OptimizationRunRequest(BaseModel):
    """Request payload to initiate an AgentEvo workflow optimization run."""
    query: Optional[str] = Field(
        None,
        description="Target query to optimize workflows for (single-query mode).",
        examples=["What is the primary clinical diagnosis and assessment for this patient?"],
    )
    workload: Optional[List[str]] = Field(
        None,
        description="Optional evaluation workload containing multiple representative queries.",
        examples=[
            "What is the primary clinical diagnosis and assessment for this patient?",
            "What medications were prescribed for this patient?",
            "What are the abnormal laboratory findings?",
            "What is the patient's treatment plan?",
            "What is the relevant cardiac history?",
        ],
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
    expected_answers: Optional[List[Optional[str]]] = Field(
        None,
        description="Optional list of ground truth reference answers matching each query in the evaluation workload.",
    )
    candidate_count: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of candidate workflow mutations to generate and evaluate.",
    )
    max_iterations: int = Field(
        1,
        ge=1,
        le=5,
        description="Maximum evolutionary loop iterations for adaptive optimization.",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional seed for deterministic candidate generation.",
    )
    strategy: Optional[str] = Field(
        "mutation",
        description="Candidate generation strategy: 'mutation' (Exploration), 'refinement' (Targeted), 'crossover' (Recombination), 'adaptive' (Adaptive Scheduler).",
    )
    parent_workflow_id: Optional[str] = Field(
        None,
        description="Parent A / primary parent workflow version ID e.g. wf_v4_b0844c or candidate ID.",
    )
    parent_b_workflow_id: Optional[str] = Field(
        None,
        description="Parent B workflow version ID for crossover recombination.",
    )
    crossover_map: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of components ('model', 'prompt', 'retrieval', 'generation', 'operators') to 'parent_a' or 'parent_b'.",
    )
    post_crossover_mutation: Optional[str] = Field(
        "none",
        description="Optional post-crossover mutation: 'none', 'prompt', 'retrieval', 'model', 'operator', 'composite'.",
    )
    refinement_strategy: Optional[str] = Field(
        None,
        description="Targeted refinement operation: e.g. 'prompt_simplification', 'retrieval_reduction', 'context_compression'.",
    )
    allowed_mutation_types: Optional[List[str]] = Field(
        None,
        description="Optional subset of mutation categories to explore: prompt, retrieval, model, operator, mixed, refinement.",
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
    status: str = Field("APPROVED", description="Workflow version governance status: APPROVED, ACTIVE, INACTIVE")
    approved_by: str = Field(..., description="Developer who approved this workflow version")
    approved_at: str = Field(..., description="ISO 8601 approval timestamp")
    activated_at: Optional[str] = Field(None, description="ISO 8601 activation timestamp if active")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    task_categories: Optional[List[str]] = Field(None, description="Clinical task categories this workflow is evaluated or optimized for")


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
# Workflow Governance & Production Lifecycle Schemas
# ---------------------------------------------------------------------------

class ActivateWorkflowResponse(BaseModel):
    """Response when an approved workflow is promoted to ACTIVE production status."""
    status: str
    version_id: str
    message: str
    previous_active_version_id: Optional[str] = None
    version: WorkflowVersionResponse


class RollbackWorkflowRequest(BaseModel):
    """Request to rollback production to a previous approved/inactive workflow version."""
    target_version_id: str = Field(..., description="The previous version ID to reactivate")
    reason: Optional[str] = Field(None, description="Reason for performing the rollback")


class RollbackWorkflowResponse(BaseModel):
    """Response confirming rollback of production workflow."""
    status: str
    active_version_id: str
    previous_active_version_id: Optional[str] = None
    message: str
    version: WorkflowVersionResponse


class ActiveWorkflowResponse(BaseModel):
    """Details of the currently active production workflow version."""
    is_active_set: bool
    active_version_id: str
    version_number: Optional[int] = None
    name: str
    model_id: str
    provider: str
    configuration: Dict[str, Any]
    status: str
    activated_at: Optional[str] = None
    evaluation_snapshot: Optional[Dict[str, Any]] = None


class WorkflowExecutionRequest(BaseModel):
    """Request payload to execute a query against the currently active production workflow."""
    query: str = Field(..., min_length=1, description="User query to process")
    document_id: Optional[str] = Field(None, description="Optional document ID for retrieval")
    expected_answer: Optional[str] = Field(None, description="Optional ground truth reference")
    workflow_version_id: Optional[str] = Field(None, description="Optional workflow version ID override")


class WorkflowExecutionResponse(BaseModel):
    """Runtime metrics and response from executing a query through the active workflow."""
    execution_id: str
    workflow_version_id: str
    query: str
    response: str
    sources: List[Dict[str, Any]] = []
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    total_execution_time_ms: float = 0.0
    status: str = "success"
    created_at: str
    task_category: Optional[str] = Field(None, description="Classified clinical task category for the query")
    selection_reason: Optional[str] = Field(None, description="Explainable selection rationale for the chosen workflow")
    groundedness: Optional[float] = Field(None, description="Evaluated factual groundedness ratio [0.0 - 1.0]")
    execution_trace: Optional[Dict[str, Any]] = Field(None, description="Structured runtime stage execution trace")


class WorkflowExecutionListResponse(BaseModel):
    """List of production workflow runtime executions."""
    total_executions: int
    executions: List[WorkflowExecutionResponse]


class WorkflowFeedbackRequest(BaseModel):
    """Feedback payload for a specific workflow execution."""
    rating: int = Field(..., ge=1, le=5, description="Numerical rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, description="Optional qualitative feedback or issue description")


class WorkflowFeedbackResponse(BaseModel):
    """Confirmation of recorded feedback."""
    feedback_id: str
    execution_id: str
    workflow_version_id: str
    rating: int
    feedback_text: Optional[str] = None
    created_at: str


class WorkflowFeedbackListResponse(BaseModel):
    """List of feedback items across production executions."""
    total_feedback: int
    feedback: List[WorkflowFeedbackResponse]


class WorkflowAuditEventResponse(BaseModel):
    """Audit log entry for lifecycle governance actions."""
    id: Optional[str] = None
    event_id: str
    workflow_version_id: str
    event_type: str
    user_email: str
    details: Optional[Dict[str, Any]] = None
    created_at: str
    triggered_by: Optional[str] = None
    previous_version_id: Optional[str] = None


class WorkflowAuditListResponse(BaseModel):
    """List of audit events."""
    total_events: int
    events: List[WorkflowAuditEventResponse]
    audit_events: Optional[List[WorkflowAuditEventResponse]] = None


class WorkflowReoptimizeRequest(BaseModel):
    """Request to trigger explicit developer re-optimization from runtime feedback context."""
    version_id: Optional[str] = Field(None, description="Optional baseline version to optimize from (defaults to ACTIVE)")
    query: Optional[str] = Field(None, description="Target query (defaults to standard or recent feedback query)")
    document_id: Optional[str] = Field(None, description="Optional target document ID")
    use_feedback_context: bool = Field(True, description="Whether to include feedback insights in rationale")
    strategy: Optional[str] = Field("mutation", description="Generation strategy: mutation, refinement, or crossover")


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
    query: Optional[str] = Field(
        None,
        description="The identical query to evaluate across all selected workflow versions.",
        examples=["What is the primary clinical diagnosis and assessment for this patient?"],
    )
    workload: Optional[List[str]] = Field(
        None,
        description="Optional evaluation workload of multiple queries to benchmark identically across versions.",
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
    expected_answers: Optional[List[Optional[str]]] = Field(
        None,
        description="Optional ground truth reference answers for workload queries.",
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
    groundedness: Optional[float] = None
    hallucination_rate: Optional[float] = None
    cost: Optional[float] = None
    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    llm_latency_ms: Optional[float] = None
    execution_time_ms: Optional[float] = None
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
    parent_a_version_id: Optional[str] = None
    parent_b_version_id: Optional[str] = None
    candidate_type: Optional[str] = "MUTATION"  # "BASELINE" | "MUTATION" | "CROSSOVER"
    crossover_strategy: Optional[str] = None
    crossover_map: Optional[Dict[str, str]] = None
    source_run_id: Optional[str] = None
    source_candidate_id: Optional[str] = None
    model_id: str
    provider: str
    mutation_type: Optional[str] = None
    refinement_operation: Optional[str] = None
    scheduler_strategy: Optional[str] = None
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
    parent_a_version_id: Optional[str] = None
    parent_b_version_id: Optional[str] = None
    candidate_type: Optional[str] = "MUTATION"
    crossover_strategy: Optional[str] = None
    crossover_map: Optional[Dict[str, str]] = None
    parent_version: Optional[Dict[str, Any]] = None
    parent_a_version: Optional[Dict[str, Any]] = None
    parent_b_version: Optional[Dict[str, Any]] = None
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


# ---------------------------------------------------------------------------
# Phase 7.4 Evolutionary Crossover Schemas
# ---------------------------------------------------------------------------

class CrossoverInheritanceItem(BaseModel):
    """Component inheritance mapping detail."""
    component: str
    source_parent: str  # "parent_a" | "parent_b"
    source_version_id: str
    value: Any


class CrossoverPreviewRequest(BaseModel):
    """Request payload to preview component inheritance and validate crossover configuration."""
    parent_a_version_id: str = Field(..., description="Version ID or name of Parent A")
    parent_b_version_id: str = Field(..., description="Version ID or name of Parent B")
    crossover_map: Dict[str, str] = Field(
        default_factory=lambda: {
            "model": "parent_a",
            "prompt": "parent_b",
            "retrieval": "parent_a",
            "generation": "parent_a",
            "operators": "parent_b",
        },
        description="Map of component to parent source ('parent_a' or 'parent_b')",
    )
    post_crossover_mutation: Optional[str] = Field(
        "none",
        description="Optional post-crossover mutation: 'none', 'prompt', 'retrieval', 'model', 'operator', 'composite'",
    )


class CrossoverPreviewResponse(BaseModel):
    """Preview of resulting child workflow genome and validation status before execution."""
    is_valid: bool
    validation_errors: List[str] = []
    parent_a: Dict[str, Any]
    parent_b: Dict[str, Any]
    crossover_strategy: str = "component_level"
    crossover_map: Dict[str, str]
    inheritance_breakdown: List[CrossoverInheritanceItem]
    resulting_configuration: Dict[str, Any]
    post_crossover_mutation: Optional[str] = "none"


class CrossoverCandidateRequest(BaseModel):
    """Request payload to construct, validate, and evaluate a crossover candidate workflow."""
    parent_a_version_id: str = Field(..., description="Version ID of Parent A")
    parent_b_version_id: str = Field(..., description="Version ID of Parent B")
    crossover_map: Dict[str, str] = Field(
        ...,
        description="Map of component name to parent ('parent_a' or 'parent_b')",
    )
    post_crossover_mutation: Optional[str] = Field(
        "none",
        description="Optional post-crossover mutation: 'none', 'prompt', 'retrieval', 'model', 'operator', 'composite'",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="The evaluation query for testing the crossover child.",
    )
    document_id: Optional[str] = Field(None, description="Optional document ID for evaluation.")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs for evaluation.")
    expected_answer: Optional[str] = Field(None, description="Optional ground truth expected answer.")
    evaluate: bool = Field(True, description="Whether to immediately evaluate the child workflow through Phase 6.")


class CrossoverCandidateResponse(BaseModel):
    """Complete record of an evaluated crossover candidate workflow."""
    candidate_id: str
    candidate_type: str = "CROSSOVER"
    name: str
    description: str
    parent_a_version_id: str
    parent_b_version_id: str
    parent_a_name: Optional[str] = None
    parent_b_name: Optional[str] = None
    parent_a_candidate_id: Optional[str] = None
    parent_b_candidate_id: Optional[str] = None
    parent_a_run_id: Optional[str] = None
    parent_b_run_id: Optional[str] = None
    crossover_strategy: str = "component_level"
    crossover_map: Dict[str, str]
    post_crossover_mutation_strategy: Optional[str] = "none"
    resulting_workflow_configuration: Dict[str, Any]
    evaluation_run_id: Optional[str] = None
    groundedness: Optional[float] = 0.0
    hallucination: Optional[float] = 0.0
    cost: float = 0.0
    cost_per_query: Optional[float] = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    llm_latency_ms: Optional[float] = None
    execution_time_ms: Optional[float] = None
    is_pareto_optimal: bool
    pareto_status: str
    dominated_by: List[str] = []
    dominates: List[str] = []
    parent_a_metrics: Optional[Dict[str, Any]] = None
    parent_b_metrics: Optional[Dict[str, Any]] = None
    inheritance_breakdown: Optional[List[CrossoverInheritanceItem]] = None
    claim_breakdown: Optional[List[Dict[str, Any]]] = None
    raw_details: Optional[Dict[str, Any]] = None
    created_at: str

    def model_post_init(self, __context: Any) -> None:
        if self.cost_per_query is None:
            self.cost_per_query = self.cost
        if self.llm_latency_ms is None:
            self.llm_latency_ms = self.latency_ms


# ---------------------------------------------------------------------------
# Reproducible Baseline vs Optimized Evidence Comparison Schemas
# ---------------------------------------------------------------------------

class PerQueryComparisonRecord(BaseModel):
    """Granular per-query comparative evidence record."""
    query_id: str
    query: str
    task_category: str
    baseline_response: str
    baseline_groundedness: float
    baseline_hallucination_rate: float
    baseline_claims: List[Any] = []
    baseline_supported_claim_count: int = 0
    baseline_unsupported_claim_count: int = 0
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_total_tokens: int = 0
    baseline_cost: float = 0.0
    baseline_latency_ms: float = 0.0
    baseline_provider: str = "Google"
    baseline_model_id: str = "gemini-3.5-flash-lite"
    baseline_status: str = "success"

    optimized_response: str
    optimized_groundedness: float
    optimized_hallucination_rate: float
    optimized_claims: List[Any] = []
    optimized_supported_claim_count: int = 0
    optimized_unsupported_claim_count: int = 0
    optimized_input_tokens: int = 0
    optimized_output_tokens: int = 0
    optimized_total_tokens: int = 0
    optimized_cost: float = 0.0
    optimized_latency_ms: float = 0.0
    optimized_provider: str = "Google"
    optimized_model_id: str = "gemini-3.5-flash-lite"
    optimized_status: str = "success"

    cost_reduction: float = 0.0
    groundedness_delta: float = 0.0


class BaselineVsOptimizedComparisonRequest(BaseModel):
    """Request payload to execute a controlled baseline vs optimized workflow workload comparison."""
    baseline_version_id: Optional[str] = Field("baseline", description="Version ID for reference baseline (default: 'baseline')")
    optimized_version_id: str = Field(..., description="Version ID for evaluated optimized workflow")
    workload: List[str] = Field(..., min_length=1, description="List of queries to evaluate identically against baseline and optimized workflows")
    document_id: Optional[str] = Field(None, description="Document ID to scope retrieval")
    expected_answers: Optional[List[Optional[str]]] = Field(None, description="Optional ground truth reference answers")


class BaselineVsOptimizedComparisonResponse(BaseModel):
    """
    Structured, reproducible experimental evidence comparing baseline and optimized workflow configurations
    across an identical evaluation workload.
    """
    baseline_version_id: str
    optimized_version_id: str
    optimized_version_number: Optional[int] = None
    optimized_workflow_name: str
    evaluation_mode: str  # "live_provider" or "mocked_or_saved_response"
    is_mocked: bool
    mode_disclaimer: str
    scope_claim: str
    total_queries_evaluated: int
    successful_evaluations: int
    failed_evaluations: int
    baseline_groundedness: float
    optimized_groundedness: float
    groundedness_diff_pp: float
    groundedness_retained: bool
    baseline_hallucination_rate: float
    optimized_hallucination_rate: float
    baseline_cost_per_query: float
    optimized_cost_per_query: float
    cost_reduction_pct: float
    cost_reduced: bool
    baseline_tokens_per_query: float
    optimized_tokens_per_query: float
    baseline_latency_ms: float
    optimized_latency_ms: float
    per_query_evidence: List[PerQueryComparisonRecord]
    finding_summary: str
    created_at: str
