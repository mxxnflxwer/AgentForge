import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User, UserRole
from app.models.workflow_version import WorkflowVersion
from app.schemas.agent_evo import (
    ActivateWorkflowResponse,
    ActiveWorkflowResponse,
    ApproveWorkflowRequest,
    ApproveWorkflowResponse,
    BaselineVsOptimizedComparisonRequest,
    BaselineVsOptimizedComparisonResponse,
    BenchmarkReportResponse,
    BenchmarkRunListResponse,
    BenchmarkRunRequest,
    CrossoverCandidateRequest,
    CrossoverCandidateResponse,
    CrossoverPreviewRequest,
    CrossoverPreviewResponse,
    OptimizationRunListResponse,
    OptimizationRunRequest,
    RollbackWorkflowRequest,
    RollbackWorkflowResponse,
    VersionInspectResponse,
    VersionLineageResponse,
    WorkflowAuditEventResponse,
    WorkflowAuditListResponse,
    WorkflowExecutionListResponse,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowFeedbackListResponse,
    WorkflowFeedbackRequest,
    WorkflowFeedbackResponse,
    WorkflowReoptimizeRequest,
    WorkflowVersionListResponse,
    WorkflowVersionResponse,
)
from app.services.agent_evo import (
    EvaluatedCandidate,
    OptimizationReport,
    get_agent_evo_optimizer,
)
from app.services.agent_evo.benchmark_service import BenchmarkService
from app.services.agent_evo.crossover_service import CrossoverService
from app.services.agent_evo.workflow_lifecycle_service import WorkflowLifecycleService


logger = logging.getLogger("agentforge.api.agent_evo")

router = APIRouter(prefix="/api/agent-evo", tags=["AgentEvo Optimization Engine"])
workflows_router = APIRouter(prefix="/api/workflows", tags=["Workflow Lifecycle & Versions"])


def format_version_response(version: WorkflowVersion) -> WorkflowVersionResponse:
    """Format SQLAlchemy WorkflowVersion model into API schema."""
    approved_at_str = (
        version.approved_at.isoformat()
        if hasattr(version.approved_at, "isoformat")
        else str(version.approved_at)
    )
    created_at_str = (
        version.created_at.isoformat()
        if hasattr(version.created_at, "isoformat")
        else str(version.created_at)
    )
    activated_at_str = (
        version.activated_at.isoformat()
        if getattr(version, "activated_at", None) and hasattr(version.activated_at, "isoformat")
        else str(version.activated_at) if getattr(version, "activated_at", None) else None
    )
    task_cats = None
    if isinstance(version.configuration, dict):
        task_cats = version.configuration.get("task_categories")
    if not task_cats and isinstance(version.evaluation_snapshot, dict):
        task_cats = version.evaluation_snapshot.get("evaluated_task_categories")

    return WorkflowVersionResponse(
        id=str(version.id),
        version_id=version.version_id,
        version_number=version.version_number,
        name=version.name,
        description=version.description,
        source_run_id=version.source_run_id,
        source_candidate_id=version.source_candidate_id,
        model_id=version.model_id,
        provider=version.provider,
        configuration=version.configuration,
        evaluation_snapshot=version.evaluation_snapshot,
        status=version.status.upper() if version.status else "APPROVED",
        approved_by=version.approved_by,
        approved_at=approved_at_str,
        activated_at=activated_at_str,
        created_at=created_at_str,
        task_categories=task_cats,
    )


@router.post(
    "/optimize",
    response_model=OptimizationReport,
    status_code=status.HTTP_200_OK,
    summary="Execute AgentEvo workflow generation, evaluation, and Pareto optimization",
)
async def optimize_workflow(
    request: OptimizationRunRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OptimizationReport:
    """
    Execute autonomous AgentEvo workflow optimization:
    Supports 3 Candidate Generation Strategies:
    1. Mutation / Exploration
    2. Refinement
    3. Crossover / Recombination
    All strategies produce candidates evaluated through Phase 6 and Pareto analysis.
    """
    optimizer = get_agent_evo_optimizer()
    target_query = request.query
    if not target_query and not request.workload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'query' or 'workload' must be provided for AgentEvo optimization.",
        )
    if not target_query and request.workload:
        target_query = request.workload[0]

    report = await optimizer.run_optimization(
        query=target_query,
        user_id=current_user.id,
        workload=request.workload,
        document_id=request.document_id,
        expected_answer=request.expected_answer,
        expected_answers=request.expected_answers,
        candidate_count=request.candidate_count,
        max_iterations=request.max_iterations,
        seed=request.seed,
        strategy=request.strategy or "mutation",
        parent_workflow_id=request.parent_workflow_id,
        parent_b_workflow_id=request.parent_b_workflow_id,
        crossover_map=request.crossover_map,
        post_crossover_mutation=request.post_crossover_mutation,
        refinement_strategy=request.refinement_strategy,
        allowed_mutation_types=request.allowed_mutation_types,
        db=db,
    )
    return report


@router.get(
    "/runs",
    response_model=OptimizationRunListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all completed AgentEvo optimization runs",
)
def list_optimization_runs(
    current_user: User = Depends(get_current_user),
) -> OptimizationRunListResponse:
    """List all completed optimization runs, newest first."""
    optimizer = get_agent_evo_optimizer()
    runs = optimizer.list_runs()
    return OptimizationRunListResponse(
        total_runs=len(runs),
        runs=runs,
    )


@router.get(
    "/runs/{run_id}",
    response_model=OptimizationReport,
    status_code=status.HTTP_200_OK,
    summary="Retrieve full Optimization Report for a specific run",
)
def get_optimization_run(
    run_id: str,
    current_user: User = Depends(get_current_user),
) -> OptimizationReport:
    """Retrieve full optimization report by run_id."""
    optimizer = get_agent_evo_optimizer()
    report = optimizer.get_run(run_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization run '{run_id}' not found.",
        )
    return report


@router.get(
    "/runs/{run_id}/pareto",
    response_model=List[EvaluatedCandidate],
    status_code=status.HTTP_200_OK,
    summary="Retrieve Pareto-optimal frontier candidates for a run",
)
def get_pareto_frontier(
    run_id: str,
    current_user: User = Depends(get_current_user),
) -> List[EvaluatedCandidate]:
    """Retrieve non-dominated Pareto frontier candidates for a run."""
    optimizer = get_agent_evo_optimizer()
    report = optimizer.get_run(run_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization run '{run_id}' not found.",
        )
    return report.pareto_frontier


@router.get(
    "/runs/{run_id}/candidates",
    response_model=List[EvaluatedCandidate],
    status_code=status.HTTP_200_OK,
    summary="Retrieve all evaluated candidates for a run",
)
def get_all_run_candidates(
    run_id: str,
    current_user: User = Depends(get_current_user),
) -> List[EvaluatedCandidate]:
    """Retrieve all evaluated candidate workflows for a run."""
    optimizer = get_agent_evo_optimizer()
    report = optimizer.get_run(run_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization run '{run_id}' not found.",
        )
    return report.all_candidates


@router.post(
    "/approve",
    response_model=ApproveWorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Record human developer approval for a candidate workflow",
)
def approve_candidate_workflow(
    request: ApproveWorkflowRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApproveWorkflowResponse:
    """
    Human developer approval governance step:
    - Validates run and candidate existence.
    - Rejects baseline approval attempts (baseline is reference, not candidate).
    - Rejects dominated candidates (only Pareto-optimal workflows are eligible).
    - Persists a new immutable WorkflowVersion in the database.
    - Idempotent: Repeated approvals return the existing workflow version without duplication.
    """
    optimizer = get_agent_evo_optimizer()
    report, wf_version = optimizer.approve_workflow(
        run_id=request.run_id,
        candidate_id=request.candidate_id,
        approved_by=current_user.email,
        db=db,
    )

    formatted_version = format_version_response(wf_version)
    return ApproveWorkflowResponse(
        status="approved",
        run_id=request.run_id,
        approved_workflow_id=request.candidate_id,
        message=f"Candidate workflow '{request.candidate_id}' approved as version '{wf_version.version_id}' by {current_user.email}.",
        report=report,
        version=formatted_version,
    )


@router.post(
    "/runs/{run_id}/candidates/{candidate_id}/approve",
    response_model=ApproveWorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve a candidate workflow via path parameters",
)
def approve_candidate_workflow_path(
    run_id: str,
    candidate_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ApproveWorkflowResponse:
    """RESTful path-based approval endpoint."""
    optimizer = get_agent_evo_optimizer()
    report, wf_version = optimizer.approve_workflow(
        run_id=run_id,
        candidate_id=candidate_id,
        approved_by=current_user.email,
        db=db,
    )

    formatted_version = format_version_response(wf_version)
    return ApproveWorkflowResponse(
        status="approved",
        run_id=run_id,
        approved_workflow_id=candidate_id,
        message=f"Candidate workflow '{candidate_id}' approved as version '{wf_version.version_id}' by {current_user.email}.",
        report=report,
        version=formatted_version,
    )


@router.get(
    "/versions",
    response_model=WorkflowVersionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all approved workflow versions",
)
def list_workflow_versions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowVersionListResponse:
    """Retrieve all approved workflow versions, newest first."""
    optimizer = get_agent_evo_optimizer()
    versions = optimizer.list_versions(db=db)
    formatted = [format_version_response(v) for v in versions]
    return WorkflowVersionListResponse(
        total_versions=len(formatted),
        versions=formatted,
    )


@router.get(
    "/versions/{version_id}",
    response_model=WorkflowVersionResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve details of a specific approved workflow version",
)
def get_workflow_version(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowVersionResponse:
    """Retrieve a specific workflow version by version_id or database UUID."""
    optimizer = get_agent_evo_optimizer()
    version = optimizer.get_version(version_id=version_id, db=db)
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow version '{version_id}' not found.",
        )
    return format_version_response(version)


# ---------------------------------------------------------------------------
# Phase 7.3+ Controlled Benchmarking, Lineage & Inspection Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/benchmark",
    response_model=BenchmarkReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute controlled multi-version benchmark comparison",
)
async def run_version_benchmark(
    request: BenchmarkRunRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BenchmarkReportResponse:
    """
    Execute controlled head-to-head benchmarking across selected workflow versions.
    Evaluates each version against the EXACT SAME queries, documents, and Phase 6 evaluation rules.
    Calculates objective Pareto dominance and persists benchmark run for reproducibility.
    """
    report = await BenchmarkService.run_benchmark(
        version_ids=request.version_ids,
        query=request.query,
        workload=request.workload,
        user_id=current_user.id,
        user_email=current_user.email,
        document_id=request.document_id,
        document_ids=request.document_ids,
        expected_answer=request.expected_answer,
        expected_answers=request.expected_answers,
        preset_name=request.preset_name,
        name=request.name or "Version Benchmark Comparison",
        description=request.description,
        iterations=request.iterations,
        repetitions=request.repetitions,
        db=db,
    )
    return report


@router.post(
    "/compare-evaluation",
    response_model=BaselineVsOptimizedComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Controlled workload evaluation comparing baseline and optimized workflows",
)
@workflows_router.post(
    "/compare-evaluation",
    response_model=BaselineVsOptimizedComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Controlled workload evaluation comparing baseline and optimized workflows (workflows alias)",
)
async def compare_baseline_vs_optimized_endpoint(
    request: BaselineVsOptimizedComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BaselineVsOptimizedComparisonResponse:
    """
    Controlled workload evaluation comparing baseline and an approved optimized workflow:
    - Evaluates identical workload for both workflows.
    - Records per-query response, groundedness, claims, tokens, cost, latency, provider, and model ID.
    - Computes aggregate metrics from raw unrounded metrics.
    - Reports cost reduction and groundedness difference honestly.
    - Strictly prevents execution of unapproved candidates.
    - Distinguishes live provider evaluations from mocked tests.
    """
    return await BenchmarkService.compare_baseline_vs_optimized(
        optimized_version_id=request.optimized_version_id,
        workload=request.workload,
        user_id=current_user.id,
        user_email=current_user.email,
        baseline_version_id=request.baseline_version_id,
        document_id=request.document_id,
        expected_answers=request.expected_answers,
        db=db,
    )


@router.get(
    "/benchmarks",
    response_model=BenchmarkRunListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all past reproducible benchmark runs",
)
def list_version_benchmarks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BenchmarkRunListResponse:
    """Retrieve all past multi-version benchmark runs, newest first."""
    return BenchmarkService.list_benchmarks(db=db)


@router.get(
    "/benchmarks/{benchmark_id}",
    response_model=BenchmarkReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a specific benchmark run report",
)
def get_version_benchmark(
    benchmark_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BenchmarkReportResponse:
    """Retrieve benchmark run details by benchmark_id."""
    report = BenchmarkService.get_benchmark(benchmark_id=benchmark_id, db=db)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{benchmark_id}' not found.",
        )
    return report


@router.get(
    "/lineage",
    response_model=VersionLineageResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve complete workflow version lineage graph",
)
def get_workflow_lineage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VersionLineageResponse:
    """
    Retrieve directed lineage graph showing actual parent-to-child relationships
    and configuration mutations across all approved versions from the database.
    """
    return BenchmarkService.get_lineage(db=db)


@router.get(
    "/versions/{version_id}/inspect",
    response_model=VersionInspectResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect complete metadata and configuration diffs for a version",
)
def inspect_workflow_version(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VersionInspectResponse:
    """
    Retrieve comprehensive metadata, parent linkage, before/after diffs,
    evaluation snapshot, and governance approval details for a specific workflow version.
    """
    return BenchmarkService.inspect_version(version_id=version_id, db=db)


# ---------------------------------------------------------------------------
# Phase 7.4 Evolutionary Crossover Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/crossover/preview",
    response_model=CrossoverPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview crossover child workflow configuration and component inheritance breakdown",
)
def preview_crossover(
    request: CrossoverPreviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CrossoverPreviewResponse:
    """
    Simulate component-level crossover recombination from Parent A and Parent B.
    Validates resulting child genome without persisting or executing LLM evaluation.
    """
    return CrossoverService.preview_crossover(request=request, db=db)


@router.post(
    "/crossover/candidates",
    response_model=CrossoverCandidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Construct, validate, evaluate, and persist evolutionary crossover candidate workflow",
)
async def create_crossover_candidate(
    request: CrossoverCandidateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CrossoverCandidateResponse:
    """
    Create a crossover candidate workflow by recombining genomes from Parent A and Parent B:
    1. Validates parent existence and distinctness.
    2. Inherits configuration components deterministically per crossover map.
    3. Applies optional post-crossover mutation.
    4. Validates resulting child genome against production safety constraints.
    5. Evaluates child workflow through Phase 6 claim-level grounding and cost evaluator.
    6. Computes 2D Pareto dominance against parents and baseline.
    7. Persists crossover candidate for review and optional human approval.
    """
    return await CrossoverService.create_crossover_candidate(
        request=request,
        user_id=current_user.id,
        user_email=current_user.email,
        db=db,
    )


@router.get(
    "/crossover/candidates",
    response_model=List[CrossoverCandidateResponse],
    status_code=status.HTTP_200_OK,
    summary="List all evaluated crossover candidates",
)
def list_crossover_candidates(
    current_user: User = Depends(get_current_user),
) -> List[CrossoverCandidateResponse]:
    """Retrieve all evaluated crossover candidate workflows."""
    return CrossoverService.list_crossover_candidates()


@router.get(
    "/crossover/candidates/{candidate_id}",
    response_model=CrossoverCandidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve details for a specific crossover candidate",
)
def get_crossover_candidate(
    candidate_id: str,
    current_user: User = Depends(get_current_user),
) -> CrossoverCandidateResponse:
    """Retrieve evaluated crossover candidate by candidate_id."""
    cand = CrossoverService.get_crossover_candidate(candidate_id=candidate_id)
    if not cand:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crossover candidate '{candidate_id}' not found.",
        )
    return cand


# Workflows router alias endpoints for /api/workflows/versions
@workflows_router.get(
    "/versions",
    response_model=WorkflowVersionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all approved workflow versions",
)
def list_workflows_versions_alias(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowVersionListResponse:
    """Alias for /api/agent-evo/versions."""
    return list_workflow_versions(current_user=current_user, db=db)


@workflows_router.get(
    "/versions/{version_id}",
    response_model=WorkflowVersionResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve details of a specific approved workflow version",
)
def get_workflows_version_alias(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowVersionResponse:
    """Alias for /api/agent-evo/versions/{version_id}."""
    return get_workflow_version(version_id=version_id, current_user=current_user, db=db)


@workflows_router.post(
    "/{version_id}/activate",
    response_model=ActivateWorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Promote an approved workflow version to ACTIVE production status",
)
def activate_workflow_version(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ActivateWorkflowResponse:
    """
    Explicit Production Promotion:
    - Marks target approved version as ACTIVE.
    - Demotes previous active version to INACTIVE.
    - Records an auditable activation event.
    - Requires Developer role.
    """
    if current_user.role != UserRole.DEVELOPER and str(current_user.role).upper() != "DEVELOPER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer privileges required to activate production workflow versions.",
        )

    target_wf, prev_id = WorkflowLifecycleService.activate_version(
        version_id=version_id,
        user_id=current_user.id,
        user_email=current_user.email,
        db=db,
    )
    return ActivateWorkflowResponse(
        status="success",
        version_id=target_wf.version_id,
        message=f"Workflow version '{target_wf.version_id}' successfully activated for production.",
        previous_active_version_id=prev_id,
        version=format_version_response(target_wf),
    )


@workflows_router.post(
    "/{version_id}/rollback",
    response_model=RollbackWorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Rollback production to a previous approved/inactive workflow version",
)
def rollback_workflow_version(
    version_id: str,
    request: Optional[RollbackWorkflowRequest] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RollbackWorkflowResponse:
    """
    Governed Rollback:
    - Reactivates target earlier approved version.
    - Demotes current active version to INACTIVE.
    - Records an auditable rollback event with reason.
    - Requires Developer role.
    """
    if current_user.role != UserRole.DEVELOPER and str(current_user.role).upper() != "DEVELOPER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer privileges required to execute workflow rollback.",
        )

    reason = request.reason if request else None
    target_wf, prev_id = WorkflowLifecycleService.rollback_version(
        target_version_id=version_id,
        user_id=current_user.id,
        user_email=current_user.email,
        reason=reason,
        db=db,
    )
    return RollbackWorkflowResponse(
        status="success",
        active_version_id=target_wf.version_id,
        previous_active_version_id=prev_id,
        message=f"Production successfully rolled back to version '{target_wf.version_id}'.",
        version=format_version_response(target_wf),
    )


@workflows_router.get(
    "/active",
    response_model=ActiveWorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Get details and configuration of the currently ACTIVE production workflow",
)
def get_active_workflow_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ActiveWorkflowResponse:
    """Retrieve currently active production workflow configuration."""
    return WorkflowLifecycleService.get_active_workflow(db=db)


@workflows_router.post(
    "/execute",
    response_model=WorkflowExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a user query using the currently ACTIVE workflow version configuration",
)
async def execute_active_workflow_endpoint(
    request: WorkflowExecutionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowExecutionResponse:
    """
    User Query Execution against Active Production Workflow:
    - Loads exact immutable active configuration.
    - Executes retrieval and LLM generation.
    - Captures separate LLM latency and Total execution time.
    - Records runtime execution metrics.
    """
    return await WorkflowLifecycleService.execute_active_workflow(
        query=request.query,
        user_id=current_user.id,
        document_id=request.document_id,
        expected_answer=request.expected_answer,
        workflow_version_id=request.workflow_version_id,
        db=db,
    )


@workflows_router.get(
    "/executions",
    response_model=WorkflowExecutionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List runtime executions against production workflows",
)
def list_workflow_executions_endpoint(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowExecutionListResponse:
    """Retrieve runtime execution history and latency/token metrics."""
    records = WorkflowLifecycleService.list_executions(limit=limit, db=db)
    return WorkflowExecutionListResponse(
        total_executions=len(records),
        executions=records,
    )


@workflows_router.post(
    "/executions/{execution_id}/feedback",
    response_model=WorkflowFeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit user feedback for a specific workflow execution",
)
def submit_execution_feedback(
    execution_id: str,
    request: WorkflowFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowFeedbackResponse:
    """
    Submit feedback against a production execution:
    - Associated with specific execution and workflow version.
    - Does NOT automatically mutate workflow or trigger optimization.
    """
    return WorkflowLifecycleService.record_feedback(
        execution_id=execution_id,
        rating=request.rating,
        feedback_text=request.feedback_text,
        user_id=current_user.id,
        db=db,
    )


@workflows_router.get(
    "/feedback",
    response_model=WorkflowFeedbackListResponse,
    status_code=status.HTTP_200_OK,
    summary="List feedback items for production workflows",
)
def list_workflow_feedback_endpoint(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowFeedbackListResponse:
    """List feedback across executions."""
    records = WorkflowLifecycleService.list_feedback(limit=limit, db=db)
    return WorkflowFeedbackListResponse(
        total_feedback=len(records),
        feedback=records,
    )


@workflows_router.get(
    "/audit",
    response_model=WorkflowAuditListResponse,
    status_code=status.HTTP_200_OK,
    summary="List workflow lifecycle governance audit events",
)
def list_workflow_audit_endpoint(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowAuditListResponse:
    """List lifecycle governance audit events (ACTIVATION, ROLLBACK, DEACTIVATION)."""
    records = WorkflowLifecycleService.list_audit_events(limit=limit, db=db)
    return WorkflowAuditListResponse(
        total_events=len(records),
        events=records,
        audit_events=records,
    )


@workflows_router.post(
    "/reoptimize",
    response_model=OptimizationReport,
    status_code=status.HTTP_200_OK,
    summary="Trigger explicit developer re-optimization from runtime metrics/feedback context",
)
async def reoptimize_workflow_endpoint(
    request: WorkflowReoptimizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OptimizationReport:
    """
    Explicit Re-Optimization Trigger:
    - Never triggered automatically by feedback.
    - Allows authorized developer to launch an optimization run using active or target version as parent.
    """
    if current_user.role != UserRole.DEVELOPER and str(current_user.role).upper() != "DEVELOPER":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer privileges required to trigger workflow re-optimization.",
        )

    # Determine baseline parent version ID
    parent_version_id = request.version_id
    if not parent_version_id:
        active_wf = WorkflowLifecycleService.get_active_workflow(db=db)
        parent_version_id = active_wf.active_version_id

    query = request.query or "What is the primary clinical diagnosis and assessment for this patient?"
    optimizer = get_agent_evo_optimizer()
    report = await optimizer.run_optimization(
        query=query,
        user_id=current_user.id,
        document_id=request.document_id,
        strategy=request.strategy or "mutation",
        parent_workflow_id=parent_version_id,
        db=db,
    )

    # Record reoptimization audit event
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    audit = WorkflowAuditEvent(
        event_id=f"audit_{uuid.uuid4().hex[:8]}",
        workflow_version_id=parent_version_id,
        event_type="REOPTIMIZATION",
        performed_by=current_user.id,
        user_email=current_user.email,
        details={
            "run_id": report.run_id,
            "query": query,
            "strategy": request.strategy,
            "feedback_context_enabled": request.use_feedback_context,
            "timestamp": now_utc.isoformat(),
        },
        created_at=now_utc,
    )
    db.add(audit)
    db.commit()

    return report


# ---------------------------------------------------------------------------
# Router Aliases on /api/agent-evo for full API flexibility
# ---------------------------------------------------------------------------

@router.post("/versions/{version_id}/activate", response_model=ActivateWorkflowResponse)
def activate_workflow_version_alias(
    version_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ActivateWorkflowResponse:
    return activate_workflow_version(version_id=version_id, current_user=current_user, db=db)


@router.post("/versions/{version_id}/rollback", response_model=RollbackWorkflowResponse)
def rollback_workflow_version_alias(
    version_id: str,
    request: Optional[RollbackWorkflowRequest] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RollbackWorkflowResponse:
    return rollback_workflow_version(version_id=version_id, request=request, current_user=current_user, db=db)


@router.get("/active", response_model=ActiveWorkflowResponse)
def get_active_workflow_alias(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ActiveWorkflowResponse:
    return get_active_workflow_endpoint(current_user=current_user, db=db)


@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_active_workflow_alias(
    request: WorkflowExecutionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowExecutionResponse:
    return await execute_active_workflow_endpoint(request=request, current_user=current_user, db=db)


@router.get("/executions", response_model=WorkflowExecutionListResponse)
def list_workflow_executions_alias(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowExecutionListResponse:
    return list_workflow_executions_endpoint(limit=limit, current_user=current_user, db=db)


@router.post("/executions/{execution_id}/feedback", response_model=WorkflowFeedbackResponse)
def submit_execution_feedback_alias(
    execution_id: str,
    request: WorkflowFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowFeedbackResponse:
    return submit_execution_feedback(execution_id=execution_id, request=request, current_user=current_user, db=db)


@router.get("/feedback", response_model=WorkflowFeedbackListResponse)
def list_workflow_feedback_alias(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowFeedbackListResponse:
    return list_workflow_feedback_endpoint(limit=limit, current_user=current_user, db=db)


@router.get("/audit", response_model=WorkflowAuditListResponse)
def list_workflow_audit_alias(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkflowAuditListResponse:
    return list_workflow_audit_endpoint(limit=limit, current_user=current_user, db=db)


@router.post("/reoptimize", response_model=OptimizationReport)
async def reoptimize_workflow_alias(
    request: WorkflowReoptimizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OptimizationReport:
    return await reoptimize_workflow_endpoint(request=request, current_user=current_user, db=db)

