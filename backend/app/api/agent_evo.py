import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.models.workflow_version import WorkflowVersion
from app.schemas.agent_evo import (
    ApproveWorkflowRequest,
    ApproveWorkflowResponse,
    BenchmarkReportResponse,
    BenchmarkRunListResponse,
    BenchmarkRunRequest,
    OptimizationRunListResponse,
    OptimizationRunRequest,
    VersionInspectResponse,
    VersionLineageResponse,
    WorkflowVersionListResponse,
    WorkflowVersionResponse,
)
from app.services.agent_evo import (
    EvaluatedCandidate,
    OptimizationReport,
    get_agent_evo_optimizer,
)
from app.services.agent_evo.benchmark_service import BenchmarkService

logger = logging.getLogger("agentforge.api.agent_evo")

router = APIRouter(prefix="/api/agent-evo", tags=["AgentEvo Optimization Engine"])
workflows_router = APIRouter(prefix="/api/workflows", tags=["Workflow Versions"])


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
        status=version.status,
        approved_by=version.approved_by,
        approved_at=approved_at_str,
        created_at=created_at_str,
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
    1. Evaluates baseline workflow against target query and context.
    2. Generates candidate workflow variations along controlled dimensions.
    3. Validates each candidate configuration.
    4. Evaluates all candidates through Phase 6 evaluator.
    5. Computes Performance vs Cost Pareto dominance and maintains Pareto archive.
    6. Returns structured Optimization Report ready for human review.
    """
    optimizer = get_agent_evo_optimizer()
    report = await optimizer.run_optimization(
        query=request.query,
        user_id=current_user.id,
        document_id=request.document_id,
        expected_answer=request.expected_answer,
        candidate_count=request.candidate_count,
        seed=request.seed,
        parent_workflow_id=request.parent_workflow_id,
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
        user_id=current_user.id,
        user_email=current_user.email,
        document_id=request.document_id,
        document_ids=request.document_ids,
        expected_answer=request.expected_answer,
        preset_name=request.preset_name,
        name=request.name or "Version Benchmark Comparison",
        description=request.description,
        iterations=request.iterations,
        repetitions=request.repetitions,
        db=db,
    )
    return report


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
