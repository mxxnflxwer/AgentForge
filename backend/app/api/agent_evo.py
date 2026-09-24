import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.agent_evo import (
    ApproveWorkflowRequest,
    ApproveWorkflowResponse,
    OptimizationRunListResponse,
    OptimizationRunRequest,
)
from app.services.agent_evo import (
    EvaluatedCandidate,
    OptimizationReport,
    get_agent_evo_optimizer,
)

logger = logging.getLogger("agentforge.api.agent_evo")

router = APIRouter(prefix="/api/agent-evo", tags=["AgentEvo Optimization Engine"])


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
    2. Generates 3-5 candidate workflow variations along controlled dimensions.
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
) -> ApproveWorkflowResponse:
    """
    Human developer approval governance step:
    Records developer approval for a Pareto-optimal workflow.
    """
    optimizer = get_agent_evo_optimizer()
    report = optimizer.approve_workflow(
        run_id=request.run_id,
        candidate_id=request.candidate_id,
    )
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{request.run_id}' or candidate '{request.candidate_id}' not found.",
        )

    return ApproveWorkflowResponse(
        status="approved",
        run_id=request.run_id,
        approved_workflow_id=request.candidate_id,
        message=f"Candidate workflow '{request.candidate_id}' approved by developer {current_user.email}.",
        report=report,
    )
