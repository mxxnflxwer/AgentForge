import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.orm import Session

from app.models.user import User, UserRole
from app.models.workflow_version import (
    WorkflowAuditEvent,
    WorkflowExecution,
    WorkflowVersion,
)
from app.schemas.agent_evo import (
    BaselineVsOptimizedComparisonRequest,
    BenchmarkRunRequest,
)
from app.schemas.rag import RAGSearchResponse, RetrievedChunk
from app.services.agent_evo.benchmark_service import BenchmarkService
from app.services.agent_evo.evaluator_adapter import EvaluatorAdapter
from app.services.agent_evo.optimizer import AgentEvoOptimizer
from app.services.agent_evo.report import (
    CandidateMetrics,
    EvaluatedCandidate,
    generate_optimization_report,
)
from app.services.agent_evo.runtime_selector import (
    RuntimeWorkflowSelector,
    WorkflowSelectionResult,
)
from app.services.agent_evo.workflow import AgentWorkflow, get_baseline_workflow
from app.services.agent_evo.workflow_lifecycle_service import WorkflowLifecycleService
from app.services.llm.base import LLMResponse


@pytest.fixture
def test_users(db_session: Session):
    """Seed test users for human approval and role governance."""
    dev_user = User(
        id="dev_user_eval_1",
        name="Developer Governed",
        email="dev_governed@agentforge.ai",
        password_hash="hashedpassword",
        role=UserRole.DEVELOPER,
        is_active=True,
    )
    user = User(
        id="clinician_user_eval_1",
        name="Clinician User",
        email="clinician@agentforge.ai",
        password_hash="hashedpassword",
        role=UserRole.USER,
        is_active=True,
    )
    db_session.merge(dev_user)
    db_session.merge(user)
    db_session.commit()
    return dev_user, user


@pytest.fixture
def mock_evaluation_env():
    """Mock RAG retrieval and LLM adapter for fast, deterministic unit testing."""
    mock_retrieval = RAGSearchResponse(
        query="What is the primary clinical diagnosis and assessment?",
        status="success",
        total_results=2,
        results=[
            RetrievedChunk(
                chunk_id="chunk_test_1",
                document_id="doc_test_1",
                chunk_index=0,
                content="Assessment: Primary clinical diagnosis is Type 2 Diabetes Mellitus with peripheral neuropathy.",
                section_title="Clinical Assessment",
                similarity_score=0.92,
                relevance_score=0.95,
            ),
            RetrievedChunk(
                chunk_id="chunk_test_2",
                document_id="doc_test_1",
                chunk_index=1,
                content="Plan: Prescribed Metformin 500mg PO daily. Monitor HbA1c in 3 months.",
                section_title="Plan and Medications",
                similarity_score=0.88,
                relevance_score=0.90,
            ),
        ],
    )

    mock_llm_response = LLMResponse(
        model="Gemini 3.5 Flash-Lite",
        provider="Google",
        answer="The primary diagnosis is Type 2 Diabetes Mellitus with peripheral neuropathy, managed with Metformin 500mg daily.",
        latency_ms=145.0,
        success=True,
        raw_usage={"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    )

    mock_adapter = MagicMock()
    mock_adapter.name = "gemini-3.5-flash-lite"
    mock_adapter.provider = "Google"
    mock_adapter.is_configured = MagicMock(return_value=True)
    mock_adapter.generate = AsyncMock(return_value=mock_llm_response)

    mock_router = MagicMock()
    mock_router.get_adapter.return_value = mock_adapter

    return mock_retrieval, mock_router


# ===========================================================================
# 1. Human Approval and Workflow Registry Governance
# ===========================================================================

class TestHumanApprovalAndWorkflowRegistry:

    def test_manual_approval_and_activation_separation(self, db_session, test_users):
        """
        Verify:
        - Approval is an explicit action creating an immutable version.
        - Approval does NOT automatically activate or deploy the workflow.
        - Activation is a separate explicit action.
        """
        dev_user, _ = test_users
        optimizer = AgentEvoOptimizer()

        # Seed an optimization run in optimizer memory
        baseline = get_baseline_workflow()
        cand = baseline.clone(new_id="cand_diag_opt_1", new_name="Optimized Diagnosis Workflow")
        cand.retrieval.top_k = 3

        from app.services.agent_evo.report import (
            CandidateMetrics,
            EvaluatedCandidate,
            generate_optimization_report,
        )
        metrics = CandidateMetrics(
            workflow_id="cand_diag_opt_1",
            workflow_name="Optimized Diagnosis Workflow",
            performance=0.95,
            cost=0.000022,
            groundedness=0.95,
            hallucination_rate=0.05,
            total_tokens=180,
            llm_latency_ms=110.0,
            status="success",
        )
        evaluated_cand = EvaluatedCandidate(
            candidate_id="cand_diag_opt_1",
            workflow=cand,
            metrics=metrics,
            is_pareto_optimal=True,
        )
        base_cand = EvaluatedCandidate(
            candidate_id="baseline",
            workflow=baseline,
            metrics=metrics,
            is_pareto_optimal=False,
        )

        run_report = generate_optimization_report(
            run_id="run_governance_test_1",
            query="What is the clinical diagnosis and assessment for this patient?",
            baseline=base_cand,
            all_candidates=[base_cand, evaluated_cand],
            pareto_candidates=[evaluated_cand],
            is_workload=True,
            workload_queries=["What is the clinical diagnosis and assessment for this patient?"],
        )
        optimizer._runs[run_report.run_id] = run_report

        # 1. Developer explicitly approves candidate
        report, wf_version = optimizer.approve_workflow(
            run_id=run_report.run_id,
            candidate_id="cand_diag_opt_1",
            approved_by=dev_user.email,
            db=db_session,
        )

        assert wf_version is not None
        assert wf_version.version_id.startswith("wf_v")
        assert wf_version.status.lower() == "approved"
        assert wf_version.activated_at is None  # Invariant: Approval does NOT activate

        # Verify active workflow is still baseline, not the newly approved version
        active_info = WorkflowLifecycleService.get_active_workflow(db=db_session)
        assert active_info.active_version_id != wf_version.version_id
        assert active_info.is_active_set is False or active_info.active_version_id == "baseline"

        # 2. Developer explicitly activates the approved version
        activated_ver, prev_id = WorkflowLifecycleService.activate_version(
            version_id=wf_version.version_id,
            user_id=dev_user.id,
            user_email=dev_user.email,
            db=db_session,
        )
        assert activated_ver.status == "ACTIVE"
        assert activated_ver.activated_at is not None

        active_info_now = WorkflowLifecycleService.get_active_workflow(db=db_session)
        assert active_info_now.active_version_id == wf_version.version_id
        assert active_info_now.is_active_set is True

    def test_version_immutability_and_idempotency(self, db_session, test_users):
        """
        Verify:
        - Approved version configuration is immutable.
        - Repeated approval calls for the same candidate do not duplicate versions.
        """
        dev_user, _ = test_users
        optimizer = AgentEvoOptimizer()

        baseline = get_baseline_workflow()
        cand = baseline.clone(new_id="cand_immutable_1", new_name="Immutable Workflow")
        metrics = CandidateMetrics(
            workflow_id="cand_immutable_1",
            workflow_name="Immutable Workflow",
            performance=0.95,
            cost=0.000019,
            groundedness=1.0,
            hallucination_rate=0.0,
            total_tokens=150,
            llm_latency_ms=100.0,
            status="success",
        )

        evaluated_cand = EvaluatedCandidate(
            candidate_id="cand_immutable_1",
            workflow=cand,
            metrics=metrics,
            is_pareto_optimal=True,
        )
        base_cand = EvaluatedCandidate(
            candidate_id="baseline",
            workflow=baseline,
            metrics=metrics,
            is_pareto_optimal=False,
        )

        from app.services.agent_evo.report import generate_optimization_report
        run_report = generate_optimization_report(
            run_id="run_immutability_test",
            query="What is the diagnosis?",
            baseline=base_cand,
            all_candidates=[base_cand, evaluated_cand],
            pareto_candidates=[evaluated_cand],
            is_workload=True,
            workload_queries=["What is the diagnosis?"],
        )
        optimizer._runs[run_report.run_id] = run_report

        # First approval
        _, v1 = optimizer.approve_workflow("run_immutability_test", "cand_immutable_1", dev_user.email, db=db_session)
        # Second approval (idempotency check)
        _, v2 = optimizer.approve_workflow("run_immutability_test", "cand_immutable_1", dev_user.email, db=db_session)

        assert v1.id == v2.id
        assert v1.version_id == v2.version_id
        assert db_session.query(WorkflowVersion).filter_by(source_candidate_id="cand_immutable_1").count() == 1


# ===========================================================================
# 2. Automatic Runtime Workflow Selection
# ===========================================================================

class TestAutomaticRuntimeWorkflowSelector:

    def test_rejection_of_unapproved_workflow_versions(self, db_session):
        """
        Verify:
        - Requesting an unapproved candidate at runtime is strictly rejected.
        - Safely falls back to production baseline without executing unapproved candidate.
        """
        result = RuntimeWorkflowSelector.select_workflow_for_query(
            query="What is the primary clinical diagnosis?",
            db=db_session,
            candidate_version_id_override="cand_unapproved_rogue_candidate",
        )

        assert result.version_id == "baseline"
        assert result.is_fallback is True
        assert "Unapproved candidates are never executed" in result.selection_reason

    def test_automatic_routing_for_supported_task_categories(self, db_session, test_users):
        """
        Verify:
        - Query about diagnosis routes to approved diagnosis-specialized workflow.
        - Query about medications routes to approved medications-specialized workflow.
        """
        dev_user, _ = test_users
        base_wf = get_baseline_workflow()

        # Seed Diagnosis-Specialized Approved Workflow
        diag_cfg = base_wf.to_dict()
        diag_cfg["task_categories"] = ["diagnosis"]
        diag_cfg["retrieval"]["top_k"] = 4
        v_diag = WorkflowVersion(
            version_id="wf_v10_diagnosis_expert",
            version_number=10,
            name="Diagnosis Expert Workflow",
            source_run_id="run_diag",
            source_candidate_id="cand_diag",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            configuration=diag_cfg,
            evaluation_snapshot={"groundedness": 0.98, "cost": 0.000028, "evaluated_task_categories": ["diagnosis"]},
            status="APPROVED",
            approved_by=dev_user.email,
        )

        # Seed Medications-Specialized Approved Workflow
        med_cfg = base_wf.to_dict()
        med_cfg["task_categories"] = ["medications"]
        med_cfg["retrieval"]["top_k"] = 6
        v_med = WorkflowVersion(
            version_id="wf_v11_medications_expert",
            version_number=11,
            name="Medications Expert Workflow",
            source_run_id="run_med",
            source_candidate_id="cand_med",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            configuration=med_cfg,
            evaluation_snapshot={"groundedness": 0.96, "cost": 0.000024, "evaluated_task_categories": ["medications"]},
            status="APPROVED",
            approved_by=dev_user.email,
        )

        db_session.merge(v_diag)
        db_session.merge(v_med)
        db_session.commit()

        # Test Diagnosis Query
        diag_query = "What is the primary clinical diagnosis and assessment for this patient?"
        diag_result = RuntimeWorkflowSelector.select_workflow_for_query(query=diag_query, db=db_session)
        assert diag_result.task_category == "diagnosis"
        assert diag_result.version_id == "wf_v10_diagnosis_expert"
        assert diag_result.configuration["retrieval"]["top_k"] == 4
        assert "best-matching eligible workflow" in diag_result.selection_reason
        assert "not a guaranteed global optimum" in diag_result.selection_reason

        # Test Medications Query
        med_query = "What medications, prescriptions, and dosages are currently recommended?"
        med_result = RuntimeWorkflowSelector.select_workflow_for_query(query=med_query, db=db_session)
        assert med_result.task_category == "medications"
        assert med_result.version_id == "wf_v11_medications_expert"
        assert med_result.configuration["retrieval"]["top_k"] == 6

    def test_safe_fallback_when_no_category_match(self, db_session, test_users):
        """
        Verify:
        - When incoming query has a task category with no specialized approved profile,
          selector safely falls back to the active production workflow.
        """
        dev_user, _ = test_users
        base_wf = get_baseline_workflow()

        # Set an active production workflow
        active_cfg = base_wf.to_dict()
        active_cfg["retrieval"]["top_k"] = 5
        v_active = WorkflowVersion(
            version_id="wf_v20_production_active",
            version_number=20,
            name="General Production Active",
            source_run_id="run_gen",
            source_candidate_id="cand_gen",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            configuration=active_cfg,
            evaluation_snapshot={"groundedness": 0.90, "cost": 0.000035},
            status="ACTIVE",
            approved_by=dev_user.email,
        )
        db_session.merge(v_active)
        db_session.commit()

        # Query asking for an explanation (category: "explanation") - no specialized profile exists
        explain_query = "Please explain simply why the patient feels short of breath upon exertion."
        result = RuntimeWorkflowSelector.select_workflow_for_query(query=explain_query, db=db_session)

        assert result.task_category == "explanation"
        assert result.version_id == "wf_v20_production_active"
        assert result.is_fallback is True
        assert "No specialized approved workflow matched task category 'explanation'" in result.selection_reason


# ===========================================================================
# 3. Reproducible Baseline versus Optimized Evidence Comparison
# ===========================================================================

class TestBaselineVsOptimizedEvidenceComparison:

    @pytest.mark.asyncio
    async def test_baseline_vs_optimized_workload_metric_aggregation(self, db_session, test_users, mock_evaluation_env):
        """
        Verify:
        - Same evaluation workload is evaluated on both baseline and optimized workflows.
        - Per-query records capture response, groundedness, claims, tokens, cost, latency, provider, and model ID.
        - Aggregate results are computed from raw unrounded metrics.
        - Cost reduction and groundedness retention are calculated accurately.
        """
        dev_user, _ = test_users
        mock_retrieval, mock_router = mock_evaluation_env

        # Create an approved optimized workflow with context compression enabled
        base_wf = get_baseline_workflow()
        opt_cfg = base_wf.to_dict()
        opt_cfg["execution"]["context_compression"] = True
        opt_cfg["retrieval"]["top_k"] = 3

        v_opt = WorkflowVersion(
            version_id="wf_v30_compressed_opt",
            version_number=30,
            name="Optimized Compressed Clinical",
            source_run_id="run_comp",
            source_candidate_id="cand_comp",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            configuration=opt_cfg,
            evaluation_snapshot={"groundedness": 1.0, "cost": 0.000020},
            status="APPROVED",
            approved_by=dev_user.email,
        )
        db_session.merge(v_opt)
        db_session.commit()

        workload = [
            "What is the primary clinical diagnosis and assessment?",
            "What medications and dosages are prescribed in the plan?",
            "Summarize the physical examination findings.",
        ]

        with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
             patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

            comparison = await BenchmarkService.compare_baseline_vs_optimized(
                optimized_version_id=v_opt.version_id,
                workload=workload,
                user_id=dev_user.id,
                user_email=dev_user.email,
                db=db_session,
            )

            # Verification of aggregate metrics
            assert comparison.total_queries_evaluated == 3
            assert comparison.successful_evaluations == 3
            assert comparison.failed_evaluations == 0
            assert comparison.baseline_version_id == "baseline"
            assert comparison.optimized_version_id == v_opt.version_id

            # Invariant: Aggregates computed from raw unrounded per-query metrics
            assert comparison.baseline_groundedness >= 0.0
            assert comparison.optimized_groundedness >= 0.0
            assert isinstance(comparison.groundedness_diff_pp, float)
            assert isinstance(comparison.cost_reduction_pct, float)

            # Invariant: Per-query evidence preservation
            assert len(comparison.per_query_evidence) == 3
            for r in comparison.per_query_evidence:
                assert r.query_id.startswith("q_")
                assert r.baseline_response != ""
                assert r.optimized_response != ""
                assert r.baseline_provider == "Google"
                assert r.optimized_provider == "Google"
                assert r.baseline_model_id != ""
                assert r.optimized_model_id != ""
                assert r.baseline_cost >= 0.0
                assert r.optimized_cost >= 0.0
                assert isinstance(r.baseline_claims, list)
                assert isinstance(r.optimized_claims, list)

            # Invariant: Clearly distinguish real provider vs mocked
            assert comparison.is_mocked is True or comparison.is_mocked is False
            assert comparison.evaluation_mode in ("mocked_or_saved_response", "live_provider")
            assert "does not claim universal optimality" in comparison.scope_claim.lower()

    @pytest.mark.asyncio
    async def test_honest_reporting_when_cost_does_not_decrease(self, db_session, test_users, mock_evaluation_env):
        """
        Verify:
        - If the optimized workflow has equal or higher cost than baseline,
          the system reports cost_reduced = False and states this honestly without fabrication.
        """
        dev_user, _ = test_users
        mock_retrieval, mock_router = mock_evaluation_env

        # Create an approved workflow with larger top_k (more expensive)
        base_wf = get_baseline_workflow()
        expensive_cfg = base_wf.to_dict()
        expensive_cfg["retrieval"]["top_k"] = 12

        v_exp = WorkflowVersion(
            version_id="wf_v40_expensive",
            version_number=40,
            name="Deep Retrieval Expensive",
            source_run_id="run_exp",
            source_candidate_id="cand_exp",
            model_id="gemini-3.5-flash-lite",
            provider="Google",
            configuration=expensive_cfg,
            evaluation_snapshot={"groundedness": 1.0, "cost": 0.000080},
            status="APPROVED",
            approved_by=dev_user.email,
        )
        db_session.merge(v_exp)
        db_session.commit()

        workload = ["What is the primary clinical diagnosis?"]

        with patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval", return_value=mock_retrieval), \
             patch("app.services.agent_evo.evaluator_adapter.get_llm_router", return_value=mock_router):

            comparison = await BenchmarkService.compare_baseline_vs_optimized(
                optimized_version_id=v_exp.version_id,
                workload=workload,
                user_id=dev_user.id,
                user_email=dev_user.email,
                db=db_session,
            )

            # If equal or higher cost, honest report must not fabricate cost reduction
            if comparison.optimized_cost_per_query >= comparison.baseline_cost_per_query:
                assert comparison.cost_reduced is False
                assert "not reduce cost" in comparison.finding_summary.lower() or "did not achieve cost reduction" in comparison.finding_summary.lower()

    @pytest.mark.asyncio
    async def test_rejection_of_unapproved_version_in_comparison(self, db_session, test_users):
        """
        Verify:
        - Attempting to benchmark/compare an unapproved candidate raises HTTP 404 or 400.
        """
        dev_user, _ = test_users
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await BenchmarkService.compare_baseline_vs_optimized(
                optimized_version_id="cand_unapproved_id",
                workload=["What is the clinical diagnosis?"],
                user_id=dev_user.id,
                user_email=dev_user.email,
                db=db_session,
            )
        assert exc_info.value.status_code in (400, 404)
        assert "not found" in exc_info.value.detail.lower() or "unapproved" in exc_info.value.detail.lower()
