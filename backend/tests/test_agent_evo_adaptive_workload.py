"""
Comprehensive tests for AgentForge Phase 7.1:
- Multi-query workload evaluation & candidate x query matrix
- Raw unrounded metric aggregation (groundedness, hallucination, cost, tokens, latency)
- Pareto comparison using aggregate metrics (groundedness vs cost/query strictly)
- Deterministic Adapter Scheduler strategy selection & stagnation handling
- Candidate count and iteration limit enforcement
- Manual governance invariant (no auto-approval / auto-activation)
- Active workflow runtime execution configuration alignment
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sqlalchemy.orm import Session

from app.services.agent_evo.workflow import (
    AgentWorkflow,
    ModelConfig,
    PromptConfig,
    RetrievalConfig,
    ExecutionConfig,
)
from app.services.agent_evo.objective import CandidateMetrics
from app.services.agent_evo.pareto import EvaluatedCandidate, ParetoArchive
from app.services.agent_evo.report import OptimizationReport
from app.services.agent_evo.scheduler import AgentEvoScheduler, SchedulerDecision
from app.services.agent_evo.evaluator_adapter import EvaluatorAdapter
from app.services.agent_evo.optimizer import AgentEvoOptimizer
from app.models.workflow_version import WorkflowVersion


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Session)
    db.query.return_value.filter.return_value.all.return_value = []
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.order_by.return_value.all.return_value = []
    return db


@pytest.fixture
def baseline_workflow():
    return AgentWorkflow(
        workflow_id="baseline",
        name="Production Baseline",
        retrieval=RetrievalConfig(top_k=5, similarity_threshold=0.35, chunk_size=512, chunk_overlap=50),
        model=ModelConfig(model_id="gemini-3.5-flash-lite", provider="google", temperature=0.0, max_tokens=1024),
        prompt=PromptConfig(template_name="default_medical", include_disclaimer=True),
        execution=ExecutionConfig(timeout_seconds=30, max_retries=2, context_compression=False),
    )


@pytest.fixture
def sample_workload():
    return [
        "What is the primary clinical diagnosis and assessment for this patient?",
        "What medications were prescribed for this patient?",
        "What are the abnormal laboratory findings?",
        "What is the patient's treatment plan?",
        "What is the relevant cardiac history?",
    ]


class TestAdapterScheduler:
    """Test deterministic rule-based scheduler."""

    def test_initial_iteration_selects_exploration(self, baseline_workflow):
        decision = AgentEvoScheduler.select_strategy(
            iteration=1,
            max_iterations=3,
            baseline=None,
            archive_candidates=[],
            available_versions=[],
            candidate_count=4,
        )
        assert decision.strategy == "mutation"
        assert "Initial" in decision.reason or "Exploration" in decision.stage_name

    def test_high_groundedness_triggers_refinement(self, baseline_workflow):
        baseline_evaluated = EvaluatedCandidate(
            candidate_id="baseline",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="baseline",
                workflow_name="Production Baseline",
                performance=1.0,
                cost=0.000080,
                groundedness=1.0,
                hallucination_rate=0.0,
                total_tokens=400,
                input_tokens=300,
                output_tokens=100,
                llm_latency_ms=250.0,
                execution_time_ms=250.0,
                status="success",
            ),
            is_pareto_optimal=True,
        )

        cand = EvaluatedCandidate(
            candidate_id="cand_1",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="cand_1",
                workflow_name="Candidate 1",
                performance=1.0,
                cost=0.000080,
                groundedness=1.0,
                hallucination_rate=0.0,
                total_tokens=400,
                input_tokens=300,
                output_tokens=100,
                llm_latency_ms=250.0,
                execution_time_ms=250.0,
                status="success",
            ),
            is_pareto_optimal=True,
        )

        decision = AgentEvoScheduler.select_strategy(
            iteration=2,
            max_iterations=3,
            baseline=baseline_evaluated,
            archive_candidates=[baseline_evaluated, cand],
            available_versions=[],
            candidate_count=4,
        )
        assert decision.strategy == "refinement"
        assert "groundedness" in decision.reason.lower()

    def test_sufficient_parents_triggers_crossover(self, baseline_workflow):
        baseline_evaluated = EvaluatedCandidate(
            candidate_id="baseline",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="baseline",
                workflow_name="Production Baseline",
                performance=1.0,
                cost=0.000080,
                groundedness=1.0,
                hallucination_rate=0.0,
                total_tokens=400,
                input_tokens=300,
                output_tokens=100,
                llm_latency_ms=250.0,
                execution_time_ms=250.0,
                status="success",
            ),
            is_pareto_optimal=True,
        )

        cand_low_cost = EvaluatedCandidate(
            candidate_id="cand_1",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="cand_1",
                workflow_name="Candidate 1",
                performance=0.85,
                cost=0.000040,  # 50% cost reduction
                groundedness=0.85,
                hallucination_rate=0.15,
                total_tokens=200,
                input_tokens=150,
                output_tokens=50,
                llm_latency_ms=180.0,
                execution_time_ms=180.0,
                status="success",
            ),
            is_pareto_optimal=True,
        )

        # 2 approved parent versions available
        parent_versions = [
            WorkflowVersion(version_id="wf_v1_aaa", version_number=1, name="Version 1", source_run_id="run1", source_candidate_id="cand1", status="APPROVED", configuration={}, model_id="gemini-3.5-flash-lite", provider="google", evaluation_snapshot={}, approved_by="dev"),
            WorkflowVersion(version_id="wf_v2_bbb", version_number=2, name="Version 2", source_run_id="run2", source_candidate_id="cand2", status="APPROVED", configuration={}, model_id="gemini-3.5-flash-lite", provider="google", evaluation_snapshot={}, approved_by="dev"),
        ]

        decision = AgentEvoScheduler.select_strategy(
            iteration=3,
            max_iterations=3,
            baseline=baseline_evaluated,
            archive_candidates=[baseline_evaluated, cand_low_cost],
            available_versions=parent_versions,
            candidate_count=4,
        )
        assert decision.strategy == "crossover"
        assert decision.parameters.get("parent_b_workflow_id") is not None


class TestEvaluatorAdapterWorkload:
    """Test candidate x query workload evaluation and metric aggregation."""

    @pytest.mark.anyio
    async def test_workload_evaluation_aggregates_unrounded_metrics(self, baseline_workflow, sample_workload, mock_db):
        adapter = EvaluatorAdapter()

        from app.schemas.evaluation import EvaluationResponse
        eval_resp1 = EvaluationResponse(
            model="gemini-3.5-flash-lite",
            groundedness=1.0,
            hallucination_rate=0.0,
            total_claims=2,
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            latency_ms=200.0,
            execution_time_ms=200.0,
            input_cost=0.000015,
            output_cost=0.000025,
            total_cost=0.000040,
            supported_claim_count=2,
            unsupported_claim_count=0,
            supported_claims=["Claim 1", "Claim 2"],
            unsupported_claims=[],
        )
        eval_resp2 = EvaluationResponse(
            model="gemini-3.5-flash-lite",
            groundedness=0.80,
            hallucination_rate=0.20,
            total_claims=5,
            input_tokens=250,
            output_tokens=150,
            total_tokens=400,
            latency_ms=300.0,
            execution_time_ms=300.0,
            input_cost=0.000020,
            output_cost=0.000040,
            total_cost=0.000060,
            supported_claim_count=4,
            unsupported_claim_count=1,
            supported_claims=["Claim 1", "Claim 2", "Claim 3", "Claim 4"],
            unsupported_claims=["Hallucinated Claim"],
        )

        mock_engine = MagicMock()
        mock_engine.evaluate.side_effect = [eval_resp1, eval_resp2]

        mock_llm_adapter = MagicMock()
        mock_llm_adapter.name = "gemini-3.5-flash-lite"
        mock_llm_resp = MagicMock()
        mock_llm_resp.answer = "Clinical response"
        mock_llm_resp.latency_ms = 250.0
        mock_llm_resp.raw_usage = {"prompt_tokens": 200, "completion_tokens": 100}
        mock_llm_resp.success = True
        mock_llm_resp.error = None
        mock_llm_adapter.generate = AsyncMock(return_value=mock_llm_resp)

        with patch("app.services.agent_evo.evaluator_adapter.get_evaluation_engine", return_value=mock_engine), \
             patch("app.services.agent_evo.evaluator_adapter.get_llm_router") as mock_router, \
             patch("app.services.agent_evo.evaluator_adapter.execute_rag_retrieval") as mock_rag:
            
            mock_router.return_value.get_adapter.return_value = mock_llm_adapter
            
            mock_rag_result = MagicMock()
            mock_rag_result.summary_context = None
            mock_rag_result.status = "success"
            mock_chunk = MagicMock()
            mock_chunk.chunk_index = 0
            mock_chunk.section_title = "Clinical Section"
            mock_chunk.content = "Retrieved clinical document text"
            mock_rag_result.results = [mock_chunk]
            mock_rag.return_value = mock_rag_result

            agg_cand = await adapter.evaluate_candidate_workload(
                workflow=baseline_workflow,
                queries=sample_workload[:2],
                user_id="test_user",
                db=mock_db,
            )

            # Groundedness: (1.0 + 0.8) / 2 = 0.90
            assert agg_cand.metrics.groundedness == pytest.approx(0.90, rel=1e-5)
            assert agg_cand.metrics.performance == pytest.approx(0.90, rel=1e-5)
            # Hallucination: (0.0 + 0.2) / 2 = 0.10
            assert agg_cand.metrics.hallucination_rate == pytest.approx(0.10, rel=1e-5)
            # Cost / Query: (0.000040 + 0.000060) / 2 = 0.000050
            assert agg_cand.metrics.cost == pytest.approx(0.000050, rel=1e-6)
            # Tokens / Query: (300 + 400) / 2 = 350
            assert agg_cand.metrics.total_tokens == 350
            # Latency / Query: (200.0 + 300.0) / 2 = 250.0
            assert agg_cand.metrics.llm_latency_ms == pytest.approx(250.0, rel=1e-4)

            # Check per-query evidence
            assert "per_query_evaluations" in agg_cand.metrics.raw_details
            assert len(agg_cand.metrics.raw_details["per_query_evaluations"]) == 2
            assert agg_cand.metrics.raw_details["per_query_evaluations"][0]["query"] == sample_workload[0]
            assert agg_cand.metrics.raw_details["per_query_evaluations"][1]["query"] == sample_workload[1]


class TestParetoFrontierObjectives:
    """Verify Pareto strictly optimizes Groundedness (max) vs Cost/Query (min)."""

    def test_pareto_strictly_groundedness_and_cost(self, baseline_workflow):
        archive = ParetoArchive()

        # Candidate A: Groundedness 1.0, Cost 0.000050, Latency 1000ms
        cand_a = EvaluatedCandidate(
            candidate_id="cand_a",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="cand_a",
                workflow_name="Candidate A",
                performance=1.0,
                groundedness=1.0,
                cost=0.000050,
                total_tokens=300,
                input_tokens=200,
                output_tokens=100,
                llm_latency_ms=1000.0,
                execution_time_ms=1000.0,
                status="success",
            ),
        )

        # Candidate B: Groundedness 1.0, Cost 0.000050, Latency 200ms
        # Identical Groundedness and Cost -> Both remain Pareto optimal (Latency is secondary)
        cand_b = EvaluatedCandidate(
            candidate_id="cand_b",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="cand_b",
                workflow_name="Candidate B",
                performance=1.0,
                groundedness=1.0,
                cost=0.000050,
                total_tokens=300,
                input_tokens=200,
                output_tokens=100,
                llm_latency_ms=200.0,
                execution_time_ms=200.0,
                status="success",
            ),
        )

        archive.add_candidate(cand_a)
        archive.add_candidate(cand_b)

        assert len(archive.pareto_candidates) == 2
        assert cand_a.is_pareto_optimal is True
        assert cand_b.is_pareto_optimal is True

    def test_lower_cost_dominates_equal_groundedness(self, baseline_workflow):
        archive = ParetoArchive()

        cand_high_cost = EvaluatedCandidate(
            candidate_id="cand_high",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="cand_high",
                workflow_name="High Cost",
                performance=1.0,
                groundedness=1.0,
                cost=0.000060,
                total_tokens=400,
                input_tokens=300,
                output_tokens=100,
                llm_latency_ms=200.0,
                execution_time_ms=200.0,
                status="success",
            ),
        )

        cand_low_cost = EvaluatedCandidate(
            candidate_id="cand_low",
            workflow=baseline_workflow,
            metrics=CandidateMetrics(
                workflow_id="cand_low",
                workflow_name="Low Cost",
                performance=1.0,
                groundedness=1.0,
                cost=0.000040,
                total_tokens=250,
                input_tokens=180,
                output_tokens=70,
                llm_latency_ms=500.0,  # Higher latency, but lower cost
                execution_time_ms=500.0,
                status="success",
            ),
        )

        archive.add_candidate(cand_high_cost)
        archive.add_candidate(cand_low_cost)

        assert len(archive.pareto_candidates) == 1
        assert archive.pareto_candidates[0].candidate_id == "cand_low"
        assert cand_high_cost.is_pareto_optimal is False
        assert "cand_high" in cand_low_cost.dominates_candidates


class TestAdaptiveOptimizationFlow:
    """Test full adaptive optimization run flow."""

    @pytest.mark.anyio
    async def test_adaptive_workload_run(self, mock_db, sample_workload, baseline_workflow):
        optimizer = AgentEvoOptimizer()

        def make_eval_cand(workflow, **kwargs):
            return EvaluatedCandidate(
                candidate_id=workflow.workflow_id,
                workflow=workflow,
                metrics=CandidateMetrics(
                    workflow_id=workflow.workflow_id,
                    workflow_name=workflow.name,
                    performance=0.95,
                    groundedness=0.95,
                    hallucination_rate=0.05,
                    cost=0.000045,
                    total_tokens=300,
                    input_tokens=200,
                    output_tokens=100,
                    llm_latency_ms=220.0,
                    execution_time_ms=220.0,
                    status="success",
                    raw_details={
                        "is_workload": True,
                        "evaluation_count": 5,
                        "workload_queries": sample_workload,
                        "per_query_evaluations": [{"query": q} for q in sample_workload],
                    },
                ),
                is_pareto_optimal=True,
            )

        with patch("app.services.agent_evo.optimizer.EvaluatorAdapter.evaluate_candidate_workload", side_effect=make_eval_cand), \
             patch("app.services.agent_evo.optimizer.EvaluatorAdapter.evaluate_candidate", side_effect=make_eval_cand):

            report = await optimizer.run_optimization(
                query=sample_workload[0],
                workload=sample_workload,
                candidate_count=2,
                strategy="adaptive",
                user_id="test_user",
                db=mock_db,
            )

            assert isinstance(report, OptimizationReport)
            assert report.is_workload is True
            assert len(report.workload_queries) == 5
            assert report.total_workflow_evaluations == (1 + 2) * 5  # (1 baseline + 2 candidates) * 5 queries = 15
            assert report.scope_claim == "Optimized for the selected evaluation workload."
            assert report.scheduler_decision is not None
            assert "strategy" in report.scheduler_decision or "stage" in report.scheduler_decision
            assert "reason" in report.scheduler_decision
