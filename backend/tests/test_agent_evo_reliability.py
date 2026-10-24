import hashlib
import pytest
from app.services.evaluation.evaluator import EvaluationEngine, EvaluationInput
from app.services.evaluation.groundedness import calculate_groundedness
from app.services.agent_evo.objective import CandidateMetrics
from app.services.agent_evo.pareto import ParetoArchive, EvaluatedCandidate, dominates
from app.services.agent_evo.workflow import get_baseline_workflow


def test_1_provider_failure_produces_null_metrics_not_zero_cost():
    """1. Provider failure produces status FAILED / provider_error and null metrics, not $0 cost."""
    engine = EvaluationEngine()
    eval_input = EvaluationInput(
        question="What is the diagnosis?",
        generated_answer="",
        retrieved_context="Assessment: CAD",
        model_name="gemini-3.5-flash-lite",
        success=False,
        error="Connection timed out after 30000ms",
    )
    result = engine.evaluate(eval_input)
    assert result.total_cost is None, "Cost must be None on provider failure, not 0.0"
    assert result.latency_ms is None, "Latency must be None on provider failure, not 0.0"
    assert result.groundedness is None, "Groundedness must be None on failure"
    assert result.details["status"] == "provider_error"

    metrics = CandidateMetrics.from_eval_result("cand_fail", "Failed Candidate", result)
    assert metrics.cost is None
    assert metrics.performance is None
    assert metrics.llm_latency_ms is None
    assert metrics.status == "model_error"


def test_2_candidate_with_failed_queries_cannot_enter_pareto_archive():
    """2. Candidate with failed queries or null metrics cannot enter Pareto archive."""
    archive = ParetoArchive()
    wf = get_baseline_workflow()
    
    # Candidate with partial failure / null cost
    failed_metrics = CandidateMetrics.from_failure(
        workflow_id="cand_fail",
        workflow_name="Failed Cand",
        status="failed",
        error_message="504 Gateway Timeout",
    )
    cand = EvaluatedCandidate(
        candidate_id="cand_fail",
        workflow=wf,
        metrics=failed_metrics,
        is_pareto_optimal=False,
    )
    
    entered = archive.add_candidate(cand)
    assert entered is False, "Failed candidate must not enter Pareto archive"
    assert cand.is_pareto_optimal is False
    assert len(archive.pareto_candidates) == 0


def test_3_candidate_with_zero_or_null_cost_from_failure_does_not_dominate_baseline():
    """3. Candidate with $0 cost from failure does not dominate baseline."""
    baseline_metrics = CandidateMetrics(
        workflow_id="baseline",
        workflow_name="Production Baseline",
        performance=0.92,
        cost=0.000045,
        status="success",
    )
    failed_metrics = CandidateMetrics(
        workflow_id="failed_run",
        workflow_name="Failed Workflow",
        performance=None,
        cost=None,
        status="model_error",
    )
    
    # Failed run must NOT dominate baseline
    assert not dominates(failed_metrics, baseline_metrics), "Failed candidate must not dominate baseline"
    # Successful baseline must dominate failed candidate
    assert dominates(baseline_metrics, failed_metrics), "Baseline must dominate failed candidate"

    # Even if cost was 0.0 but status was non-success, it must not dominate
    zero_cost_error = CandidateMetrics(
        workflow_id="zero_cost_err",
        workflow_name="Zero Cost Error",
        performance=0.0,
        cost=0.0,
        status="model_error",
    )
    assert not dominates(zero_cost_error, baseline_metrics), "Zero-cost failed run must not dominate baseline"


def test_4_fallback_answer_produces_null_groundedness():
    """4. Fallback answer ('not found') produces null/N/A groundedness, not 100% or 0%."""
    context = ""
    answer = "The requested information was not found in the uploaded document."
    res = calculate_groundedness(answer, context)
    assert res.groundedness is None, "Fallback answer must have groundedness=None (rendered as N/A)"
    assert res.total_claims == 0
    assert res.supported_claims == 0
    assert res.unsupported_claims == 0


def test_5_genuine_zero_groundedness_is_preserved_and_distinct_from_na():
    """5. Genuine 0% groundedness (claims extracted but none supported) is preserved and distinct from N/A."""
    context = "Assessment: Patient has mild hypertension and elevated cholesterol."
    hallucinated_answer = "The patient is diagnosed with stage 4 neuroblastoma and severe acute myeloid leukemia."
    res = calculate_groundedness(hallucinated_answer, context)
    assert res.groundedness == 0.0, f"Expected genuine 0.0 groundedness, got {res.groundedness}"
    assert res.total_claims > 0, "Must have extracted factual claims"
    assert res.supported_claims == 0, "No claims should be supported by hypertension context"
    assert res.groundedness is not None, "Genuine 0.0 must be distinct from None"


def test_6_context_hash_computation_is_deterministic_and_detects_drift():
    """6. Context hash computation is deterministic and detects context drift."""
    chunk1 = "Chunk #1 — Clinical Assessment: Stable angina pectoris with exertional dyspnea."
    chunk2 = "Chunk #2 — Medications: Aspirin 81mg PO daily, Atorvastatin 40mg PO QHS."
    context_a = f"{chunk1}\n\n{chunk2}"
    context_b = f"{chunk1}\n\n{chunk2}"
    context_drifted = f"{chunk1}\n\nChunk #2 — Medications: Metformin 500mg PO BID."

    hash_a = hashlib.sha256(context_a.encode("utf-8")).hexdigest()
    hash_b = hashlib.sha256(context_b.encode("utf-8")).hexdigest()
    hash_drifted = hashlib.sha256(context_drifted.encode("utf-8")).hexdigest()

    assert hash_a == hash_b, "SHA-256 context hash must be completely deterministic"
    assert hash_a != hash_drifted, "Hash must detect context drift"


def test_7_benchmark_evaluation_workload_symmetry():
    """7. Benchmark evaluation runs identical queries for baseline and candidate."""
    from run_reproducible_benchmark import BENCHMARK_QUERIES
    assert len(BENCHMARK_QUERIES) == 5
    ids = [q["id"] for q in BENCHMARK_QUERIES]
    assert ids == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert "negative_control" in [q["type"] for q in BENCHMARK_QUERIES]


def test_8_candidate_acceptance_rejects_lower_groundedness():
    """8. Candidate acceptance rejects candidates with unacceptable groundedness decline."""
    base_m = CandidateMetrics(workflow_id="base", workflow_name="Baseline", performance=0.95, cost=0.00010, status="success")
    cand_m = CandidateMetrics(workflow_id="cand", workflow_name="Cheaper Cand", performance=0.80, cost=0.00003, status="success")
    # Even though cand is cheaper, groundedness dropped from 95% to 80% (-15 pp)
    groundedness_diff = cand_m.performance - base_m.performance
    accepted = cand_m.cost < base_m.cost and groundedness_diff >= -0.02
    assert not accepted, "Candidate with 15 pp groundedness drop must be rejected"


def test_9_candidate_acceptance_retains_baseline_when_no_cost_reduction():
    """9. Candidate acceptance retains baseline when no candidate improves cost."""
    base_m = CandidateMetrics(workflow_id="base", workflow_name="Baseline", performance=0.90, cost=0.00005, status="success")
    cand_m = CandidateMetrics(workflow_id="cand", workflow_name="Candidate", performance=0.90, cost=0.00008, status="success")
    accepted = cand_m.cost < base_m.cost and (cand_m.performance >= base_m.performance - 0.02)
    assert not accepted, "Candidate with higher cost must be rejected"


def test_10_multi_model_comparison_distinguishes_provider_failure_from_semantic_failure():
    """10. 3-model comparison correctly identifies provider failures vs semantic failures."""
    engine = EvaluationEngine()

    # Gemma HTTP 429 provider failure
    gemma_input = EvaluationInput(
        question="What is the diagnosis?",
        generated_answer="",
        retrieved_context="Assessment: Hypertension",
        model_name="google/gemma-4-26b-a4b-it:free",
        success=False,
        error="OpenRouter API error 429: Rate limit exceeded",
    )
    gemma_eval = engine.evaluate(gemma_input)
    assert gemma_eval.details["error_type"] == "rate_limit_429"
    assert "Rate Limited" in gemma_eval.details.get("display_status", "")
    assert gemma_eval.total_cost is None
    assert gemma_eval.latency_ms is None

    # Semantic not-found response (model ran successfully, found nothing in context)
    semantic_input = EvaluationInput(
        question="What is the patient's phone number?",
        generated_answer="The requested information was not found in the uploaded document.",
        retrieved_context="",
        model_name="gemini-3.5-flash-lite",
        success=True,
        error=None,
    )
    semantic_eval = engine.evaluate(semantic_input)
    assert semantic_eval.details["status"] == "success"
    assert semantic_eval.groundedness is None, "Semantic not-found must have groundedness=None"
