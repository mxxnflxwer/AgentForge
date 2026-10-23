import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.benchmark_run import BenchmarkRun
from app.models.workflow_version import WorkflowVersion
from app.schemas.agent_evo import (
    BaselineVsOptimizedComparisonResponse,
    BenchmarkItemResult,
    BenchmarkReportResponse,
    BenchmarkRunListResponse,
    PerQueryComparisonRecord,
    VersionInspectResponse,
    VersionLineageEdge,
    VersionLineageNode,
    VersionLineageResponse,
)
from app.services.agent_evo.evaluator_adapter import EvaluatorAdapter
from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector
from app.services.agent_evo.workflow import AgentWorkflow, get_baseline_workflow

logger = logging.getLogger("agentforge.services.agent_evo.benchmark")


class BenchmarkService:
    """
    Controlled multi-version benchmarking and lineage service.
    Evaluates multiple approved workflow versions (and baseline) under identical conditions,
    calculates Pareto dominance, persists reproducible benchmark runs, and tracks workflow lineage.
    """

    @staticmethod
    def _compute_pareto_dominance(
        results: List[BenchmarkItemResult],
    ) -> Tuple[List[BenchmarkItemResult], List[BenchmarkItemResult]]:
        """
        Compute Pareto dominance across benchmark results based on Groundedness ↑ and Cost ↓.
        A dominates B if:
          A.groundedness >= B.groundedness and A.cost <= B.cost
          and (A.groundedness > B.groundedness or A.cost < B.cost)
        """
        for i, item_a in enumerate(results):
            for j, item_b in enumerate(results):
                if i == j:
                    continue

                # Groundedness ↑ (higher is better), Cost ↓ (lower is better)
                g_a, c_a = item_a.groundedness, item_a.cost
                g_b, c_b = item_b.groundedness, item_b.cost

                # A dominates B?
                if (g_a >= g_b and c_a <= c_b) and (g_a > g_b or c_a < c_b):
                    if item_b.version_id not in item_a.dominates:
                        item_a.dominates.append(item_b.version_id)
                    if item_a.version_id not in item_b.dominated_by:
                        item_b.dominated_by.append(item_a.version_id)

        pareto_frontier: List[BenchmarkItemResult] = []
        dominated_list: List[BenchmarkItemResult] = []

        for item in results:
            if len(item.dominated_by) == 0:
                item.is_pareto_optimal = True
                item.pareto_status = "Pareto-optimal"
                pareto_frontier.append(item)
            else:
                item.is_pareto_optimal = False
                item.pareto_status = "Dominated"
                dominated_list.append(item)

        return pareto_frontier, dominated_list

    @classmethod
    async def run_benchmark(
        cls,
        version_ids: List[str],
        user_id: str,
        user_email: str,
        query: Optional[str] = None,
        workload: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        expected_answer: Optional[str] = None,
        expected_answers: Optional[List[Optional[str]]] = None,
        preset_name: Optional[str] = "clinical",
        name: Optional[str] = "Version Benchmark Comparison",
        description: Optional[str] = None,
        iterations: int = 1,
        repetitions: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> BenchmarkReportResponse:
        """
        Execute controlled reproducible benchmark across all selected workflow versions.
        Supports single query or multi-query workload across versions, establishes baseline-relative deltas,
        and computes Pareto dominance.
        """
        if not version_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one workflow version ID must be provided for benchmarking.",
            )

        if repetitions is not None and repetitions >= 1:
            iterations = repetitions

        target_query = query
        if not target_query and workload and len(workload) > 0:
            target_query = workload[0]
        elif not target_query:
            target_query = "What is the primary clinical diagnosis and assessment for this patient?"

        benchmark_id = f"bench_{uuid.uuid4().hex[:10]}"
        logger.info(
            f"Starting Benchmark '{benchmark_id}' ({name}) across {len(version_ids)} versions ({iterations} iterations) for query: '{target_query[:40]}'"
        )

        workflows_to_evaluate: List[Tuple[str, Optional[int], str, Optional[str], Optional[str], Optional[str], AgentWorkflow]] = []

        # Step 1: Validate and load immutable workflow configurations
        for v_id in version_ids:
            if v_id in ("baseline", "Production Baseline"):
                baseline_wf = get_baseline_workflow()
                workflows_to_evaluate.append((
                    "baseline",
                    0,
                    "Production Baseline",
                    None,
                    None,
                    None,
                    baseline_wf,
                ))
            else:
                if not db:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Database session required to load workflow versions.",
                    )

                version_rec = db.query(WorkflowVersion).filter(
                    (WorkflowVersion.version_id == v_id)
                    | (WorkflowVersion.id == v_id)
                    | (WorkflowVersion.source_candidate_id == v_id)
                ).first()

                if not version_rec:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Workflow version '{v_id}' not found in database.",
                    )

                wf_config = version_rec.configuration
                wf_instance = AgentWorkflow.from_dict(wf_config)
                wf_instance.workflow_id = version_rec.version_id
                wf_instance.name = version_rec.name

                mutation_meta = wf_config.get("mutation_metadata") or {}
                parent_id = mutation_meta.get("parent_workflow_id") or "baseline"

                workflows_to_evaluate.append((
                    version_rec.version_id,
                    version_rec.version_number,
                    version_rec.name,
                    parent_id,
                    version_rec.source_candidate_id,
                    version_rec.source_run_id,
                    wf_instance,
                ))

        # Step 2: Evaluate every selected workflow under the EXACT SAME conditions
        raw_benchmark_results: List[BenchmarkItemResult] = []
        num_iters = max(1, min(iterations, 5))

        for (
            v_id,
            v_num,
            v_name,
            parent_id,
            src_cand_id,
            src_run_id,
            wf_inst,
        ) in workflows_to_evaluate:
            logger.info(f"Benchmarking version '{v_id}' ({v_name}) across {num_iters} iteration(s)...")
            iter_runs: List[Dict[str, Any]] = []
            last_evaluated = None
            last_error = None

            for it in range(num_iters):
                try:
                    if workload and len(workload) > 0:
                        evaluated = await EvaluatorAdapter.evaluate_candidate_workload(
                            workflow=wf_inst,
                            queries=workload,
                            user_id=user_id,
                            document_id=document_id,
                            expected_answers=expected_answers,
                            db=db,
                        )
                    else:
                        evaluated = await EvaluatorAdapter.evaluate_candidate(
                            workflow=wf_inst,
                            query=target_query,
                            user_id=user_id,
                            document_id=document_id,
                            expected_answer=expected_answer,
                            db=db,
                        )
                    last_evaluated = evaluated
                    m = evaluated.metrics
                    groundedness = m.groundedness if m.groundedness is not None else (m.performance or 0.0)
                    hallucination_rate = m.hallucination_rate if m.hallucination_rate is not None else 0.0
                    iter_runs.append({
                        "iteration": it + 1,
                        "groundedness": groundedness,
                        "hallucination_rate": hallucination_rate,
                        "cost": m.cost,
                        "total_tokens": m.total_tokens,
                        "input_tokens": m.input_tokens,
                        "output_tokens": m.output_tokens,
                        "llm_latency_ms": m.llm_latency_ms,
                        "execution_time_ms": m.execution_time_ms,
                    })
                except Exception as e:
                    logger.error(f"Iteration {it + 1} failed for version '{v_id}': {e}")
                    last_error = str(e)

            retrieval_summary = (
                f"top_k={wf_inst.retrieval.top_k}, sim_thresh={wf_inst.retrieval.similarity_threshold}, "
                f"chunk_size={wf_inst.retrieval.chunk_size}, rerank={getattr(wf_inst.retrieval, 'reranking', True)}"
            )
            op_config = {
                "context_compression": wf_inst.execution.context_compression,
                "temperature": wf_inst.model.temperature,
                "max_tokens": wf_inst.model.max_tokens,
            }

            if iter_runs:
                # Compute mean values across iterations
                mean_groundedness = round(sum(r["groundedness"] for r in iter_runs) / len(iter_runs), 4)
                mean_hallucination = round(sum(r["hallucination_rate"] for r in iter_runs) / len(iter_runs), 4)
                mean_cost = round(sum(r["cost"] for r in iter_runs) / len(iter_runs), 6)
                mean_total_tokens = int(round(sum(r["total_tokens"] for r in iter_runs) / len(iter_runs)))
                mean_input_tokens = int(round(sum(r["input_tokens"] for r in iter_runs) / len(iter_runs)))
                mean_output_tokens = int(round(sum(r["output_tokens"] for r in iter_runs) / len(iter_runs)))
                mean_llm_latency = round(sum(r["llm_latency_ms"] for r in iter_runs) / len(iter_runs), 2)
                mean_exec_time = round(sum(r["execution_time_ms"] for r in iter_runs) / len(iter_runs), 2)

                def _calc_std(vals: List[float], ndigits: int = 4) -> float:
                    if len(vals) <= 1:
                        return 0.0
                    m_v = sum(vals) / len(vals)
                    var = sum((x - m_v) ** 2 for x in vals) / len(vals)
                    import math
                    return round(math.sqrt(var), ndigits)

                std_dev_dict = {
                    "groundedness": _calc_std([r["groundedness"] for r in iter_runs], 4),
                    "hallucination": _calc_std([r["hallucination_rate"] for r in iter_runs], 4),
                    "cost": _calc_std([r["cost"] for r in iter_runs], 6),
                    "cost_per_query": _calc_std([r["cost"] for r in iter_runs], 6),
                    "tokens": _calc_std([float(r["total_tokens"]) for r in iter_runs], 2),
                    "latency_ms": _calc_std([r["llm_latency_ms"] for r in iter_runs], 2),
                }

                last_m = last_evaluated.metrics if last_evaluated else None
                claim_breakdown = last_m.raw_details.get("claim_breakdown") if (last_m and last_m.raw_details) else None
                raw_details = last_m.raw_details if last_m else {}

                item = BenchmarkItemResult(
                    version_id=v_id,
                    version_number=v_num,
                    workflow_name=v_name,
                    parent_workflow_id=parent_id,
                    source_candidate_id=src_cand_id,
                    source_run_id=src_run_id,
                    model_id=wf_inst.model.model_id,
                    provider=wf_inst.model.provider,
                    groundedness=mean_groundedness,
                    hallucination_rate=mean_hallucination,
                    cost=mean_cost,
                    total_tokens=mean_total_tokens,
                    input_tokens=mean_input_tokens,
                    output_tokens=mean_output_tokens,
                    llm_latency_ms=mean_llm_latency,
                    execution_time_ms=mean_exec_time,
                    retrieval={
                        "top_k": wf_inst.retrieval.top_k,
                        "similarity_threshold": wf_inst.retrieval.similarity_threshold,
                        "chunk_size": wf_inst.retrieval.chunk_size,
                        "chunk_overlap": wf_inst.retrieval.chunk_overlap,
                        "reranking": getattr(wf_inst.retrieval, "reranking", True),
                    },
                    retrieval_config_summary=retrieval_summary,
                    prompt_template=wf_inst.prompt.template_name,
                    prompt_strategy=wf_inst.prompt.template_name,
                    context_compression=wf_inst.execution.context_compression,
                    operator_config=op_config,
                    is_pareto_optimal=False,
                    pareto_status="Non-dominated",
                    dominated_by=[],
                    dominates=[],
                    claim_breakdown=claim_breakdown,
                    raw_details=raw_details,
                    configuration=wf_inst.to_dict(),
                    iterations_count=len(iter_runs),
                    runs=iter_runs,
                    relative_changes=None,
                    version=v_id,
                    workflow=v_name,
                    parent_workflow=parent_id,
                    cost_per_query=mean_cost,
                    hallucination=mean_hallucination,
                    latency_ms=mean_llm_latency,
                    model=wf_inst.model.model_id,
                    retrieval_config=retrieval_summary,
                    std_dev=std_dev_dict,
                )
                raw_benchmark_results.append(item)
            else:
                # Create a failed candidate entry to preserve evaluation record
                item = BenchmarkItemResult(
                    version_id=v_id,
                    version_number=v_num,
                    workflow_name=v_name,
                    parent_workflow_id=parent_id,
                    source_candidate_id=src_cand_id,
                    source_run_id=src_run_id,
                    model_id=wf_inst.model.model_id,
                    provider=wf_inst.model.provider,
                    groundedness=None,
                    hallucination_rate=None,
                    cost=None,
                    total_tokens=None,
                    input_tokens=None,
                    output_tokens=None,
                    llm_latency_ms=None,
                    execution_time_ms=None,
                    retrieval={},
                    retrieval_config_summary="Failed evaluation",
                    prompt_template=wf_inst.prompt.template_name,
                    prompt_strategy=wf_inst.prompt.template_name,
                    context_compression=wf_inst.execution.context_compression,
                    operator_config=op_config,
                    is_pareto_optimal=False,
                    pareto_status="Dominated",
                    dominated_by=[],
                    dominates=[],
                    claim_breakdown=[],
                    raw_details={"error": last_error or "All evaluation iterations failed"},
                    configuration=wf_inst.to_dict(),
                    iterations_count=0,
                    runs=[],
                    relative_changes=None,
                    version=v_id,
                    workflow=v_name,
                    parent_workflow=parent_id,
                    cost_per_query=None,
                    hallucination=None,
                    latency_ms=None,
                    model=wf_inst.model.model_id,
                    retrieval_config="Failed evaluation",
                    std_dev={
                        "groundedness": 0.0,
                        "hallucination": 0.0,
                        "cost": 0.0,
                        "tokens": 0.0,
                        "latency_ms": 0.0,
                    },
                )
                raw_benchmark_results.append(item)

        # Step 3: Calculate relative changes against the Production Baseline
        baseline_item = next(
            (r for r in raw_benchmark_results if r.version_id in ("baseline", "Production Baseline") or r.version_number == 0),
            None,
        )

        for item in raw_benchmark_results:
            if baseline_item:
                g_delta = round(item.groundedness - baseline_item.groundedness, 4)
                c_red = (
                    round(((baseline_item.cost - item.cost) / baseline_item.cost) * 100, 2)
                    if baseline_item.cost > 0
                    else 0.0
                )
                c_change = (
                    round(((item.cost - baseline_item.cost) / baseline_item.cost) * 100, 2)
                    if baseline_item.cost > 0
                    else 0.0
                )
                lat_red = (
                    round(((baseline_item.llm_latency_ms - item.llm_latency_ms) / baseline_item.llm_latency_ms) * 100, 2)
                    if baseline_item.llm_latency_ms > 0
                    else 0.0
                )
                lat_change = (
                    round(((item.llm_latency_ms - baseline_item.llm_latency_ms) / baseline_item.llm_latency_ms) * 100, 2)
                    if baseline_item.llm_latency_ms > 0
                    else 0.0
                )
                tok_red = (
                    round(((baseline_item.total_tokens - item.total_tokens) / baseline_item.total_tokens) * 100, 2)
                    if baseline_item.total_tokens > 0
                    else 0.0
                )
                tok_change = (
                    round(((item.total_tokens - baseline_item.total_tokens) / baseline_item.total_tokens) * 100, 2)
                    if baseline_item.total_tokens > 0
                    else 0.0
                )
                item.relative_changes = {
                    "groundedness_delta": g_delta,
                    "groundedness_change": g_delta,
                    "cost_reduction_pct": c_red,
                    "cost_change_pct": c_change,
                    "latency_reduction_pct": lat_red,
                    "latency_change_pct": lat_change,
                    "token_reduction_pct": tok_red,
                    "token_change_pct": tok_change,
                }
            else:
                item.relative_changes = {
                    "groundedness_delta": 0.0,
                    "groundedness_change": 0.0,
                    "cost_reduction_pct": 0.0,
                    "cost_change_pct": 0.0,
                    "latency_reduction_pct": 0.0,
                    "latency_change_pct": 0.0,
                    "token_reduction_pct": 0.0,
                    "token_change_pct": 0.0,
                }

        # Step 4: Compute Pareto Dominance across benchmarked versions
        pareto_frontier, dominated_list = cls._compute_pareto_dominance(raw_benchmark_results)

        # Step 5: Generate Benchmark Summary
        summary = (
            f"Benchmark '{name}' evaluated {len(raw_benchmark_results)} workflow versions against query '{target_query[:50]}...'. "
            f"{len(pareto_frontier)} version(s) are Pareto-optimal (non-dominated on Groundedness vs Cost trade-off frontier), "
            f"and {len(dominated_list)} version(s) are dominated."
        )

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat()

        report_response = BenchmarkReportResponse(
            benchmark_id=benchmark_id,
            name=name,
            description=description,
            status="completed",
            preset_name=preset_name,
            query=target_query,
            document_id=document_id,
            document_ids=document_ids,
            expected_answer=expected_answer,
            repetitions=iterations,
            selected_versions=version_ids,
            results=raw_benchmark_results,
            pareto_frontier=pareto_frontier,
            dominated_versions=dominated_list,
            summary=summary,
            created_by=user_email,
            created_at=now_iso,
        )

        # Step 5: Persist BenchmarkRun to PostgreSQL
        if db:
            try:
                db_record = BenchmarkRun(
                    benchmark_id=benchmark_id,
                    name=name,
                    description=description,
                    status="completed",
                    preset_name=preset_name,
                    query=target_query,
                    document_id=document_id,
                    expected_answer=expected_answer,
                    selected_versions=version_ids,
                    results=[r.model_dump() for r in raw_benchmark_results],
                    pareto_summary={
                        "pareto_frontier_ids": [r.version_id for r in pareto_frontier],
                        "dominated_ids": [r.version_id for r in dominated_list],
                        "summary": summary,
                    },
                    created_by=user_email,
                    created_at=now_utc,
                )
                db.add(db_record)
                db.commit()
                logger.info(f"Persisted BenchmarkRun '{benchmark_id}' in PostgreSQL.")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to persist BenchmarkRun '{benchmark_id}' to DB: {e}")

        return report_response

    @classmethod
    def list_benchmarks(cls, db: Session) -> BenchmarkRunListResponse:
        """List all past benchmark runs from database."""
        records = db.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).all()
        benchmarks: List[BenchmarkReportResponse] = []

        for rec in records:
            results = [BenchmarkItemResult(**item) for item in (rec.results or [])]
            pareto_frontier = [r for r in results if r.is_pareto_optimal]
            dominated_list = [r for r in results if not r.is_pareto_optimal]

            created_at_str = (
                rec.created_at.isoformat()
                if hasattr(rec.created_at, "isoformat")
                else str(rec.created_at)
            )

            benchmarks.append(
                BenchmarkReportResponse(
                    benchmark_id=rec.benchmark_id,
                    name=rec.name,
                    description=rec.description,
                    status=rec.status,
                    preset_name=rec.preset_name,
                    query=rec.query,
                    document_id=rec.document_id,
                    expected_answer=rec.expected_answer,
                    selected_versions=rec.selected_versions or [],
                    results=results,
                    pareto_frontier=pareto_frontier,
                    dominated_versions=dominated_list,
                    summary=rec.pareto_summary.get("summary", "") if isinstance(rec.pareto_summary, dict) else "",
                    created_by=rec.created_by,
                    created_at=created_at_str,
                )
            )

        return BenchmarkRunListResponse(
            total_benchmarks=len(benchmarks),
            benchmarks=benchmarks,
        )

    @classmethod
    def get_benchmark(cls, benchmark_id: str, db: Session) -> Optional[BenchmarkReportResponse]:
        """Retrieve a specific benchmark run by benchmark_id."""
        rec = db.query(BenchmarkRun).filter(
            (BenchmarkRun.benchmark_id == benchmark_id) | (BenchmarkRun.id == benchmark_id)
        ).first()
        if not rec:
            return None

        results = [BenchmarkItemResult(**item) for item in (rec.results or [])]
        pareto_frontier = [r for r in results if r.is_pareto_optimal]
        dominated_list = [r for r in results if not r.is_pareto_optimal]

        created_at_str = (
            rec.created_at.isoformat()
            if hasattr(rec.created_at, "isoformat")
            else str(rec.created_at)
        )

        return BenchmarkReportResponse(
            benchmark_id=rec.benchmark_id,
            name=rec.name,
            description=rec.description,
            status=rec.status,
            preset_name=rec.preset_name,
            query=rec.query,
            document_id=rec.document_id,
            expected_answer=rec.expected_answer,
            selected_versions=rec.selected_versions or [],
            results=results,
            pareto_frontier=pareto_frontier,
            dominated_versions=dominated_list,
            summary=rec.pareto_summary.get("summary", "") if isinstance(rec.pareto_summary, dict) else "",
            created_by=rec.created_by,
            created_at=created_at_str,
        )

    @classmethod
    def get_lineage(cls, db: Session) -> VersionLineageResponse:
        """
        Build complete workflow version lineage graph from PostgreSQL database.
        Uses actual parent/source relationships without assuming linear progression.
        """
        versions = db.query(WorkflowVersion).order_by(WorkflowVersion.version_number.asc()).all()
        baseline_wf = get_baseline_workflow()

        nodes: List[VersionLineageNode] = []
        edges: List[VersionLineageEdge] = []

        # Root node: Production Baseline
        nodes.append(
            VersionLineageNode(
                version_id="baseline",
                version_number=0,
                name="Production Baseline",
                parent_workflow_id=None,
                source_run_id=None,
                source_candidate_id=None,
                model_id=baseline_wf.model.model_id,
                provider=baseline_wf.model.provider,
                mutation_type="baseline",
                refinement_operation=None,
                fields_changed=[],
                rationale="Default unmutated production baseline RAG pipeline.",
                before_configuration=None,
                after_configuration=baseline_wf.to_dict(),
                evaluation_snapshot=None,
                approved_by="System",
                approved_at=None,
                status="approved",
                configuration=baseline_wf.to_dict(),
            )
        )

        for v in versions:
            cfg = v.configuration or {}
            meta = cfg.get("mutation_metadata") or {}
            gen_meta = meta.get("generation_metadata") or {}

            parent_id = meta.get("parent_workflow_id") or "baseline"
            candidate_type = gen_meta.get("candidate_type") or "MUTATION"
            parent_a_id = gen_meta.get("parent_a_version_id") or (parent_id if candidate_type == "CROSSOVER" else None)
            parent_b_id = gen_meta.get("parent_b_version_id")
            crossover_strat = gen_meta.get("crossover_strategy")
            crossover_map = gen_meta.get("crossover_map")

            mutation_type = meta.get("mutation_type")
            refinement_op = gen_meta.get("refinement_operation")
            fields_changed = gen_meta.get("fields_changed") or []
            rationale = meta.get("mutation_description") or gen_meta.get("rationale")
            before_cfg = gen_meta.get("before_configuration")
            after_cfg = gen_meta.get("after_configuration")

            approved_at_str = (
                v.approved_at.isoformat()
                if hasattr(v.approved_at, "isoformat")
                else str(v.approved_at)
            )

            sched_dec = cfg.get("scheduler_decision") or {}
            scheduler_strategy = sched_dec.get("selected_strategy") or sched_dec.get("strategy")

            node = VersionLineageNode(
                version_id=v.version_id,
                version_number=v.version_number,
                name=v.name,
                parent_workflow_id=parent_id,
                parent_a_version_id=parent_a_id,
                parent_b_version_id=parent_b_id,
                candidate_type=candidate_type,
                crossover_strategy=crossover_strat,
                crossover_map=crossover_map,
                source_run_id=v.source_run_id,
                source_candidate_id=v.source_candidate_id,
                model_id=v.model_id,
                provider=v.provider,
                mutation_type=mutation_type,
                refinement_operation=refinement_op,
                scheduler_strategy=scheduler_strategy,
                fields_changed=fields_changed,
                rationale=rationale,
                before_configuration=before_cfg,
                after_configuration=after_cfg,
                evaluation_snapshot=v.evaluation_snapshot,
                approved_by=v.approved_by,
                approved_at=approved_at_str,
                status=v.status,
                configuration=cfg,
            )
            nodes.append(node)

            # Add directed edge(s) from parent(s) to this version
            if candidate_type == "CROSSOVER" and parent_b_id:
                edges.append(
                    VersionLineageEdge(
                        source=parent_a_id or parent_id,
                        target=v.version_id,
                        label="Crossover (Parent A)",
                    )
                )
                edges.append(
                    VersionLineageEdge(
                        source=parent_b_id,
                        target=v.version_id,
                        label="Crossover (Parent B)",
                    )
                )
            else:
                edge_label = refinement_op or mutation_type or "mutation"
                edges.append(
                    VersionLineageEdge(
                        source=parent_id,
                        target=v.version_id,
                        label=edge_label,
                    )
                )

        return VersionLineageResponse(
            total_versions=len(nodes),
            nodes=nodes,
            edges=edges,
        )

    @classmethod
    def inspect_version(cls, version_id: str, db: Session) -> VersionInspectResponse:
        """
        Inspect complete metadata, lineage, before/after configuration diffs, and approval record for a version.
        """
        if version_id in ("baseline", "Production Baseline"):
            baseline_wf = get_baseline_workflow()
            return VersionInspectResponse(
                version_id="baseline",
                version_number=0,
                name="Production Baseline",
                description="Default unmutated production baseline workflow configuration.",
                parent_workflow_id=None,
                parent_version=None,
                source_run_id="N/A",
                source_candidate_id="baseline",
                model_id=baseline_wf.model.model_id,
                provider=baseline_wf.model.provider,
                mutation_type="baseline",
                refinement_operation=None,
                fields_changed=[],
                rationale="Production reference workflow.",
                before_configuration=None,
                after_configuration=baseline_wf.to_dict(),
                configuration=baseline_wf.to_dict(),
                evaluation_snapshot={
                    "groundedness": 1.0,
                    "cost": 0.000057,
                    "llm_latency_ms": 4328.0,
                },
                status="approved",
                approved_by="System",
                approved_at="2026-08-30T00:00:00Z",
                created_at="2026-08-30T00:00:00Z",
            )

        v = db.query(WorkflowVersion).filter(
            (WorkflowVersion.version_id == version_id)
            | (WorkflowVersion.id == version_id)
            | (WorkflowVersion.source_candidate_id == version_id)
        ).first()

        if not v:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow version '{version_id}' not found.",
            )

        cfg = v.configuration or {}
        meta = cfg.get("mutation_metadata") or {}
        gen_meta = meta.get("generation_metadata") or {}

        parent_id = meta.get("parent_workflow_id") or "baseline"
        mutation_type = meta.get("mutation_type")
        refinement_op = gen_meta.get("refinement_operation")
        fields_changed = gen_meta.get("fields_changed") or []
        rationale = meta.get("mutation_description") or gen_meta.get("rationale")
        before_cfg = gen_meta.get("before_configuration")
        after_cfg = gen_meta.get("after_configuration")

        candidate_type = gen_meta.get("candidate_type") or "MUTATION"
        parent_a_id = gen_meta.get("parent_a_version_id") or (parent_id if candidate_type == "CROSSOVER" else None)
        parent_b_id = gen_meta.get("parent_b_version_id")
        crossover_strat = gen_meta.get("crossover_strategy")
        crossover_map = gen_meta.get("crossover_map")

        # Resolve parent version summary if parent exists in DB
        parent_version_info = None
        if parent_id not in ("baseline", "Production Baseline"):
            p_rec = db.query(WorkflowVersion).filter(
                (WorkflowVersion.version_id == parent_id)
                | (WorkflowVersion.source_candidate_id == parent_id)
            ).first()
            if p_rec:
                parent_version_info = {
                    "version_id": p_rec.version_id,
                    "version_number": p_rec.version_number,
                    "name": p_rec.name,
                    "model_id": p_rec.model_id,
                    "provider": p_rec.provider,
                }

        parent_a_version_info = parent_version_info
        parent_b_version_info = None
        if parent_b_id and parent_b_id not in ("baseline", "Production Baseline"):
            pb_rec = db.query(WorkflowVersion).filter(
                (WorkflowVersion.version_id == parent_b_id)
                | (WorkflowVersion.source_candidate_id == parent_b_id)
            ).first()
            if pb_rec:
                parent_b_version_info = {
                    "version_id": pb_rec.version_id,
                    "version_number": pb_rec.version_number,
                    "name": pb_rec.name,
                    "model_id": pb_rec.model_id,
                    "provider": pb_rec.provider,
                }

        approved_at_str = (
            v.approved_at.isoformat()
            if hasattr(v.approved_at, "isoformat")
            else str(v.approved_at)
        )
        created_at_str = (
            v.created_at.isoformat()
            if hasattr(v.created_at, "isoformat")
            else str(v.created_at)
        )

        return VersionInspectResponse(
            version_id=v.version_id,
            version_number=v.version_number,
            name=v.name,
            description=v.description,
            parent_workflow_id=parent_id,
            parent_a_version_id=parent_a_id,
            parent_b_version_id=parent_b_id,
            candidate_type=candidate_type,
            crossover_strategy=crossover_strat,
            crossover_map=crossover_map,
            parent_version=parent_version_info,
            parent_a_version=parent_a_version_info,
            parent_b_version=parent_b_version_info,
            source_run_id=v.source_run_id,
            source_candidate_id=v.source_candidate_id,
            model_id=v.model_id,
            provider=v.provider,
            mutation_type=mutation_type,
            refinement_operation=refinement_op,
            fields_changed=fields_changed,
            rationale=rationale,
            before_configuration=before_cfg,
            after_configuration=after_cfg,
            configuration=cfg,
            evaluation_snapshot=v.evaluation_snapshot or {},
            status=v.status,
            approved_by=v.approved_by,
            approved_at=approved_at_str,
            created_at=created_at_str,
        )

    @classmethod
    async def compare_baseline_vs_optimized(
        cls,
        optimized_version_id: str,
        workload: List[str],
        user_id: str,
        user_email: str,
        baseline_version_id: Optional[str] = "baseline",
        document_id: Optional[str] = None,
        expected_answers: Optional[List[Optional[str]]] = None,
        db: Optional[Session] = None,
    ) -> BaselineVsOptimizedComparisonResponse:
        """
        Controlled, reproducible evaluation of an approved optimized workflow against the baseline
        using the exact same evaluation workload.
        Records raw per-query response, groundedness, claims, tokens, cost, latency, provider, and model ID.
        Computes aggregate metrics from raw unrounded metrics.
        Never executes an unapproved candidate.
        Clearly distinguishes real provider evaluations from mocked or simulation tests.
        Reports cost reduction and groundedness retention honestly without claiming universal optimality.
        """
        if not workload or len(workload) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Evaluation workload must contain at least one query.",
            )

        # 1. Resolve baseline workflow
        if not baseline_version_id or baseline_version_id in ("baseline", "Production Baseline"):
            baseline_wf = get_baseline_workflow()
            base_ver_id = "baseline"
        else:
            if not db:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session required to load workflow versions.",
                )
            base_rec = db.query(WorkflowVersion).filter(
                (WorkflowVersion.version_id == baseline_version_id) | (WorkflowVersion.id == baseline_version_id)
            ).first()
            if not base_rec:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Baseline workflow version '{baseline_version_id}' not found.",
                )
            baseline_wf = AgentWorkflow.from_dict(base_rec.configuration)
            base_ver_id = base_rec.version_id

        # 2. Resolve optimized workflow from database - STRICT GUARD: MUST BE APPROVED OR ACTIVE
        if not db:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session required to load optimized workflow version.",
            )

        opt_rec = db.query(WorkflowVersion).filter(
            (WorkflowVersion.version_id == optimized_version_id) | (WorkflowVersion.id == optimized_version_id)
        ).first()

        if not opt_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow version '{optimized_version_id}' not found in registry. Only approved workflows can be evaluated.",
            )

        status_norm = (opt_rec.status or "").upper()
        if status_norm not in ("APPROVED", "ACTIVE"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Workflow version '{opt_rec.version_id}' has status '{opt_rec.status}'. Only APPROVED or ACTIVE workflows may be evaluated.",
            )

        opt_wf = AgentWorkflow.from_dict(opt_rec.configuration)
        opt_wf.workflow_id = opt_rec.version_id
        opt_wf.name = opt_rec.name

        # 3. Execute baseline evaluation across workload
        baseline_eval = await EvaluatorAdapter.evaluate_candidate_workload(
            workflow=baseline_wf,
            queries=workload,
            user_id=user_id,
            document_id=document_id,
            expected_answers=expected_answers,
            db=db,
        )

        # 4. Execute optimized workflow evaluation across identical workload
        opt_eval = await EvaluatorAdapter.evaluate_candidate_workload(
            workflow=opt_wf,
            queries=workload,
            user_id=user_id,
            document_id=document_id,
            expected_answers=expected_answers,
            db=db,
        )

        base_pq = baseline_eval.metrics.raw_details.get("per_query_evaluations") or []
        opt_pq = opt_eval.metrics.raw_details.get("per_query_evaluations") or []

        # 5. Detect execution mode (live provider vs mocked)
        is_mocked = bool(
            baseline_eval.metrics.raw_details.get("is_mocked", False)
            or opt_eval.metrics.raw_details.get("is_mocked", False)
        )
        eval_mode = "mocked_or_saved_response" if is_mocked else "live_provider"
        mode_disclaimer = (
            "Evaluated using mocked adapter responses. Mocked test runs demonstrate pipeline functionality, "
            "deterministic routing, and metric aggregation, not real-world API cost savings."
            if is_mocked
            else "Evaluated using live provider API (Google / OpenRouter). Measured real API token usage, cost, and latency."
        )

        # 6. Build granular per-query records
        per_query_records: List[PerQueryComparisonRecord] = []
        total_queries = len(workload)

        for i in range(total_queries):
            b_item = base_pq[i] if i < len(base_pq) else {}
            o_item = opt_pq[i] if i < len(opt_pq) else {}
            q_text = workload[i]
            task_cat = RuntimeWorkflowSelector.classify_query_task(q_text)

            b_cost = float(b_item.get("cost", 0.0))
            o_cost = float(o_item.get("cost", 0.0))
            b_g = float(b_item.get("groundedness", 0.0))
            o_g = float(o_item.get("groundedness", 0.0))

            c_red = round(((b_cost - o_cost) / b_cost) * 100, 2) if b_cost > 0 else 0.0
            g_delta = round(o_g - b_g, 4)

            per_query_records.append(
                PerQueryComparisonRecord(
                    query_id=f"q_{i + 1}",
                    query=q_text,
                    task_category=task_cat,
                    baseline_response=b_item.get("response", ""),
                    baseline_groundedness=b_g,
                    baseline_hallucination_rate=float(b_item.get("hallucination_rate", 0.0)),
                    baseline_claims=b_item.get("supported_claims", []) + b_item.get("unsupported_claims", []),
                    baseline_supported_claim_count=int(b_item.get("supported_claim_count", 0)),
                    baseline_unsupported_claim_count=int(b_item.get("unsupported_claim_count", 0)),
                    baseline_input_tokens=int(b_item.get("input_tokens", 0)),
                    baseline_output_tokens=int(b_item.get("output_tokens", 0)),
                    baseline_total_tokens=int(b_item.get("total_tokens", 0)),
                    baseline_cost=b_cost,
                    baseline_latency_ms=float(b_item.get("llm_latency_ms", 0.0)),
                    baseline_provider=b_item.get("provider", baseline_wf.model.provider),
                    baseline_model_id=b_item.get("model_id", baseline_wf.model.model_id),
                    baseline_status=b_item.get("status", "success"),
                    optimized_response=o_item.get("response", ""),
                    optimized_groundedness=o_g,
                    optimized_hallucination_rate=float(o_item.get("hallucination_rate", 0.0)),
                    optimized_claims=o_item.get("supported_claims", []) + o_item.get("unsupported_claims", []),
                    optimized_supported_claim_count=int(o_item.get("supported_claim_count", 0)),
                    optimized_unsupported_claim_count=int(o_item.get("unsupported_claim_count", 0)),
                    optimized_input_tokens=int(o_item.get("input_tokens", 0)),
                    optimized_output_tokens=int(o_item.get("output_tokens", 0)),
                    optimized_total_tokens=int(o_item.get("total_tokens", 0)),
                    optimized_cost=o_cost,
                    optimized_latency_ms=float(o_item.get("llm_latency_ms", 0.0)),
                    optimized_provider=o_item.get("provider", opt_wf.model.provider),
                    optimized_model_id=o_item.get("model_id", opt_wf.model.model_id),
                    optimized_status=o_item.get("status", "success"),
                    cost_reduction=c_red,
                    groundedness_delta=g_delta,
                )
            )

        # 7. Compute aggregate results directly from raw unrounded metrics
        successful_evals = sum(1 for r in per_query_records if r.baseline_status == "success" and r.optimized_status == "success")
        failed_evals = total_queries - successful_evals

        raw_b_g_list = [r.baseline_groundedness for r in per_query_records]
        raw_o_g_list = [r.optimized_groundedness for r in per_query_records]
        raw_b_h_list = [r.baseline_hallucination_rate for r in per_query_records]
        raw_o_h_list = [r.optimized_hallucination_rate for r in per_query_records]
        raw_b_c_list = [r.baseline_cost for r in per_query_records]
        raw_o_c_list = [r.optimized_cost for r in per_query_records]
        raw_b_tok_list = [r.baseline_total_tokens for r in per_query_records]
        raw_o_tok_list = [r.optimized_total_tokens for r in per_query_records]
        raw_b_lat_list = [r.baseline_latency_ms for r in per_query_records]
        raw_o_lat_list = [r.optimized_latency_ms for r in per_query_records]

        agg_b_g = sum(raw_b_g_list) / max(1, total_queries)
        agg_o_g = sum(raw_o_g_list) / max(1, total_queries)
        groundedness_diff_pp = round((agg_o_g - agg_b_g) * 100, 2)
        groundedness_retained = bool(agg_o_g >= agg_b_g - 0.0001)

        agg_b_h = sum(raw_b_h_list) / max(1, total_queries)
        agg_o_h = sum(raw_o_h_list) / max(1, total_queries)

        agg_b_c = sum(raw_b_c_list) / max(1, total_queries)
        agg_o_c = sum(raw_o_c_list) / max(1, total_queries)
        cost_red_pct = round(((agg_b_c - agg_o_c) / agg_b_c) * 100, 2) if agg_b_c > 0 else 0.0
        cost_reduced = bool(agg_o_c < agg_b_c)

        agg_b_tok = sum(raw_b_tok_list) / max(1, total_queries)
        agg_o_tok = sum(raw_o_tok_list) / max(1, total_queries)

        agg_b_lat = sum(raw_b_lat_list) / max(1, total_queries)
        agg_o_lat = sum(raw_o_lat_list) / max(1, total_queries)

        # 8. Honest finding summary
        if cost_reduced and groundedness_retained:
            finding = (
                f"Optimized workflow '{opt_rec.name}' (v{opt_rec.version_number}) successfully reduced cost by "
                f"{cost_red_pct}% (${agg_o_c:.6f} vs ${agg_b_c:.6f}/query) while preserving clinical groundedness "
                f"({agg_o_g * 100:.1f}% vs {agg_b_g * 100:.1f}%, diff: {groundedness_diff_pp:+.1f} pp). "
                f"Results are scoped to the evaluated workload and do not claim universal optimality."
            )
        elif not cost_reduced and groundedness_retained:
            finding = (
                f"Optimized workflow '{opt_rec.name}' retained groundedness ({agg_o_g * 100:.1f}%), but did NOT "
                f"reduce cost (${agg_o_c:.6f} vs baseline ${agg_b_c:.6f}/query). "
                f"Honest evaluation indicates no cost advantage on this workload."
            )
        elif cost_reduced and not groundedness_retained:
            finding = (
                f"Optimized workflow '{opt_rec.name}' reduced cost by {cost_red_pct}%, but experienced a groundedness decline "
                f"of {abs(groundedness_diff_pp):.1f} percentage points ({agg_o_g * 100:.1f}% vs {agg_b_g * 100:.1f}%). "
                f"Trade-off shows lower cost at the expense of grounded accuracy."
            )
        else:
            finding = (
                f"Optimized workflow '{opt_rec.name}' did not achieve cost reduction (${agg_o_c:.6f} vs ${agg_b_c:.6f}) "
                f"and experienced lower groundedness ({agg_o_g * 100:.1f}% vs {agg_b_g * 100:.1f}%)."
            )

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return BaselineVsOptimizedComparisonResponse(
            baseline_version_id=base_ver_id,
            optimized_version_id=opt_rec.version_id,
            optimized_version_number=opt_rec.version_number,
            optimized_workflow_name=opt_rec.name,
            evaluation_mode=eval_mode,
            is_mocked=is_mocked,
            mode_disclaimer=mode_disclaimer,
            scope_claim="Optimized for the selected evaluation workload. Does not claim universal optimality.",
            total_queries_evaluated=total_queries,
            successful_evaluations=successful_evals,
            failed_evaluations=failed_evals,
            baseline_groundedness=agg_b_g,
            optimized_groundedness=agg_o_g,
            groundedness_diff_pp=groundedness_diff_pp,
            groundedness_retained=groundedness_retained,
            baseline_hallucination_rate=agg_b_h,
            optimized_hallucination_rate=agg_o_h,
            baseline_cost_per_query=agg_b_c,
            optimized_cost_per_query=agg_o_c,
            cost_reduction_pct=cost_red_pct,
            cost_reduced=cost_reduced,
            baseline_tokens_per_query=agg_b_tok,
            optimized_tokens_per_query=agg_o_tok,
            baseline_latency_ms=round(agg_b_lat, 2),
            optimized_latency_ms=round(agg_o_lat, 2),
            per_query_evidence=per_query_records,
            finding_summary=finding,
            created_at=now_iso,
        )
