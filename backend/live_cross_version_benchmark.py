import asyncio
import json
import logging
from app.core.database import SessionLocal
from app.models.workflow_version import WorkflowVersion
from app.models.user import User
from app.models.document import Document
from app.services.agent_evo.benchmark_service import BenchmarkService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_benchmark")

async def main():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == "863ffca5-e2c2-4a53-bc58-441debdd5177").first()
        if not user:
            user = db.query(User).filter(User.email == "user@example.com").first() or db.query(User).first()
        logger.info(f"Using evaluation user: {user.email} (id: {user.id})")

        docs = db.query(Document).filter(Document.owner_id == user.id).all()
        doc_ids = [str(d.id) for d in docs]
        doc_names = [d.original_filename for d in docs]
        logger.info(f"Using ingested evaluation documents: {doc_names} ({doc_ids})")

        # 9 Target Workflows: Baseline + v1 through v8
        target_versions = [
            "baseline",
            "wf_v1_60e5af",
            "wf_v2_dfce47",
            "wf_v3_4294dd",
            "wf_v4_b0844c",
            "wf_v5_0123b3",
            "wf_v6_a8b827",
            "wf_v7_adfa07",
            "wf_v8_2b0b35",
        ]
        logger.info(f"Target benchmark version IDs (9 workflows: Baseline + v1-v8): {target_versions}")

        query = "What is the primary clinical diagnosis and assessment for this patient?"
        logger.info(f"Executing controlled benchmark across 9 workflows (3 repetitions each) for query: '{query}'...")

        report = await BenchmarkService.run_benchmark(
            version_ids=target_versions,
            query=query,
            user_id=str(user.id),
            user_email=user.email,
            document_id=doc_ids[0] if doc_ids else None,
            document_ids=doc_ids,
            preset_name="clinical",
            name="Official Cross-Version Benchmark: Baseline + v1-v8",
            description="Controlled reproducible 3-repetition benchmark under identical Phase 6 evaluation conditions across Baseline and Approved Versions v1 through v8.",
            iterations=3,
            repetitions=3,
            db=db,
        )

        logger.info(f"Benchmark completed! Benchmark ID: {report.benchmark_id}")
        logger.info(f"Summary: {report.summary}")

        print("\n" + "="*175)
        print("AGENTEVO CONTROLLED CROSS-VERSION BENCHMARK (3 REPETITIONS PER WORKFLOW)")
        print(f"Benchmark Run ID: {report.benchmark_id} | Preset: {report.preset_name} | Target Documents: {', '.join(doc_names)}")
        print(f"Query: {report.query}")
        print("="*175)
        header = f"{'Version':<13} | {'Workflow Name':<32} | {'Groundedness':<12} | {'Hallucination':<13} | {'Cost/Query':<11} | {'Tokens':<7} | {'Latency':<9} | {'Retrieval Config':<30} | {'Prompt':<16} | {'Operators'}"
        print(header)
        print("-"*175)
        for r in report.results:
            op_str = f"compress={r.operator_config.get('context_compression', False)}, temp={r.operator_config.get('temperature', 0.1)}" if r.operator_config else "default"
            ret_str = r.retrieval_config_summary or "default"
            print(f"{r.version_id:<13} | {r.workflow_name[:32]:<32} | {r.groundedness*100:>10.1f}% | {r.hallucination_rate*100:>11.1f}% | ${r.cost:>9.6f} | {r.total_tokens:>7} | {r.llm_latency_ms:>7.0f}ms | {ret_str[:30]:<30} | {r.prompt_template[:16]:<16} | {op_str}")
        print("="*175)

        print("\n=== BASELINE-RELATIVE PERFORMANCE & EFFICIENCY CHANGES ===")
        print(f"{'Version':<13} | {'Workflow Name':<32} | {'Groundedness Delta':<20} | {'Cost Change %':<14} | {'Latency Change %':<17} | {'Token Change %':<15} | {'Pareto Status':<15} | {'Dominates'}")
        print("-"*165)
        for r in report.results:
            rc = r.relative_changes or {}
            doms = ", ".join(r.dominates) if r.dominates else "None"
            # Note: cost_reduction_pct is positive for reductions. Let's show both clear signs.
            c_red = rc.get('cost_reduction_pct', 0.0)
            lat_red = rc.get('latency_reduction_pct', 0.0)
            tok_red = rc.get('token_reduction_pct', 0.0)
            print(f"{r.version_id:<13} | {r.workflow_name[:32]:<32} | {rc.get('groundedness_delta', 0.0):>+18.4f} | {-c_red:>+12.2f}% | {-lat_red:>+15.2f}% | {-tok_red:>+13.2f}% | {r.pareto_status:<15} | {doms}")
        print("-"*165)

        print("\n=== INDIVIDUAL REPETITION RUNS (VARIANCE TRACKING) ===")
        for r in report.results:
            print(f"[{r.version_id}] {r.workflow_name}:")
            for run in (r.runs or []):
                print(f"   Repetition {run['iteration']}: Groundedness={run['groundedness']*100:.1f}%, Cost=${run['cost']:.6f}, Tokens={run['total_tokens']}, Latency={run['llm_latency_ms']:.0f}ms, Exec={run['execution_time_ms']:.0f}ms")

        # Print Lineage Graph
        lineage = BenchmarkService.get_lineage(db=db)
        print("\n=== VERSION LINEAGE (FROM DATABASE) ===")
        print(f"{'Version ID':<15} | {'Ver #':<5} | {'Parent Workflow':<15} | {'Source Candidate ID':<28} | {'Source Run ID':<20} | {'Mutation / Refinement Strategy'}")
        print("-"*135)
        for n in lineage.nodes:
            strat = f"{n.mutation_type or 'root'} / {n.refinement_operation or 'n/a'}"
            print(f"{n.version_id:<15} | {str(n.version_number):<5} | {str(n.parent_workflow_id):<15} | {str(n.source_candidate_id):<28} | {str(n.source_run_id):<20} | {strat}")
        print("-"*135)

        # Save output JSON for reporting
        with open("live_benchmark_results.json", "w") as f:
            json.dump(report.model_dump(), f, indent=2)

    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
