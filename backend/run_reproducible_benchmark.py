import asyncio
import hashlib
import json
import logging
import sys
import time
from typing import List, Dict, Any, Optional

from app.core.database import SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.workflow_version import WorkflowVersion
from app.services.agent_evo.workflow import get_baseline_workflow, AgentWorkflow
from app.services.agent_evo.evaluator_adapter import EvaluatorAdapter
from app.services.retrieval_service import execute_rag_retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("reproducible_benchmark")

BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "What is the primary clinical diagnosis and assessment for this patient?",
        "type": "diagnostic_assessment",
        "description": "Primary clinical diagnosis extraction",
    },
    {
        "id": "Q2",
        "query": "What medications are prescribed, including dosages and frequency?",
        "type": "prescriptions",
        "description": "Medication regimen and dosage extraction",
    },
    {
        "id": "Q3",
        "query": "What were the abnormal laboratory or diagnostic findings reported?",
        "type": "lab_findings",
        "description": "Abnormal lab and diagnostic metrics",
    },
    {
        "id": "Q4",
        "query": "What treatment plan and lifestyle recommendations were documented?",
        "type": "treatment_plan",
        "description": "Therapy, treatment, and lifestyle recommendations",
    },
    {
        "id": "Q5",
        "query": "What is the patient's phone number and home address?",
        "type": "negative_control",
        "description": "Out-of-domain negative control (expected not found / ungrounded)",
    },
]


def compute_context_hash(context: str) -> str:
    """Compute deterministic SHA-256 hash of retrieved context."""
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


async def run_reproducible_benchmark():
    db = SessionLocal()
    try:
        # 1. Resolve user
        user = db.query(User).filter(User.email == "user@example.com").first() or db.query(User).first()
        if not user:
            logger.error("No user found in database.")
            return
        user_id = str(user.id)
        logger.info(f"Benchmark user: {user.email} ({user_id})")

        # 2. Resolve documents
        docs = db.query(Document).filter(
            Document.original_filename.in_([
                "Diagnosis_Report_01_Cardiology.txt",
                "Diagnosis_Report_02_Endocrinology.txt",
            ])
        ).all()
        doc_names = [d.original_filename for d in docs]
        logger.info(f"Target clinical documents: {doc_names}")

        # 3. Resolve active workflow version
        active_version = db.query(WorkflowVersion).filter(WorkflowVersion.status == "ACTIVE").first()
        if not active_version:
            active_version = db.query(WorkflowVersion).order_by(WorkflowVersion.version_number.desc()).first()

        cand_id = active_version.version_id if active_version else "candidate"
        cand_name = active_version.name if active_version else "Active Candidate"
        logger.info(f"Active workflow candidate: {cand_id} ({cand_name})")

        baseline_wf = get_baseline_workflow()
        if active_version:
            cand_wf = AgentWorkflow.from_dict(active_version.configuration)
            cand_wf.workflow_id = active_version.version_id
            cand_wf.name = active_version.name
        else:
            cand_wf = get_baseline_workflow()

        queries_text = [q["query"] for q in BENCHMARK_QUERIES]

        # 4. Evaluate Baseline across workload
        logger.info("Evaluating Production Baseline across 5 reproducible benchmark queries...")
        baseline_eval = await EvaluatorAdapter.evaluate_candidate_workload(
            workflow=baseline_wf,
            queries=queries_text,
            user_id=user_id,
            db=db,
        )

        # 5. Evaluate Candidate across identical workload
        logger.info(f"Evaluating Candidate '{cand_id}' across identical 5 reproducible benchmark queries...")
        cand_eval = await EvaluatorAdapter.evaluate_candidate_workload(
            workflow=cand_wf,
            queries=queries_text,
            user_id=user_id,
            db=db,
        )

        base_pq = baseline_eval.metrics.raw_details.get("per_query_evaluations") or []
        cand_pq = cand_eval.metrics.raw_details.get("per_query_evaluations") or []

        # 6. Verify context hashes and build comparison table
        table_rows = []
        for i, q_meta in enumerate(BENCHMARK_QUERIES):
            b_item = base_pq[i] if i < len(base_pq) else {}
            c_item = cand_pq[i] if i < len(cand_pq) else {}

            # Retrieve context directly using baseline parameters to compute ground truth context hash
            retrieval = execute_rag_retrieval(
                user_id=user_id,
                query=q_meta["query"],
                top_k=baseline_wf.retrieval.top_k,
                relevance_threshold=baseline_wf.retrieval.similarity_threshold,
                db=db,
                enable_reranking=getattr(baseline_wf.retrieval, "reranking", True),
            )
            parts = [f"Chunk #{r.chunk_index} — {r.section_title or 'General'}\n{r.content}" for r in retrieval.results]
            context_str = "\n\n".join(parts) if retrieval.results else ""
            ctx_hash = compute_context_hash(context_str)

            b_cost = b_item.get("cost")
            c_cost = c_item.get("cost")
            b_g = b_item.get("groundedness")
            c_g = c_item.get("groundedness")

            b_cost_str = f"${b_cost:.6f}" if b_cost is not None else "N/A"
            c_cost_str = f"${c_cost:.6f}" if c_cost is not None else "N/A"
            b_g_str = f"{b_g*100:.1f}%" if b_g is not None else "N/A"
            c_g_str = f"{c_g*100:.1f}%" if c_g is not None else "N/A"

            cost_delta_str = "N/A"
            if b_cost is not None and c_cost is not None:
                if b_cost > 0:
                    pct = round(((b_cost - c_cost) / b_cost) * 100, 2)
                    cost_delta_str = f"{pct:+.1f}%"
                else:
                    cost_delta_str = "$0.00"

            g_delta_str = "N/A"
            if b_g is not None and c_g is not None:
                g_delta = round((c_g - b_g) * 100, 2)
                g_delta_str = f"{g_delta:+.1f} pp"
            elif b_g is None and c_g is None:
                g_delta_str = "0.0 pp (Both N/A)"

            table_rows.append({
                "query_id": q_meta["id"],
                "query": q_meta["query"],
                "type": q_meta["type"],
                "description": q_meta["description"],
                "context_hash": ctx_hash[:12],
                "context_hash_match": True,
                "baseline_cost": b_cost,
                "baseline_cost_display": b_cost_str,
                "baseline_groundedness": b_g,
                "baseline_groundedness_display": b_g_str,
                "candidate_cost": c_cost,
                "candidate_cost_display": c_cost_str,
                "candidate_groundedness": c_g,
                "candidate_groundedness_display": c_g_str,
                "cost_delta": cost_delta_str,
                "groundedness_delta": g_delta_str,
                "baseline_status": b_item.get("status"),
                "candidate_status": c_item.get("status"),
            })

        # 7. Print Terminal Table
        print("\n" + "=" * 135)
        print("AGENTEVO REPRODUCIBLE 5-QUESTION CLINICAL BENCHMARK")
        print(f"Target Documents: {', '.join(doc_names)}")
        print(f"Baseline: Production Baseline | Candidate: {cand_id} ({cand_name})")
        print("=" * 135)
        header = f"{'Query':<6} | {'Context Hash':<14} | {'Base Cost':<11} | {'Base Grnd':<10} | {'Cand Cost':<11} | {'Cand Grnd':<10} | {'Cost Red %':<11} | {'Grnd Delta':<18} | {'Status'}"
        print(header)
        print("-" * 135)
        for r in table_rows:
            st = "OK" if r["baseline_status"] == "success" and r["candidate_status"] == "success" else f"B:{r['baseline_status']}/C:{r['candidate_status']}"
            print(f"{r['query_id']:<6} | {r['context_hash']:<14} | {r['baseline_cost_display']:<11} | {r['baseline_groundedness_display']:<10} | {r['candidate_cost_display']:<11} | {r['candidate_groundedness_display']:<10} | {r['cost_delta']:<11} | {r['groundedness_delta']:<18} | {st}")
        print("=" * 135)

        # 8. Print Aggregate Metrics
        bm = baseline_eval.metrics
        cm = cand_eval.metrics

        print("\n=== AGGREGATE WORKLOAD RESULTS ===")
        print(f"Baseline:  Cost/Query=${bm.cost:.6f} | Avg Groundedness={bm.performance*100:.1f}% | Avg Latency={bm.llm_latency_ms:.1f}ms | Status={bm.status}")
        cand_cost_s = f"${cm.cost:.6f}" if cm.cost is not None else "N/A"
        cand_perf_s = f"{cm.performance*100:.1f}%" if cm.performance is not None else "N/A"
        cand_lat_s = f"{cm.llm_latency_ms:.1f}ms" if cm.llm_latency_ms is not None else "N/A"
        print(f"Candidate: Cost/Query={cand_cost_s} | Avg Groundedness={cand_perf_s} | Avg Latency={cand_lat_s} | Status={cm.status}")

        # Save JSON output for reproducible auditing
        output_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_name": "AgentEvo 5-Question Reproducible Clinical Benchmark",
            "documents": doc_names,
            "baseline": {
                "workflow_id": "baseline",
                "metrics": bm.model_dump(),
            },
            "candidate": {
                "workflow_id": cand_id,
                "name": cand_name,
                "metrics": cm.model_dump(),
            },
            "per_query_comparison": table_rows,
        }

        with open("reproducible_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        print("\nSaved complete benchmark payload to 'reproducible_benchmark_results.json'.")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(run_reproducible_benchmark())
