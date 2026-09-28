import json
import requests
import sys

BASE_URL = "http://localhost:8000"

def main():
    print("=" * 90)
    print("STARTING REAL AGENTEVO CLINICAL DIAGNOSIS & MEDICATIONS WORKFLOW OPTIMIZATION")
    print("=" * 90)

    # 1. Authenticate
    session = requests.Session()
    login_resp = session.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": "user@example.com", "password": "Password123!"},
    )
    if login_resp.status_code != 200:
        print(f"FAILED: Login error {login_resp.status_code}: {login_resp.text}")
        sys.exit(1)

    print("1. Authenticated as user@example.com (DEVELOPER)")

    # 2. Identify cardiology document
    docs_resp = session.get(f"{BASE_URL}/api/documents")
    if docs_resp.status_code != 200:
        print(f"FAILED: /api/documents error {docs_resp.status_code}")
        sys.exit(1)

    docs = docs_resp.json()
    cardio_doc = next((d for d in docs if "Cardiology" in (d.get("filename") or d.get("original_filename") or "")), None)
    doc_id = cardio_doc["id"] if cardio_doc else None
    print(f"2. Target Document: {cardio_doc.get('filename') if cardio_doc else 'Global'} (ID: {doc_id})")

    # 3. Target Task Query
    query = "What is the primary clinical diagnosis, prescribed medications, and detailed treatment plan for this patient?"
    expected_answer = (
        "Primary diagnosis: Exertional angina pectoris consistent with coronary artery disease (CAD) and uncontrolled hyperlipidemia. "
        "Medications and plan: Aspirin 81 mg daily, Atorvastatin 40 mg daily, Metoprolol succinate 50 mg daily, sublingual nitroglycerin PRN, "
        "and scheduled outpatient cardiac catheterization."
    )

    print(f"3. Target Query: '{query}'")
    print(f"   Parent Workflow: wf_v6_a8b827 (Cost-Optimized Composite)")

    # 4. Run Optimization from Parent Workflow wf_v6_a8b827
    opt_payload = {
        "query": query,
        "document_id": doc_id,
        "expected_answer": expected_answer,
        "candidate_count": 6,
        "seed": 101,
        "parent_workflow_id": "wf_v6_a8b827",
    }

    print("\n4. Sending POST /api/agent-evo/optimize ...")
    opt_resp = session.post(
        f"{BASE_URL}/api/agent-evo/optimize",
        json=opt_payload,
        timeout=180,
    )
    if opt_resp.status_code != 200:
        print(f"FAILED: /api/agent-evo/optimize error {opt_resp.status_code}: {opt_resp.text}")
        sys.exit(1)

    report = opt_resp.json()
    run_id = report["run_id"]
    print(f"   Optimization Run Completed: {run_id}")
    print(f"   Total Candidates Evaluated: {report['total_candidates_evaluated']}")
    print(f"   Pareto Frontier Count: {report['pareto_frontier_count']}")

    print("\n" + "=" * 115)
    print(f"OPTIMIZATION CANDIDATE EVALUATION TABLE (RUN: {run_id})")
    print("=" * 115)
    fmt = "{:<10} | {:<32} | {:<12} | {:<12} | {:<14} | {:<10} | {:<10} | {:<15}"
    print(fmt.format("Role", "Workflow Name", "Model", "Groundedness", "Cost / Query", "Latency", "Tokens", "Pareto Status"))
    print("-" * 115)

    for c in report["all_candidates"]:
        role = "Parent/Base" if c["candidate_id"] in ("baseline", "wf_v6_a8b827") else "Candidate"
        g_val = c["metrics"].get("groundedness")
        perf_str = f"{((g_val if g_val is not None else c['metrics']['performance']) * 100):.1f}%"
        cost_str = f"${c['metrics']['cost']:.6f}"
        lat_str = f"{c['metrics']['llm_latency_ms']:.0f} ms"
        tok_str = str(c["metrics"]["total_tokens"])
        model_str = c["workflow"]["model"]["model_id"][:12]
        pareto_str = "YES (Pareto)" if c["is_pareto_optimal"] else "NO (Dominated)"
        print(fmt.format(role, c["workflow"]["name"][:32], model_str, perf_str, cost_str, lat_str, tok_str, pareto_str))

    # 5. Extract Best Pareto Candidate
    pareto_candidates = [c for c in report["pareto_frontier"] if c["candidate_id"] not in ("baseline", "wf_v6_a8b827")]
    target_cand = pareto_candidates[0] if pareto_candidates else report["pareto_frontier"][0]
    cand_id = target_cand["candidate_id"]

    print("\n" + "=" * 90)
    print(f"5. PARETO-OPTIMAL CANDIDATE IDENTIFIED: {cand_id} ({target_cand['workflow']['name']})")
    print("=" * 90)
    meta = target_cand.get("mutation_metadata") or {}
    print(f"   Mutation Strategy: {meta.get('mutation_type')}")
    print(f"   Rationale: {meta.get('mutation_description')}")
    print(f"   Model: {target_cand['workflow']['model']['model_id']} (Temp: {target_cand['workflow']['model']['temperature']})")
    print(f"   Retrieval: top_k={target_cand['workflow']['retrieval']['top_k']}, threshold={target_cand['workflow']['retrieval']['similarity_threshold']}")
    print(f"   Context Compression: {target_cand['workflow']['execution'].get('context_compression')}")

    # 6. Benchmark Newly Discovered Candidate against All Approved Versions (v1-v6 + Baseline)
    print("\n" + "=" * 90)
    print("6. EXECUTING CONTROLLED HEAD-TO-HEAD BENCHMARK (ALL APPROVED VERSIONS + PARETO CANDIDATE)")
    print("=" * 90)

    # First approve the candidate to create v7 in PostgreSQL database
    appr_resp = session.post(
        f"{BASE_URL}/api/agent-evo/runs/{run_id}/candidates/{cand_id}/approve",
    )
    if appr_resp.status_code != 200:
        print(f"FAILED: Approval returned {appr_resp.status_code}: {appr_resp.text}")
        sys.exit(1)

    appr_data = appr_resp.json()
    new_version = appr_data.get("version")
    new_version_id = new_version["version_id"]
    print(f"   Candidate approved as new version: {new_version_id} (v{new_version['version_number']})")

    # Retrieve all versions from DB
    vers_resp = session.get(f"{BASE_URL}/api/agent-evo/versions")
    all_db_versions = vers_resp.json().get("versions", [])
    benchmark_v_ids = ["baseline"] + [v["version_id"] for v in sorted(all_db_versions, key=lambda x: x["version_number"])]

    print(f"   Running benchmark across {len(benchmark_v_ids)} versions: {benchmark_v_ids}")

    bench_payload = {
        "version_ids": benchmark_v_ids,
        "name": f"Clinical Diagnosis & Medications Benchmark (Baseline + v1-v{new_version['version_number']})",
        "preset_name": "medications",
        "query": query,
        "document_id": doc_id,
        "expected_answer": expected_answer,
    }

    bench_resp = session.post(
        f"{BASE_URL}/api/agent-evo/benchmark",
        json=bench_payload,
        timeout=180,
    )
    if bench_resp.status_code != 200:
        print(f"FAILED: Benchmark failed with {bench_resp.status_code}: {bench_resp.text}")
        sys.exit(1)

    final_bench = bench_resp.json()
    print("\n" + "=" * 115)
    print(f"CONTROLLED HEAD-TO-HEAD BENCHMARK RESULTS (BENCHMARK ID: {final_bench['benchmark_id']})")
    print("=" * 115)
    fmt_b = "{:<8} | {:<32} | {:<12} | {:<12} | {:<12} | {:<14} | {:<10} | {:<10} | {:<15}"
    print(fmt_b.format("Version", "Workflow Name", "Model", "Groundedness", "Hallucination", "Cost / Query", "Latency", "Tokens", "Pareto Status"))
    print("-" * 115)

    for r in final_bench["results"]:
        ver_label = "Base" if r["version_id"] == "baseline" else f"v{r['version_number']}"
        g_s = f"{(r['groundedness'] * 100):.1f}%"
        h_s = f"{(r['hallucination_rate'] * 100):.1f}%"
        c_s = f"${r['cost']:.6f}"
        l_s = f"{r['llm_latency_ms']:.0f} ms"
        t_s = str(r["total_tokens"])
        m_s = r["model_id"][:12]
        print(fmt_b.format(ver_label, r["workflow_name"][:32], m_s, g_s, h_s, c_s, l_s, t_s, r["pareto_status"]))

    print("\n" + "=" * 90)
    print("WORKFLOW OPTIMIZATION AND BENCHMARK VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 90)

if __name__ == "__main__":
    main()
