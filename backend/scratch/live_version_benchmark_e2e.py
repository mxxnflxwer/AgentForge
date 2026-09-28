import json
import requests
import sys

BASE_URL = "http://localhost:8000"

def main():
    print("=" * 85)
    print("STARTING REAL AGENTEVO MULTI-VERSION BENCHMARK (v1 - v6 + BASELINE)")
    print("=" * 85)

    # 1. Authenticate
    session = requests.Session()
    login_resp = session.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": "user@example.com", "password": "Password123!"},
    )
    if login_resp.status_code != 200:
        print(f"FAILED: Login error {login_resp.status_code}: {login_resp.text}")
        sys.exit(1)

    print("1. Authenticated as user@example.com")

    # 2. Get cardiology document
    docs_resp = session.get(f"{BASE_URL}/api/documents")
    if docs_resp.status_code != 200:
        print(f"FAILED: /api/documents error {docs_resp.status_code}")
        sys.exit(1)

    docs = docs_resp.json()
    cardio_doc = next((d for d in docs if "Cardiology" in (d.get("filename") or d.get("original_filename") or "")), None)
    doc_id = cardio_doc["id"] if cardio_doc else None
    print(f"2. Target Document: {cardio_doc.get('filename') if cardio_doc else 'Global'} (ID: {doc_id})")

    # 3. Retrieve all approved workflow versions from DB
    vers_resp = session.get(f"{BASE_URL}/api/agent-evo/versions")
    if vers_resp.status_code != 200:
        print(f"FAILED: /api/agent-evo/versions error {vers_resp.status_code}")
        sys.exit(1)

    versions_data = vers_resp.json().get("versions", [])
    print(f"3. Found {len(versions_data)} approved workflow versions in PostgreSQL:")
    for v in sorted(versions_data, key=lambda x: x["version_number"]):
        print(f"   - v{v['version_number']}: {v['version_id']} | {v['name']} ({v['model_id']})")

    selected_version_ids = ["baseline"] + [v["version_id"] for v in sorted(versions_data, key=lambda x: x["version_number"])]
    print(f"\n4. Benchmarking {len(selected_version_ids)} versions head-to-head under identical conditions:")
    print(f"   Versions: {selected_version_ids}")

    # 4. Run Benchmark
    benchmark_payload = {
        "version_ids": selected_version_ids,
        "name": "Cardiology Diagnosis Multi-Version Benchmark (Baseline + v1-v6)",
        "preset_name": "clinical",
        "query": "What is the primary clinical diagnosis and assessment for this patient?",
        "document_id": doc_id,
        "expected_answer": "Exertional angina pectoris consistent with underlying coronary artery disease (CAD), along with uncontrolled hyperlipidemia.",
    }

    print("\n   Sending POST /api/agent-evo/benchmark ...")
    bench_resp = session.post(
        f"{BASE_URL}/api/agent-evo/benchmark",
        json=benchmark_payload,
        timeout=180,
    )

    if bench_resp.status_code != 200:
        print(f"FAILED: /api/agent-evo/benchmark returned {bench_resp.status_code}: {bench_resp.text}")
        sys.exit(1)

    bench_report = bench_resp.json()
    print("\n" + "=" * 85)
    print(f"BENCHMARK COMPLETED: {bench_report['benchmark_id']}")
    print("=" * 85)
    print(f"Summary: {bench_report['summary']}")

    # 5. Print Comparison Table
    print("\n" + "=" * 115)
    fmt = "{:<8} | {:<30} | {:<12} | {:<12} | {:<12} | {:<14} | {:<10} | {:<10} | {:<15}"
    print(fmt.format("Version", "Workflow Name", "Model", "Groundedness", "Hallucination", "Cost / Query", "Latency", "Tokens", "Pareto Status"))
    print("-" * 115)

    for r in bench_report["results"]:
        ver_tag = "Base" if r["version_id"] == "baseline" else f"v{r['version_number']}"
        g_str = f"{(r['groundedness'] * 100):.1f}%"
        h_str = f"{(r['hallucination_rate'] * 100):.1f}%"
        cost_str = f"${r['cost']:.6f}"
        lat_str = f"{r['llm_latency_ms']:.0f} ms"
        tok_str = str(r["total_tokens"])
        model_str = r["model_id"][:12]
        print(fmt.format(ver_tag, r["workflow_name"][:30], model_str, g_str, h_str, cost_str, lat_str, tok_str, r["pareto_status"]))

    # 6. Lineage Graph Verification
    print("\n" + "=" * 85)
    print("6. VERIFYING WORKFLOW LINEAGE GRAPH FROM /api/agent-evo/lineage")
    print("=" * 85)
    lin_resp = session.get(f"{BASE_URL}/api/agent-evo/lineage")
    if lin_resp.status_code == 200:
        lin_data = lin_resp.json()
        print(f"Total Nodes: {len(lin_data['nodes'])}, Total Edges: {len(lin_data['edges'])}")
        print("\nDirected Edges (Parent -> Child):")
        for e in lin_data["edges"]:
            print(f"   {e['source']}  --[{e.get('label') or 'mutation'}]-->  {e['target']}")

    # 7. Inspection Verification
    print("\n" + "=" * 85)
    print("7. VERIFYING INSPECT ENDPOINT FOR v6 (COST-OPTIMIZED COMPOSITE)")
    print("=" * 85)
    v6_id = next((v["version_id"] for v in versions_data if v["version_number"] == 6), None)
    if v6_id:
        insp_resp = session.get(f"{BASE_URL}/api/agent-evo/versions/{v6_id}/inspect")
        if insp_resp.status_code == 200:
            insp = insp_resp.json()
            print(f"Version ID: {insp['version_id']} (v{insp['version_number']})")
            print(f"Parent: {insp['parent_workflow_id']}")
            print(f"Mutation Type: {insp['mutation_type']}")
            print(f"Fields Changed: {insp['fields_changed']}")
            print(f"Rationale: {insp['rationale']}")
            print(f"Approved By: {insp['approved_by']} at {insp['approved_at']}")

if __name__ == "__main__":
    main()
