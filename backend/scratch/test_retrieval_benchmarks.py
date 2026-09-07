import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from app.models.document import DocumentChunk
from app.services.vector_store import VectorStore
from app.services.retrieval_service import execute_rag_retrieval

def run_benchmarks():
    temp_store = VectorStore(
        persist_directory="storage/chromadb_benchmark_test",
        collection_name=f"benchmark_test_{uuid.uuid4().hex[:8]}"
    )
    user_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=0,
            content="Specialty: Cardiology\nReport Type: Consultation History and Physical",
            section_title="Document Overview",
            token_count=10,
            character_count=70,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=1,
            content="Chief Complaint\nPatient presents with intermittent chest discomfort and shortness of breath on exertion.",
            section_title="Chief Complaint",
            token_count=15,
            character_count=110,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=2,
            content="History of Present Illness\nA 58-year-old presents with substernal chest tightness occurring with moderate exertion.",
            section_title="History of Present Illness",
            token_count=15,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=3,
            content="Physical Examination\nBlood pressure 138/86, heart rate 78 bpm regular, respiratory rate 16. Lungs clear. No peripheral edema.",
            section_title="Physical Examination",
            token_count=20,
            character_count=130,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=4,
            content="Diagnostic Findings\nECG shows normal sinus rhythm with no acute ST-T wave changes. Troponin I negative. Lipid panel shows LDL 162 mg/dL, HDL 38 mg/dL.",
            section_title="Diagnostic Findings",
            token_count=25,
            character_count=160,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=5,
            content="Assessment\nFindings are consistent with stable angina, likely related to underlying coronary artery disease.",
            section_title="Assessment",
            token_count=16,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=6,
            content="Plan\nRecommend coronary CT angiography for further risk stratification. Start atorvastatin 40mg daily and low-dose aspirin.",
            section_title="Plan",
            token_count=20,
            character_count=135,
        ),
    ]

    temp_store.upsert_document_chunks(user_id=user_id, document_id=doc_id, chunks=chunks)

    queries = [
        ("1. Age", "What is the patient's age?", "58-year-old", "success", "History of Present Illness"),
        ("2. Diagnosis", "What is the patient's diagnosis?", "stable angina", "success", "Assessment"),
        ("3. Blood Pressure", "What is the patient's blood pressure?", "138/86", "success", "Physical Examination"),
        ("4. Heart Rate", "What is the patient's heart rate?", "78 bpm", "success", "Physical Examination"),
        ("5. LDL Level", "What is the LDL level?", "162 mg/dl", "success", "Diagnostic Findings"),
        ("6. HDL Level", "What is the HDL level?", "38 mg/dl", "success", "Diagnostic Findings"),
        ("7. ECG", "What did the ECG show?", "normal sinus rhythm", "success", "Diagnostic Findings"),
        ("8. Important Findings", "What were the important findings?", "ecg", "success", "Diagnostic Findings"),
        ("9. Blood Glucose (Not Found)", "What is the patient's blood glucose level?", None, "not_found", None),
        ("10. Phone Number (Not Found)", "What is the patient's phone number?", None, "not_found", None),
        ("11. Address (Not Found)", "What is the patient's address?", None, "not_found", None),
        ("12. Summary", "Summarize the medical report", None, "success", None),
    ]

    print("=================== RUNNING RETRIEVAL BENCHMARKS ===================")
    all_passed = True
    for label, q, expected_text, expected_status, expected_section in queries:
        resp = execute_rag_retrieval(
            user_id=user_id,
            query=q,
            document_id=doc_id,
            vector_store=temp_store,
        )
        print(f"\n[{label}] Query: '{q}'")
        print(f"Status: {resp.status} | Intent: {resp.intent} | Results Count: {resp.total_results}")
        if resp.debug:
            print(f"Debug: {resp.debug.model_dump()}")

        if resp.status != expected_status:
            print(f"FAILED: Expected status '{expected_status}', got '{resp.status}'")
            all_passed = False
            continue

        if expected_status == "not_found":
            if resp.total_results != 0 or len(resp.results) != 0:
                print(f"FAILED: Expected 0 results, got {resp.total_results}")
                all_passed = False
            elif resp.message != "The requested information was not found in the uploaded document.":
                print(f"FAILED: Missing or invalid not_found message: {resp.message}")
                all_passed = False
            else:
                print("PASSED (Correctly rejected)")

        elif label == "12. Summary":
            if resp.intent != "summary":
                print(f"FAILED: Expected intent 'summary', got '{resp.intent}'")
                all_passed = False
            elif not resp.sections or len(resp.sections) < 5:
                print(f"FAILED: Expected multiple sections, got {len(resp.sections) if resp.sections else 0}")
                all_passed = False
            elif not resp.summary_context:
                print("FAILED: summary_context is empty")
                all_passed = False
            else:
                print(f"PASSED (Summary created with {len(resp.sections)} sections)")
                print("Summary context snippet:\n" + resp.summary_context[:200] + "...")

        else:
            if resp.total_results == 0:
                print(f"FAILED: No results returned for factual query")
                all_passed = False
            else:
                top_chunk = resp.results[0]
                content_lower = top_chunk.content.lower()
                section = top_chunk.section_title
                if expected_text and expected_text.lower() not in content_lower:
                    print(f"FAILED: Expected '{expected_text}' in chunk content:\n{top_chunk.content}")
                    all_passed = False
                elif expected_section and section != expected_section:
                    print(f"WARNING/FAILED: Expected section '{expected_section}', got '{section}'")
                    all_passed = False
                else:
                    print(f"PASSED (Rank #1: [{section}] Relevance: {top_chunk.relevance_score})")

    print("\n====================================================================")
    if all_passed:
        print("ALL 12 BENCHMARK QUERIES PASSED PERFECTLY!")
    else:
        print("SOME BENCHMARK QUERIES FAILED.")
    return all_passed

if __name__ == "__main__":
    success = run_benchmarks()
    sys.exit(0 if success else 1)
