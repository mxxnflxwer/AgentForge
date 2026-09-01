import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import SessionLocal
from app.services.retrieval_service import search_similar_chunks
from app.models.document import Document

db = SessionLocal()
doc = db.query(Document).filter(Document.processing_status == "COMPLETED").first()
user_id = doc.owner_id if doc else "test_user"

queries = [
    ("what is the clinical diagnosis?", "Assessment"),
    ("what medications were prescribed?", "Plan"),
    ("did they have edema?", "Physical Examination"),
    ("what is the age of the patient?", "History of Present Illness"),
    ("what is the speciality?", "Document Overview"),
    ("what are the symptoms?", "Chief Complaint"),
    ("what did the ECG show?", "Diagnostic Findings"),
    ("what is google?", "REJECT (0 results)"),
    ("hi", "REJECT (0 results)"),
    ("what is the treatment plan?", "Plan"),
]

print("="*70)
print(f"EVALUATING 10 TEST QUERIES (User: {user_id})")
print("="*70)

for q, expected in queries:
    results = search_similar_chunks(
        user_id=user_id,
        query=q,
        top_k=5,
        db=db,
    )
    print(f"\nQuery: '{q}' -> Expected: [{expected}]")
    print(f"Total retrieved: {len(results)}")
    for i, r in enumerate(results[:3]):
        print(f"  [{i+1}] Sec: '{r.section_title}', Rel: {r.relevance_score:.4f}, Sim: {r.similarity_score:.4f}, Dist: {r.distance:.4f}")
        print(f"      Content: {repr(r.content[:75])}...")

db.close()
