import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import SessionLocal
from app.services.retrieval_service import search_similar_chunks
from app.models.document import Document

db = SessionLocal()
doc = db.query(Document).filter(Document.processing_status == "COMPLETED").first()
user_id = doc.owner_id if doc else "test_user"

short_queries = [
    ("age?", "History of Present Illness"),
    ("diagnosis?", "Assessment"),
    ("symptoms?", "Chief Complaint"),
    ("medications?", "Plan"),
    ("specialty?", "Document Overview"),
    ("edema?", "Physical Examination"),
    ("ecg?", "Diagnostic Findings"),
]

print("="*70)
print("TESTING SHORT VALID QUERIES")
print("="*70)

for q, expected in short_queries:
    results = search_similar_chunks(
        user_id=user_id,
        query=q,
        top_k=3,
        db=db,
    )
    print(f"\nShort Query: '{q}' -> Expected: [{expected}]")
    print(f"Total retrieved: {len(results)}")
    if results:
        print(f"  [1] Sec: '{results[0].section_title}', Rel: {results[0].relevance_score:.4f}, Sim: {results[0].similarity_score:.4f}")
        assert results[0].section_title == expected or expected in (results[0].section_title or ""), f"Expected {expected}, got {results[0].section_title}"

db.close()
print("\nAll short valid queries passed successfully!")
