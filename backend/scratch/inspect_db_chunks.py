import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.database import SessionLocal
from app.models.document import Document, DocumentChunk
from app.services.reranker import LocalSemanticReranker

db = SessionLocal()
chunks = db.query(DocumentChunk).all()
ranker = LocalSemanticReranker()
query = "What are the symptoms?"
tokens = ["symptoms"]

print(f"Total chunks in DB: {len(chunks)}")
for c in chunks:
    c_dict = {
        "section_title": c.section_title or "",
        "content": c.content,
        "similarity_score": 0.53,
    }
    score = ranker.compute_chunk_score(query, c_dict)
    intent = ranker.compute_intent_score(tokens, c.content.lower(), (c.section_title or "").lower())
    lexical = ranker.compute_lexical_score(tokens, c.content.lower(), (c.section_title or "").lower())
    print(f"ID: {c.document_id[:8]}.. Sec: {c.section_title:20} -> Score: {score:.4f}, Intent: {intent:.2f}, Lexical: {lexical:.2f}")
    if intent > 0 or score > 0.4:
        print(f"    Content preview: {c.content[:80]}...")
db.close()
