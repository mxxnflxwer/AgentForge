import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import SessionLocal
from app.models.document import Document, DocumentChunk, DocumentProcessingStatus
from app.services.vector_store import get_vector_store
from app.services.reindex_service import reindex_all_documents
from app.services.retrieval_service import search_similar_chunks

def run_verification():
    db = SessionLocal()
    store = get_vector_store()
    
    # 1. PostgreSQL document count before
    doc_count_before = db.query(Document).count()
    chunk_count_before = db.query(DocumentChunk).count()
    completed_docs = db.query(Document).filter(Document.processing_status == DocumentProcessingStatus.COMPLETED).all()
    print(f"[1] PostgreSQL Documents count: {doc_count_before} ({len(completed_docs)} COMPLETED), Chunks: {chunk_count_before}")
    
    # 2. Reindex all documents with fresh BGE-M3 embeddings
    print("[2] Executing reindex_all_documents(reset_chroma=True)...")
    reindex_res = reindex_all_documents(db=db, vector_store=store, reset_chroma=True)
    print(f"Reindex result: {reindex_res}")
    
    # 3. Verify PostgreSQL document count after
    doc_count_after = db.query(Document).count()
    chunk_count_after = db.query(DocumentChunk).count()
    print(f"[3] PostgreSQL Documents count after: {doc_count_after}, Chunks: {chunk_count_after}")
    assert doc_count_before == doc_count_after, "PostgreSQL document count changed!"
    assert chunk_count_before == chunk_count_after, "PostgreSQL chunk count changed!"
    
    # 4. Chroma vector count
    chroma_count = store.count()
    print(f"[4] ChromaDB total vector count: {chroma_count}")
    
    # 5. Check user 863ffca5-e2c2-4a53-bc58-441debdd5177
    target_user_id = "863ffca5-e2c2-4a53-bc58-441debdd5177"
    user_docs = db.query(Document).filter(Document.owner_id == target_user_id).all()
    print(f"[5] User '{target_user_id}' has {len(user_docs)} documents in PostgreSQL.")
    
    # Check user vectors in Chroma
    try:
        user_vectors = store._collection.get(where={"user_id": {"$eq": target_user_id}})
        user_vector_count = len(user_vectors["ids"]) if user_vectors and user_vectors.get("ids") else 0
        print(f"    User '{target_user_id}' ChromaDB vector count: {user_vector_count}")
    except Exception as e:
        print(f"    Error querying user vectors: {e}")
        user_vector_count = 0
        
    # 6. Test relevant query: What is the diagnosis?
    print("\n[6] Test Query: 'What is the diagnosis?'")
    res_diag = search_similar_chunks(
        user_id=target_user_id if user_docs else (completed_docs[0].owner_id if completed_docs else "test"),
        query="What is the diagnosis?",
        top_k=3,
        db=db,
    )
    print(f"    Total retrieved chunks: {len(res_diag)}")
    for i, r in enumerate(res_diag):
        print(f"    [{i+1}] Section: '{r.section_title}', RelScore: {r.relevance_score}, SimScore: {r.similarity_score}")
        print(f"        Content: {r.content[:90]}...")
        
    # 7. Test relevant query: What are the symptoms?
    print("\n[7] Test Query: 'What are the symptoms?'")
    res_symp = search_similar_chunks(
        user_id=target_user_id if user_docs else (completed_docs[0].owner_id if completed_docs else "test"),
        query="What are the symptoms?",
        top_k=3,
        db=db,
    )
    print(f"    Total retrieved chunks: {len(res_symp)}")
    for i, r in enumerate(res_symp):
        print(f"    [{i+1}] Section: '{r.section_title}', RelScore: {r.relevance_score}, SimScore: {r.similarity_score}")
        print(f"        Content: {r.content[:90]}...")

    # 8. Test irrelevant query: What is Google?
    print("\n[8] Test Query: 'What is Google?'")
    res_google = search_similar_chunks(
        user_id=target_user_id if user_docs else (completed_docs[0].owner_id if completed_docs else "test"),
        query="What is Google?",
        top_k=3,
        db=db,
    )
    print(f"    Total retrieved chunks: {len(res_google)} (Expected: 0)")
    assert len(res_google) == 0, f"Expected 0 results for 'What is Google?', got {len(res_google)}"
    
    # 9. Test other irrelevant queries
    for q in ["abcdefxyz123", "What is the weather?", "Explain quantum computing", "Who is Elon Musk?"]:
        res_irr = search_similar_chunks(
            user_id=target_user_id if user_docs else (completed_docs[0].owner_id if completed_docs else "test"),
            query=q,
            top_k=3,
            db=db,
        )
        print(f"    Query '{q}': {len(res_irr)} results (Expected: 0)")
        assert len(res_irr) == 0, f"Expected 0 results for '{q}', got {len(res_irr)}"

    db.close()
    print("\nAll verifications passed successfully!")

if __name__ == "__main__":
    run_verification()
