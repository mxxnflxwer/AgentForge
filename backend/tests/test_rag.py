import io
import os
import uuid
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.models.document import Document, DocumentChunk, DocumentProcessingStatus
from app.models.user import User
from app.services.embedding_service import BaseEmbeddingService, LocalChromaEmbeddingService, SentenceTransformerEmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store
from app.services.retrieval_service import search_similar_chunks
from app.services.document.pipeline import get_document_pipeline


def register_and_login(client: TestClient, email: str, name: str = "Test User") -> str:
    """Helper to register and log in a user, returning session cookie."""
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPassword123!", "name": name},
    )
    res = client.post(
        "/api/auth/login",
        json={"email": email, "password": "StrongPassword123!"},
    )
    assert res.status_code == 200
    return res.cookies.get("session_id")


@pytest.fixture
def temp_vector_store(tmp_path):
    """Provides an isolated VectorStore instance in a temp directory."""
    persist_dir = str(tmp_path / "chromadb_test")
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    return VectorStore(persist_directory=persist_dir, collection_name=collection_name)


# 1. Embedding Output and Dimensions
def test_embedding_output_and_dimensions():
    service = get_embedding_service()
    assert isinstance(service, BaseEmbeddingService)

    # Single text embedding
    text = "Machine learning and medical AI diagnostic workflow."
    embedding = service.embed_text(text)
    assert isinstance(embedding, list)
    assert len(embedding) in (1024, 384)  # 1024 for BGE-M3, 384 for MiniLM
    assert all(isinstance(val, float) for val in embedding)

    # Batch embedding
    batch_texts = ["First clinical trial chunk", "Second diagnostic imaging chunk"]
    batch_embeddings = service.embed_batch(batch_texts)
    assert len(batch_embeddings) == 2
    assert len(batch_embeddings[0]) in (1024, 384)
    assert len(batch_embeddings[1]) in (1024, 384)

    # Empty text handling
    assert service.embed_text("") == []
    assert service.embed_batch([]) == []


# 2. Vector Upsert and Metadata Preservation
def test_vector_upsert_and_metadata(temp_vector_store):
    user_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    chunk1_id = str(uuid.uuid4())
    chunk2_id = str(uuid.uuid4())

    chunk1 = DocumentChunk(
        id=chunk1_id,
        document_id=doc_id,
        chunk_index=0,
        content="Patient presents with acute chest pain and shortness of breath.",
        section_title="CHIEF COMPLAINT",
        token_count=12,
        character_count=65,
    )
    chunk2 = DocumentChunk(
        id=chunk2_id,
        document_id=doc_id,
        chunk_index=1,
        content="Prescribed nitroglycerin and ordered an immediate ECG.",
        section_title="TREATMENT PLAN",
        token_count=8,
        character_count=54,
    )

    count = temp_vector_store.upsert_document_chunks(
        user_id=user_id,
        document_id=doc_id,
        chunks=[chunk1, chunk2],
    )
    assert count == 2

    # Check raw Chroma collection contents
    raw_data = temp_vector_store._collection.get(include=["metadatas", "documents"])
    assert len(raw_data["ids"]) == 2
    assert raw_data["metadatas"][0]["user_id"] == user_id
    assert raw_data["metadatas"][0]["document_id"] == doc_id
    assert "chunk_index" in raw_data["metadatas"][0]


# 3. Semantic Retrieval and Top-K Limit
def test_semantic_retrieval_and_top_k(temp_vector_store):
    user_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=0,
            content="Cardiovascular hypertension and arterial blood pressure management guidelines.",
            section_title="CARDIOLOGY",
            token_count=10,
            character_count=75,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=1,
            content="Severe neurological migraine episodes with visual aura and photophobia.",
            section_title="NEUROLOGY",
            token_count=10,
            character_count=72,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=2,
            content="Pediatric respiratory asthma inhaler dosages and nebulizer instructions.",
            section_title="PEDIATRICS",
            token_count=10,
            character_count=73,
        ),
    ]
    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc_id, chunks=chunks)

    # Query specifically for migraine / headache
    results = temp_vector_store.query_similar_chunks(
        user_id=user_id,
        query_text="headache and visual aura symptoms",
        top_k=1,
    )
    assert len(results) == 1
    assert "migraine" in results[0]["content"].lower()
    assert results[0]["section_title"] == "NEUROLOGY"
    assert results[0]["similarity_score"] > 0.4
    assert results[0]["distance"] >= 0.0

    # Test top_k constraint
    results_k2 = temp_vector_store.query_similar_chunks(
        user_id=user_id,
        query_text="medical treatments",
        top_k=2,
    )
    assert len(results_k2) == 2


# 4. Strict User Isolation
def test_strict_user_isolation(temp_vector_store):
    user_a_id = str(uuid.uuid4())
    user_b_id = str(uuid.uuid4())

    doc_a_id = str(uuid.uuid4())
    doc_b_id = str(uuid.uuid4())

    chunk_a = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc_a_id,
        chunk_index=0,
        content="Confidential financial report of User A regarding Q3 earnings.",
        section_title="FINANCE",
        token_count=10,
        character_count=65,
    )
    chunk_b = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc_b_id,
        chunk_index=0,
        content="Public Python documentation on asyncio concurrency for User B.",
        section_title="DEV",
        token_count=10,
        character_count=63,
    )

    temp_vector_store.upsert_document_chunks(user_id=user_a_id, document_id=doc_a_id, chunks=[chunk_a])
    temp_vector_store.upsert_document_chunks(user_id=user_b_id, document_id=doc_b_id, chunks=[chunk_b])

    # User B should never receive User A's chunk
    results_b = temp_vector_store.query_similar_chunks(
        user_id=user_b_id,
        query_text="Confidential financial earnings",
        top_k=5,
    )
    assert all(r["document_id"] != doc_a_id and "financial report" not in r["content"].lower() for r in results_b)

    # User C who has no indexed documents receives 0 results
    user_c_id = str(uuid.uuid4())
    results_c = temp_vector_store.query_similar_chunks(
        user_id=user_c_id,
        query_text="Confidential financial earnings",
        top_k=5,
    )
    assert len(results_c) == 0

    # User A should receive their own chunk
    results_a = temp_vector_store.query_similar_chunks(
        user_id=user_a_id,
        query_text="Confidential financial earnings",
        top_k=5,
    )
    assert len(results_a) == 1
    assert results_a[0]["document_id"] == doc_a_id
    assert "financial report" in results_a[0]["content"].lower()


# 5. Document Isolation (Scoped Retrieval)
def test_document_isolation(temp_vector_store):
    user_id = str(uuid.uuid4())
    doc_1_id = str(uuid.uuid4())
    doc_2_id = str(uuid.uuid4())

    chunk_1 = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc_1_id,
        chunk_index=0,
        content="Document 1: Deep learning architectures and transformer attention.",
        section_title="AI",
        token_count=10,
        character_count=68,
    )
    chunk_2 = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc_2_id,
        chunk_index=0,
        content="Document 2: Classical database normalization and indexing.",
        section_title="DB",
        token_count=10,
        character_count=60,
    )

    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc_1_id, chunks=[chunk_1])
    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc_2_id, chunks=[chunk_2])

    # Query scoped to Document 2 should ONLY return chunks from Document 2 and never Doc 1
    results_scoped = temp_vector_store.query_similar_chunks(
        user_id=user_id,
        query_text="transformer attention",
        document_id=doc_2_id,
        top_k=5,
    )
    assert all(r["document_id"] == doc_2_id for r in results_scoped)
    assert all(r["document_id"] != doc_1_id for r in results_scoped)

    results_scoped_doc1 = temp_vector_store.query_similar_chunks(
        user_id=user_id,
        query_text="transformer attention",
        document_id=doc_1_id,
        top_k=5,
    )
    assert len(results_scoped_doc1) == 1
    assert results_scoped_doc1[0]["document_id"] == doc_1_id



# 6. Nonexistent / Unowned Document Ownership Validation
def test_nonexistent_or_unowned_document_404(db_session):
    user_a = User(
        id=str(uuid.uuid4()),
        name="User A",
        email="usera_rag@example.com",
        password_hash="hash",
    )
    user_b = User(
        id=str(uuid.uuid4()),
        name="User B",
        email="userb_rag@example.com",
        password_hash="hash",
    )
    doc_a = Document(
        id=str(uuid.uuid4()),
        owner_id=user_a.id,
        original_filename="doc_a.txt",
        stored_filename="doc_a_stored.txt",
        file_type="txt",
        file_size=100,
        mime_type="text/plain",
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    db_session.add_all([user_a, user_b, doc_a])
    db_session.commit()

    # Nonexistent document ID raises 404
    with pytest.raises(HTTPException) as exc_info:
        search_similar_chunks(
            user_id=user_a.id,
            query="test query",
            document_id="00000000-0000-0000-0000-000000000000",
            db=db_session,
        )
    assert exc_info.value.status_code == 404

    # Document owned by User A queried by User B raises 404
    with pytest.raises(HTTPException) as exc_info_unowned:
        search_similar_chunks(
            user_id=user_b.id,
            query="test query",
            document_id=doc_a.id,
            db=db_session,
        )
    assert exc_info_unowned.value.status_code == 404


# 7. Empty and Whitespace Query Validation
def test_empty_and_whitespace_query_validation(db_session):
    user_id = str(uuid.uuid4())

    with pytest.raises(HTTPException) as exc1:
        search_similar_chunks(user_id=user_id, query="", db=db_session)
    assert exc1.value.status_code == 400

    with pytest.raises(HTTPException) as exc2:
        search_similar_chunks(user_id=user_id, query="    ", db=db_session)
    assert exc2.value.status_code == 400


# 8. API Endpoint /api/rag/search (Authenticated vs Unauthenticated)
def test_api_rag_search_unauthenticated(client: TestClient):
    response = client.post(
        "/api/rag/search",
        json={"query": "test query", "top_k": 5},
    )
    assert response.status_code == 401


def test_api_rag_search_authenticated_end_to_end(client: TestClient, db_session):
    cookie = register_and_login(client, "rag_api_user@example.com", "RAG Tester")
    client.cookies.set("session_id", cookie)

    # 1. Upload a document
    txt_content = (
        "SECTION: PATIENT DIAGNOSIS\n"
        "The patient was diagnosed with severe acute bronchitis.\n"
        "Antibiotic treatment with amoxicillin was initiated for 10 days."
    )
    upload_res = client.post(
        "/api/documents",
        files={"file": ("bronchitis_report.txt", io.BytesIO(txt_content.encode("utf-8")), "text/plain")},
        data={"specialty": "Pulmonology"},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # 2. Process document pipeline synchronously to index in ChromaDB
    pipeline = get_document_pipeline()
    pipeline.process_document(db=db_session, document_id=doc_id)

    # 3. Perform RAG search across all documents
    search_res = client.post(
        "/api/rag/search",
        json={"query": "bronchitis antibiotic prescription", "top_k": 3},
    )
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["query"] == "bronchitis antibiotic prescription"
    assert data["total_results"] >= 1
    assert len(data["results"]) >= 1

    first_chunk = data["results"][0]
    assert first_chunk["document_id"] == doc_id
    assert "amoxicillin" in first_chunk["content"].lower() or "bronchitis" in first_chunk["content"].lower()
    assert first_chunk["similarity_score"] > 0
    assert first_chunk["distance"] >= 0

    # 4. Search with valid scoped document_id
    scoped_res = client.post(
        "/api/rag/search",
        json={"query": "antibiotic", "document_id": doc_id, "top_k": 2},
    )
    assert scoped_res.status_code == 200
    assert scoped_res.json()["total_results"] >= 1

    # 5. Search with nonexistent document_id returns 404
    bad_doc_res = client.post(
        "/api/rag/search",
        json={"query": "antibiotic", "document_id": str(uuid.uuid4())},
    )
    assert bad_doc_res.status_code == 404

    # 6. Search with empty query returns 422 (Pydantic min_length=1) or 400
    empty_query_res = client.post(
        "/api/rag/search",
        json={"query": "", "top_k": 5},
    )
    assert empty_query_res.status_code in (400, 422)


from app.services.reindex_service import reindex_all_documents, reindex_document



# 10. Reindex Service Synchronization Test
def test_reindex_service(db_session, temp_vector_store):
    user_id = str(uuid.uuid4())
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user_id,
        original_filename="reindex_test.txt",
        stored_filename="reindex_stored.txt",
        file_type="txt",
        file_size=120,
        mime_type="text/plain",
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    chunk1 = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        chunk_index=0,
        content="PostgreSQL source of truth chunk 1 for reindexing test.",
        section_title="OVERVIEW",
        token_count=10,
        character_count=55,
    )
    chunk2 = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        chunk_index=1,
        content="PostgreSQL source of truth chunk 2 for reindexing test.",
        section_title="DETAILS",
        token_count=10,
        character_count=55,
    )
    db_session.add_all([doc, chunk1, chunk2])
    db_session.commit()

    # Run reindex
    summary = reindex_all_documents(db=db_session, vector_store=temp_vector_store)
    assert summary["documents_processed"] >= 1
    assert summary["total_vectors_indexed"] >= 2

    # Check that chunks are searchable in ChromaDB
    results = temp_vector_store.query_similar_chunks(
        user_id=user_id,
        query_text="source of truth",
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["document_id"] == doc.id


# 11. Disease Query and Assessment Chunk Ranking Test
def test_disease_query_assessment_chunk_ranking(temp_vector_store, db_session):
    user_id = str(uuid.uuid4())
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user_id,
        original_filename="Cardiology_Report.txt",
        stored_filename="cardio_stored.txt",
        file_type="txt",
        file_size=500,
        mime_type="text/plain",
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    db_session.add(doc)
    db_session.commit()

    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=0,
            content="Specialty: Cardiology\nReport Type: Consultation History and Physical",
            section_title="Document Overview",
            token_count=10,
            character_count=65,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=1,
            content="Chief Complaint\nPatient presents with intermittent chest discomfort and shortness of breath on exertion.",
            section_title="Chief Complaint",
            token_count=15,
            character_count=105,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=2,
            content="History of Present Illness\nA 58-year-old presents with substernal chest tightness occurring with moderate exertion.",
            section_title="History of Present Illness",
            token_count=20,
            character_count=120,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=3,
            content="Physical Examination\nBlood pressure 138/86, heart rate 78 bpm regular. Lungs clear to auscultation.",
            section_title="Physical Examination",
            token_count=20,
            character_count=110,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=4,
            content="Assessment\nFindings are consistent with stable angina, likely related to underlying coronary artery disease.",
            section_title="Assessment",
            token_count=18,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=5,
            content="Plan\nRecommend coronary CT angiography. Start atorvastatin 40mg daily and low-dose aspirin.",
            section_title="Plan",
            token_count=16,
            character_count=98,
        ),
    ]
    db_session.add_all(chunks)
    db_session.commit()

    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc.id, chunks=chunks)

    # Perform retrieval with search_similar_chunks
    results = search_similar_chunks(
        user_id=user_id,
        query="What is the disease?",
        document_id=doc.id,
        top_k=3,
        db=db_session,
        vector_store=temp_vector_store,
    )

    assert len(results) >= 1
    # Assessment chunk (Chunk 4 with stable angina) should be ranked #1
    assert "stable angina" in results[0].content.lower()
    assert results[0].section_title == "Assessment"
    assert results[0].relevance_score is not None
    assert results[0].relevance_score > 0.5
    assert results[0].similarity_score >= 0.0
    assert results[0].distance >= 0.0


# 12. Irrelevant Query Rejection Test Suite
def test_irrelevant_query_rejection(temp_vector_store, db_session):
    user_id = str(uuid.uuid4())
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user_id,
        original_filename="Endocrinology_Report.txt",
        stored_filename="endo_stored.txt",
        file_type="txt",
        file_size=600,
        mime_type="text/plain",
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    db_session.add(doc)
    db_session.commit()

    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=0,
            content="Chief Complaint\nPatient presents with fatigue, 10 lb weight gain, cold intolerance.",
            section_title="Chief Complaint",
            token_count=12,
            character_count=75,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=1,
            content="Diagnostic Findings\nTSH elevated at 14.2 mIU/L, Free T4 0.6 ng/dL, Anti-TPO positive.",
            section_title="Diagnostic Findings",
            token_count=14,
            character_count=85,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=2,
            content="Assessment\nPrimary hypothyroidism secondary to Hashimoto thyroiditis.",
            section_title="Assessment",
            token_count=8,
            character_count=65,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=3,
            content="Plan\nStart Levothyroxine 75mcg PO daily in morning. Recheck TSH in 6 weeks.",
            section_title="Plan",
            token_count=12,
            character_count=75,
        ),
    ]
    db_session.add_all(chunks)
    db_session.commit()

    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc.id, chunks=chunks)

    irrelevant_queries = [
        "What is Google?",
        "What is the weather?",
        "Explain quantum computing",
        "Who is Elon Musk?",
        "abcdefxyz123",
    ]

    for q in irrelevant_queries:
        results = search_similar_chunks(
            user_id=user_id,
            query=q,
            document_id=doc.id,
            top_k=5,
            relevance_threshold=0.25,
            db=db_session,
            vector_store=temp_vector_store,
        )
        assert len(results) == 0, f"Query '{q}' should be rejected (returned {len(results)} chunks)"


# 13. Symptoms Query Ranking Test
def test_symptoms_query_ranking(temp_vector_store, db_session):
    user_id = str(uuid.uuid4())
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user_id,
        original_filename="Cardio_Note.txt",
        stored_filename="cardio_stored_note.txt",
        file_type="txt",
        file_size=500,
        mime_type="text/plain",
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    db_session.add(doc)
    db_session.commit()

    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=0,
            content="Specialty: Cardiology\nConsultation Note",
            section_title="Document Overview",
            token_count=5,
            character_count=35,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=1,
            content="Chief Complaint\nPatient reports frequent shortness of breath and substernal chest discomfort on exertion.",
            section_title="Chief Complaint",
            token_count=16,
            character_count=105,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=2,
            content="Plan\nStart atorvastatin 40mg daily and aspirin 81mg.",
            section_title="Plan",
            token_count=8,
            character_count=50,
        ),
    ]
    db_session.add_all(chunks)
    db_session.commit()

    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc.id, chunks=chunks)

    results = search_similar_chunks(
        user_id=user_id,
        query="What are the symptoms?",
        document_id=doc.id,
        top_k=3,
        db=db_session,
        vector_store=temp_vector_store,
    )

    assert len(results) >= 1
    assert results[0].section_title == "Chief Complaint"
    assert "shortness of breath" in results[0].content.lower()


