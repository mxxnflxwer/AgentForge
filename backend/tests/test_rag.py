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
from app.services.retrieval_service import execute_rag_retrieval, search_similar_chunks
from app.services.reranker import classify_query_intent
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


# 14. Full Clinical Retrieval Suite (10 Required Benchmark Queries)
def test_clinical_retrieval_suite_all_queries(temp_vector_store, db_session):
    user_id = str(uuid.uuid4())
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user_id,
        original_filename="Cardiology_Consultation.txt",
        stored_filename="cardio_consult_stored.txt",
        file_type="txt",
        file_size=1200,
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
            character_count=70,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=1,
            content="Chief Complaint\nPatient presents with intermittent chest discomfort and shortness of breath on exertion.",
            section_title="Chief Complaint",
            token_count=15,
            character_count=110,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=2,
            content="History of Present Illness\nA 58-year-old presents with substernal chest tightness occurring with moderate exertion.",
            section_title="History of Present Illness",
            token_count=15,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=3,
            content="Physical Examination\nBlood pressure 138/86, heart rate 78 bpm regular, respiratory rate 16. Lungs clear. No peripheral edema.",
            section_title="Physical Examination",
            token_count=20,
            character_count=130,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=4,
            content="Diagnostic Findings\nECG shows normal sinus rhythm with no acute ST-T wave changes. Troponin I negative.",
            section_title="Diagnostic Findings",
            token_count=18,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=5,
            content="Assessment\nFindings are consistent with stable angina, likely related to underlying coronary artery disease.",
            section_title="Assessment",
            token_count=16,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=6,
            content="Plan\nRecommend coronary CT angiography for further risk stratification. Start atorvastatin 40mg daily and low-dose aspirin.",
            section_title="Plan",
            token_count=20,
            character_count=135,
        ),
    ]
    db_session.add_all(chunks)
    db_session.commit()

    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc.id, chunks=chunks)

    # 1. "what is the clinical diagnosis?" -> Assessment #1
    res1 = search_similar_chunks(user_id=user_id, query="what is the clinical diagnosis?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res1) >= 1
    assert res1[0].section_title == "Assessment"

    # 2. "what medications were prescribed?" -> Plan #1
    res2 = search_similar_chunks(user_id=user_id, query="what medications were prescribed?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res2) >= 1
    assert res2[0].section_title == "Plan"

    # 3. "did they have edema?" -> Physical Examination #1
    res3 = search_similar_chunks(user_id=user_id, query="did they have edema?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res3) >= 1
    assert res3[0].section_title == "Physical Examination"
    assert "no peripheral edema" in res3[0].content.lower()

    # 4. "what is the age of the patient?" -> History of Present Illness #1
    res4 = search_similar_chunks(user_id=user_id, query="what is the age of the patient?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res4) >= 1
    assert res4[0].section_title == "History of Present Illness"
    assert "58-year-old" in res4[0].content.lower()

    # 5. "what is the speciality?" -> Document Overview #1
    res5 = search_similar_chunks(user_id=user_id, query="what is the speciality?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res5) >= 1
    assert res5[0].section_title == "Document Overview"

    # 6. "what are the symptoms?" -> Chief Complaint #1
    res6 = search_similar_chunks(user_id=user_id, query="what are the symptoms?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res6) >= 1
    assert res6[0].section_title == "Chief Complaint"

    # 7. "what did the ECG show?" -> Diagnostic Findings #1
    res7 = search_similar_chunks(user_id=user_id, query="what did the ECG show?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res7) >= 1
    assert res7[0].section_title == "Diagnostic Findings"
    assert "ecg" in res7[0].content.lower()

    # 8. "what is google?" -> 0 results
    res8 = search_similar_chunks(user_id=user_id, query="what is google?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res8) == 0

    # 9. "hi" -> 0 results
    res9 = search_similar_chunks(user_id=user_id, query="hi", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res9) == 0

    # 10. "what is the treatment plan?" -> Plan #1
    res10 = search_similar_chunks(user_id=user_id, query="what is the treatment plan?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert len(res10) >= 1
    assert res10[0].section_title == "Plan"

    # 11. Short valid queries
    assert search_similar_chunks(user_id=user_id, query="age?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)[0].section_title == "History of Present Illness"
    assert search_similar_chunks(user_id=user_id, query="specialty?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)[0].section_title == "Document Overview"
    assert search_similar_chunks(user_id=user_id, query="edema?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)[0].section_title == "Physical Examination"


# 15. Query Intent Classifier Unit Tests
def test_query_intent_classifier():
    assert classify_query_intent("What is the patient's age?") == "specific_fact"
    assert classify_query_intent("What is the patient's blood pressure?") == "specific_fact"
    assert classify_query_intent("What is the LDL level?") == "specific_fact"
    assert classify_query_intent("What is the diagnosis?") == "diagnosis"
    assert classify_query_intent("What disease does the patient have?") == "diagnosis"
    assert classify_query_intent("Summarize the medical report") == "summary"
    assert classify_query_intent("Give me a summary of the complete report") == "summary"
    assert classify_query_intent("What are the important findings?") == "findings"
    assert classify_query_intent("What did the ECG show?") == "findings"
    assert classify_query_intent("Explain the diagnosis simply") == "explanation"
    assert classify_query_intent("Why was atorvastatin prescribed?") == "explanation"
    assert classify_query_intent("What is Google?") == "unsupported_query"
    assert classify_query_intent("hi") == "unsupported_query"


# 16. Full 12 Clinical Benchmark Queries Suite via execute_rag_retrieval
def test_execute_rag_retrieval_all_twelve_benchmarks(temp_vector_store, db_session):
    user_id = str(uuid.uuid4())
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user_id,
        original_filename="Cardiology_Full_Report.txt",
        stored_filename="cardio_full_stored.txt",
        file_type="txt",
        file_size=1500,
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
            character_count=70,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=1,
            content="Chief Complaint\nPatient presents with intermittent chest discomfort and shortness of breath on exertion.",
            section_title="Chief Complaint",
            token_count=15,
            character_count=110,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=2,
            content="History of Present Illness\nA 58-year-old presents with substernal chest tightness occurring with moderate exertion.",
            section_title="History of Present Illness",
            token_count=15,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=3,
            content="Physical Examination\nBlood pressure 138/86, heart rate 78 bpm regular, respiratory rate 16. Lungs clear. No peripheral edema.",
            section_title="Physical Examination",
            token_count=20,
            character_count=130,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=4,
            content="Diagnostic Findings\nECG shows normal sinus rhythm with no acute ST-T wave changes. Troponin I negative. Lipid panel shows LDL 162 mg/dL, HDL 38 mg/dL.",
            section_title="Diagnostic Findings",
            token_count=25,
            character_count=160,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=5,
            content="Assessment\nFindings are consistent with stable angina, likely related to underlying coronary artery disease.",
            section_title="Assessment",
            token_count=16,
            character_count=115,
        ),
        DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=6,
            content="Plan\nRecommend coronary CT angiography for further risk stratification. Start atorvastatin 40mg daily and low-dose aspirin.",
            section_title="Plan",
            token_count=20,
            character_count=135,
        ),
    ]
    db_session.add_all(chunks)
    db_session.commit()

    temp_vector_store.upsert_document_chunks(user_id=user_id, document_id=doc.id, chunks=chunks)

    # 1. Age -> 58-year-old
    q1 = execute_rag_retrieval(user_id=user_id, query="What is the patient's age?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q1.status == "success"
    assert q1.intent == "specific_fact"
    assert q1.total_results >= 1
    assert "58-year-old" in q1.results[0].content.lower()

    # 2. Diagnosis -> Stable angina
    q2 = execute_rag_retrieval(user_id=user_id, query="What is the patient's diagnosis?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q2.status == "success"
    assert q2.intent == "diagnosis"
    assert q2.total_results >= 1
    assert "stable angina" in q2.results[0].content.lower()
    assert q2.results[0].section_title == "Assessment"

    # 3. Blood pressure -> 138/86
    q3 = execute_rag_retrieval(user_id=user_id, query="What is the patient's blood pressure?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q3.status == "success"
    assert q3.total_results >= 1
    assert "138/86" in q3.results[0].content
    assert q3.results[0].section_title == "Physical Examination"

    # 4. Heart rate -> 78 bpm
    q4 = execute_rag_retrieval(user_id=user_id, query="What is the patient's heart rate?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q4.status == "success"
    assert q4.total_results >= 1
    assert "78 bpm" in q4.results[0].content
    assert q4.results[0].section_title == "Physical Examination"

    # 5. LDL level -> 162 mg/dL
    q5 = execute_rag_retrieval(user_id=user_id, query="What is the LDL level?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q5.status == "success"
    assert q5.total_results >= 1
    assert "162 mg/dl" in q5.results[0].content.lower()
    assert q5.results[0].section_title == "Diagnostic Findings"

    # 6. HDL level -> 38 mg/dL
    q6 = execute_rag_retrieval(user_id=user_id, query="What is the HDL level?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q6.status == "success"
    assert q6.total_results >= 1
    assert "38 mg/dl" in q6.results[0].content.lower()
    assert q6.results[0].section_title == "Diagnostic Findings"

    # 7. ECG -> Normal sinus rhythm
    q7 = execute_rag_retrieval(user_id=user_id, query="What did the ECG show?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q7.status == "success"
    assert q7.intent == "findings"
    assert q7.total_results >= 1
    assert "normal sinus rhythm" in q7.results[0].content.lower()
    assert q7.results[0].section_title == "Diagnostic Findings"

    # 8. Important findings
    q8 = execute_rag_retrieval(user_id=user_id, query="What were the important findings?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q8.status == "success"
    assert q8.intent == "findings"
    assert q8.total_results >= 1

    # 9. Blood glucose -> NOT FOUND (Anti-hallucination)
    q9 = execute_rag_retrieval(user_id=user_id, query="What is the patient's blood glucose level?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q9.status == "not_found"
    assert q9.total_results == 0
    assert len(q9.results) == 0
    assert q9.message == "The requested information was not found in the uploaded document."

    # 10. Phone number -> NOT FOUND
    q10 = execute_rag_retrieval(user_id=user_id, query="What is the patient's phone number?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q10.status == "not_found"
    assert q10.total_results == 0
    assert len(q10.results) == 0
    assert q10.message == "The requested information was not found in the uploaded document."

    # 11. Address -> NOT FOUND
    q11 = execute_rag_retrieval(user_id=user_id, query="What is the patient's address?", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q11.status == "not_found"
    assert q11.total_results == 0
    assert len(q11.results) == 0
    assert q11.message == "The requested information was not found in the uploaded document."

    # 12. Summary query -> Multi-section aggregation
    q12 = execute_rag_retrieval(user_id=user_id, query="Summarize the medical report", document_id=doc.id, db=db_session, vector_store=temp_vector_store)
    assert q12.status == "success"
    assert q12.intent == "summary"
    assert q12.sections is not None
    assert len(q12.sections) >= 5
    assert q12.summary_context is not None
    assert "Chief Complaint" in q12.summary_context
    assert "Assessment" in q12.summary_context
    assert "Diagnostic Findings" in q12.summary_context
    assert q12.total_results == len(chunks)


# 17. HTTP RAG Search API Endpoint Testing
def test_rag_api_endpoint_standardized_responses(client: TestClient, db_session):
    email = f"rag_api_test_{uuid.uuid4().hex[:6]}@example.com"
    session_id = register_and_login(client, email, name="RAG Tester")

    # Get user
    user = db_session.query(User).filter(User.email == email).first()
    doc = Document(
        id=str(uuid.uuid4()),
        owner_id=user.id,
        original_filename="Endpoint_Cardio.txt",
        stored_filename="endpoint_cardio.txt",
        file_type="txt",
        file_size=600,
        mime_type="text/plain",
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    db_session.add(doc)
    db_session.commit()

    chunk1 = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        chunk_index=0,
        content="Assessment: Exertional stable angina secondary to CAD.",
        section_title="Assessment",
        token_count=8,
        character_count=55,
    )
    db_session.add(chunk1)
    db_session.commit()

    vstore = get_vector_store()
    vstore.upsert_document_chunks(user_id=user.id, document_id=doc.id, chunks=[chunk1])

    # 1. Valid diagnosis search
    res = client.post(
        "/api/rag/search",
        json={"query": "What is the diagnosis?", "document_id": doc.id, "top_k": 3},
        cookies={"session_id": session_id},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["intent"] == "diagnosis"
    assert data["total_results"] >= 1
    assert "stable angina" in data["results"][0]["content"].lower()

    # 2. Unsupported blood glucose search -> not_found
    res2 = client.post(
        "/api/rag/search",
        json={"query": "What is the patient's blood glucose level?", "document_id": doc.id, "top_k": 3},
        cookies={"session_id": session_id},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "not_found"
    assert data2["total_results"] == 0
    assert data2["message"] == "The requested information was not found in the uploaded document."


