import pytest
import uuid
from unittest.mock import AsyncMock, patch

from app.models.document import Document, DocumentChunk, DocumentProcessingStatus
from app.models.user import User, UserRole
from app.services.agent_evo.evaluator_adapter import (
    EvaluatorAdapter,
    get_or_create_candidate_vector_store,
)
from app.services.agent_evo.workflow import AgentWorkflow, get_baseline_workflow
from app.services.document.chunker import DocumentChunker
from app.services.document.base import DocumentSection
from app.services.vector_store import get_vector_store


@pytest.fixture
def mock_embedding_and_llm(monkeypatch):
    """Ensure tests run offline without external API dependencies."""
    from app.services.embedding_service import BaseEmbeddingService

    class MockEmbedding(BaseEmbeddingService):
        def embed_text(self, text: str):
            # Deterministic 384-dimensional unit vector
            vec = [0.0] * 384
            vec[0] = 1.0
            return vec

        def embed_batch(self, texts):
            return [self.embed_text(t) for t in texts]

    mock_emb = MockEmbedding()
    monkeypatch.setattr("app.services.vector_store.get_embedding_service", lambda: mock_emb)
    monkeypatch.setattr("app.services.embedding_service.get_embedding_service", lambda: mock_emb)


@pytest.fixture
def sample_test_doc(db_session, mock_embedding_and_llm):
    """Seed a test document with long paragraphs into the database and baseline vector store."""
    user = User(
        id=str(uuid.uuid4()),
        name="Test Chunking User",
        email="test_chunking@agentforge.internal",
        password_hash="pw",
        role=UserRole.DEVELOPER,
    )
    db_session.add(user)
    db_session.commit()

    doc_id = str(uuid.uuid4())
    # Create natural sentences totaling ~2200 characters in one section
    sentences = [
        f"Clinical observation statement number {i} specifies neurological reflexes and cognitive status in detail."
        for i in range(22)
    ]
    long_content = " ".join(sentences)

    doc = Document(
        id=doc_id,
        owner_id=user.id,
        original_filename="clinical_eval_long.txt",
        stored_filename=f"{doc_id}_clinical_eval_long.txt",
        file_type="txt",
        file_size=len(long_content),
        mime_type="text/plain",
        extracted_character_count=len(long_content),
        chunk_count=0,
        processing_status=DocumentProcessingStatus.COMPLETED,
    )
    db_session.add(doc)
    db_session.commit()

    # Ingest with baseline chunker (1000/200) into production document_chunks
    baseline_chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    sections = [DocumentSection(title="Neurological Exam", content=long_content, start_char=0, end_char=len(long_content))]
    baseline_chunks_data = baseline_chunker.chunk_sections(sections)

    prod_chunks = []
    for cd in baseline_chunks_data:
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=doc.id,
            chunk_index=cd.chunk_index,
            content=cd.content,
            section_title=cd.section_title,
            token_count=cd.token_count,
            character_count=cd.character_count,
        )
        db_session.add(chunk)
        prod_chunks.append(chunk)

    doc.chunk_count = len(prod_chunks)
    db_session.commit()

    # Index into baseline vector store
    vs = get_vector_store()
    vs.upsert_document_chunks(user_id=user.id, document_id=doc.id, chunks=prod_chunks)

    return user, doc, prod_chunks, long_content


def test_chunker_unit_behavior():
    """Verify that DocumentChunker at 500/100 and 1000/200 produce strictly distinct chunk counts."""
    sentences = [
        f"Cardiovascular telemetry finding index {i} demonstrates normal sinus rhythm with intact perfusion."
        for i in range(30)
    ]
    text = " ".join(sentences)
    assert len(text) > 2500

    sec = DocumentSection(title="Telemetry", content=text, start_char=0, end_char=len(text))

    c_base = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks_base = c_base.chunk_sections([sec])

    c_cand = DocumentChunker(chunk_size=500, chunk_overlap=100)
    chunks_cand = c_cand.chunk_sections([sec])

    # Candidate chunker (500/100) must produce more, smaller chunks than baseline (1000/200)
    assert len(chunks_cand) > len(chunks_base)
    assert all(c.character_count <= 500 for c in chunks_cand)
    assert any(c.character_count > 500 for c in chunks_base)


@pytest.mark.asyncio
async def test_candidate_chunking_creates_isolated_index_and_preserves_production(
    db_session, sample_test_doc, mock_embedding_and_llm
):
    """
    Verify:
    1. Baseline workflow (1000/200) uses baseline chunks.
    2. Candidate cand_retrieval_chunking_256 (500/100) creates an isolated candidate collection.
    3. Production document_chunks table and production vector store are 100% preserved.
    """
    user, doc, prod_chunks, raw_text = sample_test_doc
    initial_db_chunk_count = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
    assert initial_db_chunk_count == len(prod_chunks)

    # 1. Baseline workflow evaluation
    baseline_wf = get_baseline_workflow()
    assert baseline_wf.retrieval.chunk_size == 1000
    assert baseline_wf.retrieval.chunk_overlap == 200

    # Mock LLM generation to avoid external network calls
    from app.services.llm.base import LLMResponse
    with patch("app.services.agent_evo.evaluator_adapter.get_llm_router") as mock_router:
        mock_adapter = AsyncMock()
        mock_adapter.name = "Gemini 3.5 Flash-Lite"
        mock_adapter.generate.return_value = LLMResponse(
            model="Gemini 3.5 Flash-Lite",
            provider="Google",
            model_id="gemini-3.5-flash-lite",
            answer="Normal cognitive status and reflexes observed.",
            latency_ms=150.0,
            success=True,
            error=None,
            raw_usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
        mock_router.return_value.get_adapter.return_value = mock_adapter

        # Evaluate Baseline
        baseline_eval = await EvaluatorAdapter.evaluate_candidate(
            workflow=baseline_wf,
            query="What is the patient cognitive status?",
            user_id=user.id,
            document_id=doc.id,
            db=db_session,
        )
        assert baseline_eval.metrics.status == "success"

        # Verify DB records after baseline evaluation
        assert db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count() == initial_db_chunk_count

        # 2. Candidate workflow evaluation (cand_retrieval_chunking_256)
        cand_wf = baseline_wf.clone(new_id="cand_retrieval_chunking_256", new_name="Fine-Grained Chunking Retrieval")
        cand_wf.retrieval.chunk_size = 500
        cand_wf.retrieval.chunk_overlap = 100
        cand_wf.retrieval.top_k = 6

        # Evaluate Candidate
        cand_eval = await EvaluatorAdapter.evaluate_candidate(
            workflow=cand_wf,
            query="What is the patient cognitive status?",
            user_id=user.id,
            document_id=doc.id,
            db=db_session,
        )
        assert cand_eval.metrics.status == "success"

        # 3. VERIFY PRODUCTION DATABASE WAS NEVER CHANGED
        after_cand_db_count = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
        assert after_cand_db_count == initial_db_chunk_count, "Candidate evaluation altered production DB chunks!"

        # 4. VERIFY CANDIDATE STORE CREATED ISOLATED COLLECTION WITH 500/100 CHUNKS
        isolated_store = get_or_create_candidate_vector_store(
            user_id=user.id,
            chunk_size=500,
            chunk_overlap=100,
            document_id=doc.id,
            db=db_session,
        )
        assert isolated_store is not None
        assert isolated_store.collection_name == "cand_chunks_500_100"

        # Query isolated store for chunks
        cand_vectors = isolated_store._collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": str(user.id)}},
                    {"document_id": {"$eq": str(doc.id)}},
                ]
            }
        )
        cand_docs = cand_vectors.get("documents", [])
        assert len(cand_docs) > initial_db_chunk_count, "Candidate index should have more chunks due to 500-token splitting"
        assert all(len(d) <= 500 for d in cand_docs), "All candidate chunks must respect chunk_size=500"


@pytest.mark.asyncio
async def test_active_workflow_execution_with_candidate_chunking(
    db_session, sample_test_doc, mock_embedding_and_llm
):
    """Verify that when an approved workflow with 500/100 is executed, it isolates chunking safely."""
    from app.services.agent_evo.workflow_lifecycle_service import WorkflowLifecycleService
    from app.models.workflow_version import WorkflowVersion
    from app.services.llm.base import LLMResponse

    user, doc, prod_chunks, raw_text = sample_test_doc
    initial_db_chunk_count = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()

    wf_custom = get_baseline_workflow()
    wf_custom.retrieval.chunk_size = 500
    wf_custom.retrieval.chunk_overlap = 100
    wf_custom.retrieval.top_k = 6

    version_rec = WorkflowVersion(
        id=str(uuid.uuid4()),
        version_id="wf_v_custom_chunking",
        version_number=99,
        name="Custom Chunking Active Workflow",
        source_run_id="run_custom_test",
        source_candidate_id="cand_custom_test",
        model_id=wf_custom.model.model_id,
        provider=wf_custom.model.provider,
        configuration=wf_custom.to_dict(),
        evaluation_snapshot={"groundedness": 0.95, "cost": 0.0001},
        status="ACTIVE",
        approved_by=user.id,
    )
    db_session.add(version_rec)
    db_session.commit()

    with patch("app.services.agent_evo.evaluator_adapter.get_llm_router") as mock_router:
        mock_adapter = AsyncMock()
        mock_adapter.name = "Gemini 3.5 Flash-Lite"
        mock_adapter.generate.return_value = LLMResponse(
            model="Gemini 3.5 Flash-Lite",
            provider="Google",
            model_id="gemini-3.5-flash-lite",
            answer="Normal cognitive status observed under active workflow.",
            latency_ms=120.0,
            success=True,
            error=None,
            raw_usage={"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95},
        )
        mock_router.return_value.get_adapter.return_value = mock_adapter

        exec_res = await WorkflowLifecycleService.execute_active_workflow(
            query="What is the patient cognitive status?",
            user_id=user.id,
            document_id=doc.id,
            workflow_version_id="wf_v_custom_chunking",
            db=db_session,
        )
        assert exec_res.workflow_version_id == "wf_v_custom_chunking"
        assert exec_res.response != ""

    # Verify production DB chunk count is unchanged
    assert db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count() == initial_db_chunk_count

