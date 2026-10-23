import hashlib
import pytest
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.document import Document
from app.models.user import User
from app.services.retrieval_service import execute_rag_retrieval
from app.services.evaluation.groundedness import calculate_groundedness
from app.services.evaluation.hallucination import calculate_hallucination_rate
from app.services.evaluation.evaluator import EvaluationEngine, EvaluationInput
from app.services.agent_evo.runtime_selector import RuntimeWorkflowSelector


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def test_user_and_docs(db_session: Session):
    # Lookup the user and documents created for testing
    user = db_session.query(User).filter(User.email == "user@example.com").first()
    assert user is not None, "Test user user@example.com must exist in database"

    cardiology_doc = (
        db_session.query(Document)
        .filter(
            Document.owner_id == user.id,
            Document.original_filename.ilike("%Cardiology%"),
        )
        .first()
    )
    assert cardiology_doc is not None, "Cardiology document must exist in database"

    endocrinology_doc = (
        db_session.query(Document)
        .filter(
            Document.owner_id == user.id,
            Document.original_filename.ilike("%Endocrinology%"),
        )
        .first()
    )
    assert endocrinology_doc is not None, "Endocrinology document must exist in database"

    return {
        "user_id": str(user.id),
        "cardiology_id": str(cardiology_doc.id),
        "endocrinology_id": str(endocrinology_doc.id),
    }


def test_cardiology_document_retrieval(db_session: Session, test_user_and_docs: dict):
    """Verify that scoped retrieval on Cardiology document returns clinical diagnosis chunk."""
    res = execute_rag_retrieval(
        user_id=test_user_and_docs["user_id"],
        query="What is the clinical diagnosis?",
        document_id=test_user_and_docs["cardiology_id"],
        top_k=5,
        relevance_threshold=0.35,
        db=db_session,
        enable_reranking=True,
    )
    assert res.status == "success"
    assert len(res.results) > 0
    # Top chunk should be Assessment / Diagnostic Findings discussing angina
    contents = " ".join([c.content for c in res.results]).lower()
    assert "angina" in contents or "coronary" in contents or "assessment" in contents
    assert all(c.document_id == test_user_and_docs["cardiology_id"] for c in res.results)


def test_endocrinology_document_retrieval(db_session: Session, test_user_and_docs: dict):
    """Verify that scoped retrieval on Endocrinology document returns hypothyroidism diagnosis chunk."""
    res = execute_rag_retrieval(
        user_id=test_user_and_docs["user_id"],
        query="What is the clinical diagnosis?",
        document_id=test_user_and_docs["endocrinology_id"],
        top_k=5,
        relevance_threshold=0.35,
        db=db_session,
        enable_reranking=True,
    )
    assert res.status == "success"
    assert len(res.results) > 0
    contents = " ".join([c.content for c in res.results]).lower()
    assert "hypothyroid" in contents or "tsh" in contents or "assessment" in contents
    assert all(c.document_id == test_user_and_docs["endocrinology_id"] for c in res.results)


def test_all_documents_retrieval_scope(db_session: Session, test_user_and_docs: dict):
    """Verify that retrieval across All Documents (document_id=None) searches all user documents."""
    res = execute_rag_retrieval(
        user_id=test_user_and_docs["user_id"],
        query="What is the clinical diagnosis?",
        document_id=None,
        top_k=5,
        relevance_threshold=0.35,
        db=db_session,
        enable_reranking=True,
    )
    assert res.status == "success"
    assert len(res.results) >= 2
    doc_ids = {c.document_id for c in res.results}
    # Should include chunks from both cardiology and endocrinology
    assert test_user_and_docs["cardiology_id"] in doc_ids
    assert test_user_and_docs["endocrinology_id"] in doc_ids


def test_out_of_domain_retrieval_returns_not_found(db_session: Session, test_user_and_docs: dict):
    """Verify that an out-of-domain query (e.g. phone number) yields not_found with 0 chunks."""
    res = execute_rag_retrieval(
        user_id=test_user_and_docs["user_id"],
        query="What is the patient's phone number?",
        document_id=test_user_and_docs["cardiology_id"],
        top_k=5,
        relevance_threshold=0.35,
        db=db_session,
        enable_reranking=True,
    )
    assert res.status == "not_found"
    assert len(res.results) == 0


def test_evaluation_not_found_returns_none_not_100_percent():
    """Verify that safe not-found fallback returns groundedness=None and hallucination=None (N/A)."""
    answer = "The requested information was not found in the uploaded document."
    context = ""
    g_res = calculate_groundedness(answer, context)
    assert g_res.groundedness is None
    assert g_res.total_claims == 0
    assert g_res.supported_claims == 0
    assert g_res.unsupported_claims == 0

    h_res = calculate_hallucination_rate(answer, context, g_res)
    assert h_res.hallucination_rate is None
    assert h_res.total_claims == 0
    assert h_res.hallucinated_claims == 0

    engine = EvaluationEngine()
    e_input = EvaluationInput(
        question="What is the patient's phone number?",
        generated_answer=answer,
        retrieved_context=context,
        model_name="gemini-3.5-flash-lite",
        latency_ms=0.0,
        execution_time_ms=10.0,
        success=True,
    )
    e_res = engine.evaluate(e_input)
    assert e_res.groundedness is None
    assert e_res.hallucination_rate is None
    assert e_res.total_claims == 0


def test_context_hash_deterministic():
    """Verify that identical retrieved context produces an exact deterministic SHA-256 hash."""
    context_a = "Chunk #1: Blood pressure 120/80.\nChunk #2: Normal sinus rhythm."
    context_b = "Chunk #1: Blood pressure 120/80.\nChunk #2: Normal sinus rhythm."
    hash_a = hashlib.sha256(context_a.encode("utf-8")).hexdigest()
    hash_b = hashlib.sha256(context_b.encode("utf-8")).hexdigest()
    assert hash_a == hash_b
    assert len(hash_a) == 64


def test_runtime_workflow_selector_modes(db_session: Session):
    """Verify that RuntimeWorkflowSelector correctly selects active workflow vs baseline."""
    # Active workflow mode (default)
    active_selection = RuntimeWorkflowSelector.select_workflow_for_query(
        query="What is the clinical diagnosis?",
        db=db_session,
        candidate_version_id_override=None,
    )
    assert active_selection.version_id != "baseline"
    assert active_selection.version_id.startswith("wf_v")
    assert "retrieval" in active_selection.configuration

    # Baseline mode
    baseline_selection = RuntimeWorkflowSelector.select_workflow_for_query(
        query="What is the clinical diagnosis?",
        db=db_session,
        candidate_version_id_override="baseline",
    )
    assert baseline_selection.version_id == "baseline"
    assert baseline_selection.is_fallback is True
