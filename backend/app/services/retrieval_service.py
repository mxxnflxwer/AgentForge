import logging
from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.document import Document
from app.schemas.rag import RetrievedChunk
from app.services.reranker import BaseReranker, get_reranker
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger("agentforge.services.retrieval")


def search_similar_chunks(
    user_id: str,
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    relevance_threshold: float = 0.25,
    db: Optional[Session] = None,
    vector_store: Optional[VectorStore] = None,
    reranker: Optional[BaseReranker] = None,
) -> List[RetrievedChunk]:
    """
    Core retrieval pipeline:
    1. Validates query and parameters.
    2. Validates document ownership in PostgreSQL if document_id is provided and db is supplied.
    3. Retrieves candidate pool (candidate_k = max(25, top_k * 5)) from ChromaDB with strict user isolation.
    4. Reranks candidates using hybrid dense + lexical BM25 + intent scoring.
    5. Filters chunks below relevance_threshold (rejects unrelated queries).
    6. Returns up to top_k relevant chunks with similarity scores, relevance scores, and original distances.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    # Validate document existence and ownership if scoped to a specific document
    if document_id and db is not None:
        doc = (
            db.query(Document)
            .filter(Document.id == document_id, Document.owner_id == user_id)
            .first()
        )
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

    store = vector_store or get_vector_store()
    ranker = reranker or get_reranker()

    # Retrieve candidate pool (20-30+ candidates) for reranking
    candidate_k = max(25, top_k * 5)
    raw_candidates = store.query_similar_chunks(
        user_id=user_id,
        query_text=cleaned_query,
        document_id=document_id,
        top_k=candidate_k,
    )

    # Rerank candidates with hybrid scoring & threshold filtering
    reranked_results = ranker.rerank(
        query=cleaned_query,
        chunks=raw_candidates,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    results: List[RetrievedChunk] = []
    for r in reranked_results:
        results.append(
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                section_title=r.get("section_title"),
                distance=r["distance"],
                similarity_score=r["similarity_score"],
                relevance_score=r.get("relevance_score", r["similarity_score"]),
            )
        )

    logger.info(
        f"Retrieved and reranked {len(results)} chunks for user {user_id} (query: '{cleaned_query[:40]}...', doc_id: {document_id})"
    )
    return results
