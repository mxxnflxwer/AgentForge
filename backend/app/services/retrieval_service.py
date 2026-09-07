import logging
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.schemas.rag import RAGDebugInfo, RAGSearchResponse, RetrievedChunk, SectionSummary
from app.services.reranker import BaseReranker, classify_query_intent, get_reranker
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger("agentforge.services.retrieval")


def build_extractive_summary_context(sections: List[SectionSummary]) -> str:
    """
    Construct a clean structured summary context string from document sections.
    Prioritizes key clinical sections in standard clinical order.
    """
    section_order = [
        "document overview",
        "chief complaint",
        "history of present illness",
        "physical examination",
        "diagnostic findings",
        "assessment",
        "plan",
    ]

    # Map normalized section names to sections
    sec_dict = {s.section_title.lower().strip(): s for s in sections}
    ordered_parts: List[str] = []

    # Add standard sections in clinical sequence if present
    added_keys = set()
    for key in section_order:
        for sec_name, sec_obj in sec_dict.items():
            if key in sec_name and sec_name not in added_keys:
                ordered_parts.append(f"### {sec_obj.section_title}\n{sec_obj.content}")
                added_keys.add(sec_name)

    # Add remaining sections not in standard list
    for sec_name, sec_obj in sec_dict.items():
        if sec_name not in added_keys:
            ordered_parts.append(f"### {sec_obj.section_title}\n{sec_obj.content}")
            added_keys.add(sec_name)

    return "\n\n".join(ordered_parts)


def execute_rag_retrieval(
    user_id: str,
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    relevance_threshold: Optional[float] = None,
    db: Optional[Session] = None,
    vector_store: Optional[VectorStore] = None,
    reranker: Optional[BaseReranker] = None,
) -> RAGSearchResponse:
    """
    Intelligent, query-aware RAG retrieval pipeline:
    1. Validates query and document ownership.
    2. Detects query intent (specific_fact, diagnosis, findings, summary, explanation, unsupported_query).
    3. Executes intent-driven retrieval strategy:
       - Summary: aggregates full document chunks, groups by section, builds extractive summary.
       - Unsupported/Nonsense: rejects immediately with anti-hallucination not_found status.
       - Specific Fact / Diagnosis / Findings / Explanation: retrieves candidate pool, reranks with
         section & entity verification, applies relevance threshold gating.
    4. Returns standardized RAGSearchResponse with optional debug info.
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

    # Step 1: Query Intent Detection
    intent = classify_query_intent(cleaned_query)
    effective_threshold = (
        relevance_threshold
        if relevance_threshold is not None
        else getattr(settings, "MIN_RELEVANCE_SCORE", 0.35)
    )

    include_debug = getattr(settings, "DEBUG_RETRIEVAL", True) or (
        getattr(settings, "ENVIRONMENT", "development") == "development"
    )

    # Step 2: Handle Unsupported / Out-of-Domain queries immediately
    if intent == "unsupported_query":
        debug_info = (
            RAGDebugInfo(
                retrieved_chunks=0,
                filtered_chunks=0,
                threshold=effective_threshold,
                intent=intent,
            )
            if include_debug
            else None
        )
        return RAGSearchResponse(
            query=cleaned_query,
            status="not_found",
            intent=intent,
            total_results=0,
            message="The requested information was not found in the uploaded document.",
            document_id=document_id,
            results=[],
            debug=debug_info,
        )

    # Step 3: Handle Summary Queries (Multi-chunk & multi-section aggregation)
    if intent == "summary":
        if document_id:
            raw_chunks = store.get_all_document_chunks(user_id=user_id, document_id=document_id)
        else:
            # Multi-chunk retrieval across documents
            raw_chunks = store.query_similar_chunks(
                user_id=user_id,
                query_text="medical report overview clinical assessment plan diagnostic findings chief complaint",
                document_id=None,
                top_k=max(20, top_k * 4),
            )
            raw_chunks.sort(key=lambda x: x.get("chunk_index", 0))

        if not raw_chunks:
            debug_info = (
                RAGDebugInfo(
                    retrieved_chunks=0,
                    filtered_chunks=0,
                    threshold=effective_threshold,
                    intent=intent,
                )
                if include_debug
                else None
            )
            return RAGSearchResponse(
                query=cleaned_query,
                status="not_found",
                intent=intent,
                total_results=0,
                message="The requested information was not found in the uploaded document.",
                document_id=document_id,
                results=[],
                debug=debug_info,
            )

        # Group by section_title
        sections_map: Dict[str, Dict[str, Any]] = {}
        results_list: List[RetrievedChunk] = []

        for c in raw_chunks:
            sec_title = c.get("section_title") or "General"
            if sec_title not in sections_map:
                sections_map[sec_title] = {"content_parts": [], "chunk_indices": []}
            sections_map[sec_title]["content_parts"].append(c.get("content", ""))
            sections_map[sec_title]["chunk_indices"].append(c.get("chunk_index", 0))

            results_list.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    chunk_index=c["chunk_index"],
                    content=c["content"],
                    section_title=c.get("section_title"),
                    distance=c.get("distance", 0.0),
                    similarity_score=c.get("similarity_score", 1.0),
                    relevance_score=c.get("relevance_score", 1.0),
                )
            )

        sections: List[SectionSummary] = [
            SectionSummary(
                section_title=sec_name,
                content="\n".join(sec_data["content_parts"]),
                chunk_indices=sec_data["chunk_indices"],
            )
            for sec_name, sec_data in sections_map.items()
        ]

        summary_context = build_extractive_summary_context(sections)

        debug_info = (
            RAGDebugInfo(
                retrieved_chunks=len(raw_chunks),
                filtered_chunks=0,
                threshold=effective_threshold,
                intent=intent,
            )
            if include_debug
            else None
        )

        return RAGSearchResponse(
            query=cleaned_query,
            status="success",
            intent=intent,
            total_results=len(results_list),
            document_id=document_id,
            sections=sections,
            summary_context=summary_context,
            results=results_list,
            debug=debug_info,
        )

    # Step 4: Specific Fact / Diagnosis / Findings / Explanation Strategies
    candidate_k = max(25, top_k * 5)
    raw_candidates = store.query_similar_chunks(
        user_id=user_id,
        query_text=cleaned_query,
        document_id=document_id,
        top_k=candidate_k,
    )

    reranked_results = ranker.rerank(
        query=cleaned_query,
        chunks=raw_candidates,
        top_k=top_k,
        relevance_threshold=effective_threshold,
    )

    filtered_count = max(0, len(raw_candidates) - len(reranked_results))
    debug_info = (
        RAGDebugInfo(
            retrieved_chunks=len(raw_candidates),
            filtered_chunks=filtered_count,
            threshold=effective_threshold,
            intent=intent,
        )
        if include_debug
        else None
    )

    # If no results pass the threshold -> Anti-hallucination not_found response
    if not reranked_results:
        return RAGSearchResponse(
            query=cleaned_query,
            status="not_found",
            intent=intent,
            total_results=0,
            message="The requested information was not found in the uploaded document.",
            document_id=document_id,
            results=[],
            debug=debug_info,
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
        f"Retrieved {len(results)} chunks (intent: {intent}) for user {user_id} (query: '{cleaned_query[:40]}...', doc_id: {document_id})"
    )

    return RAGSearchResponse(
        query=cleaned_query,
        status="success",
        intent=intent,
        total_results=len(results),
        document_id=document_id,
        results=results,
        debug=debug_info,
    )


def search_similar_chunks(
    user_id: str,
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    relevance_threshold: Optional[float] = None,
    db: Optional[Session] = None,
    vector_store: Optional[VectorStore] = None,
    reranker: Optional[BaseReranker] = None,
) -> List[RetrievedChunk]:
    """
    Backward-compatible convenience function returning List[RetrievedChunk].
    Delegates to the intelligent retrieval pipeline.
    """
    response = execute_rag_retrieval(
        user_id=user_id,
        query=query,
        document_id=document_id,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        db=db,
        vector_store=vector_store,
        reranker=reranker,
    )
    return response.results
