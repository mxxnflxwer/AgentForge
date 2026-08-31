import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.rag import RAGSearchRequest, RAGSearchResponse
from app.services.retrieval_service import search_similar_chunks

logger = logging.getLogger("agentforge.api.rag")

router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic similarity search over indexed document chunks",
)
def rag_search(
    request: RAGSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RAGSearchResponse:
    """
    Search for relevant document chunks using semantic similarity embeddings.
    - Strictly scopes retrieval to the authenticated user's documents.
    - Optionally filters by a specific document_id.
    - Returns retrieved chunks with cosine distance and similarity score.
    """
    results = search_similar_chunks(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )
    return RAGSearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )
