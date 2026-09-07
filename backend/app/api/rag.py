import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.rag import RAGSearchRequest, RAGSearchResponse
from app.services.retrieval_service import execute_rag_retrieval

logger = logging.getLogger("agentforge.api.rag")

router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Query-aware intelligent RAG retrieval over indexed document chunks",
)
def rag_search(
    request: RAGSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RAGSearchResponse:
    """
    Search for relevant document chunks using semantic similarity embeddings,
    intent classification, entity verification, and multi-strategy retrieval:
    - Strictly scopes retrieval to the authenticated user's documents.
    - Optionally filters by a specific document_id.
    - Detects query intent (specific_fact, diagnosis, findings, summary, explanation).
    - Returns standardized responses for facts, document summaries, and missing information.
    """
    return execute_rag_retrieval(
        user_id=current_user.id,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        db=db,
    )
