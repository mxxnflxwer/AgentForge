import logging
import sys
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.document import Document, DocumentChunk, DocumentProcessingStatus
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger("agentforge.services.reindex")


def reindex_document(
    db: Session,
    document_id: str,
    vector_store: Optional[VectorStore] = None,
) -> int:
    """
    Reindex a single document's chunks from PostgreSQL into ChromaDB.
    Idempotent: removes existing vectors for the document before inserting fresh embeddings.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        logger.warning(f"Document {document_id} not found in PostgreSQL.")
        return 0

    if doc.processing_status != DocumentProcessingStatus.COMPLETED:
        logger.warning(f"Document {document_id} is not COMPLETED (status: {doc.processing_status}). Skipping.")
        return 0

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )

    if not chunks:
        logger.warning(f"Document {document_id} has no chunks in PostgreSQL.")
        return 0

    store = vector_store or get_vector_store()
    count = store.upsert_document_chunks(
        user_id=doc.owner_id,
        document_id=doc.id,
        chunks=chunks,
    )
    logger.info(f"Reindexed {count} chunks for document {doc.id} (user {doc.owner_id}).")
    return count


def reindex_all_documents(
    db: Optional[Session] = None,
    vector_store: Optional[VectorStore] = None,
    reset_chroma: bool = False,
) -> Dict[str, any]:
    """
    Read all COMPLETED documents and their chunks from PostgreSQL and synchronize them into ChromaDB.
    """
    close_db_session = False
    if db is None:
        db = SessionLocal()
        close_db_session = True

    store = vector_store or get_vector_store()
    if reset_chroma:
        logger.info("Resetting ChromaDB collection before reindexing...")
        store.reset_collection()

    try:
        documents = (
            db.query(Document)
            .filter(Document.processing_status == DocumentProcessingStatus.COMPLETED)
            .all()
        )
        total_vectors = 0
        reindexed_docs: List[str] = []

        for doc in documents:
            count = reindex_document(db=db, document_id=doc.id, vector_store=store)
            total_vectors += count
            reindexed_docs.append(doc.id)

        result = {
            "status": "success",
            "documents_processed": len(documents),
            "total_vectors_indexed": total_vectors,
            "document_ids": reindexed_docs,
        }
        logger.info(f"Reindex complete: {result}")
        return result
    finally:
        if close_db_session:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Starting AgentForge ChromaDB vector reindexing from PostgreSQL...")
    summary = reindex_all_documents()
    print("Reindex finished successfully:")
    print(f"  - Documents processed: {summary['documents_processed']}")
    print(f"  - Total vectors indexed: {summary['total_vectors_indexed']}")
    print(f"  - Document IDs: {summary['document_ids']}")
