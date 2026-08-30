import os
import uuid
from typing import List, Optional, Tuple
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.document import Document, DocumentProcessingStatus
from app.models.user import User
from app.services.document.pipeline import get_document_pipeline
from app.services.storage import get_storage_backend


VALID_MAGIC_HEADERS = {
    "pdf": [b"%PDF-"],
    "docx": [b"PK\x03\x04"],
}


def validate_file_upload(file: UploadFile, file_bytes: bytes) -> Tuple[str, str]:
    """
    Validate file extension, size, and header bytes.
    Returns sanitized (extension, mime_type).
    """
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot upload an empty file.",
        )

    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if hasattr(status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE") else 413,
            detail=f"File size exceeds maximum allowed limit of {max_mb:.0f} MB.",
        )

    filename = file.filename or ""
    parts = filename.rsplit(".", 1)
    if len(parts) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a valid extension (.pdf, .docx, .txt).",
        )

    ext = parts[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '.{ext}'. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Magic byte verification
    if ext in VALID_MAGIC_HEADERS:
        matched = any(file_bytes.startswith(magic) for magic in VALID_MAGIC_HEADERS[ext])
        if not matched:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File contents do not match valid {ext.upper()} format.",
            )

    mime_type = file.content_type or f"application/{ext}"
    return ext, mime_type


def create_and_store_document(
    db: Session,
    user: User,
    file: UploadFile,
    file_bytes: bytes,
    specialty: Optional[str] = None,
) -> Document:
    """
    Validate, store to disk/storage, and create the Document record.
    """
    ext, mime_type = validate_file_upload(file, file_bytes)

    doc_id = str(uuid.uuid4())
    sanitized_original = os.path.basename(file.filename or f"doc_{doc_id}.{ext}")
    stored_filename = f"{doc_id}_{sanitized_original}"

    storage = get_storage_backend()
    storage.save(file_bytes, stored_filename)

    document = Document(
        id=doc_id,
        owner_id=user.id,
        original_filename=sanitized_original,
        stored_filename=stored_filename,
        file_type=ext,
        file_size=len(file_bytes),
        mime_type=mime_type,
        specialty=specialty,
        processing_status=DocumentProcessingStatus.UPLOADED,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def run_pipeline_in_background(document_id: str):
    """
    Background worker function executed via FastAPI BackgroundTasks.
    Creates an isolated database session and runs DocumentPipeline.
    """
    db = SessionLocal()
    try:
        pipeline = get_document_pipeline()
        pipeline.process_document(db=db, document_id=document_id)
    except Exception as e:
        # The document record has already been marked as FAILED with error message
        pass
    finally:
        db.close()



def get_user_documents(db: Session, user: User) -> List[Document]:
    """
    Retrieve all documents owned by the user.
    """
    return (
        db.query(Document)
        .filter(Document.owner_id == user.id)
        .order_by(Document.created_at.desc())
        .all()
    )


def get_user_document_by_id(db: Session, document_id: str, user: User) -> Document:
    """
    Retrieve a specific document owned by the user.
    Raises 404 if not found or if owned by another user.
    """
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return document


def delete_user_document(db: Session, document_id: str, user: User) -> bool:
    """
    Delete a document, its database chunks, and its underlying file in storage.
    Enforces strict ownership.
    """
    document = get_user_document_by_id(db, document_id, user)
    stored_filename = document.stored_filename

    # Delete from database (chunks are cascade-deleted)
    db.delete(document)
    db.commit()

    # Delete from storage backend (graceful handling of missing files)
    storage = get_storage_backend()
    storage.delete(stored_filename)
    return True
