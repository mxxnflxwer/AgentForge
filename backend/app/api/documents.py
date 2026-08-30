from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListItem,
    DocumentUploadResponse,
)
from app.schemas.user import MessageResponse
from app.services.document_service import (
    create_and_store_document,
    delete_user_document,
    get_user_document_by_id,
    get_user_documents,
    run_pipeline_in_background,
)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for processing",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (.pdf, .docx, .txt)"),
    specialty: Optional[str] = Form(None, description="Optional medical/domain specialty"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Accepts multipart/form-data document upload.
    Validates format, magic headers, and size limits.
    Stores the binary file and enqueues the processing pipeline in the background.
    """
    file_bytes = await file.read()
    document = create_and_store_document(
        db=db,
        user=current_user,
        file=file,
        file_bytes=file_bytes,
        specialty=specialty,
    )

    # Dispatch pipeline execution in background
    background_tasks.add_task(run_pipeline_in_background, document.id)

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.processing_status,
        file_type=document.file_type,
        file_size=document.file_size,
        specialty=document.specialty,
        message="Document uploaded successfully and queued for processing.",
    )


@router.get(
    "",
    response_model=List[DocumentListItem],
    summary="List all documents owned by current user",
)
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns all documents owned strictly by the authenticated user.
    """
    return get_user_documents(db=db, user=current_user)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details, status, and chunks",
)
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve document metadata, processing status, and extracted chunks.
    Enforces user ownership isolation.
    """
    return get_user_document_by_id(db=db, document_id=document_id, user=current_user)


@router.delete(
    "/{document_id}",
    response_model=MessageResponse,
    summary="Delete a document and its stored files",
)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Deletes the document record, associated chunks, and disk file.
    Enforces strict user ownership.
    """
    delete_user_document(db=db, document_id=document_id, user=current_user)
    return MessageResponse(message="Document and associated data successfully deleted.")
