import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.models.document import DocumentProcessingStatus


class DocumentChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    content: str
    section_title: Optional[str] = None
    token_count: Optional[int] = None
    character_count: int
    created_at: datetime.datetime


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: DocumentProcessingStatus
    file_type: str
    file_size: int
    specialty: Optional[str] = None
    message: str = "Document uploaded successfully and queued for processing."


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str = Field(validation_alias="original_filename")
    status: DocumentProcessingStatus = Field(validation_alias="processing_status")
    file_type: str
    file_size: int
    specialty: Optional[str] = None
    chunk_count: int
    page_count: Optional[int] = None
    created_at: datetime.datetime
    processed_at: Optional[datetime.datetime] = None


class DocumentDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str = Field(validation_alias="original_filename")
    stored_filename: str
    status: DocumentProcessingStatus = Field(validation_alias="processing_status")
    file_type: str
    file_size: int
    mime_type: str
    specialty: Optional[str] = None
    page_count: Optional[int] = None
    extracted_character_count: Optional[int] = None
    chunk_count: int
    processing_error: Optional[str] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    processed_at: Optional[datetime.datetime] = None
    chunks: List[DocumentChunkResponse] = []
