import datetime
import logging
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk, DocumentProcessingStatus
from app.services.document.base import BaseDocumentProcessor, ExtractedDocument
from app.services.document.cleaner import TextCleaner
from app.services.document.chunker import DocumentChunker
from app.services.document.docx_processor import DocxProcessor
from app.services.document.pdf_processor import PDFProcessor
from app.services.document.section_detector import SectionDetector
from app.services.document.txt_processor import TxtProcessor
from app.services.storage import get_storage_backend

logger = logging.getLogger("agentforge.document.pipeline")


class DocumentPipeline:
    """
    End-to-end ingestion and processing pipeline for AgentForge documents.
    """

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DocxProcessor()
        self.txt_processor = TxtProcessor()
        self.cleaner = TextCleaner()
        self.section_detector = SectionDetector()
        self.chunker = DocumentChunker()
        self.storage = get_storage_backend()

    def get_processor_for_type(self, file_type: str) -> BaseDocumentProcessor:
        normalized_type = file_type.lower().lstrip(".")
        if normalized_type == "pdf":
            return self.pdf_processor
        elif normalized_type in ("docx", "doc"):
            return self.docx_processor
        elif normalized_type in ("txt", "text", "log", "md"):
            return self.txt_processor
        else:
            raise ValueError(f"Unsupported document file type: {file_type}")

    def process_document(self, db: Session, document_id: str) -> Document:
        """
        Execute document processing pipeline synchronously for the given document ID.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document {document_id} not found.")

        # Update status to PROCESSING
        doc.processing_status = DocumentProcessingStatus.PROCESSING
        doc.processing_error = None
        db.commit()
        db.refresh(doc)

        try:
            logger.info(f"Starting processing for document {doc.id} ({doc.original_filename})")

            # 1. Retrieve stored file
            file_bytes = self.storage.read(doc.stored_filename)

            # 2. Select extractor and extract text
            processor = self.get_processor_for_type(doc.file_type)
            extracted: ExtractedDocument = processor.extract(file_bytes)

            # 3. Clean text
            cleaned_text = self.cleaner.clean(extracted.raw_text)
            if not cleaned_text.strip():
                raise ValueError("Document contains no readable text or is empty.")

            # 4. Detect sections
            sections = self.section_detector.detect_sections(cleaned_text)

            # 5. Chunk text
            chunks_data = self.chunker.chunk_sections(sections)

            # 6. Delete any existing chunks if re-processing
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()

            # 7. Create DocumentChunk records
            for c_data in chunks_data:
                chunk = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=c_data.chunk_index,
                    content=c_data.content,
                    section_title=c_data.section_title,
                    token_count=c_data.token_count,
                    character_count=c_data.character_count,
                )
                db.add(chunk)

            # 8. Update Document metadata & status
            doc.page_count = extracted.page_count
            doc.extracted_character_count = len(cleaned_text)
            doc.chunk_count = len(chunks_data)
            doc.processing_status = DocumentProcessingStatus.COMPLETED
            doc.processed_at = datetime.datetime.now(datetime.timezone.utc)
            doc.processing_error = None

            db.commit()
            db.refresh(doc)
            logger.info(f"Successfully processed document {doc.id}: {doc.chunk_count} chunks created.")
            return doc

        except Exception as e:
            logger.exception(f"Document processing failed for {doc.id}: {e}")
            db.rollback()
            # Fetch fresh reference for failure update
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = DocumentProcessingStatus.FAILED
                doc.processing_error = str(e)
                db.commit()
                db.refresh(doc)
            raise e


_pipeline_instance: Optional[DocumentPipeline] = None


def get_document_pipeline() -> DocumentPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DocumentPipeline()
    return _pipeline_instance
