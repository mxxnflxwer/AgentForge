import io
import logging
from typing import Any, Union
from pypdf import PdfReader

from app.services.document.base import BaseDocumentProcessor, ExtractedDocument
from app.services.document.ocr import OCRProcessor

logger = logging.getLogger("agentforge.document.pdf")


class PDFProcessor(BaseDocumentProcessor):
    """
    Production PDF text extractor with scanned document detection and OCR fallback.
    """

    MIN_TEXT_THRESHOLD_PER_PAGE = 20  # Characters per page to consider as text-bearing

    def extract(self, file_path_or_bytes: Union[str, bytes]) -> ExtractedDocument:
        if isinstance(file_path_or_bytes, bytes):
            stream = io.BytesIO(file_path_or_bytes)
        else:
            stream = open(file_path_or_bytes, "rb")

        try:
            reader = PdfReader(stream)
            page_count = len(reader.pages)
            extracted_pages: list[str] = []
            is_scanned = False
            ocr_pages_count = 0

            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                cleaned_page_text = page_text.strip()

                # If text extraction yields insufficient characters, attempt OCR on embedded images
                if len(cleaned_page_text) < self.MIN_TEXT_THRESHOLD_PER_PAGE:
                    ocr_text_parts = []
                    # Check for embedded images in the page
                    if hasattr(page, "images") and page.images:
                        for img_file in page.images:
                            img_text = OCRProcessor.extract_text_from_image(img_file.data)
                            if img_text:
                                ocr_text_parts.append(img_text)

                    if ocr_text_parts:
                        ocr_combined = "\n".join(ocr_text_parts)
                        extracted_pages.append(ocr_combined)
                        ocr_pages_count += 1
                    else:
                        extracted_pages.append(cleaned_page_text)
                else:
                    extracted_pages.append(cleaned_page_text)

            full_text = "\n\n".join([p for p in extracted_pages if p])
            if ocr_pages_count > 0 or (page_count > 0 and len(full_text) < (page_count * self.MIN_TEXT_THRESHOLD_PER_PAGE)):
                is_scanned = True

            metadata = {}
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.title or "",
                    "author": reader.metadata.author or "",
                    "creator": reader.metadata.creator or "",
                    "producer": reader.metadata.producer or "",
                }

            return ExtractedDocument(
                raw_text=full_text,
                page_count=page_count,
                character_count=len(full_text),
                metadata=metadata,
                is_scanned=is_scanned,
            )
        finally:
            if not isinstance(file_path_or_bytes, bytes):
                stream.close()
