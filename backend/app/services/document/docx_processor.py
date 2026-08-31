import io
import logging
from typing import Union
# pyrefly: ignore [missing-import]
from docx import Document as DocxDocument

from app.services.document.base import BaseDocumentProcessor, ExtractedDocument

logger = logging.getLogger("agentforge.document.docx")


class DocxProcessor(BaseDocumentProcessor):
    """
    DOCX text extractor that preserves headings, paragraphs, and table contents.
    """

    def extract(self, file_path_or_bytes: Union[str, bytes]) -> ExtractedDocument:
        if isinstance(file_path_or_bytes, bytes):
            stream = io.BytesIO(file_path_or_bytes)
        else:
            stream = open(file_path_or_bytes, "rb")

        try:
            doc = DocxDocument(stream)
            extracted_parts: list[str] = []

            # Extract paragraphs
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    extracted_parts.append(text)

            # Extract tables
            for table in doc.tables:
                table_lines: list[str] = []
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    # Filter empty or duplicate merged cells
                    if any(row_cells):
                        table_lines.append(" | ".join(row_cells))
                if table_lines:
                    extracted_parts.append("\n".join(table_lines))

            full_text = "\n\n".join(extracted_parts)

            # Metadata extraction if present
            metadata = {}
            core_props = doc.core_properties
            if core_props:
                metadata = {
                    "title": core_props.title or "",
                    "author": core_props.author or "",
                    "category": core_props.category or "",
                    "comments": core_props.comments or "",
                }

            return ExtractedDocument(
                raw_text=full_text,
                page_count=1,  # DOCX files do not have explicit page delimiters until rendered
                character_count=len(full_text),
                metadata=metadata,
                is_scanned=False,
            )
        finally:
            if not isinstance(file_path_or_bytes, bytes):
                stream.close()
