import logging
from typing import Union

from app.services.document.base import BaseDocumentProcessor, ExtractedDocument

logger = logging.getLogger("agentforge.document.txt")


class TxtProcessor(BaseDocumentProcessor):
    """
    Plain text extractor with robust multi-encoding detection and fallback handling.
    """

    ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1", "utf-16"]

    def extract(self, file_path_or_bytes: Union[str, bytes]) -> ExtractedDocument:
        if isinstance(file_path_or_bytes, bytes):
            raw_bytes = file_path_or_bytes
        else:
            with open(file_path_or_bytes, "rb") as f:
                raw_bytes = f.read()

        decoded_text = None
        used_encoding = None

        for encoding in self.ENCODING_CANDIDATES:
            try:
                decoded_text = raw_bytes.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            # Safe ultimate fallback: decode as UTF-8 with character replacement
            logger.warning("Could not cleanly decode text with candidate encodings; using utf-8 replacement.")
            decoded_text = raw_bytes.decode("utf-8", errors="replace")
            used_encoding = "utf-8 (lossy)"

        return ExtractedDocument(
            raw_text=decoded_text,
            page_count=1,
            character_count=len(decoded_text),
            metadata={"encoding": used_encoding},
            is_scanned=False,
        )
