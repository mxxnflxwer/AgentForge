import io
import logging
from typing import Optional, Union
from PIL import Image

logger = logging.getLogger("agentforge.document.ocr")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None


class OCRProcessor:
    """OCR processor utilizing Tesseract via pytesseract and Pillow."""

    @classmethod
    def is_available(cls) -> bool:
        if not PYTESSERACT_AVAILABLE:
            return False
        try:
            # Check if tesseract binary can be queried
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @classmethod
    def extract_text_from_image(cls, image_data: Union[bytes, Image.Image]) -> str:
        """
        Extract text from an image (bytes or PIL Image).
        Handles Tesseract absence gracefully.
        """
        if not PYTESSERACT_AVAILABLE:
            logger.warning("pytesseract library is not available.")
            return ""

        try:
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data

            # Convert to RGB or Grayscale if needed
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")

            text = pytesseract.image_to_string(image)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract OCR binary not found on the host system. Skipping OCR extraction.")
            return ""
        except Exception as e:
            logger.error(f"OCR extraction failed on image: {e}")
            return ""
