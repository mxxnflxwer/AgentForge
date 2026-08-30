import re
import unicodedata


class TextCleaner:
    """Production-grade text sanitizer for extracted document text."""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # Normalize unicode (NFKC)
        cleaned = unicodedata.normalize("NFKC", text)

        # Normalize line breaks
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Fix hyphenated word breaks at end of line (e.g. "treat-\nment" -> "treatment")
        cleaned = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", cleaned)

        # Remove control characters except standard tabs and newlines
        cleaned = "".join(ch for ch in cleaned if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C"))

        # Replace non-breaking spaces and special horizontal whitespace with standard space
        cleaned = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", cleaned)

        # Collapse horizontal spaces and tabs
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Clean up whitespace on each individual line
        lines = [line.strip() for line in cleaned.split("\n")]

        # Collapse 3+ consecutive newlines into 2 (paragraphs)
        collapsed_lines: list[str] = []
        consecutive_empty = 0
        for line in lines:
            if not line:
                consecutive_empty += 1
                if consecutive_empty <= 1:
                    collapsed_lines.append("")
            else:
                consecutive_empty = 0
                collapsed_lines.append(line)

        return "\n".join(collapsed_lines).strip()
