import re
from typing import List
from app.services.document.base import DocumentSection


class SectionDetector:
    """
    Detects logical sections, clinical/medical headings, and structured titles
    within cleaned document text.
    """

    # Common clinical and domain headers
    CLINICAL_HEADERS = [
        r"CHIEF COMPLAINT|CC",
        r"HISTORY OF PRESENT ILLNESS|HPI",
        r"PAST MEDICAL HISTORY|PMH",
        r"PAST SURGICAL HISTORY|PSH",
        r"FAMILY HISTORY|FH",
        r"SOCIAL HISTORY|SH",
        r"MEDICATIONS|CURRENT MEDICATIONS|MEDICATION LIST",
        r"ALLERGIES|ALLERGIES & ADVERSE REACTIONS",
        r"REVIEW OF SYSTEMS|ROS",
        r"PHYSICAL EXAMINATION|PHYSICAL EXAM|EXAMINATION|VITAL SIGNS",
        r"DIAGNOSTIC FINDINGS|DIAGNOSTIC STUDIES|DIAGNOSTICS|LABORATORY DATA|LABS|LAB RESULTS|IMAGING",
        r"ASSESSMENT & PLAN|ASSESSMENT AND PLAN|A/P",
        r"ASSESSMENT|IMPRESSION|CLINICAL ASSESSMENT",
        r"PLAN|RECOMMENDATIONS|TREATMENT PLAN",
        r"DISCHARGE SUMMARY|DISCHARGE INSTRUCTIONS",
        r"PROCEDURE NOTE|OPERATIVE REPORT",
        r"DIAGNOSIS|PRIMARY DIAGNOSIS|SECONDARY DIAGNOSIS|FINAL DIAGNOSIS",
    ]


    # Markdown headers, e.g., "# Title", "## Subtitle"
    MARKDOWN_HEADER_PATTERN = r"^#{1,6}\s+([^\n]+)$"

    # Numbered sections, e.g., "1. Introduction", "Section 2: Overview", "Chapter 3 -"
    NUMBERED_SECTION_PATTERN = r"^(?:Section\s+\d+|Chapter\s+\d+|\d+\.(?:\d+)*)\s*[:\-\.]?\s*([^\n]+)$"

    # All uppercase headers on their own line (3 to 60 characters)
    ALL_CAPS_HEADER_PATTERN = r"^[A-Z0-9\s,\-\/\&]{3,60}:?$"

    def __init__(self):
        clinical_joined = "|".join(self.CLINICAL_HEADERS)
        self.clinical_pattern = re.compile(
            rf"^(?:{clinical_joined})\s*[:\-]?(?:\s+.*)?$",
            re.IGNORECASE,
        )
        self.markdown_pattern = re.compile(self.MARKDOWN_HEADER_PATTERN, re.MULTILINE)
        self.numbered_pattern = re.compile(self.NUMBERED_SECTION_PATTERN, re.MULTILINE | re.IGNORECASE)
        self.all_caps_pattern = re.compile(self.ALL_CAPS_HEADER_PATTERN)

    def is_header(self, line: str) -> bool:
        line_clean = line.strip()
        if not line_clean or len(line_clean) > 80:
            return False

        if line_clean.startswith("#"):
            return True

        if self.clinical_pattern.match(line_clean):
            return True

        if self.numbered_pattern.match(line_clean):
            return True

        # Check all caps candidate
        if (
            line_clean.isupper()
            and len(line_clean.split()) <= 8
            and self.all_caps_pattern.match(line_clean)
        ):
            return True

        return False

    def detect_sections(self, text: str) -> List[DocumentSection]:
        """
        Segment text into a list of DocumentSection objects based on detected headers.
        """
        if not text.strip():
            return []

        lines = text.split("\n")
        sections: List[DocumentSection] = []

        current_title = "Document Overview"
        current_lines: List[str] = []
        current_start_char = 0
        char_counter = 0

        for line in lines:
            line_len = len(line) + 1  # include newline in char tracking
            if self.is_header(line):
                # If we have accumulated content, flush the current section
                accumulated_content = "\n".join(current_lines).strip()
                if accumulated_content:
                    sections.append(
                        DocumentSection(
                            title=current_title,
                            content=accumulated_content,
                            start_char=current_start_char,
                            end_char=char_counter,
                        )
                    )
                # Clean the new header title
                cleaned_title = line.strip().lstrip("#").strip().rstrip(":-")
                current_title = cleaned_title if cleaned_title else "Section"
                current_lines = [line.strip()]
                current_start_char = char_counter
            else:
                current_lines.append(line)

            char_counter += line_len

        # Flush final section
        accumulated_content = "\n".join(current_lines).strip()
        if accumulated_content:
            sections.append(
                DocumentSection(
                    title=current_title,
                    content=accumulated_content,
                    start_char=current_start_char,
                    end_char=char_counter,
                )
            )

        # If no explicit sections were found other than the initial default with content
        if not sections:
            sections.append(
                DocumentSection(
                    title="Main Document",
                    content=text.strip(),
                    start_char=0,
                    end_char=len(text),
                )
            )

        return sections
