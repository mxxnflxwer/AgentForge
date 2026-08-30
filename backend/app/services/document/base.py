from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedDocument:
    raw_text: str
    page_count: int = 1
    character_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_scanned: bool = False

    def __post_init__(self):
        if not self.character_count:
            self.character_count = len(self.raw_text)


@dataclass
class DocumentSection:
    title: str
    content: str
    start_char: int
    end_char: int


@dataclass
class DocumentChunkData:
    chunk_index: int
    content: str
    section_title: Optional[str] = None
    character_count: int = 0
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.character_count:
            self.character_count = len(self.content)
        if self.token_count is None:
            # Approximate token count (1 token ≈ 4 characters or whitespace words)
            self.token_count = max(1, len(self.content.split()))


class BaseDocumentProcessor(ABC):
    """Common interface for format-specific document processors."""

    @abstractmethod
    def extract(self, file_path_or_bytes: Any) -> ExtractedDocument:
        """
        Extract text and metadata from the document.
        """
        pass
