import math
import re
from typing import List, Optional
from app.services.document.base import DocumentChunkData, DocumentSection


class DocumentChunker:
    """
    Section-aware, token-respecting recursive text chunker.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        self.min_chunk_size = min_chunk_size

    def _split_text_recursively(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split a large body of text cleanly on paragraph, sentence, or word boundaries."""
        if len(text) <= max_size:
            return [text]

        chunks: List[str] = []
        # Try splitting by double newline (paragraphs) first
        paragraphs = text.split("\n\n")
        current_chunk: List[str] = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If single paragraph is longer than max_size, split by sentences
            if len(para) > max_size:
                sentence_splits = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentence_splits:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(sentence) > max_size:
                        # Split by words if sentence is still too long
                        words = sentence.split(" ")
                        temp_words: List[str] = []
                        temp_len = 0
                        for word in words:
                            if temp_len + len(word) + 1 > max_size and temp_words:
                                chunks.append(" ".join(temp_words))
                                # Keep overlap words
                                overlap_words = temp_words[-max(1, overlap // 10):]
                                temp_words = list(overlap_words)
                                temp_len = sum(len(w) + 1 for w in temp_words)
                            temp_words.append(word)
                            temp_len += len(word) + 1
                        if temp_words:
                            chunks.append(" ".join(temp_words))
                    else:
                        if current_length + len(sentence) + 1 > max_size and current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            current_length = 0
                        current_chunk.append(sentence)
                        current_length += len(sentence) + 1
            else:
                if current_length + len(para) + 2 > max_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(para)
                current_length += len(para) + 2

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Add overlap if necessary and not already covered
        if not chunks:
            return [text]

        return [c.strip() for c in chunks if len(c.strip()) >= self.min_chunk_size or len(chunks) == 1]

    def chunk_sections(self, sections: List[DocumentSection]) -> List[DocumentChunkData]:
        """
        Produce a sequential list of DocumentChunkData from the detected sections.
        """
        all_chunks: List[DocumentChunkData] = []
        chunk_idx = 0

        for section in sections:
            section_text = section.content.strip()
            if not section_text:
                continue

            if len(section_text) <= self.chunk_size:
                chunk_data = DocumentChunkData(
                    chunk_index=chunk_idx,
                    content=section_text,
                    section_title=section.title,
                    character_count=len(section_text),
                    token_count=max(1, len(section_text.split())),
                    metadata={"section": section.title},
                )
                all_chunks.append(chunk_data)
                chunk_idx += 1
            else:
                sub_chunks = self._split_text_recursively(
                    section_text,
                    max_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
                for sub in sub_chunks:
                    chunk_data = DocumentChunkData(
                        chunk_index=chunk_idx,
                        content=sub,
                        section_title=section.title,
                        character_count=len(sub),
                        token_count=max(1, len(sub.split())),
                        metadata={"section": section.title},
                    )
                    all_chunks.append(chunk_data)
                    chunk_idx += 1

        return all_chunks
