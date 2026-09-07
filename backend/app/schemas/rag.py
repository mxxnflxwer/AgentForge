from typing import List, Optional
from pydantic import BaseModel, Field


class RAGSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The semantic query string to search for.",
        examples=["What is the clinical diagnosis?"],
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to scope retrieval to a single document.",
        examples=["472851a7-1959-4d2c-9a4f-d007c57b8567"],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of most relevant chunks to retrieve.",
        examples=[5],
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    section_title: Optional[str] = None
    distance: float = 0.0
    similarity_score: float = 0.0
    relevance_score: Optional[float] = None


class SectionSummary(BaseModel):
    section_title: str
    content: str
    chunk_indices: List[int] = []


class RAGDebugInfo(BaseModel):
    retrieved_chunks: int
    filtered_chunks: int
    threshold: float
    intent: str


class RAGSearchResponse(BaseModel):
    query: str
    status: str = "success"
    intent: str = "specific_fact"
    total_results: int
    message: Optional[str] = None
    document_id: Optional[str] = None
    sections: Optional[List[SectionSummary]] = None
    summary_context: Optional[str] = None
    results: List[RetrievedChunk] = []
    debug: Optional[RAGDebugInfo] = None
