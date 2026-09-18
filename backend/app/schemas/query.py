from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.evaluation import EvaluationResponse


class QueryAnswerRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The clinical or general question to ask about the document.",
        examples=["What is the clinical diagnosis?"],
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to scope query to a specific document.",
    )
    model: str = Field(
        "gemini",
        description="Selected LLM: 'gemini', 'qwen', or 'gpt_oss'.",
        examples=["gemini"],
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Optional reference answer to calculate accuracy alongside groundedness and hallucination.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve for context.",
    )


class QueryAnswerSource(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    section_title: Optional[str] = None
    similarity_score: float = 0.0
    relevance_score: Optional[float] = None


class QueryAnswerResponse(BaseModel):
    status: str = Field(..., description="'success', 'not_found', or 'error'")
    query: str
    intent: str
    model: str
    answer: str
    context: Optional[str] = Field(None, description="Retrieved document context sent to LLM.")
    sources: List[QueryAnswerSource] = []
    latency_ms: float
    disclaimer: str
    evaluation: Optional[EvaluationResponse] = Field(
        None,
        description="AgentEvo evaluation metrics (accuracy, groundedness, hallucination rate, tokens, cost, latency, execution time).",
    )


class QueryCompareRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The query to evaluate across all three LLMs simultaneously.",
        examples=["What is the clinical diagnosis?"],
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to scope retrieval to a specific document.",
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Optional reference answer to evaluate accuracy across all models.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for the common context.",
    )


class ModelComparisonResult(BaseModel):
    model: str
    provider: str
    answer: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    evaluation: Optional[EvaluationResponse] = Field(
        None,
        description="AgentEvo evaluation metrics specific to this model response.",
    )


class QueryCompareResponse(BaseModel):
    status: str
    query: str
    intent: str
    context: Optional[str] = None
    sources: List[QueryAnswerSource] = []
    results: List[ModelComparisonResult] = []
    disclaimer: str


class ModelMetadata(BaseModel):
    name: str
    provider: str
    model_id: str
    is_configured: bool


class AvailableModelsResponse(BaseModel):
    models: List[ModelMetadata]
