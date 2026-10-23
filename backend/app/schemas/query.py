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
        description="Selected LLM: 'gemini', 'qwen', or 'gemma' (legacy 'gpt_oss' mapped to gemma).",
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
    workflow_version_id: Optional[str] = Field(
        None,
        description="Optional explicit workflow version ID or 'baseline' override.",
    )


class QueryAnswerSource(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    section_title: Optional[str] = None
    similarity_score: float = 0.0
    relevance_score: Optional[float] = None


class QueryAnswerResponse(BaseModel):
    status: str = Field(..., description="'success', 'not_found', 'insufficient_credits', or 'error'")
    query: str
    intent: str
    model: str
    model_id: Optional[str] = Field(None, description="Actual model identifier used")
    provider: Optional[str] = Field(None, description="Model provider name")
    success: bool = Field(True, description="Whether model generation completed successfully")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    answer: str
    context: Optional[str] = Field(None, description="Retrieved document context sent to LLM.")
    sources: List[QueryAnswerSource] = []
    latency_ms: float
    disclaimer: str
    pipeline_type: str = Field("production", description="Pipeline category: 'production', 'baseline', or 'comparison'")
    evaluation: Optional[EvaluationResponse] = Field(
        None,
        description="AgentEvo evaluation metrics (accuracy, groundedness, hallucination rate, tokens, cost, latency, execution time).",
    )
    execution_id: Optional[str] = Field(None, description="Unique runtime execution identifier for telemetry & feedback")
    workflow_version_id: Optional[str] = Field(None, description="Active workflow version ID used for this execution")
    workflow_name: Optional[str] = Field(None, description="Active or selected workflow name")
    task_category: Optional[str] = Field(None, description="Classified clinical task category for the query")
    selection_reason: Optional[str] = Field(None, description="Explainable selection rationale for the chosen workflow")
    groundedness: Optional[float] = Field(None, description="Groundedness score directly from evaluation")
    execution_trace: Optional[Dict[str, Any]] = Field(None, description="Detailed stage execution trace")


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
    comparison_mode: Optional[str] = Field(
        "active_workflow",
        description="Comparison mode: 'active_workflow', 'baseline', or 'explicit_version'",
    )
    workflow_version_id: Optional[str] = Field(
        None,
        description="Optional explicit workflow version ID or 'baseline' override.",
    )


class ModelComparisonResult(BaseModel):
    model: str
    model_id: Optional[str] = None
    provider: str
    answer: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    pipeline_type: str = Field("comparison_shared_retrieval", description="Execution pipeline type")
    workflow_version_id: Optional[str] = Field(None, description="Actual workflow version ID (None for comparative baseline)")
    workflow_name: str = Field("Multi-LLM Comparative Baseline", description="Actual workflow name")
    context_type: str = Field("shared", description="Whether context is shared or independent")
    context_hash: Optional[str] = Field(None, description="SHA-256 hash of the shared context")
    execution_trace: Optional[Dict[str, Any]] = Field(None, description="Detailed stage execution trace for this model")
    evaluation: Optional[EvaluationResponse] = Field(
        None,
        description="AgentEvo evaluation metrics specific to this model response.",
    )


class QueryCompareResponse(BaseModel):
    status: str
    query: str
    intent: str
    context: Optional[str] = None
    context_hash: Optional[str] = None
    workflow_version_id: Optional[str] = None
    workflow_name: Optional[str] = None
    comparison_mode: Optional[str] = None
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
