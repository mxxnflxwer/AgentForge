from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The question evaluated.",
        examples=["What is the clinical diagnosis?"],
    )
    generated_answer: str = Field(
        ...,
        description="The generated answer from the LLM.",
        examples=["The patient is diagnosed with stable angina pectoris."],
    )
    retrieved_context: str = Field(
        ...,
        description="The retrieved context against which groundedness is verified.",
        examples=["Assessment: Stable angina pectoris."],
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Optional ground truth reference answer for accuracy calculation.",
        examples=["Stable angina pectoris"],
    )
    model_name: str = Field(
        "gemini-3.5-flash-lite",
        description="Evaluated model name or identifier.",
        examples=["gemini-3.5-flash-lite"],
    )
    input_tokens: Optional[int] = Field(None, ge=0, description="Optional input token count.")
    output_tokens: Optional[int] = Field(None, ge=0, description="Optional output token count.")
    latency_ms: Optional[float] = Field(None, ge=0.0, description="Optional LLM generation latency in ms.")
    execution_time_ms: Optional[float] = Field(None, ge=0.0, description="Optional total pipeline execution time in ms.")


class EvaluationResponse(BaseModel):
    model: str = Field(..., description="Evaluated model identifier.")
    accuracy: Optional[float] = Field(None, description="Accuracy score [0.0 - 1.0] if reference answer was supplied.")
    groundedness: Optional[float] = Field(None, description="Groundedness score: supported_claims / total_claims [0.0 - 1.0]. None for errors or empty responses.")
    hallucination_rate: Optional[float] = Field(None, description="Hallucination rate: unsupported_claims / total_claims [0.0 - 1.0]. None for errors or empty responses.")
    supported_claims: List[str] = Field(default_factory=list, description="List of supported factual assertions.")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unsupported or hallucinated claims.")
    supported_claim_count: int = Field(0, description="Count of supported factual claims.")
    unsupported_claim_count: int = Field(0, description="Count of unsupported or hallucinated claims.")
    total_claims: int = Field(..., description="Total evaluated claims.")
    input_tokens: int = Field(..., description="Input prompt tokens.")
    output_tokens: int = Field(..., description="Output completion tokens.")
    total_tokens: int = Field(..., description="Total tokens: input + output.")
    latency_ms: float = Field(..., description="LLM latency in milliseconds.")
    execution_time_ms: float = Field(..., description="Total pipeline execution time in milliseconds.")
    input_cost: float = Field(..., description="Estimated input cost in USD.")
    output_cost: float = Field(..., description="Estimated output cost in USD.")
    total_cost: float = Field(..., description="Estimated total cost in USD.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic claim verification details.")


class WorkflowEvaluationQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Query to execute through RAG + LLM and evaluate with AgentEvo engine.",
        examples=["What is the clinical diagnosis?"],
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID filter.",
    )
    model: str = Field(
        "gemini",
        description="Selected LLM model: 'gemini', 'qwen', or 'gpt_oss'.",
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Optional reference answer to calculate accuracy alongside groundedness and hallucination.",
    )
    top_k: int = Field(5, ge=1, le=20, description="Retrieval chunk count.")


class WorkflowEvaluationQueryResponse(BaseModel):
    query: str
    model: str
    answer: str
    context: str
    sources: List[Dict[str, Any]] = []
    evaluation: EvaluationResponse
    disclaimer: str
