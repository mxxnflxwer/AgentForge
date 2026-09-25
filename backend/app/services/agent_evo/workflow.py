from enum import Enum
import copy
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MutationType(str, Enum):
    """Supported mutation categories for AgentEvo workflow exploration."""
    PROMPT = "prompt"
    RETRIEVAL = "retrieval"
    MODEL = "model"
    OPERATOR = "operator"
    MIXED = "mixed"


class MutationMetadata(BaseModel):
    """
    Structured metadata explaining why and how a candidate was generated.
    Provides complete traceability for evolutionary search research.
    """
    mutation_type: MutationType = Field(..., description="Category of mutation applied")
    parent_workflow_id: str = Field("baseline", description="ID of the parent workflow")
    mutation_description: str = Field(..., description="Human-readable explanation of the mutation")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Detailed mutation parameters and generation context")


class RetrievalConfig(BaseModel):
    """Configuration for RAG retrieval stage."""
    chunk_size: int = Field(1000, ge=100, le=4000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap in tokens")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(0.35, ge=0.0, le=1.0, description="Minimum similarity relevance score threshold")
    reranking: bool = Field(True, description="Whether section/intent-aware reranking is enabled")


class ModelConfig(BaseModel):
    """Configuration for LLM generation stage."""
    provider: str = Field("Google", description="Provider name: Google, OpenRouter, Hugging Face")
    model_id: str = Field("gemini-3.5-flash-lite", description="Actual model identifier")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(1024, ge=64, le=4096, description="Max candidate tokens")


class PromptConfig(BaseModel):
    """Configuration for prompt construction."""
    template_name: str = Field("default_medical", description="Name of the prompt template (default_medical, concise_medical, detailed_clinical)")
    custom_instructions: Optional[str] = Field(None, description="Optional custom instruction override")


class ExecutionConfig(BaseModel):
    """Execution runtime parameters."""
    timeout_seconds: float = Field(30.0, ge=5.0, le=120.0, description="Execution timeout in seconds")
    context_compression: bool = Field(False, description="Whether to compress context before prompt construction")


class AgentWorkflow(BaseModel):
    """
    Structured representation of an executable AgentForge workflow.
    Fully serializable for evolutionary candidate generation, mutation, and storage.
    """
    workflow_id: str = Field("baseline", description="Unique workflow/candidate identifier")
    name: str = Field("Baseline Workflow", description="Human-readable workflow name")
    description: str = Field("Current production baseline configuration", description="Description of workflow changes")
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    mutation_metadata: Optional[MutationMetadata] = Field(None, description="Exploration mutation metadata if generated")

    def clone(
        self,
        new_id: str,
        new_name: str,
        new_description: str = "",
        mutation_metadata: Optional[MutationMetadata] = None,
    ) -> "AgentWorkflow":
        """Create an isolated deep copy of this workflow with a new ID, description, and mutation metadata."""
        dumped = copy.deepcopy(self.model_dump())
        dumped["workflow_id"] = new_id
        dumped["name"] = new_name
        dumped["description"] = new_description
        if mutation_metadata is not None:
            dumped["mutation_metadata"] = mutation_metadata.model_dump()
        return AgentWorkflow(**dumped)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard serializable dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentWorkflow":
        """Reconstruct workflow from dictionary."""
        return cls(**data)


def get_baseline_workflow() -> AgentWorkflow:
    """
    Create the reference Baseline Workflow representing current AgentForge production parameters.
    """
    return AgentWorkflow(
        workflow_id="baseline",
        name="Production Baseline",
        description="Standard baseline configuration: Gemini 3.5 Flash-Lite, top_k=5, threshold=0.35, default prompt.",
        retrieval=RetrievalConfig(
            chunk_size=1000,
            chunk_overlap=200,
            top_k=5,
            similarity_threshold=0.35,
            reranking=True,
        ),
        model=ModelConfig(
            provider="Google",
            model_id="gemini-3.5-flash-lite",
            temperature=0.1,
            max_tokens=1024,
        ),
        prompt=PromptConfig(
            template_name="default_medical",
            custom_instructions=None,
        ),
        execution=ExecutionConfig(
            timeout_seconds=30.0,
            context_compression=False,
        ),
        mutation_metadata=None,
    )
