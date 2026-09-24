import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from app.services.agent_evo.workflow import AgentWorkflow

logger = logging.getLogger("agentforge.services.agent_evo.validator")

SUPPORTED_MODELS = {
    "gemini-3.5-flash-lite": "Google",
    "gemini-2.5-flash-lite": "Google",
    "qwen/qwen3.6-27b": "OpenRouter",
    "qwen-3.6-27b": "OpenRouter",
    "openai/gpt-oss-120b": "Hugging Face",
    "gpt-oss-120b": "Hugging Face",
}

SUPPORTED_PROMPT_TEMPLATES = {
    "default_medical",
    "concise_medical",
    "detailed_clinical",
}


class ValidationResult(BaseModel):
    """Result of workflow configuration validation."""
    is_valid: bool = Field(..., description="Whether workflow is valid for execution")
    errors: List[str] = Field(default_factory=list, description="List of validation errors if any")
    workflow_id: str = Field(..., description="ID of validated workflow")


class WorkflowValidator:
    """
    Validates candidate workflow configurations before execution.
    Prevents execution of malformed, unsupported, or dangerous parameter combinations.
    """

    @classmethod
    def validate(cls, workflow: AgentWorkflow) -> ValidationResult:
        errors: List[str] = []

        # 1. Retrieval validation
        r = workflow.retrieval
        if r.top_k < 1 or r.top_k > 20:
            errors.append(f"top_k ({r.top_k}) must be between 1 and 20.")
        if r.similarity_threshold < 0.0 or r.similarity_threshold > 1.0:
            errors.append(f"similarity_threshold ({r.similarity_threshold}) must be between 0.0 and 1.0.")
        if r.chunk_size < 100 or r.chunk_size > 4000:
            errors.append(f"chunk_size ({r.chunk_size}) must be between 100 and 4000 tokens.")
        if r.chunk_overlap < 0 or r.chunk_overlap >= r.chunk_size:
            errors.append(f"chunk_overlap ({r.chunk_overlap}) must be non-negative and smaller than chunk_size ({r.chunk_size}).")

        # 2. Model validation
        m = workflow.model
        norm_model = (m.model_id or "").lower().strip()
        if norm_model not in SUPPORTED_MODELS:
            errors.append(
                f"Model '{m.model_id}' is unsupported. Must be one of: {list(SUPPORTED_MODELS.keys())}."
            )
        if m.temperature < 0.0 or m.temperature > 1.0:
            errors.append(f"temperature ({m.temperature}) must be between 0.0 and 1.0.")
        if m.max_tokens < 64 or m.max_tokens > 4096:
            errors.append(f"max_tokens ({m.max_tokens}) must be between 64 and 4096.")

        # 3. Prompt validation
        p = workflow.prompt
        if p.template_name not in SUPPORTED_PROMPT_TEMPLATES:
            errors.append(
                f"Prompt template '{p.template_name}' is unsupported. Must be one of: {list(SUPPORTED_PROMPT_TEMPLATES)}."
            )

        # 4. Execution validation
        e = workflow.execution
        if e.timeout_seconds < 5.0 or e.timeout_seconds > 120.0:
            errors.append(f"timeout_seconds ({e.timeout_seconds}) must be between 5.0s and 120.0s.")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Workflow '{workflow.workflow_id}' failed validation: {'; '.join(errors)}")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            workflow_id=workflow.workflow_id,
        )


def validate_workflow(workflow: AgentWorkflow) -> ValidationResult:
    """Convenience helper to validate a workflow."""
    return WorkflowValidator.validate(workflow)
