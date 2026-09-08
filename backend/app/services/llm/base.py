import abc
import logging
import re
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("agentforge.services.llm.base")

DEFAULT_MEDICAL_SYSTEM_PROMPT = """You are a medical document analysis assistant.

Answer ONLY using the provided document context.
Do not invent medical information.
If the answer is not present in the provided context, clearly say that the information was not found in the uploaded document.
Do not make unsupported assumptions.
Do not present generated information as if it came from the document."""

MEDICAL_DISCLAIMER = (
    "This document analysis is for research and demonstration purposes only. "
    "It does not constitute medical advice or substitute for professional clinical judgment."
)


def build_medical_prompt(query: str, context: str) -> str:
    """
    Format a strict context-grounded prompt separating Document Context,
    User Question, and Safety Instructions.
    """
    cleaned_context = (context or "").strip()
    cleaned_query = (query or "").strip()

    return f"""[DOCUMENT CONTEXT]
{cleaned_context}

[USER QUESTION]
{cleaned_query}

[INSTRUCTIONS]
Answer the user question strictly using the provided document context above.
If the answer is not present in the provided context, reply with:
"The requested information was not found in the uploaded document."
Do not extrapolate or assume unstated medical facts."""


class LLMResponse(BaseModel):
    """Standardized response format across all LLM adapters."""
    model: str = Field(..., description="Display model name, e.g. 'Gemini 2.5 Flash-Lite'")
    provider: str = Field(..., description="Provider name, e.g. 'Google', 'Qwen', 'OpenAI-Compatible'")
    answer: str = Field(..., description="Generated answer text")
    latency_ms: float = Field(0.0, description="Round-trip latency in milliseconds")
    success: bool = Field(True, description="Whether the generation completed successfully")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    raw_usage: Optional[Dict[str, Any]] = Field(None, description="Optional token usage data")


class BaseLLMAdapter(abc.ABC):
    """
    Abstract base adapter for LLM providers.
    Ensures provider-agnostic interchangeability and strict standardized outputs.
    """

    def __init__(self, name: str, provider: str, model_id: str):
        self.name = name
        self.provider = provider
        self.model_id = model_id

    @abc.abstractmethod
    def is_configured(self) -> bool:
        """Check if required API credentials/configurations are present."""
        pass

    @abc.abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a grounded answer for the user query given the retrieved context.
        """
        pass

    def generate_offline_fallback(
        self,
        query: str,
        context: str,
        reason: str,
    ) -> LLMResponse:
        """
        Generate a safe, grounded extractive synthesis when an API key is unconfigured
        or the external endpoint is unreachable. Ensures development/demo workflows
        remain functional without throwing 500 errors.
        """
        start = time.perf_counter()
        cleaned_context = (context or "").strip()
        cleaned_query = (query or "").strip()

        if not cleaned_context:
            ans = "The requested information was not found in the uploaded document."
        else:
            # Context-grounded synthesis
            lines = [l.strip() for l in cleaned_context.split("\n") if l.strip()]
            relevant_lines = []
            query_words = set(re.findall(r"\w+", cleaned_query.lower()))

            for line in lines:
                if line.startswith("###") or line.startswith("---"):
                    continue
                line_words = set(re.findall(r"\w+", line.lower()))
                if query_words & line_words:
                    relevant_lines.append(line)

            if relevant_lines:
                extracted = " ".join(relevant_lines[:3])
                ans = f"[Simulation / No API Key ({reason})] {extracted}"
            else:
                summary_sample = " ".join(lines[:2])
                ans = f"[Simulation / No API Key ({reason})] Based on the document context: {summary_sample}"

        latency = round((time.perf_counter() - start) * 1000, 2)
        return LLMResponse(
            model=self.name,
            provider=self.provider,
            answer=ans,
            latency_ms=latency,
            success=False,
            error=reason,
        )
