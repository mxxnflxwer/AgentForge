import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("agentforge.services.evaluation.cost")


class ModelPricing(BaseModel):
    """Configurable pricing structure per 1 Million tokens in USD."""
    input_price_per_1m: float = Field(..., description="USD cost per 1M prompt/input tokens")
    output_price_per_1m: float = Field(..., description="USD cost per 1M completion/output tokens")


class CostBreakdown(BaseModel):
    """Calculated cost breakdown for an LLM interaction."""
    input_cost: float = Field(0.0, description="Cost for prompt tokens in USD")
    output_cost: float = Field(0.0, description="Cost for completion tokens in USD")
    total_cost: float = Field(0.0, description="Total cost in USD")


# Configurable default pricing registry (USD per 1M tokens)
# Can be updated or augmented from settings/environment
DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    # Google Gemini 3.5 Flash-Lite / 2.5 Flash-Lite
    "gemini-3.5-flash-lite": ModelPricing(input_price_per_1m=0.075, output_price_per_1m=0.30),
    "gemini-2.5-flash-lite": ModelPricing(input_price_per_1m=0.075, output_price_per_1m=0.30),
    "gemini": ModelPricing(input_price_per_1m=0.075, output_price_per_1m=0.30),

    # Qwen 2.5 72B / Qwen 3.6 27B via OpenRouter
    "qwen/qwen3.6-27b": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30),
    "qwen-3.6-27b": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30),
    "qwen 3.6 27b": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30),
    "qwen/qwen-2.5-72b-instruct": ModelPricing(input_price_per_1m=0.35, output_price_per_1m=0.40),
    "qwen": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30),

    # GPT-OSS 120B via Hugging Face Inference Providers
    "openai/gpt-oss-120b": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50),
    "gpt-oss-120b": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50),
    "gpt-oss": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50),
    "gpt_oss": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50),

    # Fallback default pricing
    "default": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.40),
}


def get_model_pricing(model_name: str) -> ModelPricing:
    """Retrieve pricing for a given model identifier or display name."""
    cleaned = (model_name or "").lower().strip()
    if cleaned in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[cleaned]

    for key, pricing in DEFAULT_MODEL_PRICING.items():
        if key in cleaned or cleaned in key:
            return pricing

    return DEFAULT_MODEL_PRICING["default"]


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: Optional[ModelPricing] = None,
) -> CostBreakdown:
    """
    Calculate input cost, output cost, and total cost based on token counts
    and model pricing configuration.

    Formula:
        input_cost = (input_tokens / 1,000,000) * input_price_per_1m
        output_cost = (output_tokens / 1,000,000) * output_price_per_1m
        total_cost = input_cost + output_cost
    """
    pricing = custom_pricing or get_model_pricing(model_name)

    safe_input_tokens = max(0, input_tokens)
    safe_output_tokens = max(0, output_tokens)

    input_cost = (safe_input_tokens / 1_000_000.0) * pricing.input_price_per_1m
    output_cost = (safe_output_tokens / 1_000_000.0) * pricing.output_price_per_1m
    total_cost = input_cost + output_cost

    return CostBreakdown(
        input_cost=round(input_cost, 7),
        output_cost=round(output_cost, 7),
        total_cost=round(total_cost, 7),
    )
