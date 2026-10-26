import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("agentforge.services.evaluation.cost")


class ModelPricing(BaseModel):
    """Configurable pricing structure per 1 Million tokens in USD."""
    input_price_per_1m: float = Field(..., description="USD cost per 1M prompt/input tokens")
    output_price_per_1m: float = Field(..., description="USD cost per 1M completion/output tokens")
    is_free: bool = Field(False, description="Whether endpoint is a verified free-tier endpoint")
    pricing_verified: bool = Field(True, description="Whether pricing for this model is verified")


class CostBreakdown(BaseModel):
    """Calculated cost breakdown for an LLM interaction."""
    input_cost: Optional[float] = Field(0.0, description="Cost for prompt tokens in USD (None if unavailable)")
    output_cost: Optional[float] = Field(0.0, description="Cost for completion tokens in USD (None if unavailable)")
    total_cost: Optional[float] = Field(0.0, description="Total cost in USD (None if unavailable)")
    is_free: bool = Field(False, description="Whether this interaction used a verified free endpoint")
    pricing_status: str = Field("calculated", description="'free', 'calculated', 'provider_reported', or 'unavailable'")


# Configurable default pricing registry (USD per 1M tokens)
# Free endpoints are explicitly verified with $0.00 pricing rather than guessed.
DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    # Google Gemini 3.5 Flash-Lite / 2.5 Flash-Lite
    "gemini-3.5-flash-lite": ModelPricing(input_price_per_1m=0.075, output_price_per_1m=0.30, is_free=False, pricing_verified=True),
    "gemini-2.5-flash-lite": ModelPricing(input_price_per_1m=0.075, output_price_per_1m=0.30, is_free=False, pricing_verified=True),
    "gemini": ModelPricing(input_price_per_1m=0.075, output_price_per_1m=0.30, is_free=False, pricing_verified=True),

    # Qwen 2.5 72B / Qwen 3.6 27B via OpenRouter
    "qwen/qwen3.6-27b": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30, is_free=False, pricing_verified=True),
    "qwen-3.6-27b": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30, is_free=False, pricing_verified=True),
    "qwen 3.6 27b": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30, is_free=False, pricing_verified=True),
    "qwen/qwen-2.5-72b-instruct": ModelPricing(input_price_per_1m=0.35, output_price_per_1m=0.40, is_free=False, pricing_verified=True),
    "qwen": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.30, is_free=False, pricing_verified=True),

    # NVIDIA Nemotron 3.5 Lightning 30B A3B via NVIDIA NIM
    "nvidia/nemotron-3.5-lightning-30b-a3b": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    "nemotron-3.5-lightning-30b-a3b": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    "nvidia nemotron 3.5 lightning 30b a3b": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    "nemotron-3.5-lightning": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    "nvidia nim": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    "nemotron": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    "nvidia": ModelPricing(input_price_per_1m=0.05, output_price_per_1m=0.20, is_free=False, pricing_verified=True),
    # Retired Nemotron nano aliases preserved for historical evaluations
    "nvidia/nemotron-3-nano-30b-a3b": ModelPricing(input_price_per_1m=0.06, output_price_per_1m=0.24, is_free=False, pricing_verified=True),
    "nemotron-3-nano-30b-a3b": ModelPricing(input_price_per_1m=0.06, output_price_per_1m=0.24, is_free=False, pricing_verified=True),
    "nvidia nemotron-3-nano-30b-a3b": ModelPricing(input_price_per_1m=0.06, output_price_per_1m=0.24, is_free=False, pricing_verified=True),

    # Google Gemma 4 26B A4B via OpenRouter (Preserved for historical evaluations)
    "google/gemma-4-26b-a4b-it:free": ModelPricing(input_price_per_1m=0.0, output_price_per_1m=0.0, is_free=True, pricing_verified=True),
    "gemma-4-26b-a4b-it:free": ModelPricing(input_price_per_1m=0.0, output_price_per_1m=0.0, is_free=True, pricing_verified=True),
    "gemma 4 26b a4b (openrouter)": ModelPricing(input_price_per_1m=0.0, output_price_per_1m=0.0, is_free=True, pricing_verified=True),
    "gemma 4 26b a4b": ModelPricing(input_price_per_1m=0.0, output_price_per_1m=0.0, is_free=True, pricing_verified=True),
    "gemma-4-26b": ModelPricing(input_price_per_1m=0.0, output_price_per_1m=0.0, is_free=True, pricing_verified=True),
    "gemma": ModelPricing(input_price_per_1m=0.0, output_price_per_1m=0.0, is_free=True, pricing_verified=True),

    # Legacy: GPT-OSS 120B via Hugging Face Inference Providers (Preserved for historical evaluations)
    "openai/gpt-oss-120b": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50, is_free=False, pricing_verified=True),
    "gpt-oss-120b": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50, is_free=False, pricing_verified=True),
    "gpt-oss": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50, is_free=False, pricing_verified=True),
    "gpt_oss": ModelPricing(input_price_per_1m=0.50, output_price_per_1m=0.50, is_free=False, pricing_verified=True),

    # Fallback default pricing (marked as unverified)
    "default": ModelPricing(input_price_per_1m=0.20, output_price_per_1m=0.40, is_free=False, pricing_verified=False),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """Retrieve pricing for a given model identifier or display name."""
    cleaned = (model_name or "").lower().strip()
    if cleaned in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[cleaned]

    for key, pricing in DEFAULT_MODEL_PRICING.items():
        if key != "default" and (key in cleaned or cleaned in key):
            return pricing

    return DEFAULT_MODEL_PRICING.get("default")


def calculate_cost(
    model_name: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    custom_pricing: Optional[ModelPricing] = None,
    provider_cost: Optional[float] = None,
) -> CostBreakdown:
    """
    Calculate input cost, output cost, and total cost based on token counts
    and model pricing configuration.

    Requirements:
    - If provider reported an explicit cost, record it directly as provider-reported.
    - If verified free model (e.g. Gemma 4 26B free tier), costs evaluate to $0.00 with is_free=True.
    - If pricing is unknown or unverified, cost is None (unavailable) instead of fabricating a fake price.
    - If token counts are None or missing, cost is None.
    """
    # 1. Provider-reported cost takes precedence if explicitly provided
    if provider_cost is not None and provider_cost >= 0:
        is_free_val = (provider_cost == 0.0)
        return CostBreakdown(
            input_cost=None,
            output_cost=None,
            total_cost=round(provider_cost, 7),
            is_free=is_free_val,
            pricing_status="free" if is_free_val else "provider_reported",
        )

    # 2. If token counts are missing, cost cannot be calculated
    if input_tokens is None or output_tokens is None:
        return CostBreakdown(
            input_cost=None,
            output_cost=None,
            total_cost=None,
            is_free=False,
            pricing_status="unavailable",
        )

    pricing = custom_pricing or get_model_pricing(model_name)
    if not pricing or not pricing.pricing_verified:
        # Mark as unavailable rather than inventing arbitrary numbers
        return CostBreakdown(
            input_cost=None,
            output_cost=None,
            total_cost=None,
            is_free=False,
            pricing_status="unavailable",
        )

    # 3. Verified free tier
    if pricing.is_free or (pricing.input_price_per_1m == 0.0 and pricing.output_price_per_1m == 0.0):
        return CostBreakdown(
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            is_free=True,
            pricing_status="free",
        )

    # 4. Standard verified pricing calculation
    safe_input_tokens = max(0, input_tokens)
    safe_output_tokens = max(0, output_tokens)

    input_cost = (safe_input_tokens / 1_000_000.0) * pricing.input_price_per_1m
    output_cost = (safe_output_tokens / 1_000_000.0) * pricing.output_price_per_1m
    total_cost = input_cost + output_cost

    return CostBreakdown(
        input_cost=round(input_cost, 7),
        output_cost=round(output_cost, 7),
        total_cost=round(total_cost, 7),
        is_free=False,
        pricing_status="calculated",
    )
