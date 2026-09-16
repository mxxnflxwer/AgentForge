import logging
import re
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger("agentforge.services.evaluation.metrics")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "from",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "should",
    "would", "patient", "report", "document", "provided", "uploaded", "clinical",
}


def extract_meaningful_tokens(text: str) -> Set[str]:
    """Extract lowercased alphanumeric tokens excluding common stop words."""
    words = re.findall(r"[a-zA-Z0-9_-]+", (text or "").lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def token_matches_set(token: str, token_set: Set[str]) -> bool:
    """Check if token matches any token in set either exactly or via root prefix."""
    if token in token_set:
        return True
    for other in token_set:
        if len(token) >= 5 and len(other) >= 5:
            # Common prefix of at least 5 chars (e.g., 'diagnos' in 'diagnosis' / 'diagnosed')
            prefix_len = min(len(token), len(other), 6)
            if token[:prefix_len] == other[:prefix_len]:
                return True
    return False


def calculate_accuracy(
    generated_answer: str,
    expected_answer: Optional[str] = None,
) -> Optional[float]:
    """
    Calculate factual and semantic accuracy against a reference/expected answer.

    Accuracy evaluation:
    1. Exact normalized match or complete substring containment -> 1.0 (100%).
    2. Factual keyword recall and precision (F1-score) with morphological root matching.
    3. Numerical and clinical entity consistency verification.

    Returns:
        float score between 0.0 and 1.0, or None if no reference answer is provided.
    """
    if expected_answer is None or not expected_answer.strip():
        return None

    gen_clean = (generated_answer or "").strip().lower()
    exp_clean = expected_answer.strip().lower()

    if not gen_clean:
        return 0.0

    # Exact string match or complete substring match
    if gen_clean == exp_clean or exp_clean in gen_clean:
        return 1.0

    gen_tokens = extract_meaningful_tokens(gen_clean)
    exp_tokens = extract_meaningful_tokens(exp_clean)

    if not exp_tokens:
        return 1.0 if not gen_tokens else 0.5

    # Match tokens with morphological flexibility
    matched_exp = {t for t in exp_tokens if token_matches_set(t, gen_tokens)}
    matched_gen = {t for t in gen_tokens if token_matches_set(t, exp_tokens)}

    recall = len(matched_exp) / float(len(exp_tokens))
    precision = len(matched_gen) / float(len(gen_tokens)) if gen_tokens else 0.0

    # If all expected tokens are recalled
    if recall == 1.0:
        return 1.0

    if recall + precision == 0.0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    # Numerical alignment check (e.g. wrong dosage or vital sign)
    exp_numbers = set(re.findall(r"\b\d+(?:[./]\d+)?\b", exp_clean))
    gen_numbers = set(re.findall(r"\b\d+(?:[./]\d+)?\b", gen_clean))

    if exp_numbers:
        missing_numbers = exp_numbers - gen_numbers
        number_recall = (len(exp_numbers) - len(missing_numbers)) / float(len(exp_numbers))
        combined_score = (0.7 * recall) + (0.3 * number_recall)
    else:
        combined_score = (0.75 * recall) + (0.25 * f1)

    return round(max(0.0, min(1.0, combined_score)), 4)


class TokenUsage(BaseModel):
    """Standardized token usage counts."""
    input_tokens: int = Field(..., description="Prompt/input tokens")
    output_tokens: int = Field(..., description="Completion/output tokens")
    total_tokens: int = Field(..., description="Total tokens: input + output")


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation (~1.3 tokens per word for medical/technical text)."""
    words = (text or "").split()
    return max(1, int(len(words) * 1.3))


def extract_token_usage(
    raw_usage: Optional[Dict[str, Any]],
    generated_answer: str,
    retrieved_context: str,
    query: str = "",
) -> TokenUsage:
    """
    Extract accurate token counts from provider metadata if available,
    or calculate heuristic token metrics as a reliable fallback.
    Guarantees total_tokens = input_tokens + output_tokens.
    """
    input_tokens = 0
    output_tokens = 0

    if raw_usage and isinstance(raw_usage, dict):
        # Google Gemini format
        if "promptTokenCount" in raw_usage or "candidatesTokenCount" in raw_usage:
            input_tokens = int(raw_usage.get("promptTokenCount", 0))
            output_tokens = int(raw_usage.get("candidatesTokenCount", 0))
        # OpenAI / OpenRouter / Hugging Face format
        elif "prompt_tokens" in raw_usage or "completion_tokens" in raw_usage:
            input_tokens = int(raw_usage.get("prompt_tokens", 0))
            output_tokens = int(raw_usage.get("completion_tokens", 0))
        elif "total_tokens" in raw_usage:
            total = int(raw_usage.get("total_tokens", 0))
            output_tokens = estimate_tokens(generated_answer)
            input_tokens = max(0, total - output_tokens)

    # Heuristic fallback if provider usage is absent or zero
    if input_tokens <= 0:
        prompt_text = f"{query}\n{retrieved_context}"
        input_tokens = estimate_tokens(prompt_text)

    if output_tokens <= 0:
        output_tokens = estimate_tokens(generated_answer)

    total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
