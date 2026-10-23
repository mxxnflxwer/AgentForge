import logging
import re
import time
from typing import Any, Dict, Optional
import httpx

from app.core.config import settings
from app.services.llm.base import (
    DEFAULT_MEDICAL_SYSTEM_PROMPT,
    BaseLLMAdapter,
    LLMResponse,
    build_medical_prompt,
)

logger = logging.getLogger("agentforge.services.llm.gemma")


def _sanitize_error_message(text: str) -> str:
    """Strip any potential API keys, bearer tokens, or secrets from error messages."""
    if not text:
        return ""
    # Strip bearer tokens or key patterns (e.g. sk-or-v1-..., hf_..., AIza...)
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{10,}", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    sanitized = re.sub(r"(sk-or-v1-|key=)[A-Za-z0-9_\-\.]{10,}", r"[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()[:300]


class GemmaAdapter(BaseLLMAdapter):
    """
    Adapter for Google Gemma 4 26B A4B free tier using OpenAI-compatible Chat Completions API via OpenRouter.
    Provides robust handling for HTTP 401, 402, 429 rate limits, timeouts, and model availability.
    """

    def __init__(
        self,
        name: str = "Gemma 4 26B A4B (OpenRouter)",
        provider: str = "OpenRouter",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        model_name = model_id or getattr(settings, "OPENROUTER_MODEL_ID", "google/gemma-4-26b-a4b-it:free")
        super().__init__(name=name, provider=provider, model_id=model_name)
        # Use OPENROUTER_API_KEY, fallback to QWEN_API_KEY or legacy GPT_OSS_API_KEY
        self.api_key = (
            api_key
            if api_key is not None
            else (
                getattr(settings, "OPENROUTER_API_KEY", None)
                or getattr(settings, "QWEN_API_KEY", None)
                or getattr(settings, "GPT_OSS_API_KEY", None)
            )
        )
        self.base_url = (
            base_url
            or getattr(settings, "OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.timeout = timeout or getattr(settings, "LLM_TIMEOUT_SECONDS", 30.0)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Calls Gemma 4 26B via OpenRouter OpenAI-compatible Chat Completions endpoint.
        Accepts per-request generation overrides for temperature and max output tokens.
        """
        if not self.is_configured():
            logger.info("OpenRouter API key not configured for Gemma. Returning fallback response.")
            return self.generate_offline_fallback(
                query=query,
                context=context,
                reason="OPENROUTER_API_KEY is not configured in .env",
            )

        sys_instruction = system_prompt or DEFAULT_MEDICAL_SYSTEM_PROMPT
        user_prompt = build_medical_prompt(query=query, context=context)

        eff_temp = (
            max(0.0, min(2.0, float(temperature)))
            if temperature is not None
            else getattr(settings, "LLM_TEMPERATURE", 0.1)
        )
        eff_max_tokens = (
            max(16, min(8192, int(max_tokens)))
            if max_tokens is not None
            else getattr(settings, "LLM_MAX_TOKENS", 1024)
        )

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://agentforge.dev",
            "X-Title": "AgentForge RAG Evaluation",
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": eff_temp,
            "max_tokens": eff_max_tokens,
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                # HTTP 401 Unauthorized / Invalid Key
                if response.status_code == 401:
                    err_msg = "Invalid or unauthorized API key (HTTP 401). Please verify OPENROUTER_API_KEY in backend configuration."
                    logger.error(f"Gemma OpenRouter API 401: {err_msg}")
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer=f"Error generating answer from {self.name}: Provider returned status 401 Unauthorized.",
                        latency_ms=latency_ms,
                        success=False,
                        error=err_msg,
                        raw_usage=None,
                    )

                # HTTP 402 Payment Required / Credits Exhausted
                if response.status_code == 402 or (response.status_code != 200 and "credit" in response.text.lower()):
                    raw_text_clean = _sanitize_error_message(response.text)
                    err_msg = f"Insufficient provider credits (HTTP {response.status_code}): {raw_text_clean}"
                    logger.error(err_msg)
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer=f"Insufficient provider credits: {self.provider} requires available credits to generate answers.",
                        latency_ms=latency_ms,
                        success=False,
                        error=err_msg,
                        raw_usage=None,
                    )

                # HTTP 429 Rate Limit Exceeded
                if response.status_code == 429:
                    err_msg = (
                        "OpenRouter rate limit exceeded (HTTP 429). The free model endpoint is currently throttled; "
                        "please retry shortly."
                    )
                    logger.warning(f"Gemma OpenRouter rate limit: {err_msg}")
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer=f"Rate limit exceeded: {self.name} is temporarily throttled by OpenRouter. Please wait a moment and retry.",
                        latency_ms=latency_ms,
                        success=False,
                        error=err_msg,
                        raw_usage=None,
                    )

                # HTTP 404 / 400 Model Unavailable or Provider Error
                if response.status_code != 200:
                    raw_text_clean = _sanitize_error_message(response.text)
                    err_msg = f"OpenRouter API returned status {response.status_code}: {raw_text_clean}"
                    logger.error(err_msg)
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer=f"Error generating answer from {self.name}: Provider returned status {response.status_code}.",
                        latency_ms=latency_ms,
                        success=False,
                        error=err_msg,
                        raw_usage=None,
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer="No answer returned by model.",
                        latency_ms=latency_ms,
                        success=False,
                        error="Empty choices list in response",
                        raw_usage=data.get("usage"),
                    )

                first_choice = choices[0]
                message = first_choice.get("message") or {}
                content = message.get("content")
                if content is None:
                    content = ""
                answer_text = content.strip()
                usage = data.get("usage")

                if not answer_text:
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer="",
                        latency_ms=latency_ms,
                        success=False,
                        error="Empty response content from model",
                        raw_usage=usage,
                    )

                return LLMResponse(
                    model=self.name,
                    provider=self.provider,
                    model_id=self.model_id,
                    answer=answer_text,
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                    raw_usage=usage,
                )

        except httpx.TimeoutException:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"OpenRouter API request timed out after {self.timeout}s"
            logger.error(err_msg)
            return LLMResponse(
                model=self.name,
                provider=self.provider,
                model_id=self.model_id,
                answer="Generation request timed out.",
                latency_ms=latency_ms,
                success=False,
                error=err_msg,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"OpenRouter API exception: {_sanitize_error_message(str(exc))}"
            logger.exception(err_msg)
            return LLMResponse(
                model=self.name,
                provider=self.provider,
                model_id=self.model_id,
                answer=f"Error communicating with {self.name}.",
                latency_ms=latency_ms,
                success=False,
                error=err_msg,
            )
