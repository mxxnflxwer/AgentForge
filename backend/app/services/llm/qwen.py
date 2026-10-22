import logging
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

logger = logging.getLogger("agentforge.services.llm.qwen")


class QwenAdapter(BaseLLMAdapter):
    """
    Adapter for Qwen 3.6 27B using OpenAI-compatible Chat Completions API via OpenRouter.
    """

    def __init__(
        self,
        name: str = "Qwen 3.6 27B",
        provider: str = "OpenRouter",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        model_name = model_id or getattr(settings, "QWEN_MODEL_NAME", "qwen/qwen3.6-27b")
        super().__init__(name=name, provider=provider, model_id=model_name)
        self.api_key = api_key if api_key is not None else getattr(settings, "QWEN_API_KEY", None)
        self.base_url = (base_url or getattr(settings, "QWEN_API_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
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
        Calls Qwen API via OpenRouter OpenAI-compatible Chat Completions endpoint.
        Accepts per-request generation overrides for temperature and max output tokens.
        """
        if not self.is_configured():
            logger.info("Qwen API key not configured. Returning fallback response.")
            return self.generate_offline_fallback(
                query=query,
                context=context,
                reason="QWEN_API_KEY is not configured in .env",
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

                if response.status_code == 402 or (response.status_code != 200 and "credit" in response.text.lower()):
                    err_msg = f"Insufficient provider credits (HTTP {response.status_code}): {response.text.strip()[:300]}"
                    logger.error(err_msg)
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        model_id=self.model_id,
                        answer=f"Insufficient provider credits: {self.provider} requires pre-paid credits to generate answers.",
                        latency_ms=latency_ms,
                        success=False,
                        error=err_msg,
                        raw_usage=None,
                    )

                if response.status_code != 200:
                    err_msg = f"Qwen API returned status {response.status_code}: {response.text[:200]}"
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

        except httpx.TimeoutException as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"Qwen API request timed out after {self.timeout}s"
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
            err_msg = f"Qwen API exception: {str(exc)}"
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
