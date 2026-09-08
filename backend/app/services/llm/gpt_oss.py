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

logger = logging.getLogger("agentforge.services.llm.gpt_oss")


class GPTOSSAdapter(BaseLLMAdapter):
    """
    Adapter for GPT-OSS 120B using OpenAI-compatible Chat Completions API
    (e.g., OpenRouter, vLLM, Groq, Ollama, or custom endpoint).
    """

    def __init__(
        self,
        name: str = "GPT-OSS 120B",
        provider: str = "OpenAI-Compatible",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        model_name = model_id or getattr(settings, "GPT_OSS_MODEL_NAME", "openai/gpt-oss-120b")
        super().__init__(name=name, provider=provider, model_id=model_name)
        self.api_key = api_key or getattr(settings, "GPT_OSS_API_KEY", None)
        self.base_url = (base_url or getattr(settings, "GPT_OSS_API_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.timeout = timeout or getattr(settings, "LLM_TIMEOUT_SECONDS", 30.0)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Calls GPT-OSS 120B API via OpenAI-compatible Chat Completions endpoint.
        """
        if not self.is_configured():
            logger.info("GPT-OSS API key not configured. Returning fallback response.")
            return self.generate_offline_fallback(
                query=query,
                context=context,
                reason="GPT_OSS_API_KEY is not configured in .env",
            )

        sys_instruction = system_prompt or DEFAULT_MEDICAL_SYSTEM_PROMPT
        user_prompt = build_medical_prompt(query=query, context=context)

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
            "temperature": getattr(settings, "LLM_TEMPERATURE", 0.1),
            "max_tokens": getattr(settings, "LLM_MAX_TOKENS", 1024),
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                if response.status_code != 200:
                    err_msg = f"GPT-OSS API returned status {response.status_code}: {response.text[:200]}"
                    logger.error(err_msg)
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        answer=f"Error generating answer from {self.name}.",
                        latency_ms=latency_ms,
                        success=False,
                        error=err_msg,
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        answer="No answer returned by model.",
                        latency_ms=latency_ms,
                        success=False,
                        error="Empty choices list in response",
                    )

                first_choice = choices[0]
                message = first_choice.get("message", {})
                answer_text = message.get("content", "").strip()
                usage = data.get("usage")

                return LLMResponse(
                    model=self.name,
                    provider=self.provider,
                    answer=answer_text or "The requested information was not found in the uploaded document.",
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                    raw_usage=usage,
                )

        except httpx.TimeoutException as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"GPT-OSS API request timed out after {self.timeout}s: {exc}"
            logger.error(err_msg)
            return LLMResponse(
                model=self.name,
                provider=self.provider,
                answer="Generation request timed out.",
                latency_ms=latency_ms,
                success=False,
                error=err_msg,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = f"GPT-OSS API exception: {str(exc)}"
            logger.exception(err_msg)
            return LLMResponse(
                model=self.name,
                provider=self.provider,
                answer=f"Error communicating with {self.name}.",
                latency_ms=latency_ms,
                success=False,
                error=err_msg,
            )
