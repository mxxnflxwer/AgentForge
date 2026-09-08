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

logger = logging.getLogger("agentforge.services.llm.gemini")


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Gemini 2.5 Flash-Lite using Google Generative Language REST API.
    """

    def __init__(
        self,
        name: str = "Gemini 2.5 Flash-Lite",
        provider: str = "Google",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        model_name = model_id or getattr(settings, "GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
        super().__init__(name=name, provider=provider, model_id=model_name)
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", None)
        self.base_url = (base_url or getattr(settings, "GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")).rstrip("/")
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
        Calls Gemini 2.5 Flash-Lite API with anti-hallucination context and system prompt.
        """
        if not self.is_configured():
            logger.info("Gemini API key not configured. Returning fallback response.")
            return self.generate_offline_fallback(
                query=query,
                context=context,
                reason="GEMINI_API_KEY is not configured in .env",
            )

        sys_instruction = system_prompt or DEFAULT_MEDICAL_SYSTEM_PROMPT
        user_prompt = build_medical_prompt(query=query, context=context)

        url = f"{self.base_url}/models/{self.model_id}:generateContent?key={self.api_key}"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": sys_instruction}],
            },
            "generationConfig": {
                "temperature": getattr(settings, "LLM_TEMPERATURE", 0.1),
                "maxOutputTokens": getattr(settings, "LLM_MAX_TOKENS", 1024),
            },
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                if response.status_code != 200:
                    err_msg = f"Gemini API returned status {response.status_code}: {response.text[:200]}"
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
                candidates = data.get("candidates", [])
                if not candidates:
                    return LLMResponse(
                        model=self.name,
                        provider=self.provider,
                        answer="No answer returned by model.",
                        latency_ms=latency_ms,
                        success=False,
                        error="Empty candidates list in response",
                    )

                first_candidate = candidates[0]
                content = first_candidate.get("content", {})
                parts = content.get("parts", [])
                text_parts = [p.get("text", "") for p in parts if "text" in p]
                answer_text = "".join(text_parts).strip()

                usage = data.get("usageMetadata")

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
            err_msg = f"Gemini API request timed out after {self.timeout}s: {exc}"
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
            err_msg = f"Gemini API exception: {str(exc)}"
            logger.exception(err_msg)
            return LLMResponse(
                model=self.name,
                provider=self.provider,
                answer=f"Error communicating with {self.name}.",
                latency_ms=latency_ms,
                success=False,
                error=err_msg,
            )
