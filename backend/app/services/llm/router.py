import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.llm.base import BaseLLMAdapter, LLMResponse
from app.services.llm.gemini import GeminiAdapter
from app.services.llm.gemma import GemmaAdapter
from app.services.llm.gpt_oss import GPTOSSAdapter
from app.services.llm.nemotron import NemotronAdapter
from app.services.llm.qwen import QwenAdapter

logger = logging.getLogger("agentforge.services.llm.router")


class LLMRouter:
    """
    Central router for LLM generation and multi-model comparative evaluation.
    Enforces unified interfaces and concurrent multi-model comparisons across:
    1. Gemini 3.5 Flash-Lite (Google)
    2. Qwen 3.6 27B (OpenRouter)
    3. NVIDIA Nemotron-3-Nano-30B-A3B (NVIDIA Hosted API)
    """

    def __init__(self):
        self._adapters: Dict[str, BaseLLMAdapter] = {}
        self._initialize_adapters()

    def _initialize_adapters(self):
        gemini = GeminiAdapter()
        qwen = QwenAdapter()
        nemotron = NemotronAdapter()

        self.register_adapter("gemini", gemini)
        self.register_adapter("gemini-3.5-flash-lite", gemini)
        self.register_adapter("gemini-2.5-flash-lite", gemini)

        self.register_adapter("qwen", qwen)
        self.register_adapter("qwen-3.6-27b", qwen)
        self.register_adapter("qwen/qwen3.6-27b", qwen)
        self.register_adapter("qwen-2.5-72b", qwen)
        self.register_adapter("qwen/qwen-2.5-72b-instruct", qwen)

        # NVIDIA Nemotron 3.5 Lightning 30B A3B (NVIDIA NIM)
        self.register_adapter("nemotron", nemotron)
        self.register_adapter("nvidia", nemotron)
        self.register_adapter("nvidia nim", nemotron)
        self.register_adapter("nvidia-nim", nemotron)
        self.register_adapter("nemotron-3.5-lightning", nemotron)
        self.register_adapter("nemotron-3.5-lightning-30b-a3b", nemotron)
        self.register_adapter("nvidia/nemotron-3.5-lightning-30b-a3b", nemotron)
        # Retired Nemotron nano aliases for backward compatibility
        self.register_adapter("nemotron-3-nano-30b-a3b", nemotron)
        self.register_adapter("nvidia/nemotron-3-nano-30b-a3b", nemotron)
        self.register_adapter("nemotron-3-nano", nemotron)

        # Legacy aliases mapped to Nemotron for backward compatibility
        self.register_adapter("gemma", nemotron)
        self.register_adapter("gemma-4-26b", nemotron)
        self.register_adapter("gemma-4-26b-a4b-it:free", nemotron)
        self.register_adapter("google/gemma-4-26b-a4b-it:free", nemotron)
        self.register_adapter("openrouter_gemma", nemotron)
        self.register_adapter("openrouter", nemotron)
        self.register_adapter("gpt_oss", nemotron)
        self.register_adapter("gpt-oss", nemotron)
        self.register_adapter("gpt-oss-120b", nemotron)
        self.register_adapter("openai/gpt-oss-120b", nemotron)

    def register_adapter(self, key: str, adapter: BaseLLMAdapter):
        normalized_key = key.lower().strip()
        self._adapters[normalized_key] = adapter

    def get_adapter(self, key: str) -> Optional[BaseLLMAdapter]:
        normalized_key = (key or "").lower().strip()
        return self._adapters.get(normalized_key)

    def get_unique_adapters(self) -> List[BaseLLMAdapter]:
        """Return the unique 3 primary adapters."""
        seen = set()
        unique = []
        for adapter in self._adapters.values():
            if adapter.name not in seen:
                seen.add(adapter.name)
                unique.append(adapter)
        return unique

    def list_models(self) -> List[Dict[str, Any]]:
        """Return metadata for all supported models."""
        return [
            {
                "name": adapter.name,
                "provider": adapter.provider,
                "model_id": adapter.model_id,
                "is_configured": adapter.is_configured(),
            }
            for adapter in self.get_unique_adapters()
        ]

    async def generate_single(
        self,
        model_key: str,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Execute grounded generation on a single selected model.
        Supports per-request overrides for temperature and max_tokens.
        """
        adapter = self.get_adapter(model_key)
        if not adapter:
            logger.error(f"No adapter registered for model: '{model_key}'")
            return LLMResponse(
                model=model_key,
                provider="Unknown",
                model_id=model_key,
                answer=f"Model adapter for '{model_key}' is not registered or supported.",
                latency_ms=0.0,
                success=False,
                error=f"Unsupported model: {model_key}",
            )

        logger.info(f"Routing query to single model: {adapter.name}")
        return await adapter.generate(
            query=query,
            context=context,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def compare_all(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[LLMResponse]:
        """
        Execute grounded generation across all 3 models concurrently using the
        exact same query, context, and system instructions for fair comparison.
        """
        adapters = self.get_unique_adapters()
        logger.info(f"Dispatching comparative evaluation across {len(adapters)} models concurrently.")

        tasks = [
            adapter.generate(
                query=query,
                context=context,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for adapter in adapters
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: List[LLMResponse] = []
        for adapter, result in zip(adapters, results):
            if isinstance(result, Exception):
                logger.error(f"Error during comparison in {adapter.name}: {result}")
                responses.append(
                    LLMResponse(
                        model=adapter.name,
                        provider=adapter.provider,
                        model_id=adapter.model_id,
                        answer=f"Error running model comparison for {adapter.name}: {result}",
                        latency_ms=0.0,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                responses.append(result)

        return responses


# Singleton router instance
_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance
