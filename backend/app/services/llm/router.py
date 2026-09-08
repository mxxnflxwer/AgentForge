import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.llm.base import BaseLLMAdapter, LLMResponse
from app.services.llm.gemini import GeminiAdapter
from app.services.llm.gpt_oss import GPTOSSAdapter
from app.services.llm.qwen import QwenAdapter

logger = logging.getLogger("agentforge.services.llm.router")


class LLMRouter:
    """
    Central router for LLM generation and multi-model comparative evaluation.
    Enforces unified interfaces and concurrent multi-model comparisons.
    """

    def __init__(self):
        self._adapters: Dict[str, BaseLLMAdapter] = {}
        self._initialize_adapters()

    def _initialize_adapters(self):
        gemini = GeminiAdapter()
        qwen = QwenAdapter()
        gpt_oss = GPTOSSAdapter()

        self.register_adapter("gemini", gemini)
        self.register_adapter("gemini-2.5-flash-lite", gemini)

        self.register_adapter("qwen", qwen)
        self.register_adapter("qwen-3.6-27b", qwen)
        self.register_adapter("qwen-2.5-72b", qwen)

        self.register_adapter("gpt_oss", gpt_oss)
        self.register_adapter("gpt-oss", gpt_oss)
        self.register_adapter("gpt-oss-120b", gpt_oss)

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
    ) -> LLMResponse:
        """
        Execute grounded generation on a single selected model.
        """
        adapter = self.get_adapter(model_key)
        if not adapter:
            # Fallback to gemini as default
            adapter = self.get_adapter("gemini") or list(self.get_unique_adapters())[0]

        logger.info(f"Routing query to single model: {adapter.name}")
        return await adapter.generate(
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

    async def compare_all(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
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
                        answer=f"Error running model comparison for {adapter.name}.",
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
