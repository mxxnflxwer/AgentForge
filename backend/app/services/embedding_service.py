import abc
import logging
from typing import List, Optional
from chromadb.utils import embedding_functions

from app.core.config import settings

logger = logging.getLogger("agentforge.services.embedding")


class BaseEmbeddingService(abc.ABC):
    """
    Abstract interface for embedding generation.
    Enables swapping local ONNX, SentenceTransformers, or cloud providers seamlessly.
    """

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text strings."""
        pass


class LocalChromaEmbeddingService(BaseEmbeddingService):
    """
    Free, local embedding generator using ChromaDB's ONNX-based all-MiniLM-L6-v2 model.
    Runs locally with 0 API cost and no network dependencies after initial cache.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        logger.info(f"Initialized LocalChromaEmbeddingService with model {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        embeddings = self._ef([text])
        return [float(x) for x in embeddings[0]]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._ef(texts)
        return [[float(x) for x in emb] for emb in embeddings]


_embedding_service_instance: Optional[BaseEmbeddingService] = None


def get_embedding_service() -> BaseEmbeddingService:
    """
    Dependency injection factory for the embedding service.
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = LocalChromaEmbeddingService()
    return _embedding_service_instance
