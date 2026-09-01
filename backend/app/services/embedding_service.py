import abc
import logging
from typing import List, Optional
import torch

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


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """
    State-of-the-art local embedding generator using SentenceTransformers (e.g., BAAI/bge-m3).
    Supports 1024-dim dense embeddings, multi-lingual support, and normalized cosine similarity vectors.
    Runs locally on CPU/CUDA with 0 API cost and singleton caching.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = SentenceTransformer(self.model_name, device=device, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(self.model_name, device=device)
        logger.info(f"Initialized SentenceTransformerEmbeddingService with {self.model_name} on {device}")

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        embedding = self._model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(x) for x in embedding]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]
        embeddings = self._model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [[float(x) for x in emb] for emb in embeddings]


class LocalChromaEmbeddingService(BaseEmbeddingService):
    """
    Free, local embedding generator using ChromaDB's ONNX-based all-MiniLM-L6-v2 model.
    Runs locally with 0 API cost and no network dependencies after initial cache.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "all-MiniLM-L6-v2"
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        logger.info(f"Initialized LocalChromaEmbeddingService with model {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        embeddings = self._ef([text.strip()])
        return [float(x) for x in embeddings[0]]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]
        embeddings = self._ef(cleaned_texts)
        return [[float(x) for x in emb] for emb in embeddings]


_embedding_service_instance: Optional[BaseEmbeddingService] = None


def get_embedding_service() -> BaseEmbeddingService:
    """
    Dependency injection factory for the embedding service.
    Returns singleton instance configured according to settings.EMBEDDING_MODEL_NAME.
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        model_name = settings.EMBEDDING_MODEL_NAME
        if model_name == "all-MiniLM-L6-v2":
            _embedding_service_instance = LocalChromaEmbeddingService(model_name=model_name)
        else:
            _embedding_service_instance = SentenceTransformerEmbeddingService(model_name=model_name)
    return _embedding_service_instance
