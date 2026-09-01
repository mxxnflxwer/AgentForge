import logging
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.document import DocumentChunk
from app.services.embedding_service import BaseEmbeddingService, get_embedding_service

logger = logging.getLogger("agentforge.services.vector_store")


class VectorStore:
    """
    Persistent ChromaDB vector store wrapper for document chunk embeddings,
    strict user isolation, and semantic similarity retrieval.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_service = embedding_service or get_embedding_service()

        # Ensure persist directory exists relative to current working directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Cosine distance space for normalized semantic similarity
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized VectorStore with collection '{self.collection_name}' at '{self.persist_directory}'"
        )

    def reset_collection(self) -> None:
        """
        Delete and recreate the ChromaDB collection.
        Ensures clean separation when upgrading embedding models.
        """
        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted existing ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.debug(f"Collection deletion notice: {e}")

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Recreated fresh ChromaDB collection '{self.collection_name}'")

    def count(self) -> int:
        """Return total vector count in the collection."""
        return self._collection.count()

    def upsert_document_chunks(
        self,
        user_id: str,
        document_id: str,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Embed and upsert document chunks with strict ownership metadata into ChromaDB.
        Cleans up any existing vectors for the document before insertion.
        """
        if not chunks:
            logger.warning(f"No chunks provided to upsert for document {document_id}")
            return 0

        # Clean existing vectors for this document to avoid orphaned chunks on re-processing
        self.delete_document_vectors(user_id=user_id, document_id=document_id)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            chunk_id_str = str(chunk.id)
            vector_id = f"{document_id}_{chunk.chunk_index}_{chunk_id_str}"
            ids.append(vector_id)
            documents.append(chunk.content)
            metadatas.append({
                "user_id": str(user_id),
                "document_id": str(document_id),
                "chunk_id": chunk_id_str,
                "chunk_index": int(chunk.chunk_index),
                "section_title": str(chunk.section_title or ""),
            })

        if embeddings is None:
            logger.info(f"Generating embeddings for {len(chunks)} chunks of document {document_id}")
            embeddings = self.embedding_service.embed_batch(documents)

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"ChromaDB dimension mismatch detected ({e}). Resetting collection for new embedding model...")
                self.reset_collection()
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                raise e

        logger.info(f"Upserted {len(ids)} vectors into ChromaDB for document {document_id}")
        return len(ids)

    def delete_document_vectors(self, user_id: str, document_id: str) -> None:
        """
        Delete all vector entries belonging to the specified document and user.
        """
        try:
            self._collection.delete(
                where={
                    "$and": [
                        {"user_id": {"$eq": str(user_id)}},
                        {"document_id": {"$eq": str(document_id)}},
                    ]
                }
            )
            logger.info(f"Deleted vector chunks for document {document_id} (user {user_id})")
        except Exception as e:
            logger.warning(f"Vector deletion notice for document {document_id}: {e}")

    def query_similar_chunks(
        self,
        user_id: str,
        query_text: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search against indexed document chunks with user isolation.
        """
        if not query_text or not query_text.strip():
            return []

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query_text.strip())
        if not query_embedding:
            return []

        # Build strict user isolation filter
        if document_id:
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": str(user_id)}},
                    {"document_id": {"$eq": str(document_id)}},
                ]
            }
        else:
            where_filter = {"user_id": {"$eq": str(user_id)}}

        # Query ChromaDB collection
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"ChromaDB dimension mismatch during query ({e}).")
                return []
            logger.error(f"Error querying ChromaDB vector store: {e}")
            return []

        retrieved: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids_list = results["ids"][0]
        docs_list = results["documents"][0] if results.get("documents") else []
        metas_list = results["metadatas"][0] if results.get("metadatas") else []
        dists_list = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids_list)):
            meta = metas_list[i] if i < len(metas_list) else {}
            content = docs_list[i] if i < len(docs_list) else ""
            distance = float(dists_list[i]) if i < len(dists_list) else 0.0

            # Cosine similarity calculation: similarity = max(0, 1 - distance)
            similarity = max(0.0, 1.0 - distance)

            retrieved.append({
                "chunk_id": str(meta.get("chunk_id", "")),
                "document_id": str(meta.get("document_id", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "content": content,
                "section_title": meta.get("section_title") or None,
                "distance": round(distance, 4),
                "similarity_score": round(similarity, 4),
            })

        return retrieved


_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Dependency injection factory for the vector store.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
