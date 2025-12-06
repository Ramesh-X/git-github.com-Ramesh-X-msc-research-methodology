"""Vector store management using Qdrant."""

import hashlib
import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages Qdrant vector database operations."""

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int = 1536,
        in_memory: bool = True,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Always use in-memory for E2-E4 experiments
        self.client = QdrantClient(":memory:")
        logger.info("Initialized in-memory Qdrant vector store")

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Insert or update document chunks with embeddings.

        Args:
            chunks: List of dicts with keys: chunk_id, text, embedding, metadata.
                   Each dict must contain:
                   - chunk_id: str
                   - text: str
                   - embedding: List[float]
                   - metadata: dict (optional)
        """
        points: List[PointStruct] = []
        for chunk in chunks:
            # Ensure embedding is a list of floats
            embedding: List[float] = chunk["embedding"]
            if not isinstance(embedding, list):
                raise TypeError(f"embedding must be List[float], got {type(embedding)}")

            point = PointStruct(
                id=self._hash_to_id(chunk["chunk_id"]),
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug(f"Upserted {len(points)} chunks")

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector as list of floats
            top_k: Number of top results to return

        Returns:
            List of dicts with keys: chunk_id, text, score, metadata
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": str(r.payload.get("chunk_id", "")),
                "text": str(r.payload.get("text", "")),
                "score": float(r.score),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results
            if r.payload is not None
        ]

    def _hash_to_id(self, chunk_id: str) -> int:
        """Convert string ID to integer for Qdrant.

        Args:
            chunk_id: String identifier for the chunk

        Returns:
            Integer ID suitable for Qdrant PointStruct
        """
        hash_int = int(hashlib.md5(chunk_id.encode()).hexdigest()[:16], 16)
        # Ensure ID is a positive integer within int64 range
        return hash_int % (2**62)

    def get_collection_size(self) -> int:
        """Get number of points in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
