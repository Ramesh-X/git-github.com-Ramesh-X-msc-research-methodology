"""Reranking using cross-encoder models."""

import logging
from typing import List

from sentence_transformers import CrossEncoder

from .models import DocumentChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks using cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        logger.info(f"Initialized reranker with model: {model_name}")

    def rerank(
        self, query: str, chunks: List[DocumentChunk], top_k: int = 5
    ) -> List[DocumentChunk]:
        """Rerank chunks based on query relevance.

        Args:
            query: User query
            chunks: List of DocumentChunk objects
            top_k: Number of top chunks to return

        Returns:
            Reranked chunks (top_k)
        """
        if not chunks:
            return []

        # Prepare pairs for cross-encoder: [(query, doc1), (query, doc2), ...]
        pairs = [[query, chunk.text] for chunk in chunks]

        # Get reranking scores using predict
        scores = self.model.predict(pairs)

        # Attach new scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
            # Keep original score for comparison

        reranked = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)

        logger.debug(f"Reranked {len(chunks)} chunks, returning top {top_k}")
        return reranked[:top_k]
