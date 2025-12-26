"""Embedding service using OpenAI API with caching support."""

import json
import logging
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embeddings using OpenAI API with caching."""

    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        model: str = "text-embedding-3-small",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_cache_file = self.cache_dir / "embeddings_chunks.jsonl"
        self.queries_cache_file = self.cache_dir / "embeddings_queries.jsonl"

        # Load existing cache into memory
        self.chunks_cache: Dict[str, List[float]] = {}
        self.queries_cache: Dict[str, List[float]] = {}
        self._load_caches()

        logger.info(f"Initialized embedding service with model: {model}")
        if self.cache_dir:
            logger.info(f"Using embedding cache at: {self.cache_dir}")
            logger.info(f"Loaded {len(self.chunks_cache)} cached chunk embeddings")
            logger.info(f"Loaded {len(self.queries_cache)} cached query embeddings")

    def _load_caches(self):
        """Load existing embedding caches from disk."""
        if not self.cache_dir:
            return

        # Load chunks cache
        if self.chunks_cache_file and self.chunks_cache_file.exists():
            try:
                with open(self.chunks_cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        self.chunks_cache[data["text"]] = data["embedding"]
                logger.debug(f"Loaded {len(self.chunks_cache)} chunks from cache")
            except Exception as e:
                logger.warning(f"Failed to load chunks cache: {e}")

        # Load queries cache
        if self.queries_cache_file and self.queries_cache_file.exists():
            try:
                with open(self.queries_cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        self.queries_cache[data["text"]] = data["embedding"]
                logger.debug(f"Loaded {len(self.queries_cache)} queries from cache")
            except Exception as e:
                logger.warning(f"Failed to load queries cache: {e}")

    def embed_text(self, text: str, cache_type: str = "queries") -> List[float]:
        """Generate embedding for a single text with caching.

        Args:
            text: Text to embed
            cache_type: Either "chunks" or "queries" for cache categorization

        Returns:
            Embedding vector as list of floats
        """
        # Check appropriate cache
        cache = self.chunks_cache if cache_type == "chunks" else self.queries_cache
        if text in cache:
            logger.debug(f"Cache hit for {cache_type}: {text[:50]}...")
            return cache[text]

        # Call API
        response = self.client.embeddings.create(model=self.model, input=text)
        embedding = response.data[0].embedding

        # Store in cache
        cache[text] = embedding

        # Persist to disk
        if self.cache_dir:
            cache_file = (
                self.chunks_cache_file
                if cache_type == "chunks"
                else self.queries_cache_file
            )
            try:
                with open(cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"text": text, "embedding": embedding}) + "\n")
                logger.debug(f"Cached {cache_type} embedding for: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to persist {cache_type} cache: {e}")

        return embedding

    def embed_batch(
        self, texts: List[str], batch_size: int = 100, cache_type: str = "chunks"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches with caching.

        Args:
            texts: List of texts to embed
            batch_size: Size of API batch requests
            cache_type: Either "chunks" or "queries" for cache categorization

        Returns:
            List of embedding vectors
        """
        cache = self.chunks_cache if cache_type == "chunks" else self.queries_cache
        cache_file = (
            self.chunks_cache_file
            if cache_type == "chunks"
            else self.queries_cache_file
        )

        all_embeddings = []
        texts_to_embed = []
        text_indices = []

        # Separate cached from non-cached texts
        for idx, text in enumerate(texts):
            if text in cache:
                all_embeddings.append((idx, cache[text]))
            else:
                texts_to_embed.append(text)
                text_indices.append(idx)

        if texts_to_embed:
            logger.info(
                f"Embedding {len(texts_to_embed)} new {cache_type} (cache had {len(cache)})"
            )
        else:
            logger.info(f"All {len(texts)} {cache_type} found in cache")

        # Embed non-cached texts in batches
        new_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)

            for text, embedding_data in zip(batch, response.data):
                embedding = embedding_data.embedding
                new_embeddings.append((text, embedding))

                # Update cache
                cache[text] = embedding

                # Persist immediately
                if self.cache_dir:
                    try:
                        with open(cache_file, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps({"text": text, "embedding": embedding})
                                + "\n"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to persist {cache_type} cache: {e}")

        # Merge cached and new embeddings in original order
        cached_sorted = sorted(all_embeddings, key=lambda x: x[0])
        new_sorted = sorted(
            [
                (text_indices[i], embedding)
                for i, (_, embedding) in enumerate(new_embeddings)
            ],
            key=lambda x: x[0],
        )

        all_by_index = {}
        for idx, embedding in cached_sorted + new_sorted:
            all_by_index[idx] = embedding

        result = [all_by_index[i] for i in range(len(texts))]
        return result
