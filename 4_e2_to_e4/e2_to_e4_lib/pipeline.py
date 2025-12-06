"""Main experiment pipeline for E2-E4."""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .agents import (
    create_e2_agent,
    create_e3_agent,
    create_e4_agent,
    create_openrouter_model,
)
from .constants import REQUEST_DELAY_SECONDS
from .embeddings import EmbeddingService
from .kb_loader import MarkdownChunker, load_kb_documents
from .models import ExperimentResult, QueryInput, RetrievedChunk
from .reranker import Reranker
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Orchestrates E2-E4 experiments."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: Optional[EmbeddingService],
        reranker: Optional[Reranker],
        openrouter_api_key: str,
        model_name: str,
        dry_run: bool = False,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.model_name = model_name
        self.dry_run = dry_run

        # Create agents
        if not dry_run:
            or_model = create_openrouter_model(model_name, openrouter_api_key)
            self.e2_agent = create_e2_agent(or_model)
            self.e3_agent = create_e3_agent(or_model)
            self.e4_agent = create_e4_agent(or_model)
        else:
            self.e2_agent = None
            self.e3_agent = None
            self.e4_agent = None

    def index_kb(self, kb_dir: Path, chunk_size: int, chunk_overlap: int):
        """Index knowledge base into vector store."""
        if self.vector_store.get_collection_size() > 0:
            logger.info("Vector store already populated, skipping indexing")
            return

        if self.dry_run:
            logger.info("DRY_RUN: Skipping KB indexing")
            return

        if self.embedding_service is None:
            logger.error("Embedding service not initialized")
            return

        logger.info("Indexing knowledge base...")
        documents = load_kb_documents(kb_dir)
        chunker = MarkdownChunker(chunk_size=chunk_size, overlap=chunk_overlap)

        all_chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = chunker.chunk_document(doc["filename"], doc["content"])
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} chunks")

        # Embed chunks in batches with caching
        batch_size = 100
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding chunks"):
            batch = all_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedding_service.embed_batch(texts, cache_type="chunks")

            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding

            self.vector_store.upsert_chunks(batch)

        logger.info(f"Indexed {len(all_chunks)} chunks into vector store")

    def run_e2(
        self, queries: List[QueryInput], output_file: Path, top_k: int, overwrite: bool
    ):
        """Run E2: Standard RAG."""
        logger.info(f"Starting E2 Standard RAG; top_k={top_k}")
        self._run_experiment(
            experiment="e2",
            queries=queries,
            output_file=output_file,
            agent=self.e2_agent,
            top_k=top_k,
            use_reranking=False,
            overwrite=overwrite,
        )

    def run_e3(
        self,
        queries: List[QueryInput],
        output_file: Path,
        top_n: int,
        top_k: int,
        overwrite: bool,
    ):
        """Run E3: Filtered RAG with reranking."""
        logger.info(f"Starting E3 Filtered RAG; top_n={top_n}, top_k={top_k}")
        self._run_experiment(
            experiment="e3",
            queries=queries,
            output_file=output_file,
            agent=self.e3_agent,
            top_k=top_k,
            top_n=top_n,
            use_reranking=True,
            overwrite=overwrite,
        )

    def run_e4(
        self,
        queries: List[QueryInput],
        output_file: Path,
        top_n: int,
        top_k: int,
        overwrite: bool,
    ):
        """Run E4: Reasoning RAG with CoT."""
        logger.info(f"Starting E4 Reasoning RAG; top_n={top_n}, top_k={top_k}")
        self._run_experiment(
            experiment="e4",
            queries=queries,
            output_file=output_file,
            agent=self.e4_agent,
            top_k=top_k,
            top_n=top_n,
            use_reranking=True,
            overwrite=overwrite,
        )

    def _run_experiment(
        self,
        experiment: str,
        queries: List[QueryInput],
        output_file: Path,
        agent,
        top_k: int,
        top_n: Optional[int] = None,
        use_reranking: bool = False,
        overwrite: bool = False,
    ):
        """Generic experiment runner."""
        # Load existing results
        existing_ids = set()
        if output_file.exists() and not overwrite:
            logger.info(f"Found existing output file {output_file}; loading to resume")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    existing_ids.add(result["query_id"])

        # Process queries
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "w" if overwrite else "a"

        with open(output_file, write_mode, encoding="utf-8") as out_f:
            for query in tqdm(queries, desc=f"Processing {experiment.upper()}"):
                if query.query_id in existing_ids and not overwrite:
                    logger.info(f"Skipping existing query_id: {query.query_id}")
                    continue

                # Track start time for rate limiting
                query_start_time = time.time()

                try:
                    result = self._process_query(
                        experiment=experiment,
                        query=query,
                        agent=agent,
                        top_k=top_k,
                        top_n=top_n,
                        use_reranking=use_reranking,
                    )

                    out_f.write(result.model_dump_json() + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    logger.info(f"Processed query_id: {query.query_id}")

                    # Rate limiting for OpenRouter API (20 req/min = 1 req every 3 sec)
                    if not self.dry_run:
                        query_elapsed = time.time() - query_start_time
                        sleep_time = max(0, REQUEST_DELAY_SECONDS - query_elapsed)
                        if sleep_time > 0:
                            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                            time.sleep(sleep_time)

                except Exception as e:
                    logger.exception(f"Failed to process query {query.query_id}: {e}")

        logger.info(f"{experiment.upper()} completed; output in {output_file}")

    def _process_query(
        self,
        experiment: str,
        query: QueryInput,
        agent,
        top_k: int,
        top_n: Optional[int],
        use_reranking: bool,
    ) -> ExperimentResult:
        """Process a single query."""
        # Retrieval
        retrieval_start = time.time()

        if self.dry_run:
            retrieved_chunks = [
                {
                    "chunk_id": f"dry_run_chunk_{i}",
                    "text": f"[DRY_RUN] Chunk {i}",
                    "score": 0.9 - i * 0.1,
                    "metadata": {},
                }
                for i in range(top_k)
            ]
        else:
            if self.embedding_service is None:
                logger.error("Embedding service not initialized")
                retrieved_chunks = []
            else:
                query_embedding = self.embedding_service.embed_text(
                    query.query, cache_type="queries"
                )
                search_k = top_n if (use_reranking and top_n) else top_k
                retrieved_chunks = self.vector_store.search(
                    query_embedding, top_k=search_k
                )

                # Reranking
                if use_reranking and self.reranker:
                    retrieved_chunks = self.reranker.rerank(
                        query.query, retrieved_chunks, top_k=top_k
                    )

        retrieval_time = (time.time() - retrieval_start) * 1000

        # LLM Generation
        llm_start = time.time()

        if self.dry_run:
            llm_answer = "[DRY_RUN] No LLM call"
            reasoning_steps = "[DRY_RUN] No reasoning" if experiment == "e4" else None
        else:
            # Prepare context
            context = self._format_context(retrieved_chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {query.query}"

            try:
                result = agent.run_sync(prompt)
                if experiment == "e4":
                    llm_answer = result.output.answer
                    reasoning_steps = result.output.reasoning_steps
                else:
                    llm_answer = result.output.answer
                    reasoning_steps = None
            except Exception as e:
                logger.exception(f"LLM call failed for query {query.query_id}: {e}")
                llm_answer = f"Error: {e}"
                reasoning_steps = None

        llm_time = (time.time() - llm_start) * 1000
        total_time = retrieval_time + llm_time

        # Build result
        return ExperimentResult(
            query_id=query.query_id,
            experiment=experiment,
            query=query.query,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    score=c.get("rerank_score", c.get("score", 0.0)),
                    metadata=c.get("metadata", {}),
                )
                for c in retrieved_chunks
            ],
            llm_answer=llm_answer,
            reasoning_steps=reasoning_steps,
            ground_truth=query.ground_truth,
            context_reference=query.context_reference,
            metadata=query.metadata,
            retrieval_time_ms=retrieval_time,
            llm_time_ms=llm_time,
            total_time_ms=total_time,
            model=self.model_name,
            dry_run=self.dry_run,
        )

    def _format_context(self, chunks: List[dict]) -> str:
        """Format retrieved chunks as context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get("metadata", {}).get("filename", "unknown")
            context_parts.append(f"[Source {i}: {filename}]\n{chunk['text']}")
        return "\n\n".join(context_parts)


def load_queries(queries_file: Path) -> List[QueryInput]:
    """Load queries from JSONL file."""
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(QueryInput(**json.loads(line)))
    logger.info(f"Loaded {len(queries)} queries from {queries_file}")
    return queries
