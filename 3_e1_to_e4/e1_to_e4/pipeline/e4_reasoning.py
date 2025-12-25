import json
import logging
import os
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

from ..agents import create_e4_agent, create_openrouter_model
from ..constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    E4_TOP_K,
    E4_TOP_N,
    EMBEDDINGS_CACHE_FOLDER,
    OPENROUTER_MODEL,
    QDRANT_COLLECTION,
    REQUEST_DELAY_SECONDS,
    RERANKER_MODEL,
)
from ..embeddings import EmbeddingService
from ..kb_loader import MarkdownChunker, load_kb_documents
from ..models import DocumentChunk, ExperimentResult, QueryInput
from ..reranker import Reranker
from ..vector_store import VectorStore

logger = logging.getLogger(__name__)


def run_e4_reasoning(
    kb_dir: Path,
    queries_file: Path,
    output_file: Path,
    openrouter_api_key: str | None,
    openai_api_key: str | None,
    overwrite: bool,
    dry_run: bool,
):
    """Run E4 Reasoning RAG experiment."""
    logger.info(
        "Starting E4 Reasoning RAG; DRY_RUN=%s, KB_DIR=%s, QUERIES_FILE=%s",
        dry_run,
        kb_dir,
        queries_file,
    )

    # Load queries
    queries: List[QueryInput] = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(QueryInput(**json.loads(line)))

    # Load existing results for resume
    existing_ids = set()
    if output_file.exists() and not overwrite:
        logger.info("Found existing output file %s; loading to resume", output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                existing_ids.add(result["query_id"])

    # Initialize services if not dry run
    vector_store = None
    embedding_service = None
    reranker = None
    agent = None

    if not dry_run:
        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY required when not dry-run")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY required when not dry-run")

        # Initialize vector store
        vector_store = VectorStore(
            collection_name=QDRANT_COLLECTION,
            embedding_dim=1536,  # text-embedding-3-small dimension
        )

        # Initialize embedding service
        embedding_service = EmbeddingService(
            api_key=openai_api_key,
            cache_dir=kb_dir / EMBEDDINGS_CACHE_FOLDER,
        )

        # Initialize reranker
        reranker = Reranker(RERANKER_MODEL)

        # Index knowledge base if not already done
        if vector_store.get_collection_size() == 0:
            _index_kb(kb_dir, vector_store, embedding_service)

        # Create agent
        or_model = create_openrouter_model(OPENROUTER_MODEL, openrouter_api_key)
        agent = create_e4_agent(or_model)

    # Process queries
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"

    with open(output_file, write_mode, encoding="utf-8") as out_f:
        for query in tqdm(queries, desc="Processing E4 queries"):
            if query.query_id in existing_ids and not overwrite:
                logger.info("Skipping existing query_id: %s", query.query_id)
                continue

            # Track start time for rate limiting
            query_start_time = time.time()

            try:
                result = _process_query(
                    query=query,
                    vector_store=vector_store,
                    embedding_service=embedding_service,
                    reranker=reranker,
                    agent=agent,
                    dry_run=dry_run,
                )

                out_f.write(result.model_dump_json() + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
                logger.info("Processed query_id: %s", query.query_id)

                # Rate limiting for OpenRouter API (20 req/min = 1 req every 3 sec)
                if not dry_run:
                    query_elapsed = time.time() - query_start_time
                    sleep_time = max(0, REQUEST_DELAY_SECONDS - query_elapsed)
                    if sleep_time > 0:
                        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                        time.sleep(sleep_time)

            except Exception as e:
                logger.exception("Failed to process query %s: %s", query.query_id, e)

    logger.info("E4 Reasoning RAG completed; output in %s", output_file)


def _index_kb(
    kb_dir: Path, vector_store: VectorStore, embedding_service: EmbeddingService
):
    """Index knowledge base into vector store."""
    logger.info("Indexing knowledge base...")

    # Load documents
    documents = load_kb_documents(kb_dir)
    chunker = MarkdownChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    all_chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = chunker.chunk_document(doc["filename"], doc["content"])
        all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} chunks")

    # Embed chunks in batches
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding chunks"):
        batch = all_chunks[i : i + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = embedding_service.embed_batch(texts, cache_type="chunks")

        # Add embeddings to chunks
        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding

        vector_store.upsert_chunks(batch)

    logger.info(f"Indexed {len(all_chunks)} chunks into vector store")


def _process_query(
    query: QueryInput,
    vector_store: VectorStore | None,
    embedding_service: EmbeddingService | None,
    reranker: Reranker | None,
    agent,
    dry_run: bool,
) -> ExperimentResult:
    """Process a single query for E4."""
    # Retrieval phase (same as E3: vector search + reranking)
    retrieval_start = time.time()
    retrieved_chunks = []

    if dry_run:
        # Mock retrieval for dry run - generate 20 chunks, then "rerank" to top 5
        mock_chunks = [
            DocumentChunk(
                chunk_id=f"dry_run_chunk_{i}",
                text=f"[DRY_RUN] Chunk {i} content related to {query.query[:20]}...",
                score=0.9 - i * 0.01,  # Decreasing scores
                metadata={"filename": f"mock_doc_{i}.md"},
            )
            for i in range(E4_TOP_N)  # Generate TOP_N chunks
        ]
        # Simulate reranking by sorting and taking top TOP_K
        retrieved_chunks = sorted(mock_chunks, key=lambda x: x.score, reverse=True)[
            :E4_TOP_K
        ]
        # Set rerank_score to same as score for dry run
        for chunk in retrieved_chunks:
            chunk.rerank_score = chunk.score
    else:
        # Real retrieval: vector search + reranking
        assert embedding_service is not None
        assert vector_store is not None
        assert reranker is not None

        query_embedding = embedding_service.embed_text(
            query.query, cache_type="queries"
        )
        # Stage 1: Vector search (retrieve TOP_N chunks)
        initial_chunks = vector_store.search(query_embedding, top_k=E4_TOP_N)

        # Stage 2: Reranking (select TOP_K from TOP_N)
        retrieved_chunks = reranker.rerank(query.query, initial_chunks, top_k=E4_TOP_K)

    retrieval_time = (time.time() - retrieval_start) * 1000

    # LLM generation phase (with CoT reasoning)
    llm_start = time.time()

    if dry_run:
        llm_answer = "[DRY_RUN] No LLM call - would use CoT reasoning"
        reasoning_steps = "[DRY_RUN] Step 1: Analyze query\nStep 2: Check context\nStep 3: Provide reasoning\nStep 4: Final answer"
    else:
        # Format context from reranked chunks
        context = _format_context(retrieved_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query.query}"

        try:
            result = agent.run_sync(prompt)
            llm_answer = result.output.answer
            reasoning_steps = result.output.reasoning_steps
        except Exception as e:
            logger.exception("LLM call failed for query %s: %s", query.query_id, e)
            llm_answer = f"Error: {e}"
            reasoning_steps = f"Error during reasoning: {e}"

    llm_time = (time.time() - llm_start) * 1000
    total_time = retrieval_time + llm_time

    # Build result
    return ExperimentResult(
        query_id=query.query_id,
        experiment="E4",
        query=query.query,
        retrieved_chunks=retrieved_chunks,
        llm_answer=llm_answer,
        reasoning_steps=reasoning_steps,
        ground_truth=query.ground_truth,
        retrieval_time_ms=retrieval_time,
        llm_time_ms=llm_time,
        total_time_ms=total_time,
    )


def _format_context(chunks: List[DocumentChunk]) -> str:
    """Format reranked chunks as context for LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get("filename", "unknown")
        context_parts.append(f"[Source {i}: {filename}]\n{chunk.text}")
    return "\n\n".join(context_parts)
