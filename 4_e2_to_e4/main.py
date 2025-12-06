"""Entrypoint for E2-E4 experiments.

Configuration is specified in this file. File paths are derived from KB_DIR.
No CLI args are accepted.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from e2_to_e4_lib.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_DRY_RUN,
    DEFAULT_KB_DIR,
    DEFAULT_OVERWRITE,
    E2_OUTPUT_FILE_NAME,
    E2_TOP_K,
    E3_OUTPUT_FILE_NAME,
    E3_TOP_K,
    E3_TOP_N,
    E4_OUTPUT_FILE_NAME,
    E4_TOP_K,
    E4_TOP_N,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENROUTER_MODEL,
    QDRANT_COLLECTION,
    QDRANT_IN_MEMORY,
    QUERIES_FILE_NAME,
    RERANKER_MODEL,
)
from e2_to_e4_lib.embeddings import EmbeddingService
from e2_to_e4_lib.pipeline import ExperimentPipeline, load_queries
from e2_to_e4_lib.reranker import Reranker
from e2_to_e4_lib.vector_store import VectorStore
from logging_config import setup_logging

# ---------------- CONFIGURATION -----------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KB_DIR = os.getenv("KB_DIR", DEFAULT_KB_DIR)
OVERWRITE = os.getenv("OVERWRITE", DEFAULT_OVERWRITE).lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", DEFAULT_DRY_RUN).lower() == "true"
# ------------------------------------------------


def main():
    # Initialize file-only logging (no console handlers)
    setup_logging()
    logger = logging.getLogger(__name__)

    # Generate file paths from KB_DIR
    kb_dir = Path(KB_DIR)
    queries_file = kb_dir / QUERIES_FILE_NAME
    e2_output = kb_dir / E2_OUTPUT_FILE_NAME
    e3_output = kb_dir / E3_OUTPUT_FILE_NAME
    e4_output = kb_dir / E4_OUTPUT_FILE_NAME

    logger.info("=" * 80)
    logger.info("Starting E2-E4 Experiments")
    logger.info(f"DRY_RUN: {DRY_RUN}")
    logger.info(f"KB_DIR: {kb_dir}")
    logger.info(f"QUERIES_FILE: {queries_file}")
    logger.info(f"OVERWRITE: {OVERWRITE}")
    logger.info(f"MODEL: {OPENROUTER_MODEL}")
    logger.info("=" * 80)

    # Validate API keys
    if not DRY_RUN:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is required unless DRY_RUN=true."
            )
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required unless DRY_RUN=true."
            )

    # Load queries
    queries = load_queries(queries_file)
    logger.info(f"Loaded {len(queries)} queries")

    # Initialize services
    vector_store = VectorStore(
        collection_name=QDRANT_COLLECTION,
        embedding_dim=EMBEDDING_DIM,
        in_memory=QDRANT_IN_MEMORY,
    )

    embedding_service = (
        None
        if DRY_RUN
        else EmbeddingService(
            OPENAI_API_KEY or "",
            cache_dir=kb_dir / "embeddings_cache",
            model=EMBEDDING_MODEL,
        )
    )
    reranker = None if DRY_RUN else Reranker(RERANKER_MODEL)

    # Create pipeline
    pipeline = ExperimentPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        reranker=reranker,
        openrouter_api_key=OPENROUTER_API_KEY or "",
        model_name=OPENROUTER_MODEL,
        dry_run=DRY_RUN,
    )

    # Index KB (only if not dry run)
    if not DRY_RUN:
        pipeline.index_kb(kb_dir, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Run experiments
    logger.info("Running E2: Standard RAG")
    pipeline.run_e2(queries, e2_output, top_k=E2_TOP_K, overwrite=OVERWRITE)

    logger.info("Running E3: Filtered RAG")
    pipeline.run_e3(
        queries, e3_output, top_n=E3_TOP_N, top_k=E3_TOP_K, overwrite=OVERWRITE
    )

    logger.info("Running E4: Reasoning RAG")
    pipeline.run_e4(
        queries, e4_output, top_n=E4_TOP_N, top_k=E4_TOP_K, overwrite=OVERWRITE
    )

    logger.info("=" * 80)
    logger.info("All experiments completed successfully")
    logger.info(f"E2 output: {e2_output}")
    logger.info(f"E3 output: {e3_output}")
    logger.info(f"E4 output: {e4_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
