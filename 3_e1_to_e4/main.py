"""Entrypoint for E1 and E2 experiments.

Configuration is specified in the top section of this file. No CLI args are accepted.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from e1_to_e4.constants import (
    DATA_FOLDER,
    DEFAULT_DRY_RUN,
    DEFAULT_KB_DIR,
    DEFAULT_OVERWRITE,
    E1_OUTPUT_FILE_NAME,
    E2_OUTPUT_FILE_NAME,
    E3_OUTPUT_FILE_NAME,
    OPENROUTER_MODEL,
    QUERIES_FILE_NAME,
)
from e1_to_e4.pipeline.e1_baseline import run_e1_baseline
from e1_to_e4.pipeline.e2_standard import run_e2_standard
from e1_to_e4.pipeline.e3_filtered import run_e3_filtered
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

    # Define file paths
    kb_dir = Path(KB_DIR)
    queries_file = kb_dir / DATA_FOLDER / QUERIES_FILE_NAME
    e1_output = kb_dir / DATA_FOLDER / E1_OUTPUT_FILE_NAME
    e2_output = kb_dir / DATA_FOLDER / E2_OUTPUT_FILE_NAME
    e3_output = kb_dir / DATA_FOLDER / E3_OUTPUT_FILE_NAME

    logger.info("=" * 80)
    logger.info("Starting E1, E2, and E3 Experiments")
    logger.info(f"DRY_RUN: {DRY_RUN}")
    logger.info(f"KB_DIR: {kb_dir}")
    logger.info(f"QUERIES_FILE: {queries_file}")
    logger.info(f"OVERWRITE: {OVERWRITE}")
    logger.info(f"MODEL: {OPENROUTER_MODEL}")
    logger.info("=" * 80)

    # Run E1 baseline
    logger.info("Running E1 baseline")
    run_e1_baseline(
        queries_file=queries_file,
        output_file=e1_output,
        openrouter_api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,  # Now from constants
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )
    logger.info("E1 baseline completed")

    # Run E2 standard RAG
    logger.info("Running E2 standard RAG")
    run_e2_standard(
        kb_dir=kb_dir,
        queries_file=queries_file,
        output_file=e2_output,
        openrouter_api_key=OPENROUTER_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )
    logger.info("E2 standard RAG completed")

    # Run E3 filtered RAG
    logger.info("Running E3 filtered RAG")
    run_e3_filtered(
        kb_dir=kb_dir,
        queries_file=queries_file,
        output_file=e3_output,
        openrouter_api_key=OPENROUTER_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )
    logger.info("E3 filtered RAG completed")

    logger.info("=" * 80)
    logger.info("All experiments completed successfully")
    logger.info(f"E1 output: {e1_output}")
    logger.info(f"E2 output: {e2_output}")
    logger.info(f"E3 output: {e3_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
