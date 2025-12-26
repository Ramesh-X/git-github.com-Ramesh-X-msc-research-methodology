"""Entrypoint for E1-E4 evaluation pipeline.

Configuration is specified in environment variables. No CLI args are accepted.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from evaluation_lib.constants import DEFAULT_DRY_RUN, DEFAULT_KB_DIR, DEFAULT_OVERWRITE
from evaluation_lib.pipeline import run_evaluation_pipeline
from logging_config import setup_logging

# ---------------- CONFIGURATION -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KB_DIR = os.getenv("KB_DIR", DEFAULT_KB_DIR)
OVERWRITE = os.getenv("OVERWRITE", DEFAULT_OVERWRITE).lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", DEFAULT_DRY_RUN).lower() == "true"
# ------------------------------------------------


def main():
    print("Starting E1-E4 Evaluation Pipeline...")  # Console output for user feedback

    # Initialize file-only logging (no console handlers)
    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate API keys
    if not DRY_RUN:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required unless DRY_RUN=true."
            )

    # Define file paths
    kb_dir = Path(KB_DIR)

    logger.info("=" * 80)
    logger.info("Starting E1-E4 Evaluation")
    logger.info(f"DRY_RUN: {DRY_RUN}")
    logger.info(f"KB_DIR: {kb_dir}")
    logger.info(f"OVERWRITE: {OVERWRITE}")
    logger.info("=" * 80)

    # Run evaluation pipeline
    run_evaluation_pipeline(
        kb_dir=kb_dir,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )

    print("Evaluation completed successfully!")  # Console output for user feedback

    logger.info("=" * 80)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
