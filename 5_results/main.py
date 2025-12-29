"""Entrypoint for results generation pipeline.

Configuration is specified in environment variables. No CLI args are accepted.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from logging_config import setup_logging
from results_lib.constants import DEFAULT_KB_DIR, DEFAULT_OVERWRITE
from results_lib.pipeline import run_results_generation

# ---------------- CONFIGURATION -----------------
load_dotenv()
KB_DIR = os.getenv("KB_DIR", DEFAULT_KB_DIR)
OVERWRITE = os.getenv("OVERWRITE", DEFAULT_OVERWRITE).lower() == "true"
# ------------------------------------------------


def main():
    print("Starting Results Generation Pipeline...")  # Console output for user feedback

    # Initialize file-only logging (no console handlers)
    setup_logging()
    logger = logging.getLogger(__name__)

    kb_dir = Path(KB_DIR)

    logger.info("=" * 80)
    logger.info("Starting Results Generation")
    logger.info(f"KB_DIR: {kb_dir}")
    logger.info(f"OVERWRITE: {OVERWRITE}")
    logger.info("=" * 80)

    run_results_generation(
        kb_dir=kb_dir,
        overwrite=OVERWRITE,
    )

    print(
        "Results generation completed successfully!"
    )  # Console output for user feedback

    logger.info("=" * 80)
    logger.info("Results generation completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
