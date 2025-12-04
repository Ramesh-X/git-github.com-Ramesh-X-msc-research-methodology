"""Entrypoint for the synthetic dataset generator.

Configuration is specified in the top section of this file. No CLI args are accepted.
"""

import logging
import os

from dotenv import load_dotenv
from logging_config import setup_logging

# ---------------- CONFIGURATION -----------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")
NUM_PAGES = int(os.getenv("NUM_PAGES", "100"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/kb")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"
# ------------------------------------------------


def main():
    # Initialize file-only logging (no console handlers)
    setup_logging()
    logger = logging.getLogger(__name__)
    from create_dataset_lib.pipeline import run_generation

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not DRY_RUN and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required unless DRY_RUN=true."
        )

    logger.info(
        "Starting dataset generation; DRY_RUN=%s, NUM_PAGES=%s", DRY_RUN, NUM_PAGES
    )
    run_generation(
        openrouter_api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", MODEL),
        num_pages=NUM_PAGES,
        output_dir=OUTPUT_DIR,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )
    logger.info("Finished run_generation")


if __name__ == "__main__":
    main()
