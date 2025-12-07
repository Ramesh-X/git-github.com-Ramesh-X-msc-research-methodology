"""Entrypoint for E1 baseline experiment.

Configuration is specified in the top section of this file. No CLI args are accepted.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from e1_baseline_lib.constants import (
    DEFAULT_DRY_RUN,
    DEFAULT_KB_DIR,
    DEFAULT_MODEL,
    DEFAULT_OVERWRITE,
)
from logging_config import setup_logging

# ---------------- CONFIGURATION -----------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
KB_DIR = os.getenv("KB_DIR", DEFAULT_KB_DIR)
OVERWRITE = os.getenv("OVERWRITE", DEFAULT_OVERWRITE).lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", DEFAULT_DRY_RUN).lower() == "true"
# ------------------------------------------------


def main():
    # Initialize file-only logging (no console handlers)
    setup_logging()
    logger = logging.getLogger(__name__)
    from e1_baseline_lib.pipeline import run_e1_baseline

    if not DRY_RUN and not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required unless DRY_RUN=true."
        )
    queries_file = Path(KB_DIR) / "queries.jsonl"
    output_file = Path(KB_DIR) / "e1_baseline.jsonl"
    logger.info(
        "Starting E1 baseline; DRY_RUN=%s, QUERIES_FILE=%s", DRY_RUN, queries_file
    )
    run_e1_baseline(
        queries_file=queries_file,
        output_file=output_file,
        openrouter_api_key=OPENROUTER_API_KEY,
        model=MODEL,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )
    logger.info("Finished E1 baseline")


if __name__ == "__main__":
    main()
