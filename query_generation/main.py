"""Entrypoint for query generation evaluation queries.

Follows the same facade pattern as `create_dataset/main.py`.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from logging_config import setup_logging

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")
KB_DIR = os.getenv("KB_DIR", "output/kb")
NUM_DIRECT = int(os.getenv("NUM_DIRECT", "100"))
NUM_MULTI_HOP = int(os.getenv("NUM_MULTI_HOP", "50"))
NUM_NEGATIVE = int(os.getenv("NUM_NEGATIVE", "50"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    from query_generation_lib.pipeline import run_query_generation

    if not DRY_RUN and not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required when DRY_RUN=false")

    logger.info("Starting query generation; DRY_RUN=%s, KB_DIR=%s", DRY_RUN, KB_DIR)
    run_query_generation(
        kb_dir=Path(KB_DIR),
        output_file=Path(KB_DIR) / "queries.jsonl",
        num_direct=NUM_DIRECT,
        num_multi_hop=NUM_MULTI_HOP,
        num_negative=NUM_NEGATIVE,
        openrouter_api_key=OPENROUTER_API_KEY,
        model=MODEL,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )
    logger.info("Finished run_query_generation")


if __name__ == "__main__":
    main()
