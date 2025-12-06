"""Entrypoint for query generation evaluation queries.

Follows the same facade pattern as `create_dataset/main.py`.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from logging_config import setup_logging
from query_generation_lib.constants import (
    DEFAULT_DRY_RUN,
    DEFAULT_KB_DIR,
    DEFAULT_MODEL,
    DEFAULT_NUM_DIRECT,
    DEFAULT_NUM_MULTI_HOP,
    DEFAULT_NUM_NEGATIVE,
    DEFAULT_OVERWRITE,
)
from query_generation_lib.constants import (
    NEGATIVE_PROMPT_TOKEN_LIMIT as DEFAULT_NEG_TOKEN_LIMIT,
)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
KB_DIR = os.getenv("KB_DIR", DEFAULT_KB_DIR)
NUM_DIRECT = int(os.getenv("NUM_DIRECT", str(DEFAULT_NUM_DIRECT)))
NUM_MULTI_HOP = int(os.getenv("NUM_MULTI_HOP", str(DEFAULT_NUM_MULTI_HOP)))
NUM_NEGATIVE = int(os.getenv("NUM_NEGATIVE", str(DEFAULT_NUM_NEGATIVE)))
NEGATIVE_PROMPT_TOKEN_LIMIT = int(
    os.getenv("NEGATIVE_PROMPT_TOKEN_LIMIT", str(DEFAULT_NEG_TOKEN_LIMIT))
)
DRY_RUN = os.getenv("DRY_RUN", DEFAULT_DRY_RUN).lower() == "true"
OVERWRITE = os.getenv("OVERWRITE", DEFAULT_OVERWRITE).lower() == "true"


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
        negative_prompt_token_limit=NEGATIVE_PROMPT_TOKEN_LIMIT,
    )
    logger.info("Finished run_query_generation")


if __name__ == "__main__":
    main()
