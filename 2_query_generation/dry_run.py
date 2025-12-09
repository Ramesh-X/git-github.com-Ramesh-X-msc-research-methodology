"""
Simple script that runs the query evaluation generator in dry-run mode with a small number of queries.
"""

import os

import query_generation_lib.constants as constants
from dotenv import load_dotenv

load_dotenv()
# Ensure a safe, reproducible dry-run for testing
os.environ["DRY_RUN"] = "true"
os.environ["KB_DIR"] = "output/kb_test"
constants.NUM_DIRECT = 3
constants.NUM_MULTI_HOP = 2
constants.NUM_NEGATIVE = 2


if __name__ == "__main__":
    # Import local package entrypoint
    from main import main

    main()
