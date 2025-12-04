"""
Simple script that runs the query evaluation generator in dry-run mode with a small number of queries.
"""

import os

from dotenv import load_dotenv

load_dotenv()
# Ensure a safe, reproducible dry-run for testing
os.environ["DRY_RUN"] = "true"
os.environ["NUM_DIRECT"] = "3"
os.environ["NUM_MULTI_HOP"] = "2"
os.environ["NUM_NEGATIVE"] = "2"
os.environ["KB_DIR"] = "output/kb_test"


if __name__ == "__main__":
    # Import local package entrypoint
    from main import main

    main()
