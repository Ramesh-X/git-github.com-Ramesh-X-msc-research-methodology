"""
Simple script that runs E1-E3 experiments in dry-run mode for testing.
"""

import os

from dotenv import load_dotenv

load_dotenv()
os.environ["DRY_RUN"] = "true"
os.environ["KB_DIR"] = "output/kb_test"


if __name__ == "__main__":
    from main import main

    main()
