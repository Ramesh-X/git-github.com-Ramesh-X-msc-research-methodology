"""Simple script that runs the generator in dry-run mode with a small number of pages.

Usage: python scripts/dry_run.py
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["DRY_RUN"] = "true"
os.environ["NUM_PAGES"] = "3"
os.environ["OUTPUT_DIR"] = "output/kb_test"


if __name__ == "__main__":
    from main import main

    main()
