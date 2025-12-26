"""
Simple script that runs evaluation of the results of E1-E4 experiments in dry-run mode for testing.
"""

import os

from dotenv import load_dotenv

load_dotenv()
os.environ["DRY_RUN"] = "true"
os.environ["KB_DIR"] = "output/kb_test"


if __name__ == "__main__":
    from main import main

    main()
