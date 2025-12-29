"""
Simple script that generates results for E1-E4 experiments using test data.
"""

import os

from dotenv import load_dotenv

load_dotenv()
os.environ["KB_DIR"] = "output/kb_test"


if __name__ == "__main__":
    from main import main

    main()
