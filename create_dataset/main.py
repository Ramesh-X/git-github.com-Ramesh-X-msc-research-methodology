"""Entrypoint for the synthetic dataset generator.

Configuration is specified in the top section of this file. No CLI args are accepted.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

# ---------------- CONFIGURATION -----------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")
NUM_PAGES = int(os.getenv("NUM_PAGES", "100"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/kb")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"
GENERATE_STRUCTURE = os.getenv("GENERATE_STRUCTURE", "true").lower() == "true"
# ------------------------------------------------


def main():
    from create_dataset_lib.pipeline import run_generation

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not DRY_RUN and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required unless DRY_RUN=true."
        )

    run_generation(
        openrouter_api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", MODEL),
        num_pages=NUM_PAGES,
        output_dir=OUTPUT_DIR,
        structure_path=None
        if GENERATE_STRUCTURE
        else os.path.join(OUTPUT_DIR, "structure.json"),
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()
