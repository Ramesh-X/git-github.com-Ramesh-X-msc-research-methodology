from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from .structure_generator import generate_structure
from .prompts import build_prompt, build_placeholder_content
from .llm_client import OpenRouterClient
from .models import Structure, Page
from .constants import DEFAULT_OUTPUT_DIR
from tqdm import tqdm
from .validators import validate_kb


def _save_md(output_dir: str, page: Page, content: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / page.filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {page.title}\n\n")
        f.write(content)
    return str(filepath)


def run_generation(
    openrouter_api_key: str | None,
    model: str,
    num_pages: int = 100,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    structure_path: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
):
    if structure_path is None:
        structure = generate_structure(num_pages=num_pages, out_dir=output_dir)
    else:
        with open(structure_path) as f:
            data = json.load(f)
        structure = Structure(**data)

    client = None
    if not dry_run:
        if not openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required when dry_run is False")
        client = OpenRouterClient(api_key=openrouter_api_key, model=model)

    dataset_items = []
    for page in tqdm(structure.pages, desc="Generating pages"):
        filepath = Path(output_dir) / page.filename
        if filepath.exists() and not overwrite:
            # read metadata and append
            dataset_items.append({"filename": str(filepath), "generated": False})
            continue
        prompt = build_prompt(page, all_pages=structure.pages)
        if dry_run:
            # generate deterministic placeholder content for testing
            content = build_placeholder_content(page)
        else:
            try:
                if client is None:
                    raise RuntimeError("Client is not initialized")
                content = client.generate(prompt, max_tokens=800)
                if content is None:
                    raise RuntimeError("No content generated")
            except Exception as e:
                print(f"Failed to generate page {page.title}: {e}")
                content = ""  # continue with empty content
        saved = _save_md(output_dir, page, content)
        dataset_items.append({"filename": saved, "generated": True})

    # Save metadata
    with open(Path(output_dir) / "dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset_items:
            f.write(json.dumps(item) + "\n")

    print(f"Generation finished. Files are in {output_dir}")

    # Run validation
    try:
        results = validate_kb(output_dir, expected_rot_pairs=len(structure.rot_pairs))
        print("Validation results:")
        for k, v in results.items():
            print(f"- {k}: {v}")
    except Exception as e:
        print(f"Validation failed: {e}")
