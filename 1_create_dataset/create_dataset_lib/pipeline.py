import json
import logging
from pathlib import Path

from tqdm import tqdm

from .constants import DEFAULT_KB_DIR
from .llm_client import OpenRouterClient
from .models import Page, Structure
from .prompts import build_placeholder_content, build_prompt
from .structure_generator import generate_structure
from .validators import validate_kb

logger = logging.getLogger(__name__)


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
    output_dir: str = DEFAULT_KB_DIR,
    overwrite: bool = False,
    dry_run: bool = False,
):
    # Determine structure file path inside the output directory
    structure_file = Path(output_dir) / "structure.json"
    if structure_file.exists():
        logger.info("Loading existing structure from %s", structure_file)
        try:
            with open(structure_file) as f:
                data = json.load(f)
            structure = Structure(**data)
        except Exception:
            logger.exception("Failed to load structure file; regenerating structure")
            structure = generate_structure(num_pages=num_pages, out_dir=output_dir)
    else:
        logger.info("No existing structure found; generating new structure")
        structure = generate_structure(num_pages=num_pages, out_dir=output_dir)

    client = None
    if not dry_run:
        if not openrouter_api_key:
            msg = "OpenRouter API key is required when dry_run is False"
            logger.error(msg)
            raise RuntimeError(msg)
        logger.info("Initializing OpenRouter client with model %s", model)
        client = OpenRouterClient(api_key=openrouter_api_key, model=model)

    for page in tqdm(structure.pages, desc="Generating pages"):
        filepath = Path(output_dir) / page.filename
        if filepath.exists() and not overwrite:
            logger.info("Skipping existing file: %s", page.filename)
            continue
        prompt = build_prompt(page, all_pages=structure.pages)
        if dry_run:
            # generate deterministic placeholder content for testing
            content = build_placeholder_content(page)
            logger.info("Generated dry-run placeholder for %s", page.filename)
        else:
            try:
                if client is None:
                    raise RuntimeError("Client is not initialized")
                content = client.generate(prompt, max_tokens=800)
                if content is None:
                    raise RuntimeError("No content generated")
            except Exception as e:
                logger.exception("Failed to generate page %s: %s", page.title, e)
                print(f"Failed to generate page {page.title}: {e}")
                content = ""  # continue with empty content
        _save_md(output_dir, page, content)
        logger.info("Saved page: %s", page.filename)

    logger.info("Generation finished. Files are in %s", output_dir)
    print(f"Generation finished. Files are in {output_dir}")

    # Run validation
    try:
        results = validate_kb(output_dir, expected_rot_pairs=len(structure.rot_pairs))
        logger.info("Validation results: %s", results)
        print("Validation results:")
        for k, v in results.items():
            print(f"- {k}: {v}")
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        print(f"Validation failed: {e}")
