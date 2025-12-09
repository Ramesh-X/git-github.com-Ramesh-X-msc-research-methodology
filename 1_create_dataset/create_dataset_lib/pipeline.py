import json
import logging
from pathlib import Path

from tqdm import tqdm

from .agents import create_content_agent, create_openrouter_model
from .constants import (
    DATA_FOLDER,
    DEFAULT_KB_DIR,
    STRUCTURE_FILE_NAME,
)
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
    # Determine structure file path inside the output directory's data subfolder
    structure_file = Path(output_dir) / DATA_FOLDER / STRUCTURE_FILE_NAME
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

    # Validate the structure for duplicate filenames/ids; if duplicates exist, regenerate
    filenames = [p.filename for p in structure.pages]
    ids = [p.id for p in structure.pages]
    dup_filenames = {f for f in filenames if filenames.count(f) > 1}
    dup_ids = {i for i in ids if ids.count(i) > 1}
    if dup_filenames or dup_ids:
        # Fail fast: duplications in the structure indicate a generation bug or
        # corrupt/partial structure.json. Rather than auto-regenerating and
        # potentially overwriting content, exit with an explicit error.
        msg = (
            f"Loaded structure contains duplicate filenames {list(dup_filenames)} "
            f"or duplicate ids {list(dup_ids)}; aborting generation."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    agent = None
    if not dry_run:
        if not openrouter_api_key:
            msg = "OpenRouter API key is required when dry_run is False"
            logger.error(msg)
            raise RuntimeError(msg)
        logger.info("Initializing content agent with model %s", model)
        or_model = create_openrouter_model(model, openrouter_api_key)
        agent = create_content_agent(or_model)

    # Build a mapping of rot pairs to identify v1/v2 relationships
    rot_v1_to_v2 = {}
    rot_v2_ids = set()
    for rot_pair in structure.rot_pairs:
        rot_v1_to_v2[rot_pair.v1] = rot_pair.v2
        rot_v2_ids.add(rot_pair.v2)

    # Store generated v1 content for v2 generation. We also preload v1 content
    # from any existing markdown files in the output directory so resumed
    # runs can still produce valid v2 pages without re-generating v1 pages.
    v1_contents = {}
    for p in structure.pages:
        if p.id in rot_v1_to_v2:
            path = Path(output_dir) / p.filename
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        v1_contents[p.id] = f.read()
                    logger.info("Preloaded existing v1 content for: %s", p.filename)
                except Exception:
                    logger.exception(
                        "Failed to preload existing v1 content for: %s", p.filename
                    )

    total_pages = len(structure.pages)
    saved_count = 0
    skipped_count = 0
    failed_count = 0
    for page in tqdm(structure.pages, desc="Generating pages"):
        filepath = Path(output_dir) / page.filename
        if filepath.exists() and not overwrite:
            logger.info(
                "Output file already exists (resuming): %s; skipping generation for this page.",
                page.filename,
            )
            skipped_count += 1
            # For v1 pages, ensure their content is available to any v2 pages
            if page.id in rot_v1_to_v2 and page.id not in v1_contents:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        v1_contents[page.id] = f.read()
                    logger.info(
                        "Loaded v1 content from existing file for %s", page.filename
                    )
                except Exception:
                    logger.exception(
                        "Failed to read existing v1 file: %s", page.filename
                    )
            continue

        # Check if this is a v2 page that needs v1 content
        v1_content = None
        if page.id in rot_v2_ids:
            # Find the corresponding v1 page
            v1_id = None
            for v1, v2 in rot_v1_to_v2.items():
                if v2 == page.id:
                    v1_id = v1
                    break
            if v1_id and v1_id in v1_contents:
                v1_content = v1_contents[v1_id]
                logger.info("Using v1 content for v2 page: %s", page.filename)

        prompt = build_prompt(page, all_pages=structure.pages, v1_content=v1_content)
        if dry_run:
            # generate deterministic placeholder content for testing
            content = build_placeholder_content(page)
            logger.info("Generated dry-run placeholder for %s", page.filename)
        else:
            try:
                if agent is None:
                    raise RuntimeError("Agent is not initialized")
                # Run the agent with the user prompt. The model is configured with
                # a system prompt via agent creation, and the user prompt
                # (page-specific instructions) is passed here.
                resp = agent.run_sync(prompt)
                content = resp.output.content or ""
                logger.debug(
                    "Generated content for %s (%d chars)",
                    page.filename,
                    len(content),
                )
            except Exception as e:
                logger.warning(
                    "Failed to generate page '%s' (continuing with empty content): %s",
                    page.title,
                    e,
                )
                # Fall back to placeholder content rather than failing.
                content = ""

        # Store v1 content for later v2 generation
        if page.id in rot_v1_to_v2:
            v1_contents[page.id] = content
            logger.info("Stored v1 content for: %s", page.filename)

        try:
            _save_md(output_dir, page, content)
            logger.info("Saved page: %s", page.filename)
            saved_count += 1
        except Exception as e:
            logger.exception("Failed to save page %s: %s", page.filename, e)
            failed_count += 1

    logger.info("Generation finished. Files are in %s", output_dir)
    logger.info(
        "Summary - Total pages in structure: %s; Saved: %s; Skipped: %s; Failed to save: %s",
        total_pages,
        saved_count,
        skipped_count,
        failed_count,
    )

    # Verify output folder contains expected number of pages (excluding the data folder)
    output_dir_path = Path(output_dir)
    md_files = [p for p in output_dir_path.glob("*.md")]
    md_count = len(md_files)
    if md_count < num_pages:
        logger.warning(
            "Output dir '%s' contains %s markdown files, expected %s. Some pages may have been skipped or failed.",
            output_dir,
            md_count,
            num_pages,
        )
        logger.info(
            "Suggestion: re-run with OVERWRITE=true or delete existing files to force regeneration."
        )

    # Run validation
    try:
        results = validate_kb(output_dir, expected_rot_pairs=len(structure.rot_pairs))
        logger.info("Validation results: %s", results)
        for k, v in results.items():
            logger.info("Validation - %s: %s", k, v)
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        logger.error("Validation failed: %s", e)
