import json
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from .agents import create_baseline_agent, create_openrouter_model
from .models import E1Result, QueryInput

logger = logging.getLogger(__name__)


def run_e1_baseline(
    queries_file: Path,
    output_file: Path,
    openrouter_api_key: str | None,
    model: str,
    overwrite: bool,
    dry_run: bool,
):
    logger.info(
        "Starting E1 baseline; DRY_RUN=%s, QUERIES_FILE=%s", dry_run, queries_file
    )

    # Load queries
    queries: List[QueryInput] = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(QueryInput(**json.loads(line)))

    # Load existing results for resume
    existing_ids = set()
    if output_file.exists() and not overwrite:
        logger.info("Found existing output file %s; loading to resume", output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                existing_ids.add(result["query_id"])

    # Create agent if not dry run
    agent = None
    if not dry_run:
        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY required when not dry-run")
        or_model = create_openrouter_model(model, openrouter_api_key)
        agent = create_baseline_agent(or_model)

    # Process queries
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"
    with open(output_file, write_mode, encoding="utf-8") as f:
        for query in tqdm(queries, desc="Processing queries"):
            if query.query_id in existing_ids and not overwrite:
                logger.info("Skipping existing query_id: %s", query.query_id)
                continue

            if dry_run:
                llm_answer = "[DRY_RUN] No LLM call"
            else:
                try:
                    assert agent is not None
                    result = agent.run_sync(query.query)
                    llm_answer = result.output.answer
                except Exception as e:
                    logger.exception(
                        "Failed to generate answer for query %s: %s", query.query_id, e
                    )
                    llm_answer = f"Error: {e}"

            e1_result = E1Result(
                query_id=query.query_id,
                query=query.query,
                llm_answer=llm_answer,
                ground_truth=query.ground_truth,
                context_reference=query.context_reference,
                model=model,
                dry_run=dry_run,
            )
            f.write(e1_result.model_dump_json() + "\n")
            logger.info("Processed query_id: %s", query.query_id)

    logger.info("E1 baseline completed; output in %s", output_file)
