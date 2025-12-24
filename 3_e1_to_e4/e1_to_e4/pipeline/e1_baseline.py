import json
import logging
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from ..agents import create_baseline_agent, create_openrouter_model
from ..models import ExperimentResult, QueryInput

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
    out_f = open(output_file, write_mode, encoding="utf-8")
    try:
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

            experiment_result = ExperimentResult(
                query_id=query.query_id,
                query=query.query,
                llm_answer=llm_answer,
                ground_truth=query.ground_truth,
                experiment="E1",
            )
            try:
                out_f.write(experiment_result.model_dump_json() + "\n")
                # Ensure each result is flushed to disk so a crash/resume
                # will not lose already processed queries
                out_f.flush()
                os.fsync(out_f.fileno())
            except Exception:
                # Log but continue processing to avoid losing the run
                logger.exception(
                    "Failed to write result for query_id %s", query.query_id
                )
            logger.info("Processed query_id: %s", query.query_id)

    finally:
        try:
            out_f.close()
        except Exception:
            logger.exception("Failed to close output file %s", output_file)

    logger.info("E1 baseline completed; output in %s", output_file)
