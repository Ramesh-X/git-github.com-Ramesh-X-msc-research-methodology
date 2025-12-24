import json
import logging
import os
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

from ..agents import create_e1_agent, create_openrouter_model
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
        agent = create_e1_agent(or_model)

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
                llm_start = time.time()
                llm_answer = "[DRY_RUN] No LLM call"
                llm_time = (time.time() - llm_start) * 1000
            else:
                try:
                    assert agent is not None
                    llm_start = time.time()
                    result = agent.run_sync(query.query)
                    llm_time = (time.time() - llm_start) * 1000
                    llm_answer = result.output.answer
                except Exception as e:
                    llm_time = 0.0
                    logger.exception(
                        "Failed to generate answer for query %s: %s", query.query_id, e
                    )
                    llm_answer = f"Error: {e}"

            experiment_result = ExperimentResult(
                query_id=query.query_id,
                experiment="E1",
                query=query.query,
                retrieved_chunks=[],  # Empty list for E1 (no retrieval)
                llm_answer=llm_answer,
                ground_truth=query.ground_truth,
                retrieval_time_ms=0.0,  # No retrieval in E1
                llm_time_ms=llm_time,
                total_time_ms=llm_time,  # Total = LLM time for E1
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
