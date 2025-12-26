# evaluation_lib/utils.py

import json
import logging
from pathlib import Path
from typing import List, Literal, Set, Tuple

from .models import ExperimentMetrics, ExperimentResult, QueryEvaluationResult

logger = logging.getLogger(__name__)


def load_experiment_with_contexts(
    result_file: Path, experiment: str
) -> Tuple[List[ExperimentResult], List[List[str]]]:
    """
    Load experiment results with contexts extracted separately.

    For E1: contexts will be empty lists
    For E2-E4: contexts will contain retrieved chunk texts

    Returns:
        - List of minimal ExperimentResult objects
        - List of context lists (one per query)
    """
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    full_results = []
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    full_results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON at line {line_num} in {result_file}: {e}"
                    )
                    raise
    except Exception as e:
        logger.error(f"Error reading {result_file}: {e}")
        raise

    if not full_results:
        logger.warning(f"No results found in {result_file}")
        return [], []

    minimal_results = []
    contexts = []

    for fr in full_results:
        # Extract minimal fields
        minimal = ExperimentResult(
            query_id=fr["query_id"],
            experiment=fr["experiment"],
            query=fr["query"],
            llm_answer=fr["llm_answer"],
            ground_truth=fr["ground_truth"],
            retrieval_time_ms=fr["retrieval_time_ms"],
            llm_time_ms=fr["llm_time_ms"],
            total_time_ms=fr["total_time_ms"],
        )
        minimal_results.append(minimal)

        # Extract contexts
        if experiment == "E1":
            contexts.append([])  # No retrieval
        else:
            # For E2-E4, extract chunk texts
            retrieved_chunks = fr.get("retrieved_chunks", [])
            chunk_texts = [chunk["text"] for chunk in retrieved_chunks]
            contexts.append(chunk_texts)

    logger.info(f"Loaded {len(minimal_results)} results from {result_file}")
    logger.info(
        f"Contexts: {len(contexts)} entries, E1 has {len([c for c in contexts if not c])} empty"
    )

    return minimal_results, contexts


def load_existing_evaluation_results(output_file: Path) -> Set[str]:
    """
    Load set of already processed query_ids from evaluation output file.

    Returns set of query_ids that have already been evaluated.
    """
    if not output_file.exists():
        return set()

    processed_query_ids = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    processed_query_ids.add(result["query_id"])
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON at line {line_num} in {output_file}: {e}"
                    )
                    raise
    except Exception as e:
        logger.error(f"Error reading existing results from {output_file}: {e}")
        raise

    logger.info(
        f"Found {len(processed_query_ids)} already processed queries in {output_file}"
    )
    return processed_query_ids


def save_evaluation_result(result: QueryEvaluationResult, output_file: Path) -> None:
    """
    Save a single QueryEvaluationResult to JSONL file (append mode).
    Creates parent directories if needed.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False)
        f.write("\n")

    logger.debug(
        f"Saved evaluation result for query_id={result.query_id} to {output_file}"
    )


def save_metrics(metrics: ExperimentMetrics, output_file: Path) -> None:
    """
    Save ExperimentMetrics to JSON file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metrics to {output_file}")


def categorize_query(
    cp: float, f: float, ar: float
) -> Literal[
    "Clean Pass",
    "Hallucination",
    "Retrieval Failure",
    "Irrelevant Answer",
    "Total Failure",
]:
    """
    Categorize a query based on RAGAS scores using research thresholds.

    Returns one of: "Clean Pass", "Hallucination", "Retrieval Failure",
                   "Irrelevant Answer", "Total Failure"
    """
    from .constants import (
        CLEAN_PASS_THRESHOLD,
        HALLUCINATION_CP_THRESHOLD,
        HALLUCINATION_F_THRESHOLD,
        IRRELEVANT_AR_THRESHOLD,
        IRRELEVANT_F_THRESHOLD,
        RETRIEVAL_FAILURE_CP_THRESHOLD,
        RETRIEVAL_FAILURE_F_THRESHOLD,
        TOTAL_FAILURE_AR_THRESHOLD,
        TOTAL_FAILURE_CP_THRESHOLD,
        TOTAL_FAILURE_F_THRESHOLD,
    )

    # Clean Pass: CP > 0.7 ∧ F > 0.7 ∧ AR > 0.7
    if (
        cp > CLEAN_PASS_THRESHOLD
        and f > CLEAN_PASS_THRESHOLD
        and ar > CLEAN_PASS_THRESHOLD
    ):
        return "Clean Pass"

    # Hallucination: CP > 0.6 ∧ F < 0.4
    if cp > HALLUCINATION_CP_THRESHOLD and f < HALLUCINATION_F_THRESHOLD:
        return "Hallucination"

    # Retrieval Failure: CP < 0.4 ∧ F > 0.7
    if cp < RETRIEVAL_FAILURE_CP_THRESHOLD and f > RETRIEVAL_FAILURE_F_THRESHOLD:
        return "Retrieval Failure"

    # Irrelevant Answer: AR < 0.4 ∧ F > 0.7
    if ar < IRRELEVANT_AR_THRESHOLD and f > IRRELEVANT_F_THRESHOLD:
        return "Irrelevant Answer"

    # Total Failure: CP < 0.4 ∧ F < 0.4 ∧ AR < 0.4
    if (
        cp < TOTAL_FAILURE_CP_THRESHOLD
        and f < TOTAL_FAILURE_F_THRESHOLD
        and ar < TOTAL_FAILURE_AR_THRESHOLD
    ):
        return "Total Failure"

    # Default fallback (shouldn't happen with current thresholds)
    return "Total Failure"
