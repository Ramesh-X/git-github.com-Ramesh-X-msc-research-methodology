# evaluation_lib/pipeline.py

import logging
from pathlib import Path
from typing import List, Literal

from tqdm import tqdm

from .constants import (
    DATA_FOLDER,
    E1_EVAL_OUTPUT,
    E1_INPUT_FILE,
    E1_METRICS_OUTPUT,
    E2_EVAL_OUTPUT,
    E2_INPUT_FILE,
    E2_METRICS_OUTPUT,
    E3_EVAL_OUTPUT,
    E3_INPUT_FILE,
    E3_METRICS_OUTPUT,
    E4_EVAL_OUTPUT,
    E4_INPUT_FILE,
    E4_METRICS_OUTPUT,
    RAGAS_EMBEDDING_MODEL,
    RAGAS_LLM_MODEL,
)
from .metrics_calculator import (
    calculate_experiment_metrics,
    calculate_pareto_optimality,
    update_accuracy_gain_vs_baseline,
)
from .models import QueryEvaluationResult
from .ragas_evaluator import evaluate_batch
from .utils import (
    load_existing_evaluation_results,
    load_experiment_with_contexts,
    save_evaluation_result,
    save_metrics,
)

logger = logging.getLogger(__name__)


def run_evaluation_pipeline(
    kb_dir: Path,
    llm_model: str = RAGAS_LLM_MODEL,
    embedding_model: str = RAGAS_EMBEDDING_MODEL,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Main evaluation pipeline for E1-E4 experiments.

    Steps:
    1. For each experiment (E1-E4):
       a. Load experiment results from KB_DIR/data/e{1-4}_*.jsonl
       b. Check for existing evaluation results (resume if not overwrite)
       c. Evaluate queries using RAGAS (batch processing)
       d. Save QueryEvaluationResult to KB_DIR/eval/e{1-4}_evaluation.jsonl
       e. Calculate ExperimentMetrics
       f. Save ExperimentMetrics to KB_DIR/eval/e{1-4}_metrics.json
    2. Calculate cross-experiment metrics (Pareto optimality, accuracy gain per ms)
    3. Update metrics JSONs with final calculations

    Args:
        kb_dir: Knowledge base directory
        llm_model: LLM model for RAGAS evaluation
        embedding_model: Embedding model for RAGAS evaluation
        overwrite: Whether to overwrite existing evaluation results
        dry_run: Skip API calls and use mock scores
    """
    data_dir = kb_dir / DATA_FOLDER
    eval_dir = kb_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    experiments: List[tuple[Literal["E1", "E2", "E3", "E4"], str, str, str]] = [
        ("E1", E1_INPUT_FILE, E1_EVAL_OUTPUT, E1_METRICS_OUTPUT),
        ("E2", E2_INPUT_FILE, E2_EVAL_OUTPUT, E2_METRICS_OUTPUT),
        ("E3", E3_INPUT_FILE, E3_EVAL_OUTPUT, E3_METRICS_OUTPUT),
        ("E4", E4_INPUT_FILE, E4_EVAL_OUTPUT, E4_METRICS_OUTPUT),
    ]

    all_metrics = []

    logger.info("=" * 80)
    logger.info("Starting E1-E4 Evaluation Pipeline")
    logger.info(f"KB_DIR: {kb_dir}")
    logger.info(f"LLM_MODEL: {llm_model}")
    logger.info(f"EMBEDDING_MODEL: {embedding_model}")
    logger.info(f"OVERWRITE: {overwrite}")
    logger.info(f"DRY_RUN: {dry_run}")
    logger.info("=" * 80)

    with tqdm(total=len(experiments), desc="Processing experiments") as exp_pbar:
        for experiment, input_file, eval_output, metrics_output in experiments:
            exp_pbar.set_description(f"Processing {experiment}")
            exp_pbar.set_postfix(experiment=experiment)
            logger.info(f"Processing {experiment}")

            # Define file paths
            input_path = data_dir / input_file
            eval_output_path = eval_dir / eval_output
            metrics_output_path = eval_dir / metrics_output

            try:
                # Load experiment results with contexts
                logger.info(f"Loading {experiment} results from {input_path}")
                experiment_results, contexts_list = load_experiment_with_contexts(
                    input_path, experiment
                )

                if not experiment_results:
                    logger.warning(f"No results found for {experiment}, skipping")
                    exp_pbar.update(1)
                    continue

                logger.info(
                    f"Loaded {len(experiment_results)} queries for {experiment}"
                )

                # Check for existing evaluation results (resume functionality)
                processed_query_ids = set()
                if not overwrite:
                    processed_query_ids = load_existing_evaluation_results(
                        eval_output_path
                    )
                    if processed_query_ids:
                        logger.info(
                            f"Found {len(processed_query_ids)} existing evaluations for {experiment}, resuming"
                        )

                # Filter out already processed queries
                queries_to_evaluate = []
                contexts_to_evaluate = []

                for result, contexts in zip(experiment_results, contexts_list):
                    if result.query_id not in processed_query_ids:
                        queries_to_evaluate.append(result)
                        contexts_to_evaluate.append(contexts)

                if not queries_to_evaluate:
                    logger.info(
                        f"All {experiment} queries already evaluated, skipping evaluation"
                    )
                else:
                    logger.info(
                        f"Evaluating {len(queries_to_evaluate)} new queries for {experiment}"
                    )

                    # Evaluate queries in batches with progress tracking
                    with tqdm(
                        total=len(queries_to_evaluate),
                        desc=f"Evaluating {experiment} queries",
                        leave=False,
                    ) as query_pbar:
                        batch_size = 10  # Process in smaller batches for better progress tracking
                        for i in range(0, len(queries_to_evaluate), batch_size):
                            batch_end = min(i + batch_size, len(queries_to_evaluate))
                            batch_queries = queries_to_evaluate[i:batch_end]
                            batch_contexts = contexts_to_evaluate[i:batch_end]

                            # Evaluate batch
                            batch_results = evaluate_batch(
                                batch_queries,
                                batch_contexts,
                                llm_model,
                                embedding_model,
                                dry_run=dry_run,
                            )

                            # Save results immediately
                            for eval_result in batch_results:
                                save_evaluation_result(eval_result, eval_output_path)
                                query_pbar.set_postfix(query_id=eval_result.query_id)

                            query_pbar.update(len(batch_results))

                    logger.info(
                        f"Saved {len(queries_to_evaluate)} evaluation results for {experiment}"
                    )

                # Load all evaluation results for metrics calculation
                # Note: We need to reload because we might have resumed from existing results
                all_evaluation_results = _load_all_evaluation_results(eval_output_path)
                if not all_evaluation_results:
                    logger.warning(
                        f"No evaluation results found for {experiment} metrics calculation, skipping"
                    )
                    exp_pbar.update(1)
                    continue

                # Calculate experiment metrics
                metrics = calculate_experiment_metrics(
                    all_evaluation_results, experiment, experiment_results
                )

                # Save metrics
                save_metrics(metrics, metrics_output_path)
                all_metrics.append(metrics)

                logger.info(
                    f"Completed {experiment} evaluation and metrics calculation"
                )

            except Exception as e:
                logger.error(f"Failed to process {experiment}: {e}")
                raise

            exp_pbar.update(1)

    # Cross-experiment analysis
    if all_metrics:
        logger.info("Performing cross-experiment analysis")

        # Update accuracy gain vs baseline based on E1 baseline
        update_accuracy_gain_vs_baseline(all_metrics)

        # Calculate Pareto optimality
        calculate_pareto_optimality(all_metrics)

        # Update metrics files with final calculations
        for metrics in all_metrics:
            experiment = metrics.experiment
            metrics_output = eval_dir / f"{experiment.lower()}_metrics.json"
            save_metrics(metrics, metrics_output)

        logger.info("Cross-experiment analysis completed")

    logger.info("=" * 80)
    logger.info("E1-E4 Evaluation Pipeline completed successfully")
    logger.info(f"Results saved to: {eval_dir}")
    logger.info("=" * 80)


def _load_all_evaluation_results(
    eval_output_path: Path,
) -> List["QueryEvaluationResult"]:
    """
    Load all evaluation results from a JSONL file.

    Returns empty list if file doesn't exist.
    """
    if not eval_output_path.exists():
        return []

    results = []
    try:
        with open(eval_output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # Import here to avoid circular imports
                    from .models import QueryEvaluationResult

                    result = QueryEvaluationResult.model_validate_json(line)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Invalid evaluation result at line {line_num} in {eval_output_path}: {e}"
                    )
                    raise
    except Exception as e:
        logger.error(f"Error reading evaluation results from {eval_output_path}: {e}")
        raise

    return results
