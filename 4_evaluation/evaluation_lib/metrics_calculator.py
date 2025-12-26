# evaluation_lib/metrics_calculator.py

import logging
from typing import List, Literal

import numpy as np

from .models import ExperimentMetrics, ExperimentResult, QueryEvaluationResult

logger = logging.getLogger(__name__)


def calculate_experiment_metrics(
    evaluation_results: List[QueryEvaluationResult],
    experiment: Literal["E1", "E2", "E3", "E4"],
    baseline_mean_time_ms: float
    | None = None,  # E1 baseline time for overhead calculation
) -> ExperimentMetrics:
    """
    Calculate aggregated metrics for an experiment.

    Args:
        evaluation_results: All query evaluations for this experiment
        experiment: "E1", "E2", "E3", or "E4"
        baseline_mean_time_ms: E1 mean time for overhead calculation

    Returns:
        ExperimentMetrics with all statistics
    """
    if not evaluation_results:
        raise ValueError("No evaluation results provided")

    total_queries = len(evaluation_results)

    # Extract arrays for calculations
    context_precision_scores = [r.context_precision for r in evaluation_results]
    faithfulness_scores = [r.faithfulness for r in evaluation_results]
    answer_relevancy_scores = [r.answer_relevancy for r in evaluation_results]
    geometric_mean_scores = [r.geometric_mean for r in evaluation_results]
    hallucination_risk_scores = [r.hallucination_risk_index for r in evaluation_results]

    # Note: Performance times are not in QueryEvaluationResult, they come from original ExperimentResult
    # For now, we'll skip latency metrics since they're not available here
    # They would need to be calculated separately and passed in

    # Calculate mean scores
    mean_context_precision = float(np.mean(context_precision_scores))
    mean_faithfulness = float(np.mean(faithfulness_scores))
    mean_answer_relevancy = float(np.mean(answer_relevancy_scores))
    mean_geometric_mean = float(np.mean(geometric_mean_scores))
    mean_hallucination_risk_index = float(np.mean(hallucination_risk_scores))

    # Calculate hallucination risk percentiles
    hri_95th_percentile = float(np.percentile(hallucination_risk_scores, 95))
    hri_median = float(np.median(hallucination_risk_scores))

    # Calculate category distributions
    categories = [r.category for r in evaluation_results]
    total_clean_pass = categories.count("Clean Pass")
    total_hallucination = categories.count("Hallucination")
    total_retrieval_failure = categories.count("Retrieval Failure")
    total_irrelevant_answer = categories.count("Irrelevant Answer")
    total_total_failure = categories.count("Total Failure")

    # Convert to percentages
    pct_clean_pass = (total_clean_pass / total_queries) * 100
    pct_hallucination = (total_hallucination / total_queries) * 100
    pct_retrieval_failure = (total_retrieval_failure / total_queries) * 100
    pct_irrelevant_answer = (total_irrelevant_answer / total_queries) * 100
    pct_total_failure = (total_total_failure / total_queries) * 100

    # Placeholder latency metrics (would be calculated from original ExperimentResult data)
    # For now, set to 0.0 - these should be calculated separately
    mean_retrieval_time_ms = 0.0
    mean_llm_time_ms = 0.0
    mean_total_time_ms = 0.0

    # Calculate accuracy vs latency metrics
    quality_time_efficiency = mean_geometric_mean / max(
        mean_total_time_ms, 0.001
    )  # Avoid division by zero

    # Accuracy gain per ms (only meaningful for E2-E4)
    if baseline_mean_time_ms is not None and experiment != "E1":
        # This would need baseline geometric mean passed in
        # For now, set to None and calculate later in cross-experiment analysis
        accuracy_gain_per_ms = None
    else:
        accuracy_gain_per_ms = None

    # Latency overhead vs baseline
    if baseline_mean_time_ms is not None:
        latency_overhead_vs_baseline_ms = mean_total_time_ms - baseline_mean_time_ms
    else:
        latency_overhead_vs_baseline_ms = 0.0

    # Pareto optimality will be set later in cross-experiment analysis
    is_pareto_optimal = False

    # Create metrics object
    metrics = ExperimentMetrics(
        experiment=experiment,
        total_queries=total_queries,
        mean_context_precision=mean_context_precision,
        mean_faithfulness=mean_faithfulness,
        mean_answer_relevancy=mean_answer_relevancy,
        mean_geometric_mean=mean_geometric_mean,
        mean_hallucination_risk_index=mean_hallucination_risk_index,
        hri_95th_percentile=hri_95th_percentile,
        hri_median=hri_median,
        pct_clean_pass=pct_clean_pass,
        pct_hallucination=pct_hallucination,
        pct_retrieval_failure=pct_retrieval_failure,
        pct_irrelevant_answer=pct_irrelevant_answer,
        pct_total_failure=pct_total_failure,
        mean_retrieval_time_ms=mean_retrieval_time_ms,
        mean_llm_time_ms=mean_llm_time_ms,
        mean_total_time_ms=mean_total_time_ms,
        quality_time_efficiency=quality_time_efficiency,
        accuracy_gain_per_ms=accuracy_gain_per_ms,
        latency_overhead_vs_baseline_ms=latency_overhead_vs_baseline_ms,
        is_pareto_optimal=is_pareto_optimal,
    )

    logger.info(
        f"Calculated metrics for {experiment}: {total_queries} queries, GMean={mean_geometric_mean:.3f}"
    )

    return metrics


def calculate_latency_metrics_from_experiments(
    experiment_results: List[ExperimentResult],
) -> tuple[float, float, float]:
    """
    Calculate latency metrics from original experiment results.

    Args:
        experiment_results: List of original ExperimentResult objects

    Returns:
        Tuple of (mean_retrieval_time_ms, mean_llm_time_ms, mean_total_time_ms)
    """
    if not experiment_results:
        return 0.0, 0.0, 0.0

    retrieval_times = [r.retrieval_time_ms for r in experiment_results]
    llm_times = [r.llm_time_ms for r in experiment_results]
    total_times = [r.total_time_ms for r in experiment_results]

    mean_retrieval = float(np.mean(retrieval_times))
    mean_llm = float(np.mean(llm_times))
    mean_total = float(np.mean(total_times))

    return mean_retrieval, mean_llm, mean_total


def update_accuracy_gain_per_ms(
    metrics_list: List[ExperimentMetrics],
) -> List[ExperimentMetrics]:
    """
    Update accuracy_gain_per_ms for E2-E4 experiments based on E1 baseline.

    Args:
        metrics_list: List of all ExperimentMetrics (E1-E4)

    Returns:
        Updated metrics list with accuracy_gain_per_ms populated for E2-E4
    """
    # Find E1 baseline
    e1_metrics = next((m for m in metrics_list if m.experiment == "E1"), None)
    if not e1_metrics:
        logger.warning("E1 metrics not found, cannot calculate accuracy gain per ms")
        return metrics_list

    e1_gmean = e1_metrics.mean_geometric_mean
    e1_time = e1_metrics.mean_total_time_ms

    updated_metrics = []
    for metrics in metrics_list:
        if metrics.experiment == "E1":
            # E1 has no accuracy gain
            metrics.accuracy_gain_per_ms = None
        else:
            # Calculate accuracy gain per ms for E2-E4
            if e1_time > 0:
                accuracy_gain = (metrics.mean_geometric_mean - e1_gmean) / (
                    metrics.mean_total_time_ms - e1_time
                )
                metrics.accuracy_gain_per_ms = accuracy_gain
            else:
                metrics.accuracy_gain_per_ms = None

        updated_metrics.append(metrics)

    return updated_metrics


def calculate_pareto_optimality(
    metrics_list: List[ExperimentMetrics],
) -> List[ExperimentMetrics]:
    """
    Calculate Pareto optimality for accuracy vs latency trade-off.

    An experiment is Pareto optimal if no other experiment has both:
    - Higher or equal geometric mean (better/equal quality)
    - Lower or equal total time (better/equal speed)

    Args:
        metrics_list: List of all ExperimentMetrics (E1-E4)

    Returns:
        Updated metrics list with is_pareto_optimal flags set
    """
    updated_metrics = []

    for i, metrics_i in enumerate(metrics_list):
        is_dominated = False

        for j, metrics_j in enumerate(metrics_list):
            if i == j:
                continue

            # Check if metrics_j dominates metrics_i
            better_quality = (
                metrics_j.mean_geometric_mean >= metrics_i.mean_geometric_mean
            )
            better_time = metrics_j.mean_total_time_ms <= metrics_i.mean_total_time_ms
            at_least_one_strict = (
                metrics_j.mean_geometric_mean > metrics_i.mean_geometric_mean
                or metrics_j.mean_total_time_ms < metrics_i.mean_total_time_ms
            )

            if better_quality and better_time and at_least_one_strict:
                is_dominated = True
                break

        metrics_i.is_pareto_optimal = not is_dominated
        updated_metrics.append(metrics_i)

    # Log Pareto optimal experiments
    pareto_experiments = [m.experiment for m in updated_metrics if m.is_pareto_optimal]
    logger.info(f"Pareto optimal experiments: {pareto_experiments}")

    return updated_metrics
