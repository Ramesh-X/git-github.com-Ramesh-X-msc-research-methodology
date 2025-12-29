# evaluation_lib/metrics_calculator.py

import logging
from typing import List, Literal

import numpy as np

from .models import (
    AccuracyLatencyTradeoff,
    CategoryBreakdown,
    CorrelationAnalysis,
    DistributionStats,
    ExperimentMetrics,
    ExperimentResult,
    GenerationQualityMetrics,
    HallucinationAnalysis,
    IRMetrics,
    LatencyMetrics,
    QueryEvaluationResult,
    StatisticalConfidence,
    ThresholdMetrics,
)

logger = logging.getLogger(__name__)


def compute_distribution_stats(values: List[float]) -> DistributionStats:
    """Compute complete statistical distribution for a list of values."""
    values_array = np.array(values)

    return DistributionStats(
        mean=float(np.mean(values_array)),
        std_dev=float(np.std(values_array, ddof=1)),  # Sample standard deviation
        variance=float(np.var(values_array, ddof=1)),
        min=float(np.min(values_array)),
        max=float(np.max(values_array)),
        median=float(np.median(values_array)),
        q1=float(np.percentile(values_array, 25)),
        q3=float(np.percentile(values_array, 75)),
        p5=float(np.percentile(values_array, 5)),
        p95=float(np.percentile(values_array, 95)),
        p99=float(np.percentile(values_array, 99)),
        iqr=float(np.percentile(values_array, 75) - np.percentile(values_array, 25)),
        mad=float(np.median(np.abs(values_array - np.median(values_array)))),
    )


def compute_confidence_interval_bootstrap(
    values: List[float], n_bootstraps: int = 1000, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    values_array = np.array(values)
    n = len(values_array)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(values_array, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentiles
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_means, upper_percentile))

    return ci_lower, ci_upper


def compute_cohens_d(sample1: List[float], sample2: List[float]) -> float:
    """Compute Cohen's d effect size between two samples."""
    arr1, arr2 = np.array(sample1), np.array(sample2)

    n1, n2 = len(arr1), len(arr2)
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean2 - mean1) / pooled_std)  # Positive means sample2 > sample1


def compute_cliffs_delta(sample1: List[float], sample2: List[float]) -> float:
    """Compute Cliff's delta effect size (non-parametric)."""
    arr1, arr2 = np.array(sample1), np.array(sample2)

    n1, n2 = len(arr1), len(arr2)
    total_pairs = n1 * n2

    if total_pairs == 0:
        return 0.0

    # Count pairs where arr2 > arr1
    greater_count = sum(1 for x in arr2 for y in arr1 if x > y)
    # Count pairs where arr2 < arr1
    lesser_count = sum(1 for x in arr2 for y in arr1 if x < y)

    return float((greater_count - lesser_count) / total_pairs)


def safe_corrcoef(x: List[float], y: List[float]) -> float | None:
    """
    Safely compute Pearson correlation coefficient between two arrays.

    Handles edge cases:
    - Arrays with zero variance (all values are identical)
    - Arrays with NaN or infinite values
    - Arrays with insufficient data

    Returns:
        Correlation coefficient between -1 and 1, or None if computation is invalid
    """
    if len(x) != len(y) or len(x) < 2:
        return None

    arr_x = np.array(x)
    arr_y = np.array(y)

    # Check for NaN or infinite values
    if np.any(~np.isfinite(arr_x)) or np.any(~np.isfinite(arr_y)):
        logger.warning("NaN or infinite values detected in correlation computation")
        return None

    # Check for zero variance (all values are the same)
    if np.std(arr_x) == 0.0 or np.std(arr_y) == 0.0:
        # If both arrays are constant, correlation is undefined (return None)
        # If only one is constant, correlation is also undefined
        return None

    # Compute correlation safely
    with np.errstate(invalid="raise", divide="raise"):
        try:
            corr_matrix = np.corrcoef(arr_x, arr_y)
            corr_value = float(corr_matrix[0, 1])

            # Validate result
            if not np.isfinite(corr_value):
                return None

            return corr_value
        except (FloatingPointError, ValueError) as e:
            logger.warning(f"Error computing correlation: {e}")
            return None


def compute_brier_score(faithfulness_scores: List[float]) -> float:
    """Compute Brier score for calibration assessment."""
    # For simplicity, use faithfulness as a proxy for correctness
    # In a real implementation, you'd need ground truth correctness labels
    # Brier score = mean((predicted - actual)^2)
    # Here we approximate using (faithfulness - 1)^2 as penalty for low faithfulness
    scores_array = np.array(faithfulness_scores)
    return float(np.mean((scores_array - 1.0) ** 2))


def compute_f1_score(predictions: List[str], references: List[str]) -> float:
    """Compute token-level F1 score between predictions and references."""
    try:
        from scipy.sparse import issparse
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics import f1_score

        # Simple token-level F1 using bag-of-words with binary=True
        vectorizer = CountVectorizer(
            tokenizer=str.split,
            token_pattern=None,  # important when supplying a tokenizer
            lowercase=True,
            binary=True,  # binarize at the vectorizer level
        )
        pred_vectors = vectorizer.fit_transform(predictions)
        ref_vectors = vectorizer.transform(references)

        def to_dense_int(X):
            X_dense = X.toarray() if issparse(X) else X
            return np.asarray(X_dense, dtype=np.int8)

        pred_binary = to_dense_int(pred_vectors)
        ref_binary = to_dense_int(ref_vectors)

        # Compute F1 for each sample, then average
        f1_scores = []
        for pred, ref in zip(pred_binary, ref_binary):
            if np.sum(ref) == 0:  # No tokens in reference
                f1_scores.append(1.0 if np.sum(pred) == 0 else 0.0)
            else:
                f1_scores.append(f1_score(ref, pred))

        return float(np.mean(f1_scores))

    except ImportError:
        logger.warning("sklearn not available, returning 0.0 for F1 score")
        return 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Compute ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []

        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append(score["rougeL"].fmeasure)

        return float(np.mean(scores))

    except ImportError:
        logger.warning("rouge_score not available, returning 0.0 for ROUGE-L")
        return 0.0


def compute_bleu_score(predictions: List[str], references: List[str]) -> float | None:
    """Compute BLEU score."""
    try:
        import sacrebleu

        # Prepare references as list of lists for sacrebleu
        refs = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        return float(bleu.score / 100.0)  # Convert to 0-1 scale

    except ImportError:
        logger.warning("sacrebleu not available, skipping BLEU score")
        return None


def calculate_experiment_metrics(
    evaluation_results: List[QueryEvaluationResult],
    experiment: Literal["E1", "E2", "E3", "E4"],
    experiment_results: List[ExperimentResult],
) -> ExperimentMetrics:
    """
    Calculate comprehensive aggregated metrics for an experiment.

    Args:
        evaluation_results: All query evaluations for this experiment
        experiment: "E1", "E2", "E3", or "E4"
        experiment_results: Original experiment results for latency/text data (optional)
        baseline_metrics: E1 baseline metrics for comparison (optional)

    Returns:
        ExperimentMetrics with all comprehensive statistics
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

    # Extract latency data if available
    if experiment_results:
        retrieval_times = [r.retrieval_time_ms for r in experiment_results]
        llm_times = [r.llm_time_ms for r in experiment_results]
        total_times = [r.total_time_ms for r in experiment_results]
        predictions = [r.llm_answer for r in experiment_results]
        references = [r.ground_truth for r in experiment_results]
    else:
        retrieval_times = []
        llm_times = []
        total_times = []
        predictions = []
        references = []

    # === CATEGORY 1: DISTRIBUTION METRICS ===
    context_precision_dist = compute_distribution_stats(context_precision_scores)
    faithfulness_dist = compute_distribution_stats(faithfulness_scores)
    answer_relevancy_dist = compute_distribution_stats(answer_relevancy_scores)
    geometric_mean_dist = compute_distribution_stats(geometric_mean_scores)

    # === CATEGORY 2: STATISTICAL CONFIDENCE ===
    ci_lower, ci_upper = compute_confidence_interval_bootstrap(geometric_mean_scores)
    standard_error = geometric_mean_dist.std_dev / np.sqrt(total_queries)

    geometric_mean_confidence = StatisticalConfidence(
        standard_error=standard_error,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        # Effect sizes vs baseline will be computed later in cross-experiment analysis
        cohens_d_vs_baseline=None,
        cliffs_delta_vs_baseline=None,
        p_value_vs_baseline=None,
        is_significant_vs_baseline=None,
    )

    # === CATEGORY 3: THRESHOLDED SUCCESS RATES ===
    threshold_metrics = ThresholdMetrics(
        pct_context_precision_above_0_7=sum(
            1 for x in context_precision_scores if x >= 0.7
        )
        / total_queries
        * 100,
        pct_faithfulness_above_0_8=sum(1 for x in faithfulness_scores if x >= 0.8)
        / total_queries
        * 100,
        pct_answer_relevancy_above_0_7=sum(
            1 for x in answer_relevancy_scores if x >= 0.7
        )
        / total_queries
        * 100,
        pct_geometric_mean_above_0_7=sum(1 for x in geometric_mean_scores if x >= 0.7)
        / total_queries
        * 100,
        pct_hri_below_0_1=sum(1 for x in hallucination_risk_scores if x <= 0.1)
        / total_queries
        * 100,
        pct_partial_success=sum(
            1
            for cp, f, ar in zip(
                context_precision_scores, faithfulness_scores, answer_relevancy_scores
            )
            if cp >= 0.7 or f >= 0.8 or ar >= 0.7
        )
        / total_queries
        * 100,
    )

    # === CATEGORY 4: IR QUALITY DIAGNOSTICS ===
    ir_metrics = IRMetrics(
        retrieval_success_rate=sum(1 for x in context_precision_scores if x > 0.5)
        / total_queries
        * 100,
        high_quality_retrieval_rate=sum(1 for x in context_precision_scores if x >= 0.7)
        / total_queries
        * 100,
        retrieval_coverage=sum(1 for x in context_precision_scores if x > 0.0)
        / total_queries
        * 100,
        mean_context_recall_proxy=context_precision_dist.mean,  # CP serves as recall proxy
    )

    # === CATEGORY 5: GENERATION QUALITY ===
    generation_quality = None
    if predictions and references:
        generation_quality = GenerationQualityMetrics(
            exact_match_rate=sum(
                1
                for pred, ref in zip(predictions, references)
                if pred.strip() == ref.strip()
            )
            / total_queries
            * 100,
            mean_f1_score=compute_f1_score(predictions, references),
            mean_rouge_l_score=compute_rouge_l(predictions, references),
            mean_bleu_score=compute_bleu_score(predictions, references),
        )

    # === CATEGORY 6: HALLUCINATION ANALYSIS ===
    hallucination_analysis = HallucinationAnalysis(
        brier_score=compute_brier_score(faithfulness_scores),
        severity_high_pct=sum(1 for x in hallucination_risk_scores if x > 0.2)
        / total_queries
        * 100,
        severity_medium_pct=sum(1 for x in hallucination_risk_scores if 0.1 < x <= 0.2)
        / total_queries
        * 100,
        severity_low_pct=sum(1 for x in hallucination_risk_scores if x <= 0.1)
        / total_queries
        * 100,
        hri_distribution=compute_distribution_stats(hallucination_risk_scores),
    )

    # === CATEGORY 7: LATENCY & COST ===
    if retrieval_times and llm_times and total_times:
        latency_metrics = LatencyMetrics(
            retrieval=compute_distribution_stats(retrieval_times),
            llm=compute_distribution_stats(llm_times),
            total=compute_distribution_stats(total_times),
            # Token metrics would need to be computed separately
            mean_input_tokens=None,
            mean_output_tokens=None,
            total_tokens=None,
            estimated_cost_usd=None,
        )
    else:
        # Placeholder when no latency data available
        latency_metrics = LatencyMetrics(
            retrieval=DistributionStats(
                mean=0.0,
                std_dev=0.0,
                variance=0.0,
                min=0.0,
                max=0.0,
                median=0.0,
                q1=0.0,
                q3=0.0,
                p5=0.0,
                p95=0.0,
                p99=0.0,
                iqr=0.0,
                mad=0.0,
            ),
            llm=DistributionStats(
                mean=0.0,
                std_dev=0.0,
                variance=0.0,
                min=0.0,
                max=0.0,
                median=0.0,
                q1=0.0,
                q3=0.0,
                p5=0.0,
                p95=0.0,
                p99=0.0,
                iqr=0.0,
                mad=0.0,
            ),
            total=DistributionStats(
                mean=0.0,
                std_dev=0.0,
                variance=0.0,
                min=0.0,
                max=0.0,
                median=0.0,
                q1=0.0,
                q3=0.0,
                p5=0.0,
                p95=0.0,
                p99=0.0,
                iqr=0.0,
                mad=0.0,
            ),
        )

    # === CATEGORY 8: CORRELATIONS ===
    correlation_analysis = CorrelationAnalysis(
        cp_vs_ar=safe_corrcoef(context_precision_scores, answer_relevancy_scores),
        cp_vs_gmean=safe_corrcoef(context_precision_scores, geometric_mean_scores),
        f_vs_ar=safe_corrcoef(faithfulness_scores, answer_relevancy_scores),
        latency_vs_gmean=(
            safe_corrcoef(total_times, geometric_mean_scores) if total_times else None
        ),
        context_len_vs_hri=None,  # Would need context length data
    )

    # === CATEGORY 9: QUERY CATEGORY BREAKDOWN ===
    categories = [r.category for r in evaluation_results]
    category_counts = {
        "Clean Pass": categories.count("Clean Pass"),
        "Hallucination": categories.count("Hallucination"),
        "Retrieval Failure": categories.count("Retrieval Failure"),
        "Irrelevant Answer": categories.count("Irrelevant Answer"),
        "Total Failure": categories.count("Total Failure"),
    }

    category_breakdown = {}
    for cat_name, count in category_counts.items():
        if count > 0:
            # Get metrics for queries in this category
            cat_indices = [i for i, c in enumerate(categories) if c == cat_name]
            cat_gmeans = [geometric_mean_scores[i] for i in cat_indices]
            cat_hris = [hallucination_risk_scores[i] for i in cat_indices]
            cat_latencies = [total_times[i] for i in cat_indices] if total_times else []

            category_breakdown[cat_name] = CategoryBreakdown(
                count=count,
                percentage=count / total_queries * 100,
                mean_gmean=float(np.mean(cat_gmeans)),
                mean_hri=float(np.mean(cat_hris)),
                mean_latency_ms=float(np.mean(cat_latencies))
                if cat_latencies
                else None,
            )

    # === ACCURACY VS LATENCY TRADE-OFF ===
    total_mean = latency_metrics.total.mean
    quality_time_efficiency = geometric_mean_dist.mean / max(total_mean, 0.001)

    accuracy_latency_tradeoff = AccuracyLatencyTradeoff(
        quality_time_efficiency=quality_time_efficiency,
        accuracy_gain_vs_baseline=None,  # Will be computed in cross-experiment analysis
        latency_overhead_vs_baseline_ms=None,  # Will be computed in cross-experiment analysis
        is_pareto_optimal=False,  # Will be set in cross-experiment analysis
    )

    # Create comprehensive metrics object
    metrics = ExperimentMetrics(
        experiment=experiment,
        total_queries=total_queries,
        # Distribution metrics
        context_precision=context_precision_dist,
        faithfulness=faithfulness_dist,
        answer_relevancy=answer_relevancy_dist,
        geometric_mean=geometric_mean_dist,
        # Confidence
        geometric_mean_confidence=geometric_mean_confidence,
        # Thresholds
        threshold_metrics=threshold_metrics,
        # IR metrics
        ir_metrics=ir_metrics,
        # Generation quality
        generation_quality=generation_quality,
        # Hallucination analysis
        hallucination_analysis=hallucination_analysis,
        # Latency
        latency_metrics=latency_metrics,
        # Correlations
        correlation_analysis=correlation_analysis,
        # Category breakdown
        category_breakdown=category_breakdown,
        # Trade-off
        accuracy_latency_tradeoff=accuracy_latency_tradeoff,
    )

    logger.info(
        f"Calculated comprehensive metrics for {experiment}: {total_queries} queries, GMean={geometric_mean_dist.mean:.3f}"
    )

    return metrics


def update_accuracy_gain_vs_baseline(
    metrics_list: List[ExperimentMetrics],
) -> None:
    """
    Update accuracy_gain_vs_baseline for E2-E4 experiments based on E1 baseline.

    Args:
        metrics_list: List of all ExperimentMetrics (E1-E4) - modified in place
    """
    # Find E1 baseline
    e1_metrics = next((m for m in metrics_list if m.experiment == "E1"), None)
    if not e1_metrics:
        logger.warning(
            "E1 metrics not found, cannot calculate accuracy gain vs baseline"
        )
        return

    e1_gmean = e1_metrics.geometric_mean.mean
    e1_time = e1_metrics.latency_metrics.total.mean

    for metrics in metrics_list:
        if metrics.experiment == "E1":
            # E1 has no accuracy gain (it's the baseline)
            metrics.accuracy_latency_tradeoff.accuracy_gain_vs_baseline = None
        else:
            # Calculate accuracy gain vs baseline for E2-E4
            if e1_time > 0 and metrics.latency_metrics.total.mean > e1_time:
                accuracy_gain = (metrics.geometric_mean.mean - e1_gmean) / (
                    metrics.latency_metrics.total.mean - e1_time
                )
                metrics.accuracy_latency_tradeoff.accuracy_gain_vs_baseline = (
                    accuracy_gain
                )
            else:
                metrics.accuracy_latency_tradeoff.accuracy_gain_vs_baseline = None


def calculate_pareto_optimality(
    metrics_list: List[ExperimentMetrics],
) -> None:
    """
    Calculate Pareto optimality for accuracy vs latency trade-off.

    An experiment is Pareto optimal if no other experiment has both:
    - Higher or equal geometric mean (better/equal quality)
    - Lower or equal total time (better/equal speed)

    Args:
        metrics_list: List of all ExperimentMetrics (E1-E4) - modified in place
    """
    for i, metrics_i in enumerate(metrics_list):
        is_dominated = False

        for j, metrics_j in enumerate(metrics_list):
            if i == j:
                continue

            # Check if metrics_j dominates metrics_i
            better_quality = (
                metrics_j.geometric_mean.mean >= metrics_i.geometric_mean.mean
            )
            better_time = (
                metrics_j.latency_metrics.total.mean
                <= metrics_i.latency_metrics.total.mean
            )
            at_least_one_strict = (
                metrics_j.geometric_mean.mean > metrics_i.geometric_mean.mean
                or metrics_j.latency_metrics.total.mean
                < metrics_i.latency_metrics.total.mean
            )

            if better_quality and better_time and at_least_one_strict:
                is_dominated = True
                break

        metrics_i.accuracy_latency_tradeoff.is_pareto_optimal = not is_dominated

    # Log Pareto optimal experiments
    pareto_experiments = [
        m.experiment
        for m in metrics_list
        if m.accuracy_latency_tradeoff.is_pareto_optimal
    ]
    logger.info(f"Pareto optimal experiments: {pareto_experiments}")
