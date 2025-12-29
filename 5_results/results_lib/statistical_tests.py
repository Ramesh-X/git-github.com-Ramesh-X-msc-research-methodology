# results_lib/statistical_tests.py

import logging
from typing import List

import numpy as np
from scipy import stats

from .models import StatisticalTestResult
from .shared_models import QueryEvaluationResult

logger = logging.getLogger(__name__)


def perform_pairwise_ttest(
    scores1: List[float],
    scores2: List[float],
    comparison_name: str,
    metric_name: str,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Perform paired t-test and calculate effect sizes."""

    arr1, arr2 = np.array(scores1), np.array(scores2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(arr2, arr1)  # arr2 - arr1
    is_significant = p_value < alpha

    # Significance level notation
    if p_value < 0.001:
        sig_level = "***"
    elif p_value < 0.01:
        sig_level = "**"
    elif p_value < 0.05:
        sig_level = "*"
    else:
        sig_level = "ns"

    # Cohen's d
    mean_diff = np.mean(arr2) - np.mean(arr1)
    pooled_std = np.sqrt((np.var(arr1, ddof=1) + np.var(arr2, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Cohen's d interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        d_interp = "negligible"
    elif abs_d < 0.5:
        d_interp = "small"
    elif abs_d < 0.8:
        d_interp = "medium"
    else:
        d_interp = "large"

    # Cliff's Delta
    n1, n2 = len(arr1), len(arr2)
    greater = sum(1 for x in arr2 for y in arr1 if x > y)
    lesser = sum(1 for x in arr2 for y in arr1 if x < y)
    cliffs_delta = (greater - lesser) / (n1 * n2)

    # Cliff's Delta interpretation
    abs_cd = abs(cliffs_delta)
    if abs_cd < 0.147:
        cd_interp = "negligible"
    elif abs_cd < 0.33:
        cd_interp = "small"
    elif abs_cd < 0.474:
        cd_interp = "medium"
    else:
        cd_interp = "large"

    # 95% CI for mean difference
    ci = stats.t.interval(
        0.95, len(arr1) - 1, loc=mean_diff, scale=stats.sem(arr2 - arr1)
    )

    return StatisticalTestResult(
        comparison=comparison_name,
        metric=metric_name,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=is_significant,
        significance_level=sig_level,
        cohens_d=float(cohens_d),
        cohens_d_interpretation=d_interp,
        cliffs_delta=float(cliffs_delta),
        cliffs_delta_interpretation=cd_interp,
        mean_difference=float(mean_diff),
        ci_95_lower=float(ci[0]),
        ci_95_upper=float(ci[1]),
    )


def extract_metric_scores(
    eval_results: List[QueryEvaluationResult], metric: str
) -> List[float]:
    """Extract scores for a specific metric from evaluation results."""

    if metric == "geometric_mean":
        return [r.geometric_mean for r in eval_results]
    elif metric == "context_precision":
        return [r.context_precision for r in eval_results]
    elif metric == "faithfulness":
        return [r.faithfulness for r in eval_results]
    elif metric == "answer_relevancy":
        return [r.answer_relevancy for r in eval_results]
    elif metric == "hallucination_risk_index":
        return [r.hallucination_risk_index for r in eval_results]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run_all_pairwise_tests(
    eval_results_dict: dict, metric_name: str = "geometric_mean"
) -> List[StatisticalTestResult]:
    """Run all pairwise comparisons for a metric."""

    comparisons = [
        ("E2", "E1"),
        ("E3", "E1"),
        ("E4", "E1"),
        ("E3", "E2"),
        ("E4", "E2"),
        ("E4", "E3"),
    ]

    results = []
    for exp2, exp1 in comparisons:
        if exp1 not in eval_results_dict or exp2 not in eval_results_dict:
            logger.warning(f"Missing data for {exp1} vs {exp2}")
            continue

        # Extract per-query scores
        scores1 = extract_metric_scores(eval_results_dict[exp1], metric_name)
        scores2 = extract_metric_scores(eval_results_dict[exp2], metric_name)

        if len(scores1) != len(scores2):
            logger.warning(
                f"Mismatched sample sizes: {exp1}={len(scores1)}, {exp2}={len(scores2)}"
            )
            # Use the minimum length
            min_len = min(len(scores1), len(scores2))
            scores1 = scores1[:min_len]
            scores2 = scores2[:min_len]

        if len(scores1) < 2:
            logger.warning(f"Insufficient data for {exp1} vs {exp2}")
            continue

        logger.info(f"Computing {exp2} vs {exp1} for {metric_name}")

        results.append(
            perform_pairwise_ttest(
                scores1=scores1,
                scores2=scores2,
                comparison_name=f"{exp2} vs {exp1}",
                metric_name=metric_name,
            )
        )

    return results
