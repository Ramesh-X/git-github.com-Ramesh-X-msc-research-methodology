# results_lib/table_generators.py

import logging
from typing import List

from .models import StatisticalTestResult, TableData

logger = logging.getLogger(__name__)


def generate_table1_performance_summary(metrics_dict: dict) -> TableData:
    """Table 1: Overall Performance Summary (E1-E4)."""

    headers = [
        "Experiment",
        "Mean CP",
        "Mean F",
        "Mean AR",
        "Geometric Mean",
        "HRI Mean",
        "HRI P95",
        "95% CI",
    ]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        rows.append(
            {
                "Experiment": exp,
                "Mean CP": f"{m.context_precision.mean:.3f}",
                "Mean F": f"{m.faithfulness.mean:.3f}",
                "Mean AR": f"{m.answer_relevancy.mean:.3f}",
                "Geometric Mean": f"{m.geometric_mean.mean:.3f}",
                "HRI Mean": f"{m.hallucination_analysis.hri_distribution.mean:.3f}",
                "HRI P95": f"{m.hallucination_analysis.hri_distribution.p95:.3f}",
                "95% CI": f"[{m.geometric_mean_confidence.ci_95_lower:.3f}, {m.geometric_mean_confidence.ci_95_upper:.3f}]",
            }
        )

    return TableData(
        table_name="table1_performance_summary",
        description="Overall Performance Summary across E1-E4",
        headers=headers,
        rows=rows,
    )


def generate_table2_category_breakdown(metrics_dict: dict) -> TableData:
    """Table 2: Operational Category Breakdown."""

    headers = [
        "Experiment",
        "Clean Pass %",
        "Hallucination %",
        "Retrieval Failure %",
        "Irrelevant %",
        "Total Failure %",
    ]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        cb = m.category_breakdown

        # Import CategoryBreakdown for default values
        from .shared_models import CategoryBreakdown

        clean_pass = cb.get(
            "Clean Pass",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        hallucination = cb.get(
            "Hallucination",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        retrieval_failure = cb.get(
            "Retrieval Failure",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        irrelevant = cb.get(
            "Irrelevant Answer",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        total_failure = cb.get(
            "Total Failure",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )

        rows.append(
            {
                "Experiment": exp,
                "Clean Pass %": f"{clean_pass.percentage:.1f}",
                "Hallucination %": f"{hallucination.percentage:.1f}",
                "Retrieval Failure %": f"{retrieval_failure.percentage:.1f}",
                "Irrelevant %": f"{irrelevant.percentage:.1f}",
                "Total Failure %": f"{total_failure.percentage:.1f}",
            }
        )

    return TableData(
        table_name="table2_category_breakdown",
        description="Operational Category Breakdown",
        headers=headers,
        rows=rows,
    )


def generate_table3_statistical_significance(
    test_results: List[StatisticalTestResult],
) -> TableData:
    """Table 3: Statistical Significance Matrix."""

    headers = [
        "Comparison",
        "Mean Diff",
        "95% CI",
        "t-statistic",
        "p-value",
        "Sig",
        "Cohen's d",
        "Interpretation",
    ]

    rows = []
    for result in test_results:
        rows.append(
            {
                "Comparison": result.comparison,
                "Mean Diff": f"{result.mean_difference:.3f}",
                "95% CI": f"[{result.ci_95_lower:.3f}, {result.ci_95_upper:.3f}]",
                "t-statistic": f"{result.t_statistic:.2f}",
                "p-value": f"{result.p_value:.4f}",
                "Sig": result.significance_level,
                "Cohen's d": f"{result.cohens_d:.3f}",
                "Interpretation": result.cohens_d_interpretation,
            }
        )

    return TableData(
        table_name="table3_statistical_significance",
        description="Pairwise Statistical Comparisons",
        headers=headers,
        rows=rows,
    )


def generate_table4_latency_breakdown(metrics_dict: dict) -> TableData:
    """Table 4: Latency Breakdown."""

    headers = [
        "Experiment",
        "Retrieval Time (mean ± SD)",
        "LLM Time (mean ± SD)",
        "Total Time (P50, P95, P99)",
        "Quality/Time Efficiency",
    ]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        lat = m.latency_metrics

        rows.append(
            {
                "Experiment": exp,
                "Retrieval Time (mean ± SD)": f"{lat.retrieval.mean:.1f} ± {lat.retrieval.std_dev:.1f}",
                "LLM Time (mean ± SD)": f"{lat.llm.mean:.1f} ± {lat.llm.std_dev:.1f}",
                "Total Time (P50, P95, P99)": f"{lat.total.median:.1f}, {lat.total.p95:.1f}, {lat.total.p99:.1f}",
                "Quality/Time Efficiency": f"{m.accuracy_latency_tradeoff.quality_time_efficiency:.4f}",
            }
        )

    return TableData(
        table_name="table4_latency_breakdown",
        description="Latency Breakdown and Efficiency",
        headers=headers,
        rows=rows,
    )


def generate_table5_hallucination_analysis(metrics_dict: dict) -> TableData:
    """Table 5: Hallucination Risk Detailed Analysis."""

    headers = [
        "Experiment",
        "Brier Score",
        "High Severity %",
        "Medium Severity %",
        "Low Severity %",
        "HRI Mean",
        "HRI P95",
        "HRI P99",
    ]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        ha = m.hallucination_analysis

        rows.append(
            {
                "Experiment": exp,
                "Brier Score": f"{ha.brier_score:.4f}",
                "High Severity %": f"{ha.severity_high_pct:.1f}",
                "Medium Severity %": f"{ha.severity_medium_pct:.1f}",
                "Low Severity %": f"{ha.severity_low_pct:.1f}",
                "HRI Mean": f"{ha.hri_distribution.mean:.3f}",
                "HRI P95": f"{ha.hri_distribution.p95:.3f}",
                "HRI P99": f"{ha.hri_distribution.p99:.3f}",
            }
        )

    return TableData(
        table_name="table5_hallucination_analysis",
        description="Detailed Hallucination Risk Analysis",
        headers=headers,
        rows=rows,
    )


def generate_table6_threshold_success_rates(metrics_dict: dict) -> TableData:
    """Table 6: Threshold Success Rates."""

    headers = [
        "Experiment",
        "CP > 0.7",
        "F > 0.8",
        "AR > 0.7",
        "GMean > 0.7",
        "HRI < 0.1",
        "Partial Success",
    ]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        tm = m.threshold_metrics

        rows.append(
            {
                "Experiment": exp,
                "CP > 0.7": f"{tm.pct_context_precision_above_0_7:.1f}%",
                "F > 0.8": f"{tm.pct_faithfulness_above_0_8:.1f}%",
                "AR > 0.7": f"{tm.pct_answer_relevancy_above_0_7:.1f}%",
                "GMean > 0.7": f"{tm.pct_geometric_mean_above_0_7:.1f}%",
                "HRI < 0.1": f"{tm.pct_hri_below_0_1:.1f}%",
                "Partial Success": f"{tm.pct_partial_success:.1f}%",
            }
        )

    return TableData(
        table_name="table6_threshold_success_rates",
        description="Threshold Success Rates",
        headers=headers,
        rows=rows,
    )


def generate_table7_correlation_analysis(metrics_dict: dict) -> TableData:
    """Table 7: Correlation Analysis."""

    headers = ["Experiment", "CP vs AR", "CP vs GMean", "F vs AR", "Latency vs GMean"]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        ca = m.correlation_analysis

        rows.append(
            {
                "Experiment": exp,
                "CP vs AR": f"{ca.cp_vs_ar:.3f}" if ca.cp_vs_ar is not None else "N/A",
                "CP vs GMean": f"{ca.cp_vs_gmean:.3f}" if ca.cp_vs_gmean is not None else "N/A",
                "F vs AR": f"{ca.f_vs_ar:.3f}" if ca.f_vs_ar is not None else "N/A",
                "Latency vs GMean": f"{ca.latency_vs_gmean:.3f}"
                if ca.latency_vs_gmean
                else "N/A",
            }
        )

    return TableData(
        table_name="table7_correlation_analysis",
        description="Correlation Analysis Between Metrics",
        headers=headers,
        rows=rows,
    )


def generate_table8_generation_quality(metrics_dict: dict) -> TableData:
    """Table 8: Generation Quality Metrics."""

    headers = [
        "Experiment",
        "Exact Match %",
        "Mean F1 Score",
        "Mean ROUGE-L",
        "Mean BLEU",
    ]

    rows = []
    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        gq = m.generation_quality

        if gq:
            rows.append(
                {
                    "Experiment": exp,
                    "Exact Match %": f"{gq.exact_match_rate:.1f}%",
                    "Mean F1 Score": f"{gq.mean_f1_score:.3f}",
                    "Mean ROUGE-L": f"{gq.mean_rouge_l_score:.3f}",
                    "Mean BLEU": f"{gq.mean_bleu_score:.3f}"
                    if gq.mean_bleu_score
                    else "N/A",
                }
            )
        else:
            rows.append(
                {
                    "Experiment": exp,
                    "Exact Match %": "N/A",
                    "Mean F1 Score": "N/A",
                    "Mean ROUGE-L": "N/A",
                    "Mean BLEU": "N/A",
                }
            )

    return TableData(
        table_name="table8_generation_quality",
        description="Generation Quality Metrics",
        headers=headers,
        rows=rows,
    )
