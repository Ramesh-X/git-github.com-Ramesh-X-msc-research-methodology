# results_lib/chart_generators.py

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    CATEGORY_COLORS,
    EXPERIMENT_COLORS,
    FIGURE_SIZE_STANDARD,
    FIGURE_SIZE_TALL,
    FIGURE_SIZE_WIDE,
)

logger = logging.getLogger(__name__)


def generate_chart1_radar(metrics_dict: dict, output_path: Path, dpi: int = 300):
    """Chart 1: Radar Chart - Multi-metric Comparison."""

    categories = [
        "Context\nPrecision",
        "Faithfulness",
        "Answer\nRelevancy",
        "IR Quality",
        "Generation\nQuality",
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        values = [
            m.context_precision.mean,
            m.faithfulness.mean,
            m.answer_relevancy.mean,
            m.ir_metrics.retrieval_success_rate / 100,  # Normalize to 0-1
            m.generation_quality.mean_f1_score if m.generation_quality else 0,
        ]
        values += values[:1]  # Complete the circle

        color = EXPERIMENT_COLORS.get(exp, "blue")
        ax.plot(angles, values, "o-", linewidth=2, label=exp, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Performance Comparison", size=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart2_boxplot(eval_results_dict: dict, output_path: Path, dpi: int = 300):
    """Chart 2: Box Plot - Distribution Comparison."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    data = []
    labels = []

    for exp in ["E1", "E2", "E3", "E4"]:
        if exp in eval_results_dict:
            # Extract geometric_mean scores from evaluation results
            scores = [r.geometric_mean for r in eval_results_dict[exp]]
            data.append(scores)
            labels.append(exp)

    if data:
        # Create boxplots one by one to control colors
        positions = range(1, len(data) + 1)
        for i, (d, label) in enumerate(zip(data, labels)):
            color = EXPERIMENT_COLORS.get(label, "blue")
            ax.boxplot(
                d,
                positions=[positions[i]],
                patch_artist=True,
                showmeans=True,
                meanline=True,
                boxprops=dict(facecolor=color, alpha=0.6),
                medianprops=dict(color="black"),
                meanprops=dict(marker="o", markerfacecolor="red", markersize=6),
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    ax.set_ylabel("Geometric Mean Score", fontsize=12)
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_title("Distribution of Geometric Mean Scores (E1-E4)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart3_stacked_bar(metrics_dict: dict, output_path: Path, dpi: int = 300):
    """Chart 3: Stacked Bar Chart - Category Breakdown."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    experiments = []
    clean_pass = []
    hallucination = []
    retrieval_failure = []
    irrelevant = []
    total_failure = []

    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        experiments.append(exp)
        cb = metrics_dict[exp].category_breakdown

        # Import CategoryBreakdown for default values
        from .shared_models import CategoryBreakdown

        clean_pass_obj = cb.get(
            "Clean Pass",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        hallucination_obj = cb.get(
            "Hallucination",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        retrieval_failure_obj = cb.get(
            "Retrieval Failure",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        irrelevant_obj = cb.get(
            "Irrelevant Answer",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        total_failure_obj = cb.get(
            "Total Failure",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )

        clean_pass.append(clean_pass_obj.percentage)
        hallucination.append(hallucination_obj.percentage)
        retrieval_failure.append(retrieval_failure_obj.percentage)
        irrelevant.append(irrelevant_obj.percentage)
        total_failure.append(total_failure_obj.percentage)

    x = np.arange(len(experiments))

    ax.bar(x, clean_pass, label="Clean Pass", color=CATEGORY_COLORS["Clean Pass"])
    ax.bar(
        x,
        hallucination,
        bottom=clean_pass,
        label="Hallucination",
        color=CATEGORY_COLORS["Hallucination"],
    )
    ax.bar(
        x,
        retrieval_failure,
        bottom=np.array(clean_pass) + np.array(hallucination),
        label="Retrieval Failure",
        color=CATEGORY_COLORS["Retrieval Failure"],
    )
    ax.bar(
        x,
        irrelevant,
        bottom=np.array(clean_pass)
        + np.array(hallucination)
        + np.array(retrieval_failure),
        label="Irrelevant Answer",
        color=CATEGORY_COLORS["Irrelevant Answer"],
    )
    ax.bar(
        x,
        total_failure,
        bottom=np.array(clean_pass)
        + np.array(hallucination)
        + np.array(retrieval_failure)
        + np.array(irrelevant),
        label="Total Failure",
        color=CATEGORY_COLORS["Total Failure"],
    )

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Query Category Breakdown by Experiment", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart4_pareto_frontier(
    metrics_dict: dict, output_path: Path, dpi: int = 300
):
    """Chart 4: Scatter Plot - Pareto Frontier Analysis."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    x_vals = []  # Total time
    y_vals = []  # Geometric mean
    sizes = []  # Clean pass percentage
    labels = []
    colors = []

    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        x_vals.append(m.latency_metrics.total.mean)
        y_vals.append(m.geometric_mean.mean)
        # Import CategoryBreakdown for default values
        from .shared_models import CategoryBreakdown

        clean_pass_obj = m.category_breakdown.get(
            "Clean Pass",
            CategoryBreakdown(count=0, percentage=0, mean_gmean=0, mean_hri=0),
        )
        sizes.append(clean_pass_obj.percentage * 10 + 50)
        labels.append(exp)
        colors.append(EXPERIMENT_COLORS.get(exp, "blue"))

    ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7, edgecolors="black")

    # Add Pareto frontier line (simplified)
    if len(x_vals) > 1:
        # Sort by increasing x (time) and decreasing y (quality)
        sorted_points = sorted(zip(x_vals, y_vals), key=lambda p: (p[0], -p[1]))
        pareto_x = [p[0] for p in sorted_points]
        pareto_y = [p[1] for p in sorted_points]

        # Keep only Pareto optimal points
        pareto_frontier_x = [pareto_x[0]]
        pareto_frontier_y = [pareto_y[0]]

        for i in range(1, len(pareto_x)):
            if pareto_y[i] > pareto_frontier_y[-1]:
                pareto_frontier_x.append(pareto_x[i])
                pareto_frontier_y.append(pareto_y[i])

        ax.plot(
            pareto_frontier_x,
            pareto_frontier_y,
            "--",
            color="red",
            alpha=0.7,
            label="Pareto Frontier",
        )

    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x_vals[i], y_vals[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Total Latency (ms)", fontsize=12)
    ax.set_ylabel("Geometric Mean Score", fontsize=12)
    ax.set_title("Accuracy vs Latency Trade-off (Pareto Frontier)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart5_heatmap(eval_results_dict: dict, output_path: Path, dpi: int = 300):
    """Chart 5: Heatmap - Per-Query Category Performance."""

    # This is a simplified version - in practice, you'd want to show
    # a subset or aggregate by query type
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    # Create a simple summary heatmap
    experiments = ["E1", "E2", "E3", "E4"]
    categories = [
        "Clean Pass",
        "Hallucination",
        "Retrieval Failure",
        "Irrelevant Answer",
        "Total Failure",
    ]

    data = []
    for exp in experiments:
        if exp not in eval_results_dict:
            data.append([0] * len(categories))
            continue

        row = []
        for cat in categories:
            count = sum(1 for r in eval_results_dict[exp] if r.category == cat)
            percentage = (
                (count / len(eval_results_dict[exp])) * 100
                if eval_results_dict[exp]
                else 0
            )
            row.append(percentage)
        data.append(row)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(categories)):
            ax.text(j, i, ".1f", ha="center", va="center", color="black", fontsize=8)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(experiments)
    ax.set_title("Category Distribution by Experiment (%)", fontsize=14)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom")

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart6_efficiency_scatter(
    metrics_dict: dict, output_path: Path, dpi: int = 300
):
    """Chart 6: Efficiency Scatter Plot."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    x_vals = []  # Latency overhead vs E1
    y_vals = []  # Accuracy gain vs E1
    labels = []
    colors = []

    # Get E1 baseline
    e1_gmean = (
        metrics_dict.get("E1", {}).geometric_mean.mean if "E1" in metrics_dict else 0
    )
    e1_time = (
        metrics_dict.get("E1", {}).latency_metrics.total.mean
        if "E1" in metrics_dict
        else 1
    )

    for exp in ["E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        x_vals.append(m.latency_metrics.total.mean - e1_time)
        y_vals.append(m.geometric_mean.mean - e1_gmean)
        labels.append(exp)
        colors.append(EXPERIMENT_COLORS.get(exp, "blue"))

    if x_vals:
        ax.scatter(x_vals, y_vals, c=colors, s=100, alpha=0.7, edgecolors="black")

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (x_vals[i], y_vals[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        # Add reference lines
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Latency Overhead vs E1 (ms)", fontsize=12)
    ax.set_ylabel("Accuracy Gain vs E1", fontsize=12)
    ax.set_title("Quality vs Latency Trade-off (E2-E4 vs E1)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart7_hri_distributions(
    eval_results_dict: dict, output_path: Path, dpi: int = 300
):
    """Chart 7: HRI Distribution Histograms."""

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_TALL)
    axes = axes.flatten()

    for i, exp in enumerate(["E1", "E2", "E3", "E4"]):
        ax = axes[i]

        if exp in eval_results_dict:
            hri_scores = [r.hallucination_risk_index for r in eval_results_dict[exp]]

            if hri_scores:
                ax.hist(
                    hri_scores,
                    bins=20,
                    alpha=0.7,
                    color=EXPERIMENT_COLORS.get(exp, "blue"),
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add mean and median lines
                mean_val = np.mean(hri_scores)
                median_val = np.median(hri_scores)
                ax.axvline(
                    mean_val, color="red", linestyle="--", linewidth=2, label=".3f"
                )
                ax.axvline(
                    median_val, color="green", linestyle="-", linewidth=2, label=".3f"
                )

        ax.set_xlabel("Hallucination Risk Index", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{exp} HRI Distribution", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()


def generate_chart8_correlation_scatter(
    metrics_dict: dict, output_path: Path, dpi: int = 300
):
    """Chart 8: Correlation Scatter Plots."""

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

    # Subplot 1: CP vs GMean
    ax1 = axes[0]
    cp_scores = []
    gmean_scores = []
    labels = []

    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        cp_scores.append(m.context_precision.mean)
        gmean_scores.append(m.geometric_mean.mean)
        labels.append(exp)

    if cp_scores:
        colors = [
            EXPERIMENT_COLORS.get(exp, "blue")
            for exp in ["E1", "E2", "E3", "E4"]
            if exp in metrics_dict
        ]
        ax1.scatter(
            cp_scores, gmean_scores, c=colors, s=100, alpha=0.7, edgecolors="black"
        )

        for i, label in enumerate(labels):
            ax1.annotate(
                label,
                (cp_scores[i], gmean_scores[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

    ax1.set_xlabel("Context Precision", fontsize=12)
    ax1.set_ylabel("Geometric Mean", fontsize=12)
    ax1.set_title("CP vs Geometric Mean Correlation", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Latency vs GMean
    ax2 = axes[1]
    latency_scores = []
    gmean_scores2 = []
    labels2 = []

    for exp in ["E1", "E2", "E3", "E4"]:
        if exp not in metrics_dict:
            continue

        m = metrics_dict[exp]
        latency_scores.append(m.latency_metrics.total.mean)
        gmean_scores2.append(m.geometric_mean.mean)
        labels2.append(exp)

    if latency_scores:
        colors = [
            EXPERIMENT_COLORS.get(exp, "blue")
            for exp in ["E1", "E2", "E3", "E4"]
            if exp in metrics_dict
        ]
        ax2.scatter(
            latency_scores,
            gmean_scores2,
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        for i, label in enumerate(labels2):
            ax2.annotate(
                label,
                (latency_scores[i], gmean_scores2[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

    ax2.set_xlabel("Total Latency (ms)", fontsize=12)
    ax2.set_ylabel("Geometric Mean", fontsize=12)
    ax2.set_title("Latency vs Geometric Mean Correlation", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    from .utils import save_chart

    save_chart(fig, output_path, dpi)
    plt.close()
