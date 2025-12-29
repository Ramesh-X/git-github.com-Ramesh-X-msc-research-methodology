# results_lib/pipeline.py

import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .chart_generators import (
    generate_chart1_radar,
    generate_chart2_boxplot,
    generate_chart3_stacked_bar,
    generate_chart4_pareto_frontier,
    generate_chart5_heatmap,
    generate_chart6_efficiency_scatter,
    generate_chart7_hri_distributions,
    generate_chart8_correlation_scatter,
)
from .constants import CHARTS_SUBDIR, TABLES_SUBDIR
from .models import ResultsSummary
from .statistical_tests import run_all_pairwise_tests
from .table_generators import (
    generate_table1_performance_summary,
    generate_table2_category_breakdown,
    generate_table3_statistical_significance,
    generate_table4_latency_breakdown,
    generate_table5_hallucination_analysis,
    generate_table6_threshold_success_rates,
    generate_table7_correlation_analysis,
    generate_table8_generation_quality,
)
from .utils import load_evaluation_results, load_experiment_metrics, save_table_json

logger = logging.getLogger(__name__)


def run_results_generation(kb_dir: Path, overwrite: bool = False) -> None:
    """Main pipeline for results generation.

    Steps:
    1. Load experiment metrics and evaluation results
    2. Generate all 8 tables as JSON
    3. Generate all 8 charts as PNG images
    4. Run statistical tests (t-tests, effect sizes)
    5. Create summary report
    """

    eval_dir = kb_dir / "eval"
    output_dir = kb_dir / "output"
    tables_dir = output_dir / TABLES_SUBDIR
    charts_dir = output_dir / CHARTS_SUBDIR

    # Create output directories
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting Results Generation Pipeline")
    logger.info(f"KB_DIR: {kb_dir}")
    logger.info(f"OVERWRITE: {overwrite}")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info("Loading experiment metrics and evaluation results...")

    metrics_dict = load_experiment_metrics(eval_dir)
    eval_results_dict = load_evaluation_results(eval_dir)

    if not metrics_dict:
        raise ValueError("No experiment metrics found. Run evaluation first.")

    logger.info(f"Loaded metrics for experiments: {list(metrics_dict.keys())}")
    logger.info(
        f"Loaded evaluation results for experiments: {list(eval_results_dict.keys())}"
    )

    # Step 2: Run statistical tests
    logger.info("Running statistical tests...")
    test_results = run_all_pairwise_tests(
        eval_results_dict, metric_name="geometric_mean"
    )
    logger.info(f"Completed {len(test_results)} pairwise statistical tests")

    # Track generated files
    generated_tables = []
    generated_charts = []

    # Step 3: Generate tables
    logger.info("Generating tables...")
    with tqdm(total=8, desc="Tables") as pbar:
        # Table 1: Performance Summary
        table1 = generate_table1_performance_summary(metrics_dict)
        save_table_json(table1, tables_dir / "table1_performance_summary.json")
        generated_tables.append("table1_performance_summary.json")
        pbar.update(1)

        # Table 2: Category Breakdown
        table2 = generate_table2_category_breakdown(metrics_dict)
        save_table_json(table2, tables_dir / "table2_category_breakdown.json")
        generated_tables.append("table2_category_breakdown.json")
        pbar.update(1)

        # Table 3: Statistical Significance
        table3 = generate_table3_statistical_significance(test_results)
        save_table_json(table3, tables_dir / "table3_statistical_significance.json")
        generated_tables.append("table3_statistical_significance.json")
        pbar.update(1)

        # Table 4: Latency Breakdown
        table4 = generate_table4_latency_breakdown(metrics_dict)
        save_table_json(table4, tables_dir / "table4_latency_breakdown.json")
        generated_tables.append("table4_latency_breakdown.json")
        pbar.update(1)

        # Table 5: Hallucination Analysis
        table5 = generate_table5_hallucination_analysis(metrics_dict)
        save_table_json(table5, tables_dir / "table5_hallucination_analysis.json")
        generated_tables.append("table5_hallucination_analysis.json")
        pbar.update(1)

        # Table 6: Threshold Success Rates
        table6 = generate_table6_threshold_success_rates(metrics_dict)
        save_table_json(table6, tables_dir / "table6_threshold_success_rates.json")
        generated_tables.append("table6_threshold_success_rates.json")
        pbar.update(1)

        # Table 7: Correlation Analysis
        table7 = generate_table7_correlation_analysis(metrics_dict)
        save_table_json(table7, tables_dir / "table7_correlation_analysis.json")
        generated_tables.append("table7_correlation_analysis.json")
        pbar.update(1)

        # Table 8: Generation Quality
        table8 = generate_table8_generation_quality(metrics_dict)
        save_table_json(table8, tables_dir / "table8_generation_quality.json")
        generated_tables.append("table8_generation_quality.json")
        pbar.update(1)

    # Step 4: Generate charts
    logger.info("Generating charts...")
    with tqdm(total=8, desc="Charts") as pbar:
        # Chart 1: Radar Chart
        generate_chart1_radar(metrics_dict, charts_dir / "chart1_radar.png")
        generated_charts.append("chart1_radar.png")
        pbar.update(1)

        # Chart 2: Box Plot
        generate_chart2_boxplot(eval_results_dict, charts_dir / "chart2_boxplot.png")
        generated_charts.append("chart2_boxplot.png")
        pbar.update(1)

        # Chart 3: Stacked Bar Chart
        generate_chart3_stacked_bar(metrics_dict, charts_dir / "chart3_stacked_bar.png")
        generated_charts.append("chart3_stacked_bar.png")
        pbar.update(1)

        # Chart 4: Pareto Frontier
        generate_chart4_pareto_frontier(
            metrics_dict, charts_dir / "chart4_pareto_frontier.png"
        )
        generated_charts.append("chart4_pareto_frontier.png")
        pbar.update(1)

        # Chart 5: Heatmap
        generate_chart5_heatmap(eval_results_dict, charts_dir / "chart5_heatmap.png")
        generated_charts.append("chart5_heatmap.png")
        pbar.update(1)

        # Chart 6: Efficiency Scatter
        generate_chart6_efficiency_scatter(
            metrics_dict, charts_dir / "chart6_efficiency_scatter.png"
        )
        generated_charts.append("chart6_efficiency_scatter.png")
        pbar.update(1)

        # Chart 7: HRI Distributions
        generate_chart7_hri_distributions(
            eval_results_dict, charts_dir / "chart7_hri_distributions.png"
        )
        generated_charts.append("chart7_hri_distributions.png")
        pbar.update(1)

        # Chart 8: Correlation Scatter
        generate_chart8_correlation_scatter(
            metrics_dict, charts_dir / "chart8_correlation_scatter.png"
        )
        generated_charts.append("chart8_correlation_scatter.png")
        pbar.update(1)

    # Step 5: Create summary
    summary = ResultsSummary(
        tables_generated=generated_tables,
        charts_generated=generated_charts,
        statistical_tests_performed=len(test_results),
        output_directory=str(output_dir),
        generation_timestamp=datetime.now().isoformat(),
    )

    # Save summary
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    logger.info("=" * 80)
    logger.info("Results generation completed successfully")
    logger.info(f"Tables: {len(generated_tables)} ({TABLES_SUBDIR}/)")
    logger.info(f"Charts: {len(generated_charts)} ({CHARTS_SUBDIR}/)")
    logger.info(f"Statistical tests: {len(test_results)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
