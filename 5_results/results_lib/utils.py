# results_lib/utils.py

import logging
from pathlib import Path
from typing import Dict, List

from .shared_models import ExperimentMetrics, QueryEvaluationResult

logger = logging.getLogger(__name__)


def load_experiment_metrics(
    eval_dir: Path, experiments: List[str] = ["E1", "E2", "E3", "E4"]
) -> Dict:
    """Load all ExperimentMetrics from JSON files."""

    metrics_dict = {}
    for exp in experiments:
        metrics_file = eval_dir / f"{exp.lower()}_metrics.json"

        if not metrics_file.exists():
            logger.warning(f"Metrics file not found: {metrics_file}")
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_dict[exp] = ExperimentMetrics.model_validate_json(f.read())

        logger.info(f"Loaded metrics for {exp}")

    return metrics_dict


def load_evaluation_results(
    eval_dir: Path, experiments: List[str] = ["E1", "E2", "E3", "E4"]
) -> Dict:
    """Load all QueryEvaluationResult lists from JSONL files."""

    eval_dict = {}
    for exp in experiments:
        eval_file = eval_dir / f"{exp.lower()}_evaluation.jsonl"

        if not eval_file.exists():
            logger.warning(f"Evaluation file not found: {eval_file}")
            continue

        results = []
        with open(eval_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = QueryEvaluationResult.model_validate_json(line)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Invalid evaluation result at line {line_num} in {eval_file}: {e}"
                    )
                    raise

        eval_dict[exp] = results
        logger.info(f"Loaded {len(results)} evaluation results for {exp}")

    return eval_dict


def save_table_json(table_data, output_path: Path):
    """Save table as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table_data.model_dump_json(indent=2))

    logger.info(f"Saved table: {output_path}")


def save_chart(fig, output_path: Path, dpi: int = 300):
    """Save matplotlib figure as image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved chart: {output_path}")
