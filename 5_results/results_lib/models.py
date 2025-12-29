# results_lib/models.py

from typing import Literal

from pydantic import BaseModel


class StatisticalTestResult(BaseModel):
    """Pairwise statistical comparison between experiments."""

    comparison: str  # e.g., "E2 vs E1"
    metric: str  # e.g., "Geometric Mean"

    # T-test results
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    significance_level: str  # "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "ns"

    # Effect sizes
    cohens_d: float
    cohens_d_interpretation: str  # "negligible", "small", "medium", "large"
    cliffs_delta: float
    cliffs_delta_interpretation: str

    # Mean differences
    mean_difference: float
    ci_95_lower: float
    ci_95_upper: float


class TableData(BaseModel):
    """Generic table data structure."""

    table_name: str
    description: str
    headers: list[str]
    rows: list[dict]  # Each row is a dict with header keys


class ChartMetadata(BaseModel):
    """Metadata for generated charts."""

    chart_name: str
    description: str
    file_path: str
    chart_type: Literal["radar", "box", "bar", "scatter", "heatmap", "histogram"]


class ResultsSummary(BaseModel):
    """Summary of all generated results."""

    tables_generated: list[str]
    charts_generated: list[str]
    statistical_tests_performed: int
    output_directory: str
    generation_timestamp: str
