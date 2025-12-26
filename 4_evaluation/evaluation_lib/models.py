# evaluation_lib/models.py

from typing import Literal

from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    """Minimal experiment result needed for evaluation."""

    query_id: str
    experiment: Literal["E1", "E2", "E3", "E4"]
    query: str
    llm_answer: str
    ground_truth: str
    # Performance metrics
    retrieval_time_ms: float
    llm_time_ms: float
    total_time_ms: float


class RAGASScores(BaseModel):
    """RAGAS evaluation scores for a single query."""

    context_precision: float = Field(
        ge=0.0, le=1.0, description="Context Precision score"
    )
    faithfulness: float = Field(ge=0.0, le=1.0, description="Faithfulness score")
    answer_relevancy: float = Field(
        ge=0.0, le=1.0, description="Answer Relevancy score"
    )


class QueryEvaluationResult(BaseModel):
    """Evaluation result for a single query."""

    query_id: str
    experiment: Literal["E1", "E2", "E3", "E4"]

    # RAGAS scores
    context_precision: float
    faithfulness: float
    answer_relevancy: float

    # Computed metrics
    geometric_mean: float  # (CP * F * AR)^(1/3)
    hallucination_risk_index: float  # (1 - F) * AR

    # Operational category
    category: Literal[
        "Clean Pass",
        "Hallucination",
        "Retrieval Failure",
        "Irrelevant Answer",
        "Total Failure",
    ]


class ExperimentMetrics(BaseModel):
    """Aggregated metrics for an experiment (E1-E4)."""

    experiment: Literal["E1", "E2", "E3", "E4"]
    total_queries: int

    # Mean RAGAS scores
    mean_context_precision: float
    mean_faithfulness: float
    mean_answer_relevancy: float

    # Mean computed metrics
    mean_geometric_mean: float
    mean_hallucination_risk_index: float

    # Hallucination Risk Index percentiles
    hri_95th_percentile: float
    hri_median: float

    # Category distribution (percentages)
    pct_clean_pass: float
    pct_hallucination: float
    pct_retrieval_failure: float
    pct_irrelevant_answer: float
    pct_total_failure: float

    # Latency Metrics
    mean_retrieval_time_ms: float
    mean_llm_time_ms: float
    mean_total_time_ms: float

    # Accuracy vs Latency Trade-off Metrics
    quality_time_efficiency: float  # Geometric Mean / total_time_ms (higher is better)
    accuracy_gain_per_ms: float | None = (
        None  # (Gmean_En - Gmean_E1) / (Time_En - Time_E1)
    )
    latency_overhead_vs_baseline_ms: float  # total_time_ms - E1_total_time_ms
    is_pareto_optimal: bool = False  # Calculated during cross-experiment analysis
