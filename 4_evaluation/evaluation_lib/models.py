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


class DistributionStats(BaseModel):
    """Complete statistical distribution for a metric."""

    mean: float
    std_dev: float
    variance: float
    min: float
    max: float
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    p5: float
    p95: float
    p99: float
    iqr: float  # Interquartile Range
    mad: float  # Median Absolute Deviation


class StatisticalConfidence(BaseModel):
    """Statistical confidence and comparison metrics."""

    standard_error: float
    ci_95_lower: float
    ci_95_upper: float
    # Comparison vs E1 baseline (None for E1)
    cohens_d_vs_baseline: float | None = None
    cliffs_delta_vs_baseline: float | None = None
    p_value_vs_baseline: float | None = None
    is_significant_vs_baseline: bool | None = None  # p < 0.05


class ThresholdMetrics(BaseModel):
    """Success rates based on quality thresholds."""

    pct_context_precision_above_0_7: float
    pct_faithfulness_above_0_8: float
    pct_answer_relevancy_above_0_7: float
    pct_geometric_mean_above_0_7: float
    pct_hri_below_0_1: float  # Low hallucination risk
    pct_partial_success: float  # At least 2 of 3 RAGAS metrics above threshold


class IRMetrics(BaseModel):
    """Information Retrieval quality diagnostics."""

    retrieval_success_rate: float  # Pct with Context Precision > 0.5
    high_quality_retrieval_rate: float  # Pct with CP > 0.7
    retrieval_coverage: float  # Pct with any relevant context (CP > 0.0)
    mean_context_recall_proxy: float  # CP serves as recall proxy


class GenerationQualityMetrics(BaseModel):
    """Text generation quality metrics."""

    exact_match_rate: float
    mean_f1_score: float
    mean_rouge_l_score: float
    mean_bleu_score: float | None = None  # Optional


class HallucinationAnalysis(BaseModel):
    """Detailed hallucination risk analysis."""

    brier_score: float  # Calibration: mean((1-faithfulness - actual_error)^2)
    severity_high_pct: float  # HRI > 0.2
    severity_medium_pct: float  # 0.1 < HRI ≤ 0.2
    severity_low_pct: float  # HRI ≤ 0.1
    hri_distribution: DistributionStats


class LatencyMetrics(BaseModel):
    """Latency distribution and tail behavior."""

    retrieval: DistributionStats  # Full distribution for retrieval time
    llm: DistributionStats  # Full distribution for LLM time
    total: DistributionStats  # Full distribution for total time
    # Token metrics
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    total_tokens: float | None = None
    estimated_cost_usd: float | None = None


class CorrelationAnalysis(BaseModel):
    """Correlation between key metrics."""

    cp_vs_ar: float  # Context Precision vs Answer Relevancy
    cp_vs_gmean: float  # Context Precision vs Geometric Mean
    f_vs_ar: float  # Faithfulness vs Answer Relevancy
    latency_vs_gmean: float | None = None  # If per-query latency available
    context_len_vs_hri: float | None = None  # If context lengths available


class CategoryBreakdown(BaseModel):
    """Metrics for a query category."""

    count: int
    percentage: float
    mean_gmean: float
    mean_hri: float
    mean_latency_ms: float | None = None


class AccuracyLatencyTradeoff(BaseModel):
    """Accuracy vs latency trade-off analysis."""

    quality_time_efficiency: float  # gmean / total_time_ms
    accuracy_gain_vs_baseline: float | None = None  # For E2-E4
    latency_overhead_vs_baseline_ms: float | None = None  # For E2-E4
    is_pareto_optimal: bool


class ExperimentMetrics(BaseModel):
    """Comprehensive aggregated metrics for an experiment (E1-E4)."""

    # === BASIC INFO ===
    experiment: Literal["E1", "E2", "E3", "E4"]
    total_queries: int

    # === CATEGORY 1: DISTRIBUTION METRICS ===
    # Full statistical distributions for all RAGAS metrics
    context_precision: DistributionStats
    faithfulness: DistributionStats
    answer_relevancy: DistributionStats
    geometric_mean: DistributionStats

    # === CATEGORY 2: STATISTICAL CONFIDENCE ===
    geometric_mean_confidence: StatisticalConfidence

    # === CATEGORY 3: THRESHOLDED SUCCESS RATES ===
    threshold_metrics: ThresholdMetrics

    # === CATEGORY 4: IR QUALITY DIAGNOSTICS ===
    ir_metrics: IRMetrics

    # === CATEGORY 5: GENERATION QUALITY ===
    generation_quality: GenerationQualityMetrics | None = None

    # === CATEGORY 6: HALLUCINATION ANALYSIS ===
    hallucination_analysis: HallucinationAnalysis

    # === CATEGORY 7: LATENCY & COST ===
    latency_metrics: LatencyMetrics

    # === CATEGORY 8: CORRELATIONS ===
    correlation_analysis: CorrelationAnalysis

    # === CATEGORY 9: QUERY CATEGORY BREAKDOWN ===
    category_breakdown: dict[
        str, CategoryBreakdown
    ]  # Keys: "Clean Pass", "Hallucination", etc.

    # === ACCURACY VS LATENCY TRADE-OFF ===
    accuracy_latency_tradeoff: AccuracyLatencyTradeoff
