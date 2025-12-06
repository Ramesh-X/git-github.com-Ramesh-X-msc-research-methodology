"""Pydantic models for E2-E4 experiments."""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    """Input query from queries.jsonl file."""

    query_id: str
    query_type: str
    query: str
    ground_truth: str
    context_reference: List[str] = []
    metadata: dict = {}


class RetrievedChunk(BaseModel):
    """Represents a retrieved document chunk."""

    chunk_id: str
    text: str
    score: float
    metadata: dict = {}


class E2Response(BaseModel):
    """Standard RAG response."""

    answer: str


class E3Response(BaseModel):
    """Filtered RAG response."""

    answer: str


class E4Response(BaseModel):
    """Reasoning RAG response with CoT."""

    reasoning_steps: str = Field(description="Step-by-step reasoning process")
    answer: str = Field(description="Final answer based on reasoning")


class ExperimentResult(BaseModel):
    """Result model for all experiments (E2-E4)."""

    query_id: str
    experiment: str  # "e2", "e3", or "e4"
    query: str
    retrieved_chunks: List[RetrievedChunk]
    llm_answer: str
    reasoning_steps: Optional[str] = None  # Only populated for E4
    ground_truth: str
    context_reference: List[str] = []
    metadata: dict = {}
    # Performance metrics
    retrieval_time_ms: float
    llm_time_ms: float
    total_time_ms: float
    model: str
    dry_run: bool
