from typing import List, Literal

from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    query_id: str
    query: str
    ground_truth: str


class DocumentChunk(BaseModel):
    """Represents a document chunk for storage and retrieval."""

    chunk_id: str
    text: str
    embedding: List[float] = []  # Will be populated after embedding
    score: float = 0.0  # Will be populated after retrieval
    rerank_score: float = 0.0  # Will be populated after reranking
    metadata: dict = {}


class E1Response(BaseModel):
    """Baseline response format (no retrieval)."""

    answer: str


class E2Response(BaseModel):
    """Standard RAG response format."""

    answer: str


class E3Response(BaseModel):
    """Filtered RAG response format."""

    answer: str


class E4Response(BaseModel):
    """Reasoning RAG response with Chain-of-Thought."""

    reasoning_steps: str = Field(description="Step-by-step reasoning process")
    answer: str = Field(description="Final answer based on reasoning")


class ExperimentResult(BaseModel):
    query_id: str
    experiment: Literal["E1", "E2", "E3", "E4"]  # Strict typing - E1, E2, E3, or E4
    query: str
    retrieved_chunks: List[DocumentChunk]  # Empty list for E1
    llm_answer: str
    reasoning_steps: str | None = None  # Only populated for E4 experiments
    ground_truth: str
    # Performance metrics
    retrieval_time_ms: float  # Will be 0.0 for E1
    llm_time_ms: float
    total_time_ms: float
