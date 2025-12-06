from typing import List

from pydantic import BaseModel


class QueryInput(BaseModel):
    query_id: str
    query: str
    ground_truth: str
    context_reference: List[str] = []


class E1Response(BaseModel):
    answer: str


class E1Result(BaseModel):
    query_id: str
    query: str
    llm_answer: str
    ground_truth: str
    context_reference: List[str] = []
    model: str
    dry_run: bool
