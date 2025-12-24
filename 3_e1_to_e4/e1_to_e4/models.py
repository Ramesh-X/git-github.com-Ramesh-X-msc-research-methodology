from typing import Literal

from pydantic import BaseModel


class QueryInput(BaseModel):
    query_id: str
    query: str
    ground_truth: str


class E1Response(BaseModel):
    answer: str


class ExperimentResult(BaseModel):
    query_id: str
    query: str
    llm_answer: str
    ground_truth: str
    experiment: Literal["E1"]
