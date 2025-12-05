from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    DIRECT = "direct"
    MULTI_HOP = "multi_hop"
    NEGATIVE = "negative"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QueryMetadata(BaseModel):
    difficulty: Difficulty = Difficulty.MEDIUM
    category: Optional[str] = None


class Query(BaseModel):
    query_id: str
    query_type: QueryType
    query: str
    ground_truth: str
    context_reference: List[str] = []
    metadata: QueryMetadata = QueryMetadata()


class PageMeta(BaseModel):
    id: str
    title: str
    filename: str
    category: Optional[str] = None
    primary_topic: Optional[str] = None
    secondary_topics: List[str] = []
    links_to: List[str] = []


class QueryResponse(BaseModel):
    query: str
    ground_truth: str
    difficulty: Difficulty = Difficulty.MEDIUM
    category: Optional[str] = None


class NegativeQueryList(BaseModel):
    queries: List[QueryResponse]


class Structure(BaseModel):
    num_pages: int = Field(..., gt=0)
    pages: List[PageMeta] = []
