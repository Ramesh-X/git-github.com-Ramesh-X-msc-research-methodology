from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    DIRECT = "direct"
    MULTI_HOP = "multi_hop"
    NEGATIVE = "negative"


class DirectQuerySubtype(str, Enum):
    SIMPLE_FACT = "simple_fact"  # Single fact extraction
    TABLE_LOOKUP = "table_lookup"  # Answer requires reading table
    TABLE_AGGREGATION = (
        "table_aggregation"  # Requires multi-row aggregation/conditional logic
    )
    PROCESS_STEP = "process_step"  # Extract specific steps from procedure
    CONDITIONAL_LOGIC = "conditional_logic"  # "What happens if..." scenarios
    LIST_ENUMERATION = "list_enumeration"  # "What are all...", "List the..."
    ROT_AWARE = "rot_aware"  # Target current (v2) versioned content


class MultiHopQuerySubtype(str, Enum):
    SEQUENTIAL_PROCESS = "sequential_process"  # Steps from different procedural pages
    POLICY_FAQ_CROSS = "policy_faq_cross"  # Policy + FAQ clarification
    COMPARATIVE = "comparative"  # Compare data from two pages
    HUB_TO_DETAIL = "hub_to_detail"  # Hub page provides overview, detail has answer
    CROSS_CATEGORY = "cross_category"  # Span different categories
    ROT_AWARE = "rot_aware"  # One page is v2, other is regular


class NegativeQuerySubtype(str, Enum):
    ADJACENT_TOPIC = "adjacent_topic"  # Topic similar but not covered
    MISSING_DATA = "missing_data"  # Specific data point not in KB
    OUT_OF_SCOPE_PROCEDURE = "out_of_scope_procedure"  # Procedure not documented
    CROSS_CATEGORY_GAP = "cross_category_gap"  # Info gap between categories


class QueryMetadata(BaseModel):
    subtype: Optional[str] = None  # Subtype of query
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
    category: Optional[str] = None


class NegativeQueryList(BaseModel):
    queries: List[QueryResponse]


class Structure(BaseModel):
    num_pages: int = Field(..., gt=0)
    pages: List[PageMeta] = []
