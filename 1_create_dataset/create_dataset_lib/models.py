from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PageType(str, Enum):
    TABULAR = "tabular"
    LOGICAL = "logical"
    UNSTRUCTURED = "unstructured"


class MistakeType(str, Enum):
    INCONSISTENCY = "inconsistency"
    OMISSION = "omission"
    POOR_UX = "poor_ux"
    OUTDATED_INFO = "outdated_info"
    ACCESSIBILITY = "accessibility_issue"


class Severity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class RotConflictType(str, Enum):
    """Legacy conflict types for backward compatibility."""

    PRICE_CHANGE = "price_change"
    TIMEFRAME_CHANGE = "timeframe_change"
    PROCESS_CHANGE = "process_change"
    CONTACT_CHANGE = "contact_change"
    ELIGIBILITY_CHANGE = "eligibility_change"


class SemanticDriftType(str, Enum):
    """Advanced semantic drift types for adversarial dataset hardening."""

    # Threshold/quantity changes that are subtle (not just numeric swap)
    CONDITIONAL_THRESHOLD = (
        "conditional_threshold"  # e.g., "free shipping $50 → $75 AND excludes AK/HI"
    )
    SCOPE_NARROWING = (
        "scope_narrowing"  # e.g., "returns 30 days all items → 30 days online only"
    )
    ELIGIBILITY_TIGHTENING = "eligibility_tightening"  # e.g., "pristine condition → pristine + original tags"
    EXCEPTION_ADDITION = (
        "exception_addition"  # e.g., "free returns → free except clearance items"
    )
    DEFINITION_SHIFT = "definition_shift"  # e.g., "business days → calendar days" or policy logic reframe
    HIDDEN_DEPENDENCY = (
        "hidden_dependency"  # Cross-page relationship not explicitly linked
    )


class Mistake(BaseModel):
    type: MistakeType
    severity: Severity


class RotPair(BaseModel):
    v1: str
    v2: str
    conflict_type: Optional[RotConflictType] = None  # Legacy
    semantic_drift_type: Optional[SemanticDriftType] = None  # New
    conflict_description: str
    lexical_overlap: Optional[float] = (
        None  # 0.0-1.0: how much text is identical (should be ~0.7)
    )
    semantic_confusion_level: Optional[str] = None  # "subtle", "moderate", "obvious"


class Page(BaseModel):
    id: str
    title: str
    filename: str
    category: Optional[str] = None
    type: PageType
    primary_topic: Optional[str] = None
    secondary_topics: Optional[List[str]] = None
    style: Optional[str] = None
    length: Optional[str] = None
    mistake: Optional[Mistake] = None
    links_to: List[str] = []
    requires_table: bool = False
    requires_mermaid: bool = False
    is_hub_page: bool = False  # Overview/index page (contains no specific answers)
    is_detail_page: bool = False  # Specific/detail page (contains actual data)


class Structure(BaseModel):
    num_pages: int = Field(..., gt=0)
    page_types: dict
    rot_pairs: List[RotPair] = []
    pages: List[Page]
    entity_anchors: List[
        dict
    ] = []  # Hidden dependencies: entities appearing across non-linked pages
