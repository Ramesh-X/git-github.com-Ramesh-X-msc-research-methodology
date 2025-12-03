from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


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


class Mistake(BaseModel):
    type: MistakeType
    severity: Severity


class RotPair(BaseModel):
    v1: str
    v2: str
    conflict: Optional[str]


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


class Structure(BaseModel):
    num_pages: int = Field(..., gt=0)
    page_types: dict
    rot_pairs: List[RotPair] = []
    pages: List[Page]
