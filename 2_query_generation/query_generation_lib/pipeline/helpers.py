import logging
import random
from collections import deque
from typing import List

from ..models import (
    DirectQuerySubtype,
    MultiHopQuerySubtype,
    NegativeQuerySubtype,
)

logger = logging.getLogger(__name__)


class QueryIDAllocator:
    """Unified ID allocation for query generation.

    Handles both missing ID recovery and new sequential ID assignment.
    Uses deque for O(1) missing ID retrieval.
    """

    def __init__(self, prefix: str, num_target: int, existing_ids: set[str]):
        """Initialize allocator for a query type.

        Args:
            prefix: Query ID prefix (e.g., 'DIR')
            num_target: Target number of queries for this type
            existing_ids: Set of all existing query IDs across all types
        """
        self.prefix = prefix
        self.num_target = num_target
        self.next_new_id = 1

        # Find missing IDs in the target range [1..num_target]
        missing_list = [
            i
            for i in range(1, num_target + 1)
            if f"{prefix}_{i:03d}" not in existing_ids
        ]
        # Use deque for O(1) popleft() operations
        self.missing_ids = deque(missing_list)

    def get_next_id(self) -> str:
        """Get next available query ID.

        Returns missing IDs first (recovery), then sequential new IDs.
        """
        if self.missing_ids:
            idx = self.missing_ids.popleft()  # O(1) operation
        else:
            idx = self.next_new_id
            self.next_new_id += 1
        return format_query_id(self.prefix, idx)

    def has_pending(self) -> bool:
        """Check if more IDs are available to allocate."""
        return bool(self.missing_ids) or (self.next_new_id <= self.num_target)

    def total_missing(self) -> int:
        """Get count of missing IDs that need to be filled."""
        return len(self.missing_ids)


def format_query_id(prefix: str, idx: int) -> str:
    """Return zero-padded query id string for the given prefix and index."""
    return f"{prefix}_{idx:03d}"


def choose_direct_subtype() -> str:
    subtypes: List[DirectQuerySubtype] = [
        DirectQuerySubtype.SIMPLE_FACT,
        DirectQuerySubtype.TABLE_LOOKUP,
        DirectQuerySubtype.TABLE_AGGREGATION,
        DirectQuerySubtype.PROCESS_STEP,
        DirectQuerySubtype.CONDITIONAL_LOGIC,
        DirectQuerySubtype.LIST_ENUMERATION,
        DirectQuerySubtype.ROT_AWARE,
    ]
    weights = [20, 15, 15, 12, 12, 12, 14]
    return random.choices(subtypes, weights=weights, k=1)[0].value


def choose_multi_hop_subtype() -> str:
    subtypes: List[MultiHopQuerySubtype] = [
        MultiHopQuerySubtype.SEQUENTIAL_PROCESS,
        MultiHopQuerySubtype.POLICY_FAQ_CROSS,
        MultiHopQuerySubtype.COMPARATIVE,
        MultiHopQuerySubtype.HUB_TO_DETAIL,
        MultiHopQuerySubtype.CROSS_CATEGORY,
        MultiHopQuerySubtype.ROT_AWARE,
    ]
    weights = [15, 15, 15, 10, 10, 5]
    return random.choices(subtypes, weights=weights, k=1)[0].value


def choose_negative_subtype() -> str:
    subtypes: List[NegativeQuerySubtype] = [
        NegativeQuerySubtype.ADJACENT_TOPIC,
        NegativeQuerySubtype.MISSING_DATA,
        NegativeQuerySubtype.OUT_OF_SCOPE_PROCEDURE,
        NegativeQuerySubtype.CROSS_CATEGORY_GAP,
    ]
    weights = [12, 10, 8, 10]
    return random.choices(subtypes, weights=weights, k=1)[0].value


def is_rot_page(page_filename: str) -> bool:
    """Returns true if the filename suggests a ROT (v1/Outdated) version."""
    return "v1" in page_filename or "Outdated" in page_filename


__all__ = [
    "QueryIDAllocator",
    "format_query_id",
    "choose_direct_subtype",
    "choose_multi_hop_subtype",
    "choose_negative_subtype",
    "is_rot_page",
]
