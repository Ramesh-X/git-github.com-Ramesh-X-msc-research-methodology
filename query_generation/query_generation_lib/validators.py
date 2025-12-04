import json
import logging
from typing import Iterable

from .models import Query

logger = logging.getLogger(__name__)


def validate_query(q: Query) -> bool:
    # Basic validation: id, query, ground_truth present
    if not q.query_id:
        logger.error("Query missing id: %s", q)
        return False
    if not q.query or q.query.strip() == "":
        logger.error("Query text missing for id %s", q.query_id)
        return False
    if q.ground_truth is None:
        logger.error("Ground truth missing for id %s", q.query_id)
        return False
    return True


def validate_query_set(queries: Iterable[Query]) -> dict:
    queries = list(queries)
    # Unique ID check
    ids = [q.query_id for q in queries]
    if len(ids) != len(set(ids)):
        dup = [x for x in ids if ids.count(x) > 1]
        logger.error("Duplicate query IDs found: %s", list(set(dup)))
    # Return simple stats for the caller
    counts = {"total": len(queries)}
    types = {"direct": 0, "multi_hop": 0, "negative": 0}
    for q in queries:
        types[q.query_type.value] = types.get(q.query_type.value, 0) + 1
    counts.update(types)
    return counts


def load_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(filepath: str, iterables: Iterable[dict]):
    with open(filepath, "w", encoding="utf-8") as f:
        for d in iterables:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
