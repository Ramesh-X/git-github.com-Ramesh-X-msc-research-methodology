import json
import logging
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

from tqdm import tqdm

from ..constants import MAX_ATTEMPTS
from ..kb_loader import find_linked_pairs, load_page_content
from ..models import Query, QueryMetadata, QueryType
from ..prompts import build_multi_hop_prompt
from ..validators import validate_query
from .helpers import choose_multi_hop_subtype

logger = logging.getLogger(__name__)


def generate_multi_hop_queries(
    kb_dir: Path,
    structure,
    out_f: TextIO,
    dry_run: bool,
    num_multi_hop: int,
    multi_hop_agent,
    id_allocator,
    existing_ids: set,
    generated: List[Dict],
):
    """Generate multi-hop queries based on linked page pairs.

    Returns the number of multi-hop queries generated.
    """
    pairs = find_linked_pairs(structure)
    pair_keys = {(a.filename, b.filename) for a, b in pairs}
    pair_list: List[Tuple] = [
        (a, b) for a, b in pairs if (a.filename, b.filename) in pair_keys
    ]

    generated_multi_hop_count = len(
        [q for q in generated if q["query_type"] == "multi_hop"]
    )

    if generated_multi_hop_count >= num_multi_hop or not pair_list:
        return generated_multi_hop_count

    with tqdm(
        total=num_multi_hop, desc="Multi-hop queries", initial=generated_multi_hop_count
    ) as pbar:
        for a, b in pair_list:
            if generated_multi_hop_count >= num_multi_hop:
                break

        attempts = 0
        while attempts < MAX_ATTEMPTS:
            # Get next ID from allocator (handles missing + sequential automatically)
            query_id = id_allocator.get_next_id()
            if query_id in existing_ids:
                continue

            pbar.set_postfix(id=query_id)

            subtype = choose_multi_hop_subtype()
            content_a = load_page_content(kb_dir, a.filename)
            content_b = load_page_content(kb_dir, b.filename)
            prompt = build_multi_hop_prompt(content_a, content_b, subtype=subtype)

            if dry_run:
                qobj = {
                    "query_id": query_id,
                    "query_type": "multi_hop",
                    "query": f"(DRY) [{subtype}] How can I combine info from {a.title} and {b.title}?",
                    "ground_truth": "Combine information from both pages.",
                    "context_reference": [a.filename, b.filename],
                    "metadata": {
                        "subtype": subtype,
                        "category": a.category or b.category,
                    },
                }
                generated.append(qobj)
                out_f.write(json.dumps(qobj, ensure_ascii=False) + "\n")
                out_f.flush()
                generated_multi_hop_count += 1
                pbar.update(1)
                break
            else:
                try:
                    assert multi_hop_agent is not None
                    resp = multi_hop_agent.run_sync(prompt)
                    qresp = resp.output
                    qobj = Query(
                        query_id=query_id,
                        query_type=QueryType.MULTI_HOP,
                        query=qresp.query,
                        ground_truth=qresp.ground_truth,
                        context_reference=[a.filename, b.filename],
                        metadata=QueryMetadata(
                            subtype=subtype,
                            category=qresp.category or a.category or b.category,
                        ),
                    )
                    if not validate_query(qobj):
                        logger.warning("Validation failed for %s", qobj.query_id)
                        attempts += 1
                        if attempts >= MAX_ATTEMPTS:
                            logger.warning(
                                "Exceeded attempts for multi-hop pair %s/%s; skipping",
                                a.filename,
                                b.filename,
                            )
                            break
                        continue
                    parsed = json.loads(qobj.model_dump_json())
                    generated.append(parsed)
                    out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    out_f.flush()
                    generated_multi_hop_count += 1
                    pbar.update(1)
                    break
                except Exception as e:
                    logger.exception(
                        "Failed to generate multi-hop query %s: %s", query_id, e
                    )
                    attempts += 1
                    if attempts >= MAX_ATTEMPTS:
                        logger.warning(
                            "Exceeded attempts for multi-hop pair %s/%s (errors), skipping",
                            a.filename,
                            b.filename,
                        )
                        break
                    continue

    return generated_multi_hop_count


__all__ = ["generate_multi_hop_queries"]
