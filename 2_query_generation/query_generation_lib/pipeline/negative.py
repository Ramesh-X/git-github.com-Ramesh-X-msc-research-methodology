import json
import logging
from pathlib import Path
from typing import Dict, List, TextIO

from tqdm import tqdm

from ..constants import MAX_ATTEMPTS
from ..constants import NEGATIVE_PROMPT_TOKEN_LIMIT as DEFAULT_NEG_TOKEN_LIMIT
from ..kb_loader import (
    build_kb_topic_summary,
    get_linked_page_contents,
    load_page_content,
    stratified_sample_pages,
)
from ..models import Query, QueryMetadata, QueryType
from ..prompts import build_anchored_negative_prompt
from ..validators import validate_query
from .helpers import choose_negative_subtype

logger = logging.getLogger(__name__)


def generate_negative_queries(
    kb_dir: Path,
    structure,
    out_f: TextIO,
    dry_run: bool,
    num_negative: int,
    anchored_negative_agent,
    id_allocator,
    existing_ids: set,
    generated: List[Dict],
    negative_prompt_token_limit: int | None = None,
):
    """Generate anchored negative queries.

    Returns the number of negative queries generated.
    """
    kb_summary = build_kb_topic_summary(structure)
    existing_negative_count = len(
        [q for q in generated if q["query_type"] == "negative"]
    )
    num_to_generate = max(0, num_negative - existing_negative_count)
    if num_to_generate <= 0:
        return existing_negative_count

    token_limit = negative_prompt_token_limit or DEFAULT_NEG_TOKEN_LIMIT
    remaining = num_to_generate
    anchors = stratified_sample_pages(structure, num_to_generate)

    with tqdm(
        total=num_negative, desc="Negative queries", initial=existing_negative_count
    ) as pbar:
        for anchor in anchors:
            if remaining <= 0:
                break

        attempts = 0
        while attempts < MAX_ATTEMPTS:
            # Get next ID from allocator (handles missing + sequential automatically)
            query_id = id_allocator.get_next_id()
            if query_id in existing_ids:
                continue

            pbar.set_postfix(id=query_id)

            subtype = choose_negative_subtype()
            anchor_content = load_page_content(kb_dir, anchor.filename)
            linked_cts = get_linked_page_contents(kb_dir, anchor)
            linked_contents_joined = "\n\n---\n\n".join(linked_cts)
            anchor_meta = (
                f"Title: {anchor.title}\nFilename: {anchor.filename}\nCategory: {anchor.category or 'Uncategorized'}\n"
                f"Primary topic: {anchor.primary_topic or 'None'}\nSecondary topics: {', '.join(anchor.secondary_topics) if anchor.secondary_topics else 'None'}"
            )

            anchor_block = anchor_content
            if linked_contents_joined:
                anchor_block = (anchor_block + "\n\n" + linked_contents_joined).strip()
            if len(anchor_block) > token_limit:
                anchor_block = anchor_block[:token_limit]

            prompt = build_anchored_negative_prompt(
                anchor_content=anchor_block,
                linked_contents=linked_contents_joined,
                anchor_meta=anchor_meta,
                kb_summary=kb_summary,
                num_queries=1,
                subtype=subtype,
            )

            if dry_run:
                qobj = {
                    "query_id": query_id,
                    "query_type": "negative",
                    "query": f"(DRY) [{subtype}] Anchor: {anchor.title}; Question: Is there an undocumented feature?",
                    "ground_truth": "I don't know based on the KB.",
                    "context_reference": [anchor.filename],
                    "metadata": {
                        "subtype": subtype,
                        "category": anchor.category or "general",
                    },
                }
                generated.append(qobj)
                out_f.write(json.dumps(qobj, ensure_ascii=False) + "\n")
                out_f.flush()
                remaining -= 1
                pbar.update(1)
                break

            try:
                assert anchored_negative_agent is not None
                resp = anchored_negative_agent.run_sync(prompt)
                qresp = resp.output
                qobj = Query(
                    query_id=query_id,
                    query_type=QueryType.NEGATIVE,
                    query=qresp.query,
                    ground_truth=qresp.ground_truth,
                    context_reference=[anchor.filename],
                    metadata=QueryMetadata(
                        subtype=subtype,
                        category=qresp.category or anchor.category or "general",
                    ),
                )
                if not validate_query(qobj):
                    logger.warning("Validation failed for %s", qobj.query_id)
                    attempts += 1
                    if attempts >= MAX_ATTEMPTS:
                        logger.warning(
                            "Exceeded attempts for negative anchor %s; skipping",
                            anchor.filename,
                        )
                        break
                    continue
                parsed = json.loads(qobj.model_dump_json())
                generated.append(parsed)
                out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                out_f.flush()
                remaining -= 1
                pbar.update(1)
                break
            except Exception as e:
                logger.exception(
                    "Failed to generate negative query for %s: %s",
                    anchor.filename,
                    e,
                )
                attempts += 1
                if attempts >= MAX_ATTEMPTS:
                    logger.warning(
                        "Exceeded attempts for negative anchor %s (errors), skipping",
                        anchor.filename,
                    )
                    break
                continue

    return len([q for q in generated if q["query_type"] == "negative"])


__all__ = ["generate_negative_queries"]
