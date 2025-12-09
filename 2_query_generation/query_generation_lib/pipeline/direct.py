import json
import logging
from pathlib import Path
from typing import Dict, List, TextIO

from tqdm import tqdm

from ..constants import MAX_ATTEMPTS
from ..kb_loader import load_page_content
from ..models import Query, QueryMetadata, QueryType
from ..prompts import build_direct_prompt
from ..validators import validate_query
from .helpers import choose_direct_subtype

logger = logging.getLogger(__name__)


def generate_direct_queries(
    kb_dir: Path,
    structure,
    out_f: TextIO,
    dry_run: bool,
    num_direct: int,
    direct_agent,
    id_allocator,
    existing_ids: set,
    generated: List[Dict],
):
    """Generate direct queries and append results to `generated` list and write to `out_f`.

    Returns the number of direct queries generated.
    """
    all_pages = list(structure.pages)
    current_pages = [
        p for p in all_pages if not any(x in p.filename for x in ("v1", "Outdated"))
    ]
    logger.info(
        "Filtered pages for direct queries: %d current pages (excluded %d rot pages)",
        len(current_pages),
        len(all_pages) - len(current_pages),
    )

    if not current_pages:
        return 0

    query_counts = {p.filename: 0 for p in current_pages}
    for q in generated:
        if q["query_type"] == "direct" and q.get("context_reference"):
            filename = q["context_reference"][0]
            if filename in query_counts:
                query_counts[filename] += 1

    generated_direct_count = len([q for q in generated if q["query_type"] == "direct"])
    remaining_direct = max(0, num_direct - generated_direct_count)

    if remaining_direct <= 0:
        return generated_direct_count

    with tqdm(
        total=num_direct, desc="Direct queries", initial=generated_direct_count
    ) as pbar:
        while generated_direct_count < num_direct and current_pages:
            max_count = max(query_counts.values()) if query_counts.values() else 0
            weights = [max_count + 1 - query_counts[p.filename] for p in current_pages]
            page = (
                current_pages[0]
                if len(current_pages) == 1
                else __import__("random").choices(current_pages, weights=weights, k=1)[
                    0
                ]
            )

            attempts = 0
            success = False
            while attempts < MAX_ATTEMPTS:
                # Get next ID from allocator (handles missing + sequential automatically)
                query_id = id_allocator.get_next_id()
                if query_id in existing_ids:
                    logger.info("Skipping existing query id %s", query_id)
                    continue

                pbar.set_postfix(id=query_id)

                subtype = choose_direct_subtype()
                content = load_page_content(kb_dir, page.filename)
                prompt = build_direct_prompt(content, subtype=subtype)

                if dry_run:
                    qobj = {
                        "query_id": query_id,
                        "query_type": "direct",
                        "query": f"(DRY) [{subtype}] What is the primary topic of {page.title}?",
                        "ground_truth": page.primary_topic or "Unknown",
                        "context_reference": [page.filename],
                        "metadata": {"subtype": subtype, "category": page.category},
                    }
                    generated.append(qobj)
                    out_f.write(json.dumps(qobj, ensure_ascii=False) + "\n")
                    out_f.flush()
                    query_counts[page.filename] += 1
                    generated_direct_count += 1
                    pbar.update(1)
                    success = True
                    break
                else:
                    try:
                        assert direct_agent is not None
                        resp = direct_agent.run_sync(prompt)
                        qresp = resp.output
                        qobj = Query(
                            query_id=query_id,
                            query_type=QueryType.DIRECT,
                            query=qresp.query,
                            ground_truth=qresp.ground_truth,
                            context_reference=[page.filename],
                            metadata=QueryMetadata(
                                subtype=subtype,
                                category=qresp.category or page.category,
                            ),
                        )
                        if not validate_query(qobj):
                            logger.warning("Validation failed for %s", qobj.query_id)
                            attempts += 1
                            if attempts >= MAX_ATTEMPTS:
                                logger.warning(
                                    "Exceeded attempts for page %s; skipping",
                                    page.filename,
                                )
                                break
                            continue
                        parsed = json.loads(qobj.model_dump_json())
                        generated.append(parsed)
                        out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        out_f.flush()
                        query_counts[page.filename] += 1
                        generated_direct_count += 1
                        pbar.update(1)
                        success = True
                        break
                    except Exception as e:
                        logger.exception(
                            "Failed to generate direct query for %s: %s",
                            page.filename,
                            e,
                        )
                        attempts += 1
                        if attempts >= MAX_ATTEMPTS:
                            logger.warning(
                                "Exceeded attempts for page %s (errors), skipping",
                                page.filename,
                            )
                            break
                        continue

            if not success:
                logger.warning(
                    "Failed to generate direct query for page %s after %d attempts",
                    page.filename,
                    MAX_ATTEMPTS,
                )

    return generated_direct_count


__all__ = ["generate_direct_queries"]
