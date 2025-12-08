import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .agents import (
    create_anchored_negative_agent,
    create_direct_agent,
    create_multi_hop_agent,
    create_openrouter_model,
)
from .constants import (
    MAX_ATTEMPTS,
    QUERY_ID_PREFIXES,
)
from .constants import NEGATIVE_PROMPT_TOKEN_LIMIT as DEFAULT_NEG_TOKEN_LIMIT
from .kb_loader import (
    build_kb_topic_summary,
    find_linked_pairs,
    get_linked_page_contents,
    load_page_content,
    load_structure,
    stratified_sample_pages,
)
from .models import (
    DirectQuerySubtype,
    MultiHopQuerySubtype,
    NegativeQuerySubtype,
    Query,
    QueryMetadata,
    QueryType,
)
from .prompts import (
    build_anchored_negative_prompt,
    build_direct_prompt,
    build_multi_hop_prompt,
)
from .validators import validate_query, validate_query_set

logger = logging.getLogger(__name__)


def _format_query_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


def _select_direct_subtype() -> str:
    """Randomly select a direct query subtype with weighted distribution."""
    subtypes = [
        DirectQuerySubtype.SIMPLE_FACT,
        DirectQuerySubtype.TABLE_LOOKUP,
        DirectQuerySubtype.TABLE_AGGREGATION,
        DirectQuerySubtype.PROCESS_STEP,
        DirectQuerySubtype.CONDITIONAL_LOGIC,
        DirectQuerySubtype.LIST_ENUMERATION,
        DirectQuerySubtype.ROT_AWARE,
    ]
    # Weights: simple_fact(20), table_lookup(15), table_aggregation(15), process_step(12), conditional(12), list(12), rot_aware(14)
    weights = [20, 15, 15, 12, 12, 12, 14]
    return random.choices(subtypes, weights=weights, k=1)[0].value


def _select_multi_hop_subtype() -> str:
    """Randomly select a multi-hop query subtype with weighted distribution."""
    subtypes = [
        MultiHopQuerySubtype.SEQUENTIAL_PROCESS,
        MultiHopQuerySubtype.POLICY_FAQ_CROSS,
        MultiHopQuerySubtype.COMPARATIVE,
        MultiHopQuerySubtype.HUB_TO_DETAIL,
        MultiHopQuerySubtype.CROSS_CATEGORY,
        MultiHopQuerySubtype.ROT_AWARE,
    ]
    # Weights: sequential(15), policy_faq(15), comparative(15), hub_detail(10), cross_cat(10), rot(5)
    weights = [15, 15, 15, 10, 10, 5]
    return random.choices(subtypes, weights=weights, k=1)[0].value


def _select_negative_subtype() -> str:
    """Randomly select a negative query subtype with weighted distribution."""
    subtypes = [
        NegativeQuerySubtype.ADJACENT_TOPIC,
        NegativeQuerySubtype.MISSING_DATA,
        NegativeQuerySubtype.OUT_OF_SCOPE_PROCEDURE,
        NegativeQuerySubtype.CROSS_CATEGORY_GAP,
    ]
    # Weights: adjacent(12), missing(10), out_of_scope(8), cross_gap(10)
    weights = [12, 10, 8, 10]
    return random.choices(subtypes, weights=weights, k=1)[0].value


def _is_rot_page(page_filename: str) -> bool:
    """Check if page is a rot version (contains 'v1' or 'Outdated')."""
    return "v1" in page_filename or "Outdated" in page_filename


def run_query_generation(
    kb_dir: Path,
    output_file: Path,
    num_direct: int,
    num_multi_hop: int,
    num_negative: int,
    openrouter_api_key: str | None,
    model: str,
    overwrite: bool,
    dry_run: bool,
    negative_prompt_token_limit: int | None = None,
):
    logger.info("Starting query generation; DRY_RUN=%s, KB_DIR=%s", dry_run, kb_dir)

    structure = load_structure(kb_dir)

    if not dry_run:
        if not openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when not in dry-run mode"
            )
        or_model = create_openrouter_model(model, openrouter_api_key)
        direct_agent = create_direct_agent(or_model)
        multi_hop_agent = create_multi_hop_agent(or_model)
        anchored_negative_agent = create_anchored_negative_agent(or_model)
    else:
        direct_agent = None
        multi_hop_agent = None
        anchored_negative_agent = None

    # Load existing queries for resume support
    existing_queries: List[Dict] = []
    if output_file.exists() and not overwrite:
        logger.info("Found existing output file %s; loading to resume", output_file)
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    existing_queries.append(json.loads(line))
        except Exception:
            logger.exception(
                "Failed to load existing queries; continuing with empty list"
            )
            existing_queries = []
    existing_ids = {q["query_id"] for q in existing_queries}

    generated: List[Dict] = []

    next_idx = {"direct": 1, "multi_hop": 1, "negative": 1}
    for q in existing_queries:
        qtype = q.get("query_type")
        if not qtype:
            continue
        if qtype == "direct":
            val = int(q["query_id"].split("_")[-1])
            next_idx["direct"] = max(next_idx["direct"], val + 1)
        elif qtype == "multi_hop":
            val = int(q["query_id"].split("_")[-1])
            next_idx["multi_hop"] = max(next_idx["multi_hop"], val + 1)
        elif qtype == "negative":
            val = int(q["query_id"].split("_")[-1])
            next_idx["negative"] = max(next_idx["negative"], val + 1)
        generated.append(q)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"
    out_f = open(output_file, write_mode, encoding="utf-8")

    # --- DIRECT QUERIES ---
    # Filter to current pages only (exclude v1/Outdated)
    all_pages = list(structure.pages)
    current_pages = [p for p in all_pages if not _is_rot_page(p.filename)]
    logger.info(
        "Filtered pages for direct queries: %d current pages (excluded %d rot pages)",
        len(current_pages),
        len(all_pages) - len(current_pages),
    )

    query_counts = {p.filename: 0 for p in current_pages}
    for q in generated:
        if q["query_type"] == "direct" and q.get("context_reference"):
            filename = q["context_reference"][0]
            if filename in query_counts:
                query_counts[filename] += 1

    generated_direct_count = len([q for q in generated if q["query_type"] == "direct"])

    while generated_direct_count < num_direct and current_pages:
        max_count = max(query_counts.values()) if query_counts.values() else 0
        weights = [max_count + 1 - query_counts[p.filename] for p in current_pages]
        page = random.choices(current_pages, weights=weights, k=1)[0]

        attempts = 0
        success = False
        while attempts < MAX_ATTEMPTS:
            idx = next_idx["direct"]
            query_id = _format_query_id(QUERY_ID_PREFIXES["direct"], idx)

            if query_id in existing_ids:
                logger.info("Skipping existing query id %s", query_id)
                next_idx["direct"] += 1
                break

            subtype = _select_direct_subtype()
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
                next_idx["direct"] += 1
                query_counts[page.filename] += 1
                generated_direct_count += 1
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
                                "Exceeded attempts for page %s; skipping", page.filename
                            )
                            break
                        continue
                    parsed = json.loads(qobj.model_dump_json())
                    generated.append(parsed)
                    out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    out_f.flush()
                    next_idx["direct"] += 1
                    query_counts[page.filename] += 1
                    generated_direct_count += 1
                    success = True
                    break
                except Exception as e:
                    logger.exception(
                        "Failed to generate direct query for %s: %s", page.filename, e
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

    # --- MULTI-HOP QUERIES ---
    pairs = find_linked_pairs(structure)
    pair_keys = {(a.filename, b.filename) for a, b in pairs}
    pair_list = [(a, b) for a, b in pairs if (a.filename, b.filename) in pair_keys]

    generated_multi_hop_count = len(
        [q for q in generated if q["query_type"] == "multi_hop"]
    )

    for a, b in tqdm(
        pair_list, desc="Multi-hop queries", total=min(num_multi_hop, len(pair_list))
    ):
        if generated_multi_hop_count >= num_multi_hop:
            break

        attempts = 0
        while attempts < MAX_ATTEMPTS:
            idx = next_idx["multi_hop"]
            query_id = _format_query_id(QUERY_ID_PREFIXES["multi_hop"], idx)

            if query_id in existing_ids:
                next_idx["multi_hop"] += 1
                break

            subtype = _select_multi_hop_subtype()
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
                next_idx["multi_hop"] += 1
                generated_multi_hop_count += 1
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
                    next_idx["multi_hop"] += 1
                    generated_multi_hop_count += 1
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

    # --- NEGATIVE QUERIES ---
    kb_summary = build_kb_topic_summary(structure)
    num_to_generate = num_negative - len(
        [q for q in generated if q["query_type"] == "negative"]
    )

    if num_to_generate > 0:
        token_limit = negative_prompt_token_limit or DEFAULT_NEG_TOKEN_LIMIT
        remaining = num_to_generate
        anchors = stratified_sample_pages(structure, num_to_generate)

        for anchor in tqdm(anchors, desc="Negative queries"):
            if remaining <= 0:
                break

            attempts = 0
            while attempts < MAX_ATTEMPTS:
                idx = next_idx["negative"]
                query_id = _format_query_id(QUERY_ID_PREFIXES["negative"], idx)

                if query_id in existing_ids:
                    next_idx["negative"] += 1
                    break

                subtype = _select_negative_subtype()
                anchor_content = load_page_content(kb_dir, anchor.filename)
                linked_cts = get_linked_page_contents(kb_dir, anchor)
                linked_contents_joined = "\n\n---\n\n".join(linked_cts)
                anchor_meta = (
                    f"Title: {anchor.title}\nFilename: {anchor.filename}\nCategory: {anchor.category or 'Uncategorized'}\n"
                    f"Primary topic: {anchor.primary_topic or 'None'}\nSecondary topics: {', '.join(anchor.secondary_topics) if anchor.secondary_topics else 'None'}"
                )

                anchor_block = anchor_content
                if linked_contents_joined:
                    anchor_block = (
                        anchor_block + "\n\n" + linked_contents_joined
                    ).strip()
                if len(anchor_block) > token_limit:
                    anchor_block = anchor_block[:token_limit]

                prompt = build_anchored_negative_prompt(
                    anchor_content=anchor_block,
                    linked_contents="",
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
                    next_idx["negative"] += 1
                    remaining -= 1
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
                    next_idx["negative"] += 1
                    remaining -= 1
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

    out_f.close()

    stats = validate_query_set([Query(**q) for q in generated if q is not None])
    logger.info("Generation stats: %s", stats)
    print("Generation stats: ", stats)
