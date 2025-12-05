import json
import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .agents import (
    create_direct_agent,
    create_multi_hop_agent,
    create_negative_agent,
    create_openrouter_model,
)
from .constants import (
    QUERY_ID_PREFIXES,
    MAX_ATTEMPTS,
)
from .kb_loader import (
    build_kb_topic_summary,
    find_linked_pairs,
    load_page_content,
    load_structure,
)
from .models import (
    Query,
    QueryMetadata,
    QueryType,
)
from .prompts import (
    build_direct_prompt,
    build_multi_hop_prompt,
    build_negative_prompt,
)
from .validators import validate_query, validate_query_set

logger = logging.getLogger(__name__)


def _format_query_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


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
):
    logger.info("Starting query generation; DRY_RUN=%s, KB_DIR=%s", dry_run, kb_dir)

    structure = load_structure(kb_dir)

    if not dry_run:
        if not openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when not in dry-run mode"
            )
        or_model = create_openrouter_model(model, openrouter_api_key)
        # Create the agents bound to the configured OpenRouter model
        direct_agent = create_direct_agent(or_model)
        multi_hop_agent = create_multi_hop_agent(or_model)
        negative_agent = create_negative_agent(or_model)
    else:
        # In dry-run mode, no model is required; we can create no-op agents or
        # use None so that any accidental access raises a clearer error.
        direct_agent = None
        multi_hop_agent = None
        negative_agent = None

    # load existing queries file for resume support
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

    next_idx = {
        "direct": 1,
        "multi_hop": 1,
        "negative": 1,
    }
    # ensure we continue numbering after existing files
    for q in existing_queries:
        pref = q.get("query_type")
        if not pref:
            continue
        if pref == "direct":
            val = int(q["query_id"].split("_")[-1])
            next_idx["direct"] = max(next_idx["direct"], val + 1)
        elif pref == "multi_hop":
            val = int(q["query_id"].split("_")[-1])
            next_idx["multi_hop"] = max(next_idx["multi_hop"], val + 1)
        elif pref == "negative":
            val = int(q["query_id"].split("_")[-1])
            next_idx["negative"] = max(next_idx["negative"], val + 1)
        # carry forward existing objects
        generated.append(q)

    # Prepare output file for append (support resume/interruptions)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"
    out_f = open(output_file, write_mode, encoding="utf-8")

    # Retry policy is defined in constants.MAX_ATTEMPTS

    # --- Direct queries ---
    pages = list(structure.pages)
    for page in tqdm(pages, desc="Direct queries", total=min(num_direct, len(pages))):
        if len([q for q in generated if q["query_type"] == "direct"]) >= num_direct:
            break
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            idx = next_idx["direct"]
            query_id = _format_query_id(QUERY_ID_PREFIXES["direct"], idx)
            # skip if already exists (consume the id)
            if query_id in existing_ids:
                logger.info("Skipping existing query id %s", query_id)
                next_idx["direct"] += 1
                break
            content = load_page_content(kb_dir, page.filename)
            prompt = build_direct_prompt(content)
            if dry_run:
                # generate deterministic fallback
                qobj = {
                    "query_id": query_id,
                    "query_type": "direct",
                    "query": f"(DRY) What is the primary topic of {page.title}?",
                    "ground_truth": page.primary_topic or "Unknown",
                    "context_reference": [page.filename],
                    "metadata": {"difficulty": "easy", "category": page.category},
                }
                generated.append(qobj)
                out_f.write(json.dumps(qobj, ensure_ascii=False) + "\n")
                out_f.flush()
                next_idx["direct"] += 1
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
                            difficulty=qresp.difficulty,
                            category=qresp.category or page.category,
                        ),
                    )
                    if not validate_query(qobj):
                        logger.warning("Validation failed for %s", qobj.query_id)
                        attempts += 1
                        if attempts >= MAX_ATTEMPTS:
                            logger.warning("Exceeded attempts for page %s; skipping", page.filename)
                            break
                        continue
                    parsed = json.loads(qobj.model_dump_json())
                    generated.append(parsed)
                    out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    out_f.flush()
                    next_idx["direct"] += 1
                    break
                except Exception as e:
                    logger.exception(
                        "Failed to generate direct query for %s: %s", page.filename, e
                    )
                    attempts += 1
                    if attempts >= MAX_ATTEMPTS:
                        logger.warning("Exceeded attempts for page %s (errors), skipping", page.filename)
                        break
                    continue

    # --- Multi-hop queries ---
    pairs = find_linked_pairs(structure)
    # Basic dedupe: unique pair filenames
    pair_keys = {(a.filename, b.filename) for a, b in pairs}
    pair_list = [
        (find, to) for find, to in pairs if (find.filename, to.filename) in pair_keys
    ]
    for a, b in tqdm(
        pair_list, desc="Multi-hop queries", total=min(num_multi_hop, len(pair_list))
    ):
        if (
            len([q for q in generated if q["query_type"] == "multi_hop"])
            >= num_multi_hop
        ):
            break
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            idx = next_idx["multi_hop"]
            query_id = _format_query_id(QUERY_ID_PREFIXES["multi_hop"], idx)
            # skip existing (consume id)
            if query_id in existing_ids:
                next_idx["multi_hop"] += 1
                break
            content_a = load_page_content(kb_dir, a.filename)
            content_b = load_page_content(kb_dir, b.filename)
            prompt = build_multi_hop_prompt(content_a, content_b)
            if dry_run:
                qobj = {
                    "query_id": query_id,
                    "query_type": "multi_hop",
                    "query": f"(DRY) How can I combine info from {a.title} and {b.title} to find the order id?",
                    "ground_truth": "Use order tracking and account email to find order id.",
                    "context_reference": [a.filename, b.filename],
                    "metadata": {
                        "difficulty": "medium",
                        "category": a.category or b.category,
                    },
                }
                generated.append(qobj)
                out_f.write(json.dumps(qobj, ensure_ascii=False) + "\n")
                out_f.flush()
                next_idx["multi_hop"] += 1
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
                            difficulty=qresp.difficulty,
                            category=qresp.category or a.category or b.category,
                        ),
                    )
                    if not validate_query(qobj):
                        logger.warning("Validation failed for %s", qobj.query_id)
                        attempts += 1
                        if attempts >= MAX_ATTEMPTS:
                            logger.warning("Exceeded attempts for multi-hop pair %s/%s; skipping", a.filename, b.filename)
                            break
                        continue
                    parsed = json.loads(qobj.model_dump_json())
                    generated.append(parsed)
                    out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    out_f.flush()
                    next_idx["multi_hop"] += 1
                    break
                except Exception as e:
                    logger.exception(
                        "Failed to generate multi-hop query %s: %s", query_id, e
                    )
                    attempts += 1
                    if attempts >= MAX_ATTEMPTS:
                        logger.warning("Exceeded attempts for multi-hop pair %s/%s (errors), skipping", a.filename, b.filename)
                        break
                    continue

    # --- Negative queries ---
    # Build full KB topic summary for adversarial negative query generation
    kb_summary = build_kb_topic_summary(structure)
    # For negative queries, generate based on full KB summary (no specific page context)
    num_to_generate = num_negative - len(
        [q for q in generated if q["query_type"] == "negative"]
    )
    if num_to_generate > 0:
        attempts = 0
        remaining = num_to_generate
        while remaining > 0 and attempts < MAX_ATTEMPTS:
            prompt = build_negative_prompt(kb_summary, remaining)
            if dry_run:
                for _ in tqdm(range(remaining), desc="Negative queries"):
                    idx = next_idx["negative"]
                    query_id = _format_query_id(QUERY_ID_PREFIXES["negative"], idx)
                    next_idx["negative"] += 1
                    qobj = {
                        "query_id": query_id,
                        "query_type": "negative",
                        "query": "(DRY) Is there a 60-day warranty that extends to accidental damage?",
                        "ground_truth": "I don't know based on the KB.",
                        "context_reference": [],
                        "metadata": {"difficulty": "hard", "category": "general"},
                    }
                    generated.append(qobj)
                    out_f.write(json.dumps(qobj, ensure_ascii=False) + "\n")
                    out_f.flush()
                    remaining -= 1
                break
            else:
                try:
                    assert negative_agent is not None
                    resp = negative_agent.run_sync(prompt)
                    returned = 0
                    for qresp in tqdm(resp.output.queries, desc="Negative queries"):
                        idx = next_idx["negative"]
                        query_id = _format_query_id(QUERY_ID_PREFIXES["negative"], idx)
                        # build object
                        qobj = Query(
                            query_id=query_id,
                            query_type=QueryType.NEGATIVE,
                            query=qresp.query,
                            ground_truth=qresp.ground_truth,
                            context_reference=[],
                            metadata=QueryMetadata(
                                difficulty=qresp.difficulty,
                                category=qresp.category or "general",
                            ),
                        )
                        if not validate_query(qobj):
                            logger.warning("Validation failed for %s", qobj.query_id)
                            continue
                        parsed = json.loads(qobj.model_dump_json())
                        generated.append(parsed)
                        out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        out_f.flush()
                        next_idx["negative"] += 1
                        remaining -= 1
                        returned += 1
                        if remaining <= 0:
                            break
                    if remaining > 0:
                        attempts += 1
                        logger.info("Negative generator returned %d queries; remaining=%d; attempt=%d", returned, remaining, attempts)
                        if attempts >= MAX_ATTEMPTS:
                            logger.warning("Exceeded attempts while generating negative queries; remaining=%d", remaining)
                            break
                        continue
                    break
                except Exception as e:
                    attempts += 1
                    logger.exception("Failed to generate negative queries: %s", e)
                    if attempts >= MAX_ATTEMPTS:
                        break
                    continue

    # Close output file
    out_f.close()

    stats = validate_query_set([Query(**q) for q in generated if q is not None])
    logger.info("Generation stats: %s", stats)
    print("Generation stats: ", stats)
