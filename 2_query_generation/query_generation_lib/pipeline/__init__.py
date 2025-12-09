import json
import logging
from pathlib import Path
from typing import Dict, List

from ..agents import (
    create_anchored_negative_agent,
    create_direct_agent,
    create_multi_hop_agent,
    create_openrouter_model,
)
from ..constants import QUERY_ID_PREFIXES
from ..kb_loader import load_structure
from ..models import Query
from ..validators import validate_query_set
from .direct import generate_direct_queries
from .helpers import QueryIDAllocator
from .multi_hop import generate_multi_hop_queries
from .negative import generate_negative_queries

logger = logging.getLogger(__name__)


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
    generated: List[Dict] = []
    existing_ids = set()
    if output_file.exists() and not overwrite:
        logger.info("Found existing output file %s; loading to resume", output_file)
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    generated.append(q)
                    existing_ids.add(q["query_id"])
        except Exception:
            logger.exception(
                "Failed to load existing queries; continuing with empty list"
            )
            generated = []
            existing_ids = set()

    # Initialize ID allocators for each query type
    id_allocators = {
        "direct": QueryIDAllocator(
            QUERY_ID_PREFIXES["direct"], num_direct, existing_ids
        ),
        "multi_hop": QueryIDAllocator(
            QUERY_ID_PREFIXES["multi_hop"], num_multi_hop, existing_ids
        ),
        "negative": QueryIDAllocator(
            QUERY_ID_PREFIXES["negative"], num_negative, existing_ids
        ),
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"
    out_f = open(output_file, write_mode, encoding="utf-8")

    # --- DIRECT QUERIES ---
    generate_direct_queries(
        kb_dir=kb_dir,
        structure=structure,
        out_f=out_f,
        dry_run=dry_run,
        num_direct=num_direct,
        direct_agent=direct_agent,
        id_allocator=id_allocators["direct"],
        existing_ids=existing_ids,
        generated=generated,
    )

    # --- MULTI-HOP QUERIES ---
    generate_multi_hop_queries(
        kb_dir=kb_dir,
        structure=structure,
        out_f=out_f,
        dry_run=dry_run,
        num_multi_hop=num_multi_hop,
        multi_hop_agent=multi_hop_agent,
        id_allocator=id_allocators["multi_hop"],
        existing_ids=existing_ids,
        generated=generated,
    )

    # --- NEGATIVE QUERIES ---
    generate_negative_queries(
        kb_dir=kb_dir,
        structure=structure,
        out_f=out_f,
        dry_run=dry_run,
        num_negative=num_negative,
        anchored_negative_agent=anchored_negative_agent,
        id_allocator=id_allocators["negative"],
        existing_ids=existing_ids,
        generated=generated,
        negative_prompt_token_limit=negative_prompt_token_limit,
    )

    out_f.close()

    stats = validate_query_set([Query(**q) for q in generated if q is not None])
    logger.info("Generation stats: %s", stats)
