import logging
import os
import random
from typing import List

from slugify import slugify

from .constants import (
    MISTAKE_INJECTION_RATE,
    PAGE_TYPE_DISTRIBUTION,
    ROT_RATE,
    TOPIC_DISTRIBUTION,
)
from .models import Mistake, MistakeType, Page, PageType, RotPair, Severity, Structure

logger = logging.getLogger(__name__)


def _choose_topics(n: int) -> List[str]:
    keys = list(TOPIC_DISTRIBUTION.keys())
    weights = list(TOPIC_DISTRIBUTION.values())
    choices = random.choices(keys, weights=weights, k=n)
    return choices


def generate_structure(num_pages: int = 100, out_dir: str = "output/kb") -> Structure:
    pages: List[Page] = []
    os.makedirs(out_dir, exist_ok=True)
    # Build page types according to distribution
    page_types = []
    for ptype, count in PAGE_TYPE_DISTRIBUTION.items():
        page_types.extend([ptype] * count)
    # If counts don't sum to num_pages, fill with 'unstructured'
    if len(page_types) < num_pages:
        page_types.extend(["unstructured"] * (num_pages - len(page_types)))
    random.shuffle(page_types)

    topics = _choose_topics(num_pages)

    for i in range(num_pages):
        t = page_types[i]
        primary = topics[i]
        title = f"{primary.replace('_', ' ').title()} Page {i + 1}"
        filename = slugify(title) + ".md"
        requires_table = t == "tabular"
        requires_mermaid = t == "logical"
        mistake = None
        if random.random() < MISTAKE_INJECTION_RATE:
            mistake_type = random.choice(list(MistakeType))
            severity = random.choices(list(Severity), weights=[58, 33, 9], k=1)[0]
            mistake = Mistake(type=mistake_type, severity=severity)
        p = Page(
            id=slugify(title),
            title=title,
            filename=filename,
            category=random.choice(
                ["general_retail", "fashion", "electronics", "grocery", "home_goods"]
            ),
            type=PageType(t),
            primary_topic=primary,
            secondary_topics=random.sample(list(TOPIC_DISTRIBUTION.keys()), k=2),
            style=random.choice(
                ["conversational_friendly", "corporate_formal", "technical_detailed"]
            ),
            length=random.choice(["brief", "medium", "comprehensive"]),
            mistake=mistake,
            links_to=[],
            requires_table=requires_table,
            requires_mermaid=requires_mermaid,
        )
        pages.append(p)

    # Create rot pairs: 1 pairs = 2 pages
    # Each pair consists of v1 (outdated) and v2 (current) versions
    num_rot_pairs = int(
        (num_pages * ROT_RATE) / 2
    )  # Divide by 2 since each pair = 2 pages
    rot_pairs = []
    indices = list(range(num_pages))
    random.shuffle(indices)

    for pair_idx in range(num_rot_pairs):
        # Select a page to create versioned rot for
        base_idx = indices.pop()
        base_page = pages[base_idx]

        # Create v1 (outdated) version
        v1_title = f"{base_page.title} v1"
        v1_filename = slugify(v1_title) + ".md"
        v1_page = Page(
            id=slugify(v1_title),
            title=v1_title,
            filename=v1_filename,
            category=base_page.category,
            type=base_page.type,
            primary_topic=base_page.primary_topic,
            secondary_topics=base_page.secondary_topics,
            style=base_page.style,
            length=base_page.length,
            mistake=base_page.mistake,
            links_to=[],
            requires_table=base_page.requires_table,
            requires_mermaid=base_page.requires_mermaid,
        )

        # Update base page to be v2 (current) version
        base_page.title = f"{base_page.title} v2"
        base_page.filename = slugify(base_page.title) + ".md"
        base_page.id = slugify(base_page.title)

        # Add cross-links between versions
        v1_page.links_to.append(base_page.filename)
        base_page.links_to.append(v1_filename)

        # Record the rot pair with conflict description
        conflict = f"Versioned conflict: {base_page.primary_topic} policy/data changed"
        rot_pairs.append(RotPair(v1=v1_page.id, v2=base_page.id, conflict=conflict))

        # Insert v1 page into pages list
        pages.append(v1_page)

    # Add a few additional cross-links
    for i in range(0, num_pages, 10):
        src = pages[i]
        targets = random.sample(pages, k=3)
        for t in targets:
            if t.filename not in src.links_to and t.filename != src.filename:
                src.links_to.append(t.filename)

    structure = Structure(
        num_pages=num_pages,
        page_types=PAGE_TYPE_DISTRIBUTION,
        rot_pairs=rot_pairs,
        pages=pages,
    )
    with open(os.path.join(out_dir, "structure.json"), "w", encoding="utf-8") as f:
        # Pydantic v2: use model_dump_json() to serialize with indent
        f.write(structure.model_dump_json(indent=2))
        logger.info(
            "Wrote structure.json to %s", os.path.join(out_dir, "structure.json")
        )
    return structure
