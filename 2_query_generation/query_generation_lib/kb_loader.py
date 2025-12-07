import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .constants import DATA_FOLDER, STRUCTURE_FILE_NAME
from .models import PageMeta, Structure

logger = logging.getLogger(__name__)


def load_structure(kb_dir: str | Path) -> Structure:
    kb_dir = Path(kb_dir)
    structure_file = kb_dir / DATA_FOLDER / STRUCTURE_FILE_NAME
    if not structure_file.exists():
        raise FileNotFoundError(f"structure.json not found in {structure_file}")
    with open(structure_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Map to our PageMeta (we accept sub-selection of fields)
    pages: List[PageMeta] = []
    for p in data.get("pages", []):
        page = PageMeta(
            id=p.get("id"),
            title=p.get("title"),
            filename=p.get("filename"),
            category=p.get("category"),
            primary_topic=p.get("primary_topic"),
            secondary_topics=p.get("secondary_topics", []),
            links_to=p.get("links_to", []),
        )
        pages.append(page)
    structure = Structure(num_pages=data.get("num_pages", len(pages)), pages=pages)
    return structure


def load_page_content(kb_dir: str | Path, filename: str) -> str:
    kb_dir = Path(kb_dir)
    filepath = kb_dir / filename
    if not filepath.exists():
        logger.warning("Page file not found: %s", filepath)
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def find_linked_pairs(structure: Structure) -> List[Tuple[PageMeta, PageMeta]]:
    """Return pairs of pages where p1 links to p2 (i.e. multi-hop candidates).

    In our naive approach, we return (a,b) pairs where a.links_to contains b.filename.
    """
    pairs: List[Tuple[PageMeta, PageMeta]] = []
    pages_by_filename = {p.filename: p for p in structure.pages}
    for p in structure.pages:
        for link in p.links_to:
            target = pages_by_filename.get(link)
            if target:
                pairs.append((p, target))
    return pairs


def find_page_by_filename(structure: Structure, filename: str) -> Optional[PageMeta]:
    for p in structure.pages:
        if p.filename == filename:
            return p
    return None


def get_linked_page_contents(kb_dir: str | Path, page_meta: PageMeta) -> List[str]:
    """Return the content of pages listed in `page_meta.links_to` (direct links only)."""
    contents: List[str] = []
    for link in page_meta.links_to:
        contents.append(load_page_content(kb_dir, link))
    return contents


def stratified_sample_pages(
    structure: Structure, count: int, seed: int | None = None
) -> List[PageMeta]:
    """Stratified sample pages across categories. Returns up to `count` PageMeta objects.

    Strategy:
    - Group pages by category (or 'Uncategorized').
    - Compute allocation per category proportional to category size.
    - Sample unique pages from each category.
    - If fewer than `count` pages are selected because of rounding, add more from largest categories.
    """
    import random

    pages_by_cat = {}
    for p in structure.pages:
        cat = p.category or "Uncategorized"
        pages_by_cat.setdefault(cat, []).append(p)

    total_pages = len(structure.pages)
    if count >= total_pages:
        return list(structure.pages)

    rng = random.Random(seed)
    # initial allocation by proportion
    allocated = {}
    for cat, pages in pages_by_cat.items():
        alloc = int(round((len(pages) / total_pages) * count))
        allocated[cat] = min(alloc, len(pages))

    # fix total allocated to equal count
    current = sum(allocated.values())
    remaining = count - current
    # Sort categories by size descending for distributing remainder
    sorted_cats = sorted(pages_by_cat.items(), key=lambda kv: len(kv[1]), reverse=True)
    idx = 0
    while remaining > 0 and idx < len(sorted_cats):
        cat = sorted_cats[idx][0]
        if allocated[cat] < len(pages_by_cat[cat]):
            allocated[cat] += 1
            remaining -= 1
        idx = (idx + 1) % len(sorted_cats)

    # Now sample pages per category
    sampled: List[PageMeta] = []
    for cat, pages in pages_by_cat.items():
        k = allocated.get(cat, 0)
        if k <= 0:
            continue
        sampled.extend(rng.sample(pages, k))

    # If we've undershot due to rounding, sample more from categories with leftover pages
    if len(sampled) < count:
        leftover = [p for p in structure.pages if p not in sampled]
        extra_needed = count - len(sampled)
        sampled.extend(rng.sample(leftover, extra_needed))

    return sampled[:count]


def build_kb_topic_summary(structure: Structure) -> str:
    """Build a summary of all KB topics for adversarial negative query generation.

    Returns a string listing all categories and their page titles/primary topics.
    This helps the LLM generate questions that are semantically close but unanswerable.
    """
    summary_parts = []
    categories = {}
    for page in structure.pages:
        cat = page.category or "Uncategorized"
        if cat not in categories:
            categories[cat] = []
        primary = page.primary_topic or "No topic"
        secondary = (
            ", ".join(page.secondary_topics)
            if page.secondary_topics
            else "No secondary topics"
        )
        categories[cat].append(
            f"- {page.title} (Primary: {primary}; Secondary: {secondary})"
        )

    for cat, pages in categories.items():
        summary_parts.append(f"Category: {cat}")
        summary_parts.extend(pages)
        summary_parts.append("")  # blank line

    return "\n".join(summary_parts).strip()
