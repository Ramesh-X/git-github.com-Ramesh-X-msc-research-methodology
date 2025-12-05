import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .models import PageMeta, Structure

logger = logging.getLogger(__name__)


def load_structure(kb_dir: str | Path) -> Structure:
    kb_dir = Path(kb_dir)
    structure_file = kb_dir / "structure.json"
    if not structure_file.exists():
        raise FileNotFoundError(f"structure.json not found in {kb_dir}")
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
