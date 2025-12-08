import logging
import os
import random
from typing import List

from slugify import slugify

from .constants import (
    DATA_FOLDER,
    MISTAKE_INJECTION_RATE,
    PAGE_TYPE_DISTRIBUTION,
    ROT_RATE,
    STRUCTURE_FILE_NAME,
    TOPIC_DISTRIBUTION,
)
from .models import (
    Mistake,
    MistakeType,
    Page,
    PageType,
    RotPair,
    SemanticDriftType,
    Severity,
    Structure,
)

logger = logging.getLogger(__name__)


def _choose_topics(n: int) -> List[str]:
    keys = list(TOPIC_DISTRIBUTION.keys())
    weights = list(TOPIC_DISTRIBUTION.values())
    choices = random.choices(keys, weights=weights, k=n)
    return choices


def _generate_descriptive_title(primary_topic: str, is_rot: bool = False) -> str:
    """Generate descriptive, purpose-driven page titles based on topic."""
    topic_clean = primary_topic.replace("_", " ").title()

    # Map topics to common page types/patterns
    titles = {
        "orders": [
            "Order Management Guide",
            "Order Tracking and Status",
            "Order Processing Steps",
            "Order Modification Policy",
        ],
        "returns_refunds": [
            "Returns and Refunds Policy",
            "How to Return Items",
            "Refund Processing Timeline",
            "Return Eligibility Criteria",
        ],
        "shipping_delivery": [
            "Shipping Options and Rates",
            "Delivery Timeline Information",
            "International Shipping Guide",
            "Shipping Zone Rates",
        ],
        "contact": [
            "Customer Support Contacts",
            "Department Contact Directory",
            "Support Hours and Availability",
            "How to Reach Support",
        ],
        "faq": [
            f"{topic_clean} Frequently Asked Questions",
            f"Common {topic_clean} Questions",
            f"{topic_clean} Q&A Guide",
        ],
        "account": [
            "Account Management Guide",
            "Account Registration Process",
            "Password and Security",
            "Profile Settings",
        ],
        "payments_billing": [
            "Payment Methods and Billing",
            "Billing Cycle Explanation",
            "Accepted Payment Options",
            "Invoice and Payment History",
        ],
        "membership_loyalty": [
            "Loyalty Program Guide",
            "Membership Tiers and Benefits",
            "Points Redemption",
            "Loyalty Account Management",
        ],
        "product_info": [
            "Product Specifications and Details",
            "Product Availability",
            "Product Comparison Guide",
        ],
        "warranty": [
            "Warranty Coverage Information",
            "Warranty Claims Process",
            "Extended Warranty Options",
        ],
        "store_services": [
            "In-Store Services Guide",
            "Service Availability",
            "How to Schedule Services",
        ],
        "accessibility": [
            "Accessibility Features",
            "Accessibility Support",
            "Screen Reader Compatibility",
        ],
        "installation": [
            "Installation Guide",
            "Professional Installation",
            "DIY Setup Instructions",
        ],
        "sustainability": [
            "Sustainability Initiatives",
            "Environmental Responsibility",
            "Green Shipping Options",
        ],
        "recycling": [
            "Recycling Program",
            "Product Recycling Guide",
            "Disposal and Recycling",
        ],
    }

    title = random.choice(titles.get(primary_topic, [f"{topic_clean} Information"]))
    if is_rot:
        title += " (Current)"
    return title


def generate_structure(num_pages: int = 100, out_dir: str = "output/kb") -> Structure:
    pages: List[Page] = []
    os.makedirs(out_dir, exist_ok=True)

    # Calculate rot pairs upfront to adjust initial page count
    # (rot pairs add v1 pages that count toward total)
    num_rot_pairs = int((num_pages * ROT_RATE) / 2)  # 5 pairs for 100 pages
    base_pages_to_generate = num_pages - num_rot_pairs  # 95 base pages for 100 total

    # Build page types according to distribution
    page_types = []
    for ptype, count in PAGE_TYPE_DISTRIBUTION.items():
        page_types.extend([ptype] * count)
    # If counts don't sum to base_pages_to_generate, fill with 'unstructured'
    if len(page_types) < base_pages_to_generate:
        page_types.extend(["unstructured"] * (base_pages_to_generate - len(page_types)))
    random.shuffle(page_types)

    topics = _choose_topics(base_pages_to_generate)

    used_filenames = set()
    used_ids = set()
    for i in range(base_pages_to_generate):
        t = page_types[i]
        primary = topics[i]
        title = _generate_descriptive_title(primary)
        base_slug = slugify(title)
        filename = f"{base_slug}.md"
        suffix_counter = 1
        while filename in used_filenames:
            filename = f"{base_slug}-{suffix_counter}.md"
            suffix_counter += 1
        used_filenames.add(filename)
        requires_table = t == "tabular"
        requires_mermaid = t == "logical"
        mistake = None
        if random.random() < MISTAKE_INJECTION_RATE:
            mistake_type = random.choice(list(MistakeType))
            severity = random.choices(list(Severity), weights=[58, 33, 9], k=1)[0]
            mistake = Mistake(type=mistake_type, severity=severity)

        # TRANSITIVE MULTI-HOP: Designate 20% of pages as "hub" pages (no specific data)
        is_hub = random.random() < 0.20
        is_detail = (
            not is_hub and random.random() < 0.25
        )  # 20% of remaining are detail pages

        # Ensure unique ids as well
        id_slug = base_slug
        suffix_counter = 1
        while id_slug in used_ids:
            id_slug = f"{base_slug}-{suffix_counter}"
            suffix_counter += 1
        p = Page(
            id=id_slug,
            title=title,
            filename=filename,
            category=primary,  # Use primary topic as category
            type=PageType(t),
            primary_topic=primary,
            secondary_topics=random.sample(
                list(TOPIC_DISTRIBUTION.keys()), k=min(2, len(TOPIC_DISTRIBUTION))
            ),
            style=random.choice(
                ["conversational_friendly", "corporate_formal", "technical_detailed"]
            ),
            length=random.choice(["brief", "medium", "comprehensive"]),
            mistake=mistake,
            links_to=[],
            is_hub_page=is_hub,
            is_detail_page=is_detail,
            requires_table=requires_table,
            requires_mermaid=requires_mermaid,
        )
        pages.append(p)
        used_ids.add(id_slug)

    # Create rot pairs: 5 pairs = 10 pages total
    # Each pair consists of v1 (outdated) and v2 (current) versions with SEMANTIC DRIFT
    rot_pairs = []

    # Define semantic drift distribution: focus on SUBTLE, CONFUSING conflicts
    semantic_drift_distribution = [
        (
            SemanticDriftType.CONDITIONAL_THRESHOLD,
            "Free shipping now requires $75 minimum AND excludes Alaska/Hawaii (was: $50 all locations)",
            "moderate",
        ),
        (
            SemanticDriftType.SCOPE_NARROWING,
            "30-day returns apply to online orders only; in-store purchases have 14-day window",
            "subtle",
        ),
        (
            SemanticDriftType.ELIGIBILITY_TIGHTENING,
            "Returns require pristine condition, original tags, and original packaging (was: pristine condition only)",
            "moderate",
        ),
        (
            SemanticDriftType.EXCEPTION_ADDITION,
            "Free returns on all items except clearance/final-sale merchandise (was: all returns free)",
            "subtle",
        ),
        (
            SemanticDriftType.DEFINITION_SHIFT,
            "Refund processing time now measured in calendar days (not business days); expedited refunds added",
            "moderate",
        ),
    ]

    indices = list(range(num_pages))
    random.shuffle(indices)

    for pair_idx in range(num_rot_pairs):
        # Select a page to create versioned rot for
        base_idx = indices.pop()
        base_page = pages[base_idx]

        # Create v1 (outdated) version
        v1_title = f"{base_page.title} (Outdated)"
        v1_base_slug = slugify(v1_title)
        v1_filename = f"{v1_base_slug}.md"
        suffix_counter = 1
        while v1_filename in used_filenames:
            v1_filename = f"{v1_base_slug}-{suffix_counter}.md"
            suffix_counter += 1
        used_filenames.add(v1_filename)
        # Ensure unique id for v1 pages
        v1_id_slug = v1_base_slug
        suffix_counter = 1
        while v1_id_slug in used_ids:
            v1_id_slug = f"{v1_base_slug}-{suffix_counter}"
            suffix_counter += 1
        v1_page = Page(
            id=v1_id_slug,
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
        used_ids.add(v1_id_slug)

        # Update base page to be v2 (current) version
        v2_title = f"{base_page.title} (Current)"
        base_page.title = v2_title
        v2_base_slug = slugify(v2_title)
        # allocate unique filename and id for v2 (base_page updated)
        v2_filename = f"{v2_base_slug}.md"
        suffix_counter = 1
        while v2_filename in used_filenames:
            v2_filename = f"{v2_base_slug}-{suffix_counter}.md"
            suffix_counter += 1
        used_filenames.add(v2_filename)
        base_page.filename = v2_filename
        v2_id_slug = v2_base_slug
        suffix_counter = 1
        while v2_id_slug in used_ids:
            v2_id_slug = f"{v2_base_slug}-{suffix_counter}"
            suffix_counter += 1
        used_ids.add(v2_id_slug)
        base_page.id = v2_id_slug
        used_ids.add(v2_id_slug)

        # Add cross-links between versions
        v1_page.links_to.append(base_page.filename)
        base_page.links_to.append(v1_filename)

        # Select semantic drift type for this rot pair (adversarial)
        drift_type, conflict_desc, confusion_level = semantic_drift_distribution[
            pair_idx % len(semantic_drift_distribution)
        ]

        # Record the rot pair with SEMANTIC DRIFT metadata
        rot_pairs.append(
            RotPair(
                v1=v1_page.id,
                v2=base_page.id,
                semantic_drift_type=drift_type,
                conflict_description=conflict_desc,
                lexical_overlap=0.70,  # 70% of text should be identical
                semantic_confusion_level=confusion_level,
            )
        )

        # Insert v1 page into pages list
        pages.append(v1_page)

    # TRANSITIVE MULTI-HOP ENFORCEMENT: Create hub-to-detail relationships
    # Hub pages link to detail pages but contain NO specific data themselves
    hub_pages = [p for p in pages if p.is_hub_page]
    detail_pages = [p for p in pages if p.is_detail_page]

    if hub_pages and detail_pages:
        for hub in hub_pages[:10]:  # Link up to 10 hubs
            # Each hub links to 2-3 detail pages
            targets = random.sample(detail_pages, k=min(3, len(detail_pages)))
            for target in targets:
                if target.filename not in hub.links_to:
                    hub.links_to.append(target.filename)

    # CIRCULAR REFERENCE TRAPS: Create small circular links (A -> B -> A) for definitions
    # This tests if RAG systems get stuck in loops or properly resolve circular references
    circular_pairs = random.sample(pages, k=min(4, len(pages)))
    for i in range(0, len(circular_pairs) - 1, 2):
        page_a = circular_pairs[i]
        page_b = circular_pairs[i + 1]
        # Create circular link: A -> B -> A (for specific definition sections)
        if page_b.filename not in page_a.links_to:
            page_a.links_to.append(page_b.filename)
        if page_a.filename not in page_b.links_to:
            page_b.links_to.append(page_a.filename)

    # Add a few additional cross-links
    for i in range(0, num_pages, 10):
        src = pages[i]
        targets = random.sample(pages, k=3)
        for t in targets:
            if t.filename not in src.links_to and t.filename != src.filename:
                src.links_to.append(t.filename)

    # HIDDEN ENTITY DEPENDENCIES: Create entity anchors that appear across non-linked pages
    entity_anchors = []
    entity_names = [
        "Platinum Membership",
        "Zone B Shipping",
        "Defective Item",
        "Clearance Sale",
        "Express Shipping",
    ]

    for entity_name in entity_names:
        # Select 3-4 random non-linked pages to reference this entity
        anchor_pages = random.sample(pages, k=min(4, len(pages)))
        anchor_pages_ids = [p.id for p in anchor_pages]
        entity_anchors.append(
            {
                "entity_name": entity_name,
                "appearing_in_pages": anchor_pages_ids,
                "is_explicitly_linked": False,  # No hyperlinks between these pages
            }
        )

    structure = Structure(
        num_pages=num_pages,
        page_types=PAGE_TYPE_DISTRIBUTION,
        rot_pairs=rot_pairs,
        pages=pages,
        entity_anchors=entity_anchors,
    )
    data_dir = os.path.join(out_dir, DATA_FOLDER)
    os.makedirs(data_dir, exist_ok=True)
    structure_path = os.path.join(data_dir, STRUCTURE_FILE_NAME)
    with open(structure_path, "w", encoding="utf-8") as f:
        # Pydantic v2: use model_dump_json() to serialize with indent
        f.write(structure.model_dump_json(indent=2))
        logger.info("Wrote structure.json to %s", structure_path)
    return structure
