"""Prompt construction for knowledge base content generation.

Splits prompts into user-facing instructions that provide page-specific
requirements, while the system prompt (defined in agents.py) provides
general context and quality standards.
"""

from typing import List

from .models import Page


def build_prompt(
    page: Page, all_pages: List[Page] | None = None, v1_content: str | None = None
) -> str:
    """Build a user prompt for page-specific generation instructions.

    The system prompt (defined in agents.py) provides general guidance on content
    quality, structure, and formatting. This user prompt focuses on:
    - Page metadata (title, category, topic)
    - Page type and style requirements
    - Special structural requirements (tables, diagrams)
    - Versioning and semantic drift for adversarial content
    - Intentional mistakes for dataset enrichment
    - Cross-page linking requirements

    Args:
        page: Page object specifying fields and requirements.
        all_pages: Optional list of pages for link context.
        v1_content: If this is a v2 (current) version, provide v1 content to create contradicting data.

    Returns:
        User prompt with page-specific generation instructions.
    """
    parts = []

    # Page metadata section
    parts.append("## Page Specification")
    parts.append(f"**Title**: {page.title}")
    parts.append(f"**Category**: {page.category or 'general'}")
    parts.append(f"**Primary topic**: {page.primary_topic or 'unspecified'}")

    if page.secondary_topics:
        parts.append(f"**Secondary topics**: {', '.join(page.secondary_topics)}")

    # Content requirements section
    parts.append("\n## Content Requirements")
    parts.append(f"**Page type**: {page.type.value}")
    parts.append(f"**Tone/style**: {page.style or 'conversational_friendly'}")
    parts.append(f"**Length**: {page.length or 'medium'}")
    parts.append(
        "**Length guidance (approx)**: short=200-350 words, medium=350-700 words, long=700-1200 words"
    )

    # Hub page constraints (prevents multi-hop over-answering)
    if page.is_hub_page:
        parts.append("\n## Hub Page Constraints")
        parts.append(
            "This is an OVERVIEW/HUB page that introduces topics and points users to DETAIL PAGES."
        )
        parts.append(
            "**CRITICAL**: Do NOT include specific prices, dates, timeframes, SKUs, or exact policies."
        )
        parts.append(
            "Use placeholder language like: 'See [Related Page] for specific rates' or 'Check [Policy Page] for details'."
        )
        parts.append(
            "Provide navigation context and topic summary ONLY, not the answers."
        )

    # Tables with conditional logic
    if page.requires_table:
        parts.append("\n## Table Requirements")
        parts.append(
            "MUST include at least TWO markdown tables with realistic, complex data structures:"
        )
        parts.append("\n**Primary table** (minimum 5 columns × 6 rows):")
        parts.append(
            "- Include at least ONE conditional column where values depend on row conditions"
        )
        parts.append(
            "  - Example: 'Shipping Cost' depends on 'Weight' and 'Zone' (e.g., $5.99 base + $2/lb for weight >5lb)"
        )
        parts.append(
            "- Add footnotes or nested logic within cells (e.g., 'Excludes AK/HI')"
        )

        parts.append("\n**Secondary table** (minimum 4 columns × 5 rows):")
        parts.append("- Reference or discount table for cross-table lookups")
        parts.append("  - Example: Loyalty tiers that modify primary table values")

        parts.append("\n**Adversarial complexity**:")
        parts.append(
            "- 30% of data should require multi-row aggregation or conditional evaluation to answer queries"
        )
        parts.append(
            "- Add implicit cross-table dependencies with notes like 'See Discount Table for tier adjustments'"
        )
        parts.append(
            "- Use realistic thresholds that create edge cases (e.g., 'exactly 5lb' boundary conditions)"
        )

    # Mermaid diagrams
    if page.requires_mermaid:
        parts.append("\n## Diagram Requirements")
        parts.append("MUST include at least ONE Mermaid flowchart or decision diagram:")
        parts.append("- Use proper Mermaid syntax (flowchart TD or LR)")
        parts.append("- Minimum 8 nodes showing decision points and outcomes")
        parts.append("- Include conditional logic (if/then paths)")
        parts.append("- Make the diagram realistic for the process described")

    # Semantic drift for versioned content (data rot)
    if v1_content:
        parts.append("\n## Semantic Drift Instruction (v2 Versioning)")
        parts.append(
            "This is version 2 (CURRENT version). Below is the OUTDATED version 1 content:"
        )
        parts.append(f"\n```\n{v1_content}\n```")
        parts.append(
            "\nGenerate NEW v2 content that SUBTLY CONTRADICTS v1 via semantic drift:"
        )

        parts.append("\n**Structure preservation** (70% of content):")
        parts.append(
            "- Keep sentence structure and overall flow largely identical to v1"
        )
        parts.append(
            "- Reuse 70% of exact keywords and phrases (creates lexical traps)"
        )
        parts.append("- Do NOT add 'Updated [DATE]' markers or version labels")

        parts.append("\n**Subtle shifts** (30% of content):")
        parts.append(
            "- Conditional Threshold: Add constraints (e.g., 'now requires $75 minimum AND excludes AK/HI')"
        )
        parts.append(
            "- Scope Narrowing: Limit applicability (e.g., 'online orders only' not all orders)"
        )
        parts.append(
            "- Eligibility Tightening: Add requirements (e.g., 'original tags AND packaging' not just pristine)"
        )
        parts.append(
            "- Exception Addition: Add exclusions (e.g., 'free returns except clearance items')"
        )
        parts.append(
            "- Definition Shift: Reframe logic (e.g., 'calendar days vs. business days' or policy scope change)"
        )

        parts.append("\n**Adversarial goal**:")
        parts.append(
            "- If both v1 and v2 are retrieved together, an LLM should struggle to reconcile which rule applies"
        )
        parts.append(
            "- Avoid explicitly highlighting changes; let the semantic confusion arise naturally"
        )

    # Intentional mistakes for dataset hardening
    if page.mistake:
        parts.append("\n## Intentional Mistake Requirement")
        parts.append(
            f"Embed a {page.mistake.type.value.upper()} mistake of severity {page.mistake.severity.value.upper()}:"
        )
        parts.append("- Make it realistic and plausible within the domain")
        parts.append("  - Inconsistency: Contradictory statements within the page")
        parts.append("  - Omission: Important information left out")
        parts.append("  - Poor UX: Confusing or unclear instructions")
        parts.append(
            "  - Outdated info: Information that appears current but is subtly wrong"
        )
        parts.append(
            "  - Accessibility: Content that assumes specific context or knowledge"
        )
        parts.append("- Do NOT make it catastrophic unless severity is major")

    # Cross-page linking
    if page.links_to:
        parts.append("\n## Cross-Page Links")
        parts.append(
            f"Include natural in-text relative links to these pages: {', '.join(page.links_to)}"
        )
        parts.append(
            "Use markdown link syntax [text](page.md) where contextually relevant."
        )

    # Final formatting guidance
    parts.append("\n## Final Output Format")
    parts.append("- Generate valid, semantically meaningful markdown")
    parts.append(
        "- Use hierarchical headers, bullet points, code blocks where appropriate"
    )
    parts.append("- No placeholder text, '[FILL IN]' markers, or version metadata")
    parts.append("- Ensure content is realistic, readable, and properly structured")

    # Best-practices 'forbidden' checklist to avoid accidental wrapper text
    parts.append("\n## Forbidden (Do Not Include)")
    parts.append(
        "- Do NOT include YAML front matter, JSON wrappers, or explanatory text outside the markdown content"
    )
    parts.append(
        "- Do NOT include 'Note to raters' or labels like 'Version 2' or 'Updated' in the page body"
    )
    parts.append(
        "- Do NOT include top/bottom code fences that wrap the entire document"
    )

    # Minimal example to help the model follow a consistent structure
    parts.append("\n## Example Output Structure (minimal)")
    parts.append(
        "## <Title>\n\n### Summary\nShort summary of the page purpose and scope.\n\n### Details\nDetailed sections with headings and subheadings.\n\n### Example Table\n| ColumnA | ColumnB | ColumnC |\n|---|---|---|\n|val1|val2|val3|\n\n### Diagram (Mermaid)\n```mermaid\nflowchart TD\n    A[Start] --> B{Decision}?\n```\n\n### Related Links\n- [Other Page](other-page.md)"
    )

    return "\n".join(parts)


def build_placeholder_content(page: Page) -> str:
    """Create deterministic placeholder content for DRY_RUN testing.

    This avoids using the LLM and provides a readable content stub.
    """
    lines = [
        f"**DRY RUN**\n\nThis is a placeholder for **{page.title}** (topic: {page.primary_topic}).",
        "## Summary",
        "This placeholder simulates generated content for testing purposes.",
    ]
    if page.requires_table:
        lines.append("\n| Example | Info |\n|---|---|\n| Sample | Data |")
    if page.requires_mermaid:
        lines.append("\n```mermaid\nflowchart TD\n    A[Start] --> B[Done]\n```")
    if page.links_to:
        for line in page.links_to[:3]:
            lines.append(f"[Related]({line})")
    return "\n\n".join(lines)
