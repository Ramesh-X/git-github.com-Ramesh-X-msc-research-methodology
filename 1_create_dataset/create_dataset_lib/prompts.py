from typing import List

from .models import Page

BASE_PROMPT = (
    "You are a content generator for a retail customer support knowledge base. "
    "Generate a well-structured Markdown page that adheres to the requested page type, uses the requested tone and length, "
    "and includes semantic links to the pages listed. Use realistic, specific data and examples."
)


def build_prompt(
    page: Page, all_pages: List[Page] | None = None, v1_content: str | None = None
) -> str:
    """Build a prompt for the LLM to generate the requested page.

    Args:
        page: Page object specifying fields.
        all_pages: optional list of pages to link to.
        v1_content: If this is a v2 (current) version, provide v1 content to create contradicting data.
    """
    parts = [BASE_PROMPT]
    parts.append(f"Title: {page.title}")
    parts.append(f"Category: {page.category or 'general'}")
    parts.append(f"Primary topic: {page.primary_topic or ''}")

    # TRANSITIVE MULTI-HOP: Hub pages must be overviews, NOT contain specific data
    if page.is_hub_page:
        parts.append(
            "\n--- HUB PAGE INSTRUCTION (TRANSITIVE MULTI-HOP ENFORCEMENT) ---"
        )
        parts.append(
            "This is an OVERVIEW/HUB page that introduces topics and points users to DETAIL PAGES for specific information."
        )
        parts.append("CRITICAL CONSTRAINTS:")
        parts.append(
            "1. Do NOT include specific prices, dates, timeframes, SKUs, or exact policies."
        )
        parts.append(
            "2. Use placeholder language: 'See [Related Page] for specific rates', 'Check [Policy Page] for details'"
        )
        parts.append(
            "3. Provide navigation context and topic summary ONLY, not the answers."
        )
        parts.append(
            "4. Multi-hop queries MUST require reading both this hub and its linked detail pages."
        )
    if page.secondary_topics:
        parts.append("Secondary topics: " + ", ".join(page.secondary_topics))
    parts.append(f"Page type: {page.type.value}")
    parts.append(f"Tone/style: {page.style or 'conversational_friendly'}")
    parts.append(f"Length: {page.length or 'medium'}")

    # MANDATORY table requirement with ADVERSARIAL COMPLEXITY
    if page.requires_table:
        parts.append("\n--- MANDATORY: INCLUDE TABLES WITH CONDITIONAL LOGIC ---")
        parts.append(
            "You MUST include at least TWO markdown tables with REALISTIC and COMPLEX data structures:"
        )
        parts.append("1. PRIMARY TABLE (minimum 5 columns × 6 rows):")
        parts.append(
            "   - Include at least ONE conditional column where values depend on row conditions"
        )
        parts.append(
            "   - Example: 'Shipping Cost' depends on 'Weight' and 'Zone' (e.g., $5.99 base + $2/lb for weight >5lb)"
        )
        parts.append(
            "   - Add footnotes or nested logic within cells (e.g., 'Excludes AK/HI')"
        )
        parts.append(
            "2. SECONDARY TABLE (minimum 4 columns × 5 rows): Reference or discount table for cross-table lookups"
        )
        parts.append(
            "   - Example: Loyalty discounts that apply to primary table values"
        )
        parts.append("\nCONDITIONAL LOGIC EXAMPLES:")
        parts.append(
            "| Method | Base Rate | Weight Surcharge | Zone A | Zone B | Notes |"
        )
        parts.append(
            "|--------|-----------|-----------------|--------|--------|-------|"
        )
        parts.append(
            "| Standard | $5.99 | +$2/lb (>5lb) | -$0 | +$3 | Excludes AK/HI |"
        )
        parts.append("| Express | $12.99 | +$3/lb (>3lb) | +$2 | +$4 | Expedited |")
        parts.append("\nADVARSARIAL REQUIREMENT:")
        parts.append(
            "- 30% of table data should require MULTI-ROW AGGREGATION or CONDITIONAL EVALUATION to answer queries"
        )
        parts.append(
            "- Add implicit cross-table dependencies: 'See Discount Table for tier adjustments'"
        )
        parts.append(
            "- Use realistic thresholds that create edge cases (e.g., 'exactly 5lb')"
        )

    # MANDATORY Mermaid requirement
    if page.requires_mermaid:
        parts.append("\n--- MANDATORY: INCLUDE MERMAID DIAGRAM ---")
        parts.append(
            "You MUST include at least ONE Mermaid flowchart or decision diagram:"
        )
        parts.append("- Use proper Mermaid syntax (flowchart TD or LR)")
        parts.append("- Minimum 8 nodes showing decision points and outcomes")
        parts.append("- Include conditional logic (if/then paths)")
        parts.append("\nExample Mermaid format:")
        parts.append("```mermaid")
        parts.append("flowchart TD")
        parts.append("    A[Start Process] --> B{Check Condition?}")
        parts.append("    B -->|Yes| C[Follow Path A]")
        parts.append("    B -->|No| D[Follow Path B]")
        parts.append("    C --> E[Result]")
        parts.append("```")
        parts.append(
            "\nMake the diagram relevant to the topic and realistic for the process described."
        )

    # Handle versioned pages (data rot simulation with SEMANTIC DRIFT)
    if v1_content:
        parts.append("\n--- ADVERSARIAL: VERSION SEMANTIC DRIFT INSTRUCTION ---")
        parts.append(
            "This is version 2 (CURRENT version) of a page. Below is the OUTDATED version 1 content:"
        )
        parts.append(f"\n```\n{v1_content}\n```\n")
        parts.append(
            "Generate NEW content for version 2 that SUBTLY CONTRADICTS version 1 via SEMANTIC DRIFT:"
        )
        parts.append(
            "1. PRESERVE STRUCTURE: Keep 70% of sentences and overall flow identical to v1. Do NOT add 'Updated [DATE]' markers."
        )
        parts.append(
            "2. SUBTLE SHIFTS (30% of content): Introduce CONDITIONAL CONSTRAINTS, SCOPE NARROWING, or DEFINITION CHANGES:"
        )
        parts.append(
            "   - Conditional Threshold: Add constraints (e.g., 'now requires $75 minimum AND excludes AK/HI')"
        )
        parts.append(
            "   - Scope Narrowing: Limit applicability (e.g., 'applies to online orders only' not all orders)"
        )
        parts.append(
            "   - Eligibility Tightening: Add requirements (e.g., 'requires original tags AND packaging' not just pristine)"
        )
        parts.append(
            "   - Exception Addition: Add exceptions (e.g., 'free returns except clearance items')"
        )
        parts.append(
            "   - Definition Shift: Reframe logic (e.g., 'calendar days vs. business days' or change policy scope)"
        )
        parts.append(
            "3. LEXICAL TRAP: Reuse 70% of exact keywords/phrases from v1 so the changes are not immediately obvious."
        )
        parts.append(
            "4. AVOID: Do NOT highlight changes, do NOT add version labels, do NOT explicitly state what changed."
        )
        parts.append(
            "5. GOAL: Create plausible-but-confusing content. If both v1 and v2 are retrieved, an LLM should struggle"
        )
        parts.append("   to reconcile which rule applies and to which subset of cases.")

    if page.mistake:
        parts.append(
            f"Intentionally add a {page.mistake.type.value} mistake of severity {page.mistake.severity.value}. "
            "Make it realistic but not catastrophic (unless severity is major)."
        )
    if page.links_to:
        parts.append("Add in-text relative links to: " + ", ".join(page.links_to))
    parts.append(
        "Write the markdown with headers, bold text, bullet points, and code blocks where relevant. Keep it realistic, readable, and comprehensive."
    )
    return "\n\n".join(parts)


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
