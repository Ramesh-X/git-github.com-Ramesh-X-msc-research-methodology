from typing import List

from .models import Page

BASE_PROMPT = (
    "You are a content generator for a retail customer support knowledge base. "
    "Generate a well-structured Markdown page that adheres to the requested page type, uses the requested tone and length, "
    "and includes semantic links to the pages listed."
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
    if page.secondary_topics:
        parts.append("Secondary topics: " + ", ".join(page.secondary_topics))
    parts.append(f"Page type: {page.type.value}")
    parts.append(f"Tone/style: {page.style or 'conversational_friendly'}")
    parts.append(f"Length: {page.length or 'medium'}")
    if page.requires_table:
        parts.append("Include at least one Markdown table relevant to the topic.")
    if page.requires_mermaid:
        parts.append("Include a Mermaid flowchart representing a relevant process.")

    # Handle versioned pages (data rot simulation)
    if v1_content:
        parts.append("\n--- IMPORTANT: VERSION CONFLICT INSTRUCTION ---")
        parts.append(
            "This is version 2 (current version) of a page. Below is the outdated version 1 content:"
        )
        parts.append(f"\n```\n{v1_content}\n```\n")
        parts.append(
            "Generate NEW content for version 2 that CONTRADICTS the version 1 content in meaningful ways:"
        )
        parts.append("- Change numerical values (prices, timeframes, quantities)")
        parts.append("- Modify policies or procedures")
        parts.append("- Update requirements or eligibility criteria")
        parts.append("- Alter contact information or availability")
        parts.append(
            "Make the contradictions realistic and significant enough to cause confusion if both versions are retrieved."
        )

    if page.mistake:
        parts.append(
            f"Intentionally add a {page.mistake.type.value} mistake of severity {page.mistake.severity.value}. "
            "Make it realistic but not catastrophic (unless severity is major)."
        )
    if page.links_to:
        parts.append("Add in-text relative links to: " + ", ".join(page.links_to))
    parts.append(
        "Write the markdown with headers, links, bullets, and short code blocks where relevant. Keep it realistic and readable."
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
