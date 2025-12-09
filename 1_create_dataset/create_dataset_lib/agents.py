"""Agent factory for pydantic-ai based content generation.

Creates and manages the content generation agent that produces markdown pages
for the synthetic knowledge base using OpenRouter models.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .models import ContentResponse


def create_openrouter_model(model_name: str, api_key: str) -> OpenRouterModel:
    """Create and return an `OpenRouterModel` configured with the provided API key.

    This helper centralizes creation of the model/provider and mirrors the
    examples in pydantic_ai documentation. Agents should be created with
    this model instance via the factory functions below.

    Args:
        model_name: OpenRouter model identifier (e.g., 'x-ai/grok-4.1-fast:free')
        api_key: OpenRouter API key

    Returns:
        OpenRouterModel configured and ready for agent instantiation
    """
    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)


def create_content_agent(model: OpenRouterModel) -> Agent[None, ContentResponse]:
    """Create a content generation agent bound to the given model.

    This agent generates markdown content for knowledge base pages with the
    system prompt providing role context and generation guidelines, while
    user prompts contain page-specific instructions.

    Args:
        model: OpenRouterModel instance to use for generation

    Returns:
        Agent configured for markdown content generation with retry logic
    """
    system_prompt = """# Knowledge Base Content Generator

You are a professional content generator for retail customer support knowledge bases.
Your role is to create well-structured, comprehensive markdown documentation that is:

- **Clear and accessible**: Written for end users of varying technical proficiency
- **Accurate and specific**: Include realistic data, examples, and edge cases
- **Properly formatted**: Use markdown syntax with headers, lists, tables, and diagrams
- **Linked and contextual**: Reference related pages where appropriate
- **Adversarially complex**: Create content that requires careful reading and comparison

## Core Principles

1. **Structure**: Organize content with clear hierarchical headers, logical flow, and scannable sections
2. **Completeness**: Generate comprehensive pages that address multiple aspects of a topic
3. **Realism**: Use plausible, specific data (prices, dates, thresholds) drawn from realistic business contexts
4. **Cross-references**: Include semantic links to related pages when specified
5. **Format compliance**: Generate valid markdown that renders correctly in standard viewers

## Special Requirements

When instructed to include tables, create realistic, multi-dimensional tables with:
- Conditional logic (e.g., costs that depend on multiple factors)
- Cross-table dependencies (footnotes, tiered adjustments)
- 30% of data requiring aggregation or multi-row evaluation

When instructed to include mermaid diagrams, create valid flowcharts with:
- 8+ decision nodes showing realistic workflows
- Conditional logic paths
- Properly formatted mermaid syntax

When generating versioned content (v2 of outdated v1), introduce subtle semantic drift:
- Preserve 70% of structure and phrasing
- Add conditional constraints, scope narrowing, or definition shifts
- Reuse 70% of exact keywords to create lexical traps
- Avoid explicit version markers or change callouts

## Content Quality Standards

    - Markdown must be syntactically valid and semantically meaningful
    - No placeholder text or "[FILL IN]" markers
    - No version labels or metadata comments within content
    - Consistent tone and vocabulary throughout
    - Realistic but accessible complexity level

    IMPORTANT: Return only the raw markdown content as the agent's output. Do not
    include any explanation, commentary, surrounding JSON, or diagnostic text.
    The agent's output will be captured as `ContentResponse.content`.
"""
    # Important: require the model to return only the raw markdown content; no surrounding
    # comments, JSON, or explanation text. The agent's output will be mapped to the
    # ContentResponse.content field.

    agent = Agent(
        model,
        output_type=ContentResponse,
        system_prompt=system_prompt,
        retries=5,
    )
    return agent
