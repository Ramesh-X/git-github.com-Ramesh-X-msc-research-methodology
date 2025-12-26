from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .models import E1Response, E2Response, E3Response, E4Response


def create_openrouter_model(model_name: str, api_key: str) -> OpenRouterModel:
    """Create and return an `OpenRouterModel` configured with the provided API key.

    This helper centralizes creation of the model/provider and mirrors the
    examples in pydantic_ai documentation. Agents should be created with
    this model instance via the factory functions below.
    """
    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)


def create_e1_agent(model: OpenRouterModel) -> Agent[None, E1Response]:
    """Create a baseline agent for E1 experiment with minimal system prompt."""
    agent = Agent(
        model,
        output_type=E1Response,
        system_prompt="You are a retail customer support assistant. Answer the question concisely. If you are unsure or don't have information to answer accurately, say 'I don't know'.",
        retries=2,
    )
    return agent


def create_e2_agent(model: OpenRouterModel) -> Agent[None, E2Response]:
    """Create E2 Standard RAG agent."""
    system_prompt = """You are a retail customer support assistant.
Use the provided context to answer the user's question accurately.

CRITICAL RULES:
1. If the context does not contain enough information to answer, say "I don't know" or "The information is not available."
2. Always cite sources from the context when providing answers (e.g., "According to [Source 1]...").
3. Do not make up information or hallucinate facts not present in the context.
4. Be concise and accurate.

Context will be provided in the user message."""

    agent = Agent(
        model,
        output_type=E2Response,
        system_prompt=system_prompt,
        retries=5,
    )
    return agent


def create_e3_agent(model: OpenRouterModel) -> Agent[None, E3Response]:
    """Create E3 Filtered RAG agent."""
    system_prompt = """You are a retail customer support assistant.
Use the provided context to answer the user's question accurately.

CRITICAL RULES:
1. If the context does not contain enough information to answer, say "I don't know" or "The information is not available."
2. Always cite sources from the context when providing answers (e.g., "According to [Source 1]...").
3. Do not make up information or hallucinate facts not present in the context.
4. Be concise and accurate.
5. The context has been filtered and reranked for relevance - prioritize information from higher-ranked sources.

Context will be provided in the user message."""

    agent = Agent(
        model,
        output_type=E3Response,
        system_prompt=system_prompt,
        retries=5,
    )
    return agent


def create_e4_agent(model: OpenRouterModel) -> Agent[None, E4Response]:
    """Create E4 Reasoning RAG agent with Chain-of-Thought."""
    system_prompt = """You are a retail customer support assistant with advanced reasoning capabilities.
Use Chain-of-Thought (CoT) reasoning to analyze the provided context and answer questions systematically.

CRITICAL RULES:
1. Break down the problem: First, identify what information is needed to answer the question
2. Analyze the context: Check if the context contains relevant, up-to-date information
3. Check for conflicts: If multiple sources provide contradictory information, note the discrepancy
4. Verify completeness: Ensure all parts of the question can be answered from the context
5. Cite sources: Reference specific sources for each claim
6. Handle missing data: If critical information is missing, outdated, or conflicting, explicitly state this
7. Provide reasoning: Explain your thought process step-by-step in the reasoning_steps field
8. Final answer: Provide a clear, concise answer in the answer field

If you cannot provide a confident answer based on the context, say "I don't know" and explain why in your reasoning.

Context will be provided in the user message."""

    agent = Agent(
        model,
        output_type=E4Response,
        system_prompt=system_prompt,
        retries=5,
    )
    return agent
