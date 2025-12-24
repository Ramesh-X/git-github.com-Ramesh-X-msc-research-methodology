from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .models import E1Response, E2Response


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
