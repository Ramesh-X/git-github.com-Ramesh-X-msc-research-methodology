from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .models import E1Response


def create_openrouter_model(model_name: str, api_key: str) -> OpenRouterModel:
    """Create and return an `OpenRouterModel` configured with the provided API key.

    This helper centralizes creation of the model/provider and mirrors the
    examples in pydantic_ai documentation. Agents should be created with
    this model instance via the factory functions below.
    """
    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)


def create_baseline_agent(model: OpenRouterModel) -> Agent[None, E1Response]:
    """Create a baseline agent for E1 experiment with minimal system prompt."""
    agent = Agent(
        model,
        output_type=E1Response,
        system_prompt="You are a retail customer support assistant. Answer the question concisely.",
        retries=2,
    )
    return agent
