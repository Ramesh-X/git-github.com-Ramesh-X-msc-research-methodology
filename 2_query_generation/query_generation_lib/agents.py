from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .models import Difficulty, NegativeQueryList, QueryResponse


def create_openrouter_model(model_name: str, api_key: str) -> OpenRouterModel:
    """Create and return an `OpenRouterModel` configured with the provided API key.

    This helper centralizes creation of the model/provider and mirrors the
    examples in pydantic_ai documentation. Agents should be created with
    this model instance via the factory functions below.
    """
    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)


def create_direct_agent(model: OpenRouterModel) -> Agent[None, QueryResponse]:
    """Create a direct query agent bound to the given model and attach validators."""
    agent = Agent(model, output_type=QueryResponse, retries=2)

    @agent.output_validator
    def _validate_direct_output(
        ctx: RunContext, output: QueryResponse
    ) -> QueryResponse:
        if output.difficulty not in [
            Difficulty.EASY,
            Difficulty.MEDIUM,
            Difficulty.HARD,
        ]:
            raise ModelRetry(
                f"Difficulty must be one of {list(Difficulty)}, got {output.difficulty}"
            )
        if not output.category or not isinstance(output.category, str):
            raise ModelRetry("Category must be a non-empty string")
        return output

    return agent


def create_multi_hop_agent(model: OpenRouterModel) -> Agent[None, QueryResponse]:
    """Create a multi-hop agent bound to the given model and attach validators."""
    agent = Agent(model, output_type=QueryResponse, retries=2)

    @agent.output_validator
    def _validate_multi_hop_output(
        ctx: RunContext, output: QueryResponse
    ) -> QueryResponse:
        if output.difficulty not in [
            Difficulty.EASY,
            Difficulty.MEDIUM,
            Difficulty.HARD,
        ]:
            raise ModelRetry(
                f"Difficulty must be one of {list(Difficulty)}, got {output.difficulty}"
            )
        if not output.category or not isinstance(output.category, str):
            raise ModelRetry("Category must be a non-empty string")
        return output

    return agent


def create_negative_agent(model: OpenRouterModel) -> Agent[None, NegativeQueryList]:
    """Create a negative-query agent bound to the given model and attach validators."""
    agent = Agent(model, output_type=NegativeQueryList, retries=2)

    @agent.output_validator
    def _validate_negative_output(
        ctx: RunContext, output: NegativeQueryList
    ) -> NegativeQueryList:
        for i, query in enumerate(output.queries):
            if query.difficulty not in [
                Difficulty.EASY,
                Difficulty.MEDIUM,
                Difficulty.HARD,
            ]:
                raise ModelRetry(
                    f"Query {i}: Difficulty must be one of {list(Difficulty)}, got {query.difficulty}"
                )
            if not query.category or not isinstance(query.category, str):
                raise ModelRetry(f"Query {i}: Category must be a non-empty string")
        return output

    return agent
