from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .models import Difficulty, QueryResponse


def create_openrouter_model(model_name: str, api_key: str) -> OpenRouterModel:
    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)


direct_agent = Agent(
    output_type=QueryResponse,
    retries=2,
)


@direct_agent.output_validator
def validate_direct_output(ctx: RunContext, output: QueryResponse) -> QueryResponse:
    if output.difficulty not in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        raise ModelRetry(
            f"Difficulty must be one of {list(Difficulty)}, got {output.difficulty}"
        )
    if not output.category or not isinstance(output.category, str):
        raise ModelRetry("Category must be a non-empty string")
    return output


multi_hop_agent = Agent(
    output_type=QueryResponse,
    retries=2,
)


@multi_hop_agent.output_validator
def validate_multi_hop_output(ctx: RunContext, output: QueryResponse) -> QueryResponse:
    if output.difficulty not in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        raise ModelRetry(
            f"Difficulty must be one of {list(Difficulty)}, got {output.difficulty}"
        )
    if not output.category or not isinstance(output.category, str):
        raise ModelRetry("Category must be a non-empty string")
    return output


negative_agent = Agent(
    output_type=QueryResponse,
    retries=2,
)


@negative_agent.output_validator
def validate_negative_output(ctx: RunContext, output: QueryResponse) -> QueryResponse:
    if output.difficulty not in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        raise ModelRetry(
            f"Difficulty must be one of {list(Difficulty)}, got {output.difficulty}"
        )
    if not output.category or not isinstance(output.category, str):
        raise ModelRetry("Category must be a non-empty string")
    return output
