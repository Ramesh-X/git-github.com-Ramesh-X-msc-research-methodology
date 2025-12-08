import logging
import os
import time
from typing import Optional

from openai import OpenAI

DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"
logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_OPENROUTER_BASE,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=3,
        )
        self.model = model

    def generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **kwargs
    ) -> Optional[str]:
        """Generate content from the configured model with retry logic."""
        retry = 3
        backoff = 1.0
        for attempt in range(1, retry + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                # Defensive response parsing: OpenRouter/OpenAI SDK may return
                # different structures (object-like or dict-like). Make a best-effort
                # attempt to extract generated content and raise a helpful
                # exception if absent to trigger retries.
                if not resp:
                    raise RuntimeError("Empty response from OpenRouter")
                choices = getattr(resp, "choices", None)
                if choices is None and isinstance(resp, dict):
                    choices = resp.get("choices")
                if not choices:
                    raise RuntimeError("No choices in OpenRouter response")
                choice = choices[0]
                message = getattr(choice, "message", None)
                if message is None and isinstance(choice, dict):
                    message = choice.get("message")
                content = None
                if message is not None:
                    content = getattr(message, "content", None)
                    if content is None and isinstance(message, dict):
                        content = message.get("content")
                if content is None:
                    # Fall back to older 'text' field or direct string return
                    content = getattr(choice, "text", None)
                    if content is None and isinstance(choice, dict):
                        content = choice.get("text")
                if not content:
                    raise RuntimeError(
                        "No message content returned in OpenRouter response"
                    )
                return content
            except Exception as e:
                logger.warning(
                    "OpenRouter request failed on attempt %s: %s", attempt, e
                )
                if attempt == retry:
                    logger.exception("Final attempt failed; raising")
                    raise
                logger.warning("Attempt %s failed; will retry: %s", attempt, e)
                time.sleep(backoff)
                backoff *= 2
