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
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(
                    "OpenRouter request failed on attempt %s: %s", attempt, e
                )
                if attempt == retry:
                    logger.exception("Final attempt failed; raising")
                    raise
                print(f"Attempt {attempt} failed: {e}. Retrying...")
                time.sleep(backoff)
                backoff *= 2
