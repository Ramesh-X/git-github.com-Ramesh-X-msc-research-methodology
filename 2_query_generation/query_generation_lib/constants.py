# Naming conventions
QUERY_ID_PREFIXES = {
    "direct": "q_direct",
    "multi_hop": "q_multi_hop",
    "negative": "q_negative",
}

# Number of attempts for transient LLM/validation failures
MAX_ATTEMPTS = 5

# Token limit for building prompts (characters) - overridden by env var NEGATIVE_PROMPT_TOKEN_LIMIT
NEGATIVE_PROMPT_TOKEN_LIMIT = 200000

# Defaults used by the entrypoint (main.py)
DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"
DEFAULT_KB_DIR = "output/kb"
DEFAULT_NUM_DIRECT = 100
DEFAULT_NUM_MULTI_HOP = 25
DEFAULT_NUM_NEGATIVE = 25
DEFAULT_OVERWRITE = False
DEFAULT_DRY_RUN = "true"
