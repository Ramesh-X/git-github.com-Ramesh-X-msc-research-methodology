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

# Fixed data folder and file names (never change - must be consistent across all steps)
DATA_FOLDER = "data"
STRUCTURE_FILE_NAME = "structure.json"
QUERIES_FILE_NAME = "queries.jsonl"

# Defaults used by the entrypoint (main.py)
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_KB_DIR = "output/kb"
DEFAULT_NUM_DIRECT = 120
DEFAULT_NUM_MULTI_HOP = 40
DEFAULT_NUM_NEGATIVE = 40
DEFAULT_OVERWRITE = "false"
DEFAULT_DRY_RUN = "false"
