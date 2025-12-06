"""Constants for E2-E4 experiments.

File names and model configurations are hardcoded here.
Only KB_DIR is configurable via environment variables.
"""

# Default values
DEFAULT_KB_DIR = "output/kb"
DEFAULT_DRY_RUN = "false"
DEFAULT_OVERWRITE = "false"

# Fixed file names (never change)
QUERIES_FILE_NAME = "queries.jsonl"
E2_OUTPUT_FILE_NAME = "e2_standard_rag.jsonl"
E3_OUTPUT_FILE_NAME = "e3_filtered_rag.jsonl"
E4_OUTPUT_FILE_NAME = "e4_reasoning_rag.jsonl"

# Model configurations (hardcoded)
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters (hardcoded)
E2_TOP_K = 5
E3_TOP_N = 20
E3_TOP_K = 5
E4_TOP_N = 20
E4_TOP_K = 5

# Qdrant configuration (hardcoded)
QDRANT_COLLECTION = "retail_kb_e2_e4"
QDRANT_IN_MEMORY = True  # Always use in-memory for consistency
EMBEDDING_DIM = 1536  # text-embedding-3-small dimension

# Chunking parameters (hardcoded)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Rate limiting (OpenRouter: 20 requests per minute)
# Using 0.5 second delay = 120 requests per minute (safe margin)
REQUEST_DELAY_SECONDS = 3.0  # 20 req/min = 1 req every 3 seconds
