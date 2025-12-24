# Fixed paths (never change - must be consistent across all steps)
DATA_FOLDER = "data"
EMBEDDINGS_CACHE_FOLDER = "embeddings_cache"
STRUCTURE_FILE_NAME = "structure.json"
QUERIES_FILE_NAME = "queries.jsonl"
E1_OUTPUT_FILE_NAME = "e1_baseline.jsonl"
E2_OUTPUT_FILE_NAME = "e2_standard_rag.jsonl"

DEFAULT_KB_DIR = "output/kb"
DEFAULT_DRY_RUN = "false"
DEFAULT_OVERWRITE = "false"

# Model configurations (moved from env)
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimension for text-embedding-3-small

# Retrieval parameters for E2
E2_TOP_K = 5

# Qdrant configuration
QDRANT_COLLECTION = "retail_kb"
QDRANT_IN_MEMORY = True

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Rate limiting (OpenRouter: 20 requests per minute)
REQUEST_DELAY_SECONDS = 3.0
