# Fixed paths (never change - must be consistent across all steps)
DATA_FOLDER = "data"
EMBEDDINGS_CACHE_FOLDER = "embeddings_cache"
STRUCTURE_FILE_NAME = "structure.json"
QUERIES_FILE_NAME = "queries.jsonl"
E1_OUTPUT_FILE_NAME = "e1_baseline.jsonl"
E2_OUTPUT_FILE_NAME = "e2_standard_rag.jsonl"
E3_OUTPUT_FILE_NAME = "e3_filtered_rag.jsonl"
E4_OUTPUT_FILE_NAME = "e4_reasoning_rag.jsonl"

DEFAULT_KB_DIR = "output/kb"
DEFAULT_DRY_RUN = "false"
DEFAULT_OVERWRITE = "false"

# Model configurations
OPENROUTER_MODEL = "amazon/nova-2-lite-v1:free"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimension for text-embedding-3-small
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
E2_TOP_K = 5  # E2 - Standard RAG
E3_TOP_N = 20  # E3 - Initial vector search retrieval
E3_TOP_K = 5  # E3 - After reranking
E4_TOP_N = 20  # E4 - Initial vector search retrieval
E4_TOP_K = 5  # E4 - After reranking

# Qdrant configuration
QDRANT_COLLECTION = "retail_kb"
QDRANT_IN_MEMORY = True

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Rate limiting (OpenRouter: 20 requests per minute)
REQUEST_DELAY_SECONDS = 3.0
