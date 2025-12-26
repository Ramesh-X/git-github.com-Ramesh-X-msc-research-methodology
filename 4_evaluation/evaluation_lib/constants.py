# evaluation_lib/constants.py

# Fixed paths (must match 3_e1_to_e4 structure)
DATA_FOLDER = "data"
EVAL_FOLDER = "eval"  # New: KB_DIR/eval/

# Input files from 3_e1_to_e4
E1_INPUT_FILE = "e1_baseline.jsonl"
E2_INPUT_FILE = "e2_standard_rag.jsonl"
E3_INPUT_FILE = "e3_filtered_rag.jsonl"
E4_INPUT_FILE = "e4_reasoning_rag.jsonl"

# Output files (evaluation results with RAGAS scores)
E1_EVAL_OUTPUT = "e1_evaluation.jsonl"
E2_EVAL_OUTPUT = "e2_evaluation.jsonl"
E3_EVAL_OUTPUT = "e3_evaluation.jsonl"
E4_EVAL_OUTPUT = "e4_evaluation.jsonl"

# Metrics files (aggregated statistics)
E1_METRICS_OUTPUT = "e1_metrics.json"
E2_METRICS_OUTPUT = "e2_metrics.json"
E3_METRICS_OUTPUT = "e3_metrics.json"
E4_METRICS_OUTPUT = "e4_metrics.json"

# Default values
DEFAULT_KB_DIR = "output/kb"
DEFAULT_DRY_RUN = "false"
DEFAULT_OVERWRITE = "false"

# LLM Models for RAGAS evaluation
RAGAS_LLM_MODEL = "gpt-4o-mini"  # GPT-4 as judge (from research)
RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"  # Same as retrieval

# RAGAS Configuration
RAGAS_BATCH_SIZE = 10  # Evaluate 10 queries at a time
RAGAS_METRICS = ["context_precision", "faithfulness", "answer_relevancy"]

# Category thresholds (from research)
CLEAN_PASS_THRESHOLD = 0.7
HALLUCINATION_CP_THRESHOLD = 0.6
HALLUCINATION_F_THRESHOLD = 0.4
RETRIEVAL_FAILURE_CP_THRESHOLD = 0.4
RETRIEVAL_FAILURE_F_THRESHOLD = 0.7
IRRELEVANT_AR_THRESHOLD = 0.4
IRRELEVANT_F_THRESHOLD = 0.7
TOTAL_FAILURE_CP_THRESHOLD = 0.4
TOTAL_FAILURE_F_THRESHOLD = 0.4
TOTAL_FAILURE_AR_THRESHOLD = 0.4

# Rate limiting
REQUEST_DELAY_SECONDS = 1.0  # Between RAGAS API calls
