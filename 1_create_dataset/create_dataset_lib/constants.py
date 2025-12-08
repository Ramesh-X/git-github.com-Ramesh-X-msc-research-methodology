from typing import Dict

# Topic distributions - rebalanced for broader coverage
# Increased from 11 to 15 topics for better diversity
TOPIC_DISTRIBUTION: Dict[str, float] = {
    "orders": 11.0,
    "returns_refunds": 10.0,
    "shipping_delivery": 10.0,
    "contact": 9.0,
    "faq": 11.0,
    "account": 8.0,
    "payments_billing": 8.0,
    "membership_loyalty": 7.0,
    "product_info": 6.0,
    "warranty": 5.0,
    "store_services": 4.0,
    "accessibility": 3.0,
    "installation": 3.0,
    "sustainability": 2.0,
    "recycling": 2.0,
}

PAGE_TYPE_DISTRIBUTION = {"tabular": 40, "logical": 30, "unstructured": 30}
MISTAKE_INJECTION_RATE = 0.15  # Reduced to account for intentional rot contradictions
ROT_RATE = 0.10  # 10% of pages = 10 rot pages total (5 pairs × 2 versions)
DEFAULT_MAX_TOKENS = 2000  # Increased from 800 to allow rich content (tables, Mermaid)

# Defaults for main.py
NUM_PAGES = 100
DEFAULT_KB_DIR = "output/kb"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_DRY_RUN = "false"
DEFAULT_OVERWRITE = "false"

# Fixed data folder and file names (never change - must be consistent across all steps)
DATA_FOLDER = "data"
STRUCTURE_FILE_NAME = "structure.json"
