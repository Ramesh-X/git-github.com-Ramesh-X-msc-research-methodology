from typing import Dict

# Topic distributions (rounded to nearest integers when generating structure.json)
TOPIC_DISTRIBUTION: Dict[str, float] = {
    "orders": 14.0,
    "returns_refunds": 13.5,
    "shipping_delivery": 13.0,
    "contact": 12.5,
    "faq": 17.0,
    "account": 6.5,
    "payments_billing": 5.5,
    "membership_loyalty": 4.5,
    "product_info": 3.5,
    "warranty": 2.0,
    "store_services": 2.5,
    "accessibility": 1.5,
    "installation": 2.0,
    "sustainability": 1.0,
    "recycling": 1.0,
}

PAGE_TYPE_DISTRIBUTION = {"tabular": 40, "logical": 30, "unstructured": 30}
MISTAKE_INJECTION_RATE = 0.30
ROT_RATE = 0.10  # 10% of pages will have rot (1 pairs = 2 pages)

STYLE_DISTRIBUTION = {
    "conversational_friendly": 0.41,
    "corporate_formal": 0.35,
    "technical_detailed": 0.24,
}

LENGTH_DISTRIBUTION = {
    "brief": 0.25,
    "medium": 0.51,
    "comprehensive": 0.24,
}

DEFAULT_KB_DIR = "output/kb"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_NUM_PAGES = 100
DEFAULT_DRY_RUN = "false"
DEFAULT_OVERWRITE = "false"
