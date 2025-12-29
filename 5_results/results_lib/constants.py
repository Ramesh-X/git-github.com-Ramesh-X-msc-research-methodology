# results_lib/constants.py

# File paths
EVAL_FOLDER = "eval"
OUTPUT_FOLDER = "output"  # KB_DIR/output for results

# Input files from evaluation
E1_METRICS_FILE = "e1_metrics.json"
E2_METRICS_FILE = "e2_metrics.json"
E3_METRICS_FILE = "e3_metrics.json"
E4_METRICS_FILE = "e4_metrics.json"

# Output subdirectories
TABLES_SUBDIR = "tables"
CHARTS_SUBDIR = "charts"

# Chart settings
CHART_DPI = 300  # Publication quality
CHART_FORMAT = "png"  # Can be "png" or "jpg"
FIGURE_SIZE_STANDARD = (10, 6)  # inches
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_TALL = (10, 10)

# Statistical significance levels
ALPHA_LEVEL = 0.05
BONFERRONI_COMPARISONS = 6  # E1vsE2, E1vsE3, E1vsE4, E2vsE3, E2vsE4, E3vsE4

# Color schemes (colorblind-friendly)
CATEGORY_COLORS = {
    "Clean Pass": "#2ecc71",  # Green
    "Hallucination": "#e74c3c",  # Red
    "Retrieval Failure": "#f39c12",  # Orange
    "Irrelevant Answer": "#f1c40f",  # Yellow
    "Total Failure": "#8b0000",  # Dark red
}

EXPERIMENT_COLORS = {
    "E1": "#3498db",  # Blue
    "E2": "#9b59b6",  # Purple
    "E3": "#e67e22",  # Orange
    "E4": "#27ae60",  # Green
}

# Default values
DEFAULT_KB_DIR = "output/kb"
DEFAULT_OVERWRITE = "false"
