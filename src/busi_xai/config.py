from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
XAI_DIR = OUTPUTS_DIR / "xai"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Inspection
INSPECTION_OUTPUT_DIR = OUTPUTS_DIR / "inspection"

# Processed data
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_MASKS_DIR = PROCESSED_DATA_DIR / "masks"