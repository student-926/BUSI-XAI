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

# Model input / preprocessing
IMAGE_SIZE = (224, 224)

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

HORIZONTAL_FLIP_PROBABILITY = 0.5

RANDOM_SEED = 42

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 20

BEST_MODEL_FILENAME = "resnet50_best.pt"

# ---------------------------------------------------------------------
# XAI experiment configuration
# ---------------------------------------------------------------------

XAI_RANDOM_SEED = 42

# Evaluation
XAI_EVALUATION_SPLIT = "test"
XAI_NUM_TEST_SAMPLES = 97

# Common image representation
XAI_IMAGE_SIZE = (224, 224)
XAI_MAP_DTYPE = "float32"
XAI_MAP_MIN = 0.0
XAI_MAP_MAX = 1.0

# Explanation target
XAI_TARGET = "predicted_class"

# Ground-truth masks
XAI_MASK_RESIZE_INTERPOLATION = "nearest"
XAI_MASK_BINARY = True

# Methods
XAI_METHODS = (
    "gradcam",
    "lime",
    "shap",
)

# Runtime measurement
XAI_RECORD_RUNTIME = True

# Output directories
XAI_OUTPUT_DIR = PROJECT_ROOT / "outputs/xai"

GRADCAM_OUTPUT_DIR = XAI_OUTPUT_DIR / "gradcam"
LIME_OUTPUT_DIR = XAI_OUTPUT_DIR / "lime"
SHAP_OUTPUT_DIR = XAI_OUTPUT_DIR / "shap"

# ---------------------------------------------------------------------
# XAI stability experiment configuration
# ---------------------------------------------------------------------

XAI_STABILITY_RANDOM_SEED = 42

# Controlled input perturbation
XAI_STABILITY_NOISE_STD = 0.05

# Stability evaluation
XAI_STABILITY_PRIMARY_METRIC = "spearman"
XAI_STABILITY_SECONDARY_METRIC = "mae"

# Output directories
XAI_STABILITY_OUTPUT_DIR = XAI_OUTPUT_DIR / "stability"

STABILITY_GRADCAM_OUTPUT_DIR = XAI_STABILITY_OUTPUT_DIR / "gradcam"

STABILITY_LIME_OUTPUT_DIR = XAI_STABILITY_OUTPUT_DIR / "lime"
STABILITY_SHAP_OUTPUT_DIR = XAI_STABILITY_OUTPUT_DIR / "shap"