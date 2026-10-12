import pandas as pd
import torch
from torch.utils.data import DataLoader
from busi_xai.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    SPLITS_DIR,
    RANDOM_SEED,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    MODELS_DIR,
    METRICS_DIR,
    BEST_MODEL_FILENAME,
)
from busi_xai.data import (
    BUSIDataset,
    build_image_transform,
    compute_class_weights
)

from busi_xai.model import (
    train_model,
    evaluate_model
)

print("=" * 60, flush=True)
print("Start Loading training & validation data", flush=True)
print("=" * 60, flush=True)
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
validation_df = pd.read_csv(SPLITS_DIR / "validation.csv")

train_dataset = BUSIDataset(
    dataframe=train_df,
    image_column="processed_image",
    class_column="class_name",
    mask_column="processed_mask",
    image_transform=build_image_transform(train=True),
)

validation_dataset = BUSIDataset(
    dataframe=validation_df,
    image_column="processed_image",
    class_column="class_name",
    mask_column="processed_mask",
    image_transform=build_image_transform(train=False),
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
print("=" * 60, flush=True)
print("Finish Loading training & validation data", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start Training model", flush=True)
print("=" * 60, flush=True)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint_path = MODELS_DIR / BEST_MODEL_FILENAME
history_path = METRICS_DIR / "training_history.csv"
class_weights = compute_class_weights(
    train_df,
    class_column="class_name",
)
history, best_epoch, best_validation_loss = train_model(
    train_loader=train_loader,
    validation_loader=validation_loader,
    class_weights=class_weights,
    device=device,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    checkpoint_path=checkpoint_path,
    history_path=history_path,
    seed=RANDOM_SEED,
)
print("=" * 60, flush=True)
print("Finish Training model", flush=True)
print("=" * 60, flush=True)