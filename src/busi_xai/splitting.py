from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from busi_xai.config import SPLITS_DIR


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SPLIT_CONFIG = {
    "random_seed": RANDOM_SEED,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VALIDATION_RATIO,
    "test_ratio": TEST_RATIO,
    "stratified": True,
}


def create_splits(manifest_path):
    """
    Create a reproducible stratified train/validation/test split.

    Only records marked as included in the experimental dataset
    are eligible for splitting.
    """

    if not Path(manifest_path).exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    # Use only records approved for the experiment
    df = df[df["include_in_experiment"] == True].copy()

    if len(df) == 0:
        raise ValueError("No records available for experimental splitting.")

    # First split: 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        stratify=df["class_name"],
        random_state=RANDOM_SEED,
    )

    # Second split: divide remaining 30% equally into
    # 15% validation and 15% test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["class_name"],
        random_state=RANDOM_SEED,
    )

    # Add split labels
    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    validation_df["split"] = "validation"
    test_df["split"] = "test"

    # Save exact assignments
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "train.csv"
    validation_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, validation_df, test_df