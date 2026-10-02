from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from busi_xai.config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    HORIZONTAL_FLIP_PROBABILITY,
)


def build_image_transform(train: bool = False):
    """Build the image transform for training or evaluation."""

    transform_steps = [
        transforms.Resize(
            IMAGE_SIZE,
            interpolation=InterpolationMode.BILINEAR,
        )
    ]

    if train:
        transform_steps.append(
            transforms.RandomHorizontalFlip(
                p=HORIZONTAL_FLIP_PROBABILITY
            )
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_MEAN,
                std=IMAGE_STD,
            ),
        ]
    )

    return transforms.Compose(transform_steps)


def transform_mask(mask: Image.Image) -> torch.Tensor:
    """Resize a binary ground-truth mask to model input size."""

    mask = mask.convert("L")

    mask = transforms.Resize(
        IMAGE_SIZE,
        interpolation=InterpolationMode.NEAREST,
    )(mask)

    mask = transforms.PILToTensor()(mask)

    # Convert to binary 0/1 tensor.
    mask = (mask > 0).float()

    return mask


class BUSIDataset(Dataset):
    """BUSI dataset for model input and optional ground-truth masks."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_column: str,
        class_column: str,
        image_transform,
        mask_column: str | None = None,
    ):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_column = image_column
        self.class_column = class_column
        self.mask_column = mask_column
        self.image_transform = image_transform

        self.class_to_index = {
            "benign": 0,
            "malignant": 1,
        }

        unknown_classes = set(self.dataframe[class_column]) - set(
            self.class_to_index
        )

        if unknown_classes:
            raise ValueError(
                f"Unknown class labels found: {sorted(unknown_classes)}"
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row[self.image_column])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.image_transform(image)

        label = self.class_to_index[row[self.class_column]]

        item = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "index": index,
        }

        if self.mask_column is not None:
            mask_path = Path(row[self.mask_column])

            with Image.open(mask_path) as mask:
                item["mask"] = transform_mask(mask)

        return item


def compute_class_weights(
    dataframe: pd.DataFrame,
    class_column: str,
) -> torch.Tensor:
    """Calculate balanced class weights from the supplied dataframe only."""

    class_counts = dataframe[class_column].value_counts()

    required_classes = ["benign", "malignant"]

    missing = set(required_classes) - set(class_counts.index)

    if missing:
        raise ValueError(
            f"Missing classes in dataframe: {sorted(missing)}"
        )

    total = len(dataframe)
    number_of_classes = len(required_classes)

    weights = [
        total / (number_of_classes * class_counts[class_name])
        for class_name in required_classes
    ]

    return torch.tensor(weights, dtype=torch.float32)