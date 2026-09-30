from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from busi_xai.config import (
    LOGS_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

classes = ["benign", "malignant"]

@dataclass
class ImageRecord:
    """Metadata for one BUSI image and its associated masks."""

    image_path: Path
    class_name: str
    mask_paths: list[Path]
    include_in_experiment: bool = True
    exclusion_reason: str = ""
    image_sha256 : str = ""


def configure_logging() -> None:
    """Configure logging for preprocessing."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "preprocessing.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def is_image_file(path: Path) -> bool:
    """Return True when path has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_mask_file(path: Path) -> bool:
    """Identify BUSI mask files by the conventional '_mask' suffix."""
    return "_mask" in path.stem.lower()


def get_image_id(path: Path) -> str:
    """
    Return the base BUSI image identifier.

    Example:
        benign_1.png       -> benign_1
        benign_1_mask.png  -> benign_1
    """
    stem = path.stem

    while "_mask" in stem.lower():
        index = stem.lower().rfind("_mask")
        stem = stem[:index]

    return stem


def discover_dataset() -> list[ImageRecord]:
    """
    Discover BUSI images and associate all corresponding masks.

    Raw files are only read; they are never modified.
    """
    records: list[ImageRecord] = []

    class_dirs = sorted(
        path
        for path in RAW_DATA_DIR.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise FileNotFoundError(
            f"No class directories found in {RAW_DATA_DIR}"
        )

    for class_dir in class_dirs:
        class_name = class_dir.name

        if class_name not in classes:
          continue
        
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and is_image_file(path)
        )

        images = [
            path
            for path in files
            if not is_mask_file(path)
        ]

        masks = [
            path
            for path in files
            if is_mask_file(path)
        ]

        masks_by_id: dict[str, list[Path]] = {}

        for mask_path in masks:
            image_id = get_image_id(mask_path)
            masks_by_id.setdefault(image_id, []).append(mask_path)

        for image_path in images:
            image_id = get_image_id(image_path)

            records.append(
                ImageRecord(
                    image_path=image_path,
                    class_name=class_name,
                    mask_paths=sorted(masks_by_id.get(image_id, [])),
                )
            )

    return records


def load_image(image_path: Path) -> Image.Image:
    """Load an image and convert it to RGB."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """
    Load a mask and convert it to a binary uint8 array.

    Output:
        0 = background
        1 = lesion
    """
    with Image.open(mask_path) as mask:
        array = np.asarray(mask.convert("L"))

    return (array > 0).astype(np.uint8)


def merge_masks(mask_paths: list[Path]) -> np.ndarray:
    """
    Merge one or more masks using pixel-wise logical OR.
    """
    if not mask_paths:
        raise ValueError("Cannot merge an empty mask list.")

    merged = load_binary_mask(mask_paths[0])

    for mask_path in mask_paths[1:]:
        current = load_binary_mask(mask_path)

        if current.shape != merged.shape:
            raise ValueError(
                "Mask dimension mismatch: "
                f"{mask_path} has shape {current.shape}, "
                f"expected {merged.shape}."
            )

        merged = np.logical_or(
            merged,
            current,
        ).astype(np.uint8)

    return merged


def calculate_file_hash(path: Path) -> str:
    """Calculate SHA-256 hash for exact duplicate detection."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_output_directories() -> None:
    """Create processed-data directories."""
    for class_name in ("benign", "malignant"):
        (PROCESSED_IMAGES_DIR / class_name).mkdir(
            parents=True,
            exist_ok=True,
        )
        (PROCESSED_MASKS_DIR / class_name).mkdir(
            parents=True,
            exist_ok=True,
        )


def process_record(record: ImageRecord) -> dict:
    """Process one image and its associated masks."""
    image = load_image(record.image_path)

    if not record.mask_paths:
        raise ValueError(
            f"No mask found for image: {record.image_path}"
        )

    merged_mask = merge_masks(record.mask_paths)

    image_array = np.asarray(image)

    expected_shape = image_array.shape[:2]

    if merged_mask.shape != expected_shape:
        raise ValueError(
            f"Image/mask dimension mismatch for "
            f"{record.image_path.name}: "
            f"image={expected_shape}, "
            f"mask={merged_mask.shape}"
        )

    image_id = get_image_id(record.image_path)

    output_image = (
        PROCESSED_IMAGES_DIR
        / record.class_name
        / f"{image_id}.png"
    )

    output_mask = (
        PROCESSED_MASKS_DIR
        / record.class_name
        / f"{image_id}_mask.png"
    )

    image.save(output_image)
    Image.fromarray(
        merged_mask * 255,
        mode="L",
    ).save(output_mask)

    lesion_pixels = int(np.count_nonzero(merged_mask))
    total_pixels = int(merged_mask.size)

    return {
        "image_id": image_id,
        "class_name": record.class_name,
        "source_image": str(record.image_path),
        "processed_image": str(output_image),
        "processed_mask": str(output_mask),
        "num_masks": len(record.mask_paths),
        "mask_paths": ";".join(
            str(path) for path in record.mask_paths
        ),
        "image_width": image.width,
        "image_height": image.height,
        "mask_width": merged_mask.shape[1],
        "mask_height": merged_mask.shape[0],
        "mask_empty": lesion_pixels == 0,
        "lesion_pixels": lesion_pixels,
        "lesion_area_ratio": lesion_pixels / total_pixels,
        "image_sha256": calculate_file_hash(record.image_path),
    }

def identify_duplicate_label_conflicts(records):
    """
    Identify exact duplicate images that have conflicting class labels.

    A conflict occurs when the same image SHA-256 hash appears
    under more than one class.
    """
    hash_to_classes = {}

    for record in records:
        image_hash = record["image_sha256"]
        class_name = record["class_name"]

        hash_to_classes.setdefault(image_hash, set()).add(class_name)

    conflicting_hashes = {
        image_hash
        for image_hash, classes in hash_to_classes.items()
        if len(classes) > 1
    }

    return conflicting_hashes


def preprocess_dataset() -> list[dict]:
    """Run the complete BUSI preprocessing pipeline."""
    configure_logging()

    logger.info("Starting BUSI preprocessing.")
    logger.info("Raw dataset: %s", RAW_DATA_DIR)

    records = discover_dataset()

    logger.info("Discovered %d images.", len(records))

    prepare_output_directories()

    results: list[dict] = []

    for index, record in enumerate(records, start=1):
        result = process_record(record)
        
        results.append(result)

        logger.info(
            "Processed %d/%d: %s",
            index,
            len(records),
            record.image_path.name,
        )

    # Identify exact duplicate images assigned to different classes
    conflicting_hashes = identify_duplicate_label_conflicts(results)
    for result in results:
      if result["image_sha256"] in conflicting_hashes:
          result["include_in_experiment"] = False
          result["exclusion_reason"] = "duplicate_image_conflicting_labels"
      else:
          result["include_in_experiment"] = True
          result["exclusion_reason"] = ""
        
    logger.info("Preprocessing completed.")

    return results