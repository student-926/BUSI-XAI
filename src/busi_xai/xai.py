from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def normalize_attribution_map(
    attribution_map: np.ndarray,
) -> np.ndarray:
    """
    Convert an attribution map to float32 in the range [0, 1].
    """

    attribution_map = np.asarray(attribution_map, dtype=np.float32)

    if attribution_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D attribution map, got shape {attribution_map.shape}."
        )

    attribution_map = np.maximum(attribution_map, 0.0)

    min_value = float(attribution_map.min())
    max_value = float(attribution_map.max())

    if max_value > min_value:
        attribution_map = (
            attribution_map - min_value
        ) / (max_value - min_value)
    else:
        attribution_map = np.zeros_like(attribution_map, dtype=np.float32)

    return attribution_map.astype(np.float32)


def get_resnet50_gradcam_target_layer(
    model: nn.Module,
) -> nn.Module:
    """
    Return the final convolutional layer used by Grad-CAM for ResNet-50.
    """

    try:
        return model.layer4[-1]
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            "The supplied model does not expose the expected ResNet-50 "
            "layer4[-1] target layer."
        ) from exc


def generate_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """
    Generate a Grad-CAM attribution map for one image.

    The model weights are never updated. Gradients are enabled because
    Grad-CAM requires them to compute the attribution.
    """

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            f"Expected image shape [1, C, H, W], got {tuple(image.shape)}."
        )

    model.eval()

    # Grad-CAM requires gradients even though the model is not trained.
    for parameter in model.parameters():
        parameter.requires_grad = True

    image = image.to(device)

    target_layer = get_resnet50_gradcam_target_layer(model)
    targets = [ClassifierOutputTarget(int(target_class))]

    start_time = time.perf_counter()

    with GradCAM(
        model=model,
        target_layers=[target_layer],
    ) as cam:
        grayscale_cam = cam(
            input_tensor=image,
            targets=targets,
        )

    runtime_seconds = time.perf_counter() - start_time

    attribution_map = normalize_attribution_map(grayscale_cam[0])

    if attribution_map.shape != (224, 224):
        raise ValueError(
            "Grad-CAM output must have shape (224, 224), "
            f"got {attribution_map.shape}."
        )

    return attribution_map, runtime_seconds

def explain_single_prediction(
    model: nn.Module,
    image: torch.Tensor,
    true_label: int,
    image_id: str,
    device: torch.device,
) -> dict:
    """
    Generate a Grad-CAM explanation for one preprocessed image.

    The explanation targets the model's predicted class.
    """

    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)

    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    predicted_probability = float(
        probabilities[0, predicted_class].item()
    )

    attribution_map, runtime_seconds = generate_gradcam(
        model=model,
        image=image,
        target_class=predicted_class,
        device=device,
    )

    return {
        "image_id": image_id,
        "true_label": int(true_label),
        "predicted_label": predicted_class,
        "predicted_probability": predicted_probability,
        "correct": predicted_class == int(true_label),
        "method": "gradcam",
        "runtime_seconds": float(runtime_seconds),
        "attribution_map": attribution_map,
    }

def save_attribution_map(
    attribution_map: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Save a normalized attribution map as a NumPy array.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attribution_map = np.asarray(
        attribution_map,
        dtype=np.float32,
    )

    if attribution_map.shape != (224, 224):
        raise ValueError(
            f"Expected attribution map shape (224, 224), "
            f"got {attribution_map.shape}."
        )

    if not np.isfinite(attribution_map).all():
        raise ValueError("Attribution map contains non-finite values.")

    if attribution_map.min() < 0.0 or attribution_map.max() > 1.0:
        raise ValueError(
            "Attribution map must be normalized to [0, 1]."
        )

    np.save(output_path, attribution_map)

    return output_path

def create_xai_overlay(
    image: torch.Tensor,
    attribution_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a visual overlay of an attribution map on the input image.

    Returns an RGB uint8 image with shape (224, 224, 3).
    """
    import matplotlib.cm as cm

    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape [C, H, W], got {tuple(image.shape)}."
        )

    attribution_map = np.asarray(attribution_map, dtype=np.float32)

    if attribution_map.shape != (224, 224):
        raise ValueError(
            f"Expected attribution map shape (224, 224), "
            f"got {attribution_map.shape}."
        )

    if not np.isfinite(attribution_map).all():
        raise ValueError("Attribution map contains non-finite values.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    # Convert normalized tensor back to displayable RGB.
    image = image.detach().cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image.transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)

    # Convert attribution map to an RGB heatmap.
    heatmap = cm.get_cmap("jet")(attribution_map)[..., :3]

    overlay = (
        (1.0 - alpha) * image
        + alpha * heatmap
    )

    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return overlay

def attribution_to_binary_mask(
    attribution_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert a normalized attribution map into a binary explanation mask.
    """
    attribution_map = np.asarray(
        attribution_map,
        dtype=np.float32,
    )

    if attribution_map.shape != (224, 224):
        raise ValueError(
            f"Expected attribution map shape (224, 224), "
            f"got {attribution_map.shape}."
        )

    if not np.isfinite(attribution_map).all():
        raise ValueError(
            "Attribution map contains non-finite values."
        )

    if attribution_map.min() < 0.0 or attribution_map.max() > 1.0:
        raise ValueError(
            "Attribution map must be normalized to [0, 1]."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    binary_mask = (
        attribution_map >= threshold
    ).astype(np.uint8)

    return binary_mask

def calculate_iou(
    explanation_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    """
    explanation_mask = np.asarray(explanation_mask, dtype=bool)
    ground_truth_mask = np.asarray(ground_truth_mask, dtype=bool)

    if explanation_mask.shape != (224, 224):
        raise ValueError(
            f"Expected explanation mask shape (224, 224), "
            f"got {explanation_mask.shape}."
        )

    if ground_truth_mask.shape != (224, 224):
        raise ValueError(
            f"Expected ground-truth mask shape (224, 224), "
            f"got {ground_truth_mask.shape}."
        )

    intersection = np.logical_and(
        explanation_mask,
        ground_truth_mask,
    ).sum()

    union = np.logical_or(
        explanation_mask,
        ground_truth_mask,
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def calculate_dice(
    explanation_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """
    Calculate Dice similarity coefficient between two binary masks.
    """
    explanation_mask = np.asarray(explanation_mask, dtype=bool)
    ground_truth_mask = np.asarray(ground_truth_mask, dtype=bool)

    if explanation_mask.shape != (224, 224):
        raise ValueError(
            f"Expected explanation mask shape (224, 224), "
            f"got {explanation_mask.shape}."
        )

    if ground_truth_mask.shape != (224, 224):
        raise ValueError(
            f"Expected ground-truth mask shape (224, 224), "
            f"got {ground_truth_mask.shape}."
        )

    intersection = np.logical_and(
        explanation_mask,
        ground_truth_mask,
    ).sum()

    explanation_area = explanation_mask.sum()
    ground_truth_area = ground_truth_mask.sum()

    denominator = explanation_area + ground_truth_area

    if denominator == 0:
        return 1.0

    return float(
        2.0 * intersection / denominator
    )