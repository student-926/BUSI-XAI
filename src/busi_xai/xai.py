from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
import shap


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

def attribution_to_top_percentile_mask(
    attribution_map: np.ndarray,
    percentile: float = 90.0,
) -> np.ndarray:
    """
    Convert a normalized attribution map into a binary mask
    containing the top attribution pixels.

    Parameters
    ----------
    attribution_map : np.ndarray
        Attribution map with shape (224, 224), expected to be
        normalized to [0, 1].

    percentile : float, default=90.0
        Percentile threshold used to define the top-attribution
        region. A value of 90 means the top 10% of pixels.

    Returns
    -------
    np.ndarray
        Binary uint8 mask with shape (224, 224).
    """
    if not isinstance(attribution_map, np.ndarray):
        raise TypeError("attribution_map must be a NumPy array.")

    if attribution_map.shape != (224, 224):
        raise ValueError(
            f"Expected attribution map shape (224, 224), "
            f"got {attribution_map.shape}."
        )

    if not 0.0 <= float(attribution_map.min()) <= 1.0:
        raise ValueError("Attribution map values must be within [0, 1].")

    if not 0.0 <= float(attribution_map.max()) <= 1.0:
        raise ValueError("Attribution map values must be within [0, 1].")

    if not 0.0 < percentile < 100.0:
        raise ValueError(
            f"percentile must be between 0 and 100, got {percentile}."
        )

    threshold = np.percentile(attribution_map, percentile)

    binary_mask = (attribution_map >= threshold).astype(np.uint8)

    return binary_mask
    

def calculate_deletion_faithfulness(
    model,
    image_tensor,
    attribution_map,
    target_class,
    device,
    percentile=90.0,
    baseline_value=0.0,
):
    """
    Measure deletion-based faithfulness of an attribution map.

    The top-attribution pixels are replaced by a fixed baseline value,
    and the change in probability assigned to the original target class
    is measured.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.

    image_tensor : torch.Tensor
        Preprocessed image tensor with shape (1, 3, 224, 224).

    attribution_map : np.ndarray
        Normalized attribution map with shape (224, 224).

    target_class : int
        Original predicted class whose probability is evaluated.

    device : torch.device
        Computation device.

    percentile : float, default=90.0
        Attribution percentile used to define the pixels to delete.
        90.0 means the top 10% of attribution pixels.

    baseline_value : float, default=0.0
        Fixed baseline value used to replace selected pixels.

    Returns
    -------
    dict
        Faithfulness measurements including original probability,
        perturbed probability, probability drop, and deletion size.
    """
    if image_tensor.shape != (1, 3, 224, 224):
        raise ValueError(
            "Expected image_tensor shape (1, 3, 224, 224), "
            f"got {tuple(image_tensor.shape)}."
        )

    if not isinstance(attribution_map, np.ndarray):
        raise TypeError("attribution_map must be a NumPy array.")

    if attribution_map.shape != (224, 224):
        raise ValueError(
            "Expected attribution map shape (224, 224), "
            f"got {attribution_map.shape}."
        )

    binary_mask = attribution_to_top_percentile_mask(
        attribution_map,
        percentile=percentile,
    )

    perturbed_image = image_tensor.clone()

    mask_tensor = torch.from_numpy(
        binary_mask
    ).to(
        device=device,
        dtype=torch.bool,
    )

    perturbed_image[:, :, mask_tensor] = baseline_value

    model.eval()

    with torch.no_grad():
        original_logits = model(image_tensor)
        original_probabilities = torch.softmax(
            original_logits,
            dim=1,
        )

        perturbed_logits = model(perturbed_image)
        perturbed_probabilities = torch.softmax(
            perturbed_logits,
            dim=1,
        )

    original_probability = float(
        original_probabilities[0, target_class].item()
    )

    perturbed_probability = float(
        perturbed_probabilities[0, target_class].item()
    )

    probability_drop = (
        original_probability - perturbed_probability
    )

    return {
        "original_probability": original_probability,
        "perturbed_probability": perturbed_probability,
        "probability_drop": float(probability_drop),
        "deletion_percentile": float(percentile),
        "deleted_pixels": int(binary_mask.sum()),
        "total_pixels": int(binary_mask.size),
        "deleted_fraction": float(
            binary_mask.sum() / binary_mask.size
        ),
        "baseline_value": float(baseline_value),
    }

def evaluate_faithfulness_records(
    model,
    test_dataset,
    records,
    xai_output_dir,
    device,
    percentile=90.0,
    baseline_value=0.0,
):
    """
    Evaluate deletion-based faithfulness for existing XAI records.

    Existing attribution maps are loaded from:
        xai_output_dir / method / dataset_index_XXX.npy

    No explanations are regenerated and existing XAI records are not modified.

    Parameters
    ----------
    model : torch.nn.Module
        The already-trained classification model.

    test_dataset : Dataset
        The existing test dataset used for the XAI experiment.

    records : list of dict
        Existing XAI records loaded from xai_records.json.

    xai_output_dir : str or pathlib.Path
        Root directory containing the existing XAI attribution maps.

    device : torch.device
        Device used for model inference.

    percentile : float, default=90.0
        Attribution percentile used to define deleted pixels.

    baseline_value : float, default=0.0
        Fixed replacement value for deleted pixels.

    Returns
    -------
    list of dict
        One faithfulness record for each image/method pair.
    """
    from pathlib import Path

    if not records:
        raise ValueError("records must not be empty.")

    xai_output_dir = Path(xai_output_dir)

    required_methods = {"gradcam", "lime", "shap"}

    record_pairs = set()
    faithfulness_records = []

    for record in sorted(
        records,
        key=lambda item: (
            int(item["dataset_index"]),
            str(item["method"]),
        ),
    ):
        dataset_index = int(record["dataset_index"])
        method = str(record["method"])

        if method not in required_methods:
            raise ValueError(
                f"Unexpected XAI method '{method}' "
                f"for dataset index {dataset_index}."
            )

        pair = (dataset_index, method)

        if pair in record_pairs:
            raise ValueError(
                f"Duplicate image/method pair found: {pair}"
            )

        record_pairs.add(pair)

        if dataset_index < 0 or dataset_index >= len(test_dataset):
            raise IndexError(
                f"dataset_index {dataset_index} is outside "
                f"the test dataset range."
            )

        image_item = test_dataset[dataset_index]

        if not isinstance(image_item, dict):
            raise TypeError(
                "Expected test_dataset items to be dictionaries."
            )

        if "image" not in image_item:
            raise KeyError(
                "Test dataset item is missing the 'image' key."
            )

        image_tensor = image_item["image"]

        if not torch.is_tensor(image_tensor):
            raise TypeError(
                "Expected test dataset 'image' to be a torch.Tensor."
            )

        if image_tensor.shape != (3, 224, 224):
            raise ValueError(
                "Expected test image shape (3, 224, 224), "
                f"got {tuple(image_tensor.shape)} "
                f"for dataset index {dataset_index}."
            )

        image_tensor = image_tensor.unsqueeze(0).to(device)

        map_path = (
            xai_output_dir
            / method
            / f"dataset_index_{dataset_index:03d}.npy"
        )

        if not map_path.exists():
            raise FileNotFoundError(
                f"Attribution map not found: {map_path}"
            )

        attribution_map = np.load(map_path)

        if attribution_map.shape != (224, 224):
            raise ValueError(
                f"Invalid attribution map shape for "
                f"{method}, dataset index {dataset_index}: "
                f"{attribution_map.shape}"
            )

        faithfulness = calculate_deletion_faithfulness(
            model=model,
            image_tensor=image_tensor,
            attribution_map=attribution_map,
            target_class=int(record["predicted_class"]),
            device=device,
            percentile=percentile,
            baseline_value=baseline_value,
        )

        faithfulness_record = {
            "image_id": int(record["image_id"]),
            "dataset_index": dataset_index,
            "true_label": int(record["true_label"]),
            "predicted_class": int(record["predicted_class"]),
            "correct": bool(record["correct"]),
            "method": method,
            **faithfulness,
        }

        faithfulness_records.append(faithfulness_record)

    expected_pairs = {
        (dataset_index, method)
        for dataset_index in range(len(test_dataset))
        for method in required_methods
    }

    if record_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - record_pairs)
        extra_pairs = sorted(record_pairs - expected_pairs)

        raise ValueError(
            "Faithfulness record pairing is incomplete or inconsistent. "
            f"Missing pairs: {missing_pairs}; "
            f"Unexpected pairs: {extra_pairs}"
        )

    return faithfulness_records


def generate_lime(
    model,
    image_tensor,
    device,
    target_class,
    random_seed,
    num_samples=1000,
):
    """
    Generate a LIME explanation for a single image.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen classification model.

    image_tensor : torch.Tensor
        Preprocessed image tensor with shape (1, 3, 224, 224).

    device : torch.device
        Device used for model inference.

    target_class : int
        Class whose prediction is being explained.

    random_seed : int
        Random seed for reproducibility.

    num_samples : int, default=1000
        Number of perturbed samples generated by LIME.

    Returns
    -------
    np.ndarray
        Normalized non-negative attribution map with shape (224, 224).
    """

    if image_tensor.ndim != 4 or image_tensor.shape != (1, 3, 224, 224):
        raise ValueError(
            "image_tensor must have shape (1, 3, 224, 224)."
        )

    if not isinstance(target_class, int):
        target_class = int(target_class)

    model.eval()

    start_time = time.perf_counter()

    # ImageNet normalization used by the classifier.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_np = image_tensor.detach().cpu().numpy()[0]
    image_np = np.transpose(image_np, (1, 2, 0))

    # Convert normalized tensor back to [0, 1] RGB.
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0.0, 1.0)

    explainer = lime_image.LimeImageExplainer(
        random_state=random_seed
    )

    def predict_fn(images):
        batch = np.asarray(images, dtype=np.float32)

        batch = (batch - mean) / std

        batch_tensor = torch.from_numpy(
            np.transpose(batch, (0, 3, 1, 2))
        ).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.detach().cpu().numpy()

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=None,
        labels=(target_class,),
        hide_color=0,
        num_samples=num_samples,
    )

    # Retrieve superpixel weights for the predicted target class.
    local_exp = explanation.local_exp[target_class]

    attribution = np.zeros(
        (image_np.shape[0], image_np.shape[1]),
        dtype=np.float32,
    )

    segments = explanation.segments

    for segment_id, weight in local_exp:
        attribution[segments == segment_id] = abs(weight)

    runtime_seconds = time.perf_counter() - start_time
    attribution = normalize_attribution_map(attribution)

    return attribution.astype(np.float32), runtime_seconds

def generate_shap(
    model,
    image_tensor,
    device,
    target_class,
    random_seed,
    num_samples=100,
):
    """
    Generate a SHAP GradientExplainer attribution map
    for one image.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen classification model.

    image_tensor : torch.Tensor
        Preprocessed image tensor with shape (1, 3, 224, 224).

    device : torch.device
        Device used for inference.

    target_class : int
        Class whose prediction is being explained.

    random_seed : int
        Random seed for reproducibility.

    num_samples : int, default=100
        Number of SHAP sampling evaluations.

    Returns
    -------
    np.ndarray
        Normalized non-negative attribution map with shape (224, 224).
    """

    if image_tensor.ndim != 4 or image_tensor.shape != (1, 3, 224, 224):
        raise ValueError(
            "image_tensor must have shape (1, 3, 224, 224)."
        )

    target_class = int(target_class)

    model.eval()

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    start_time = time.perf_counter()

    # Neutral baseline in the same normalized input space.
    background = torch.zeros(
        (1, 3, 224, 224),
        dtype=image_tensor.dtype,
        device=device,
    )

    explainer = shap.GradientExplainer(
        model,
        background,
    )

    shap_values = explainer.shap_values(
        image_tensor,
        nsamples=num_samples,
    )

    shap_values = np.asarray(shap_values)

    # SHAP versions can return either:
    # (samples, channels, height, width, outputs)
    # or
    # (samples, channels, height, width)
    if shap_values.ndim == 5:
        values = shap_values[0, :, :, :, target_class]
    elif shap_values.ndim == 4:
        values = shap_values[0]
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    runtime_seconds = time.perf_counter() - start_time
    # Aggregate channel-level attribution into one spatial map.
    attribution = np.abs(values).sum(axis=0)

    attribution = normalize_attribution_map(
        attribution.astype(np.float32)
    )

    return attribution.astype(np.float32), runtime_seconds

def build_xai_record(
    image_id,
    dataset_index,
    true_label,
    predicted_class,
    predicted_probability,
    method,
    runtime,
    attribution,
    ground_truth_mask,
    localisation_percentile=90.0,
):
    """
    Build a standardized XAI evaluation record for one image and one method.
    """

    attribution = np.asarray(attribution, dtype=np.float32)
    ground_truth_mask = np.asarray(ground_truth_mask, dtype=np.uint8)

    binary_mask = attribution_to_top_percentile_mask(
        attribution,
        percentile=localisation_percentile,
    )

    intersection = int(
        np.logical_and(binary_mask == 1, ground_truth_mask == 1).sum()
    )

    union = int(
        np.logical_or(binary_mask == 1, ground_truth_mask == 1).sum()
    )

    iou = calculate_iou(binary_mask, ground_truth_mask)
    dice = calculate_dice(binary_mask, ground_truth_mask)

    return {
        "image_id": image_id,
        "dataset_index": int(dataset_index),
        "true_label": int(true_label),
        "predicted_class": int(predicted_class),
        "predicted_probability": float(predicted_probability),
        "correct": bool(int(true_label) == int(predicted_class)),
        "method": method,
        "runtime_seconds": float(runtime),
        "localisation_percentile": float(localisation_percentile),
        "explanation_pixels": int(binary_mask.sum()),
        "ground_truth_pixels": int(ground_truth_mask.sum()),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(iou),
        "dice": float(dice),
    }

def save_xai_evaluation_outputs(
    records,
    attribution_maps,
    output_dir,
):
    """
    Save XAI evaluation records and raw attribution maps.

    Attribution maps are saved separately by method.
    """

    output_dir = Path(output_dir)

    for method in ("gradcam", "lime", "shap"):
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

    # Save raw attribution maps.
    for (method, dataset_index), attribution in attribution_maps.items():
        method_dir = output_dir / method

        attribution_path = (
            method_dir / f"dataset_index_{dataset_index:03d}.npy"
        )

        np.save(
            attribution_path,
            np.asarray(attribution, dtype=np.float32),
        )

    # Save structured records.
    records_path = output_dir / "xai_records.json"

    import json

    with records_path.open("w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            indent=2,
        )

    return records_path

def evaluate_xai_methods(
    model,
    dataset,
    device,
    methods=("gradcam", "lime", "shap"),
    random_seed=42,
    lime_num_samples=1000,
    shap_num_samples=100,
    localisation_percentile=90.0,
    max_samples=None,
    output_dir=None,
):    
    """
    Evaluate multiple XAI methods on the same dataset under a
    controlled experimental protocol.

    Returns
    -------
    records : list[dict]
        One standardized record per image/method.
    attribution_maps : dict
        Raw attribution maps keyed by (method, dataset_index).
    """

    model.eval()

    if output_dir is not None:
        output_dir = Path(output_dir)

        for method in methods:
            (output_dir / method).mkdir(
                parents=True,
                exist_ok=True,
            )

    if max_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(int(max_samples), len(dataset))

    records = []
    attribution_maps = {}

    for sample_position in range(num_samples):
        sample = dataset[sample_position]

        image = sample["image"]
        true_label = int(sample["label"].item())
        dataset_index = int(sample["index"])

        ground_truth_mask = sample["mask"].squeeze(0).cpu().numpy()
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

        image_batch = image.unsqueeze(0).to(device)

        # ---------------------------------------------------------
        # Model prediction is calculated once and shared by all
        # XAI methods for this image.
        # ---------------------------------------------------------
        with torch.no_grad():
            logits = model(image_batch)
            probabilities = torch.softmax(logits, dim=1)

        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        predicted_probability = float(
            probabilities[0, predicted_class].item()
        )

        print(
            f"[{sample_position + 1}/{num_samples}] "
            f"index={dataset_index}, "
            f"true={true_label}, "
            f"pred={predicted_class}"
        )

        for method in methods:

            method = method.lower()

            if method == "gradcam":
                attribution, runtime = generate_gradcam(
                    model=model,
                    image=image_batch,
                    target_class=predicted_class,
                    device=device,
                )

            elif method == "lime":
                attribution, runtime = generate_lime(
                    model=model,
                    image_tensor=image_batch,
                    device=device,
                    target_class=predicted_class,
                    random_seed=random_seed,
                    num_samples=lime_num_samples,
                )

            elif method == "shap":
                attribution, runtime = generate_shap(
                    model=model,
                    image_tensor=image_batch,
                    device=device,
                    target_class=predicted_class,
                    random_seed=random_seed,
                    num_samples=shap_num_samples,
                )

            else:
                raise ValueError(
                    f"Unsupported XAI method: {method}"
                )

            attribution = np.asarray(
                attribution,
                dtype=np.float32,
            )

            record = build_xai_record(
                image_id=dataset_index,
                dataset_index=dataset_index,
                true_label=true_label,
                predicted_class=predicted_class,
                predicted_probability=predicted_probability,
                method=method,
                runtime=runtime,
                attribution=attribution,
                ground_truth_mask=ground_truth_mask,
                localisation_percentile=localisation_percentile,
            )

            records.append(record)

            attribution_maps[(method, dataset_index)] = attribution
                        
            if output_dir is not None:
                attribution_path = (
                    output_dir
                    / method
                    / f"dataset_index_{dataset_index:03d}.npy"
                )

                np.save(
                    attribution_path,
                    attribution.astype(np.float32),
                )
    return records, attribution_maps

def aggregate_xai_results(records):
    """
    Aggregate paired XAI evaluation records.

    Computes mean, median, and standard deviation for IoU, Dice,
    and runtime overall and stratified by prediction correctness
    and true class.

    Parameters
    ----------
    records : list[dict]
        Individual XAI evaluation records. Each record must contain:
        method, dataset_index, iou, dice, runtime_seconds,
        correct, and true_label.

    Returns
    -------
    dict
        Nested aggregate results.
    """
    import numpy as np

    required_fields = {
        "method",
        "dataset_index",
        "iou",
        "dice",
        "runtime_seconds",
        "correct",
        "true_label",
    }

    if not records:
        raise ValueError("records must not be empty.")

    missing = required_fields - set(records[0].keys())
    if missing:
        raise ValueError(
            f"Records are missing required fields: {sorted(missing)}"
        )

    methods = sorted({record["method"] for record in records})

    def summarize(group):
        if not group:
            return {
                "n": 0,
                "iou_mean": None,
                "iou_median": None,
                "iou_std": None,
                "dice_mean": None,
                "dice_median": None,
                "dice_std": None,
                "runtime_mean": None,
                "runtime_median": None,
                "runtime_std": None,
            }

        iou = np.asarray([r["iou"] for r in group], dtype=float)
        dice = np.asarray([r["dice"] for r in group], dtype=float)
        runtime = np.asarray(
            [r["runtime_seconds"] for r in group],
            dtype=float,
        )

        return {
            "n": len(group),
            "iou_mean": float(np.mean(iou)),
            "iou_median": float(np.median(iou)),
            "iou_std": float(np.std(iou, ddof=1)) if len(iou) > 1 else 0.0,
            "dice_mean": float(np.mean(dice)),
            "dice_median": float(np.median(dice)),
            "dice_std": float(np.std(dice, ddof=1)) if len(dice) > 1 else 0.0,
            "runtime_mean": float(np.mean(runtime)),
            "runtime_median": float(np.median(runtime)),
            "runtime_std": (
                float(np.std(runtime, ddof=1))
                if len(runtime) > 1
                else 0.0
            ),
        }

    results = {}

    for method in methods:
        method_records = [
            record for record in records
            if record["method"] == method
        ]

        results[method] = {
            "overall": summarize(method_records),

            "correct": summarize([
                record for record in method_records
                if record["correct"] is True
            ]),

            "incorrect": summarize([
                record for record in method_records
                if record["correct"] is False
            ]),

            "benign": summarize([
                record for record in method_records
                if record["true_label"] == 0
            ]),

            "malignant": summarize([
                record for record in method_records
                if record["true_label"] == 1
            ]),
        }

    return results

# ---------------------------------------------------------------------
# XAI stability evaluation
# ---------------------------------------------------------------------

def generate_stability_noise(
    image_tensor: torch.Tensor,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    """
    Generate a reproducible additive Gaussian-noise perturbation.

    The same noise tensor can be reused across all XAI methods for
    the same image, ensuring a paired stability comparison.
    """
    generator = torch.Generator(device=image_tensor.device)
    generator.manual_seed(seed)

    noise = torch.randn(
        image_tensor.shape,
        generator=generator,
        device=image_tensor.device,
        dtype=image_tensor.dtype,
    )

    return image_tensor + (noise * noise_std)


def calculate_map_spearman_correlation(
    original_map: np.ndarray,
    perturbed_map: np.ndarray,
) -> float:
    """
    Calculate Spearman rank correlation between two attribution maps.
    """
    from scipy.stats import spearmanr

    original_flat = np.asarray(original_map, dtype=np.float64).reshape(-1)
    perturbed_flat = np.asarray(perturbed_map, dtype=np.float64).reshape(-1)

    correlation = spearmanr(original_flat, perturbed_flat).statistic

    if not np.isfinite(correlation):
        return 0.0

    return float(correlation)


def calculate_map_mae(
    original_map: np.ndarray,
    perturbed_map: np.ndarray,
) -> float:
    """
    Calculate mean absolute error between two attribution maps.
    """
    original_array = np.asarray(original_map, dtype=np.float32)
    perturbed_array = np.asarray(perturbed_map, dtype=np.float32)

    if original_array.shape != perturbed_array.shape:
        raise ValueError(
            "Original and perturbed attribution maps must have "
            "the same shape."
        )

    return float(
        np.mean(np.abs(original_array - perturbed_array))
    )

def build_xai_stability_record(
    image_id: int,
    dataset_index: int,
    true_label: int,
    original_predicted_class: int,
    perturbed_predicted_class: int,
    original_probability: float,
    perturbed_probability: float,
    method: str,
    spearman_correlation: float,
    mae: float,
    noise_std: float,
    random_seed: int,
) -> dict:
    """
    Build one reproducible stability-evaluation record.
    """
    return {
        "image_id": int(image_id),
        "dataset_index": int(dataset_index),
        "true_label": int(true_label),
        "original_predicted_class": int(original_predicted_class),
        "perturbed_predicted_class": int(perturbed_predicted_class),
        "prediction_changed": bool(
            original_predicted_class != perturbed_predicted_class
        ),
        "original_probability": float(original_probability),
        "perturbed_probability_for_original_target": float(
            perturbed_probability
        ),
        "method": str(method),
        "spearman_correlation": float(spearman_correlation),
        "mae": float(mae),
        "noise_std": float(noise_std),
        "random_seed": int(random_seed),
    }

def validate_xai_stability_records(
    records: list[dict],
    expected_num_samples: int = 97,
    expected_methods: tuple[str, ...] = (
        "gradcam",
        "lime",
        "shap",
    ),
) -> dict:
    """
    Validate stability evaluation records before persistence.
    """
    expected_record_count = (
        expected_num_samples * len(expected_methods)
    )

    errors = []

    if len(records) != expected_record_count:
        errors.append(
            f"Expected {expected_record_count} records, "
            f"found {len(records)}."
        )

    methods = sorted({
        record.get("method")
        for record in records
    })

    if methods != sorted(expected_methods):
        errors.append(
            f"Expected methods {sorted(expected_methods)}, "
            f"found {methods}."
        )

    seen_pairs = set()

    for record in records:
        required_fields = {
            "image_id",
            "dataset_index",
            "true_label",
            "original_predicted_class",
            "perturbed_predicted_class",
            "prediction_changed",
            "original_probability",
            "perturbed_probability_for_original_target",
            "method",
            "spearman_correlation",
            "mae",
            "noise_std",
            "random_seed",
        }

        missing_fields = required_fields - set(record.keys())

        if missing_fields:
            errors.append(
                f"Missing fields: {sorted(missing_fields)}"
            )
            continue

        pair = (
            record["dataset_index"],
            record["method"],
        )

        if pair in seen_pairs:
            errors.append(
                f"Duplicate dataset_index/method pair: {pair}"
            )

        seen_pairs.add(pair)

        if record["method"] not in expected_methods:
            errors.append(
                f"Unexpected method: {record['method']}"
            )

        if not (
            np.isfinite(record["spearman_correlation"])
        ):
            errors.append(
                f"Non-finite Spearman value for {pair}"
            )

        if not np.isfinite(record["mae"]):
            errors.append(
                f"Non-finite MAE value for {pair}"
            )

        if record["mae"] < 0:
            errors.append(
                f"Negative MAE value for {pair}"
            )

        if not np.isfinite(record["original_probability"]):
            errors.append(
                f"Non-finite original probability for {pair}"
            )

        if not np.isfinite(
            record["perturbed_probability_for_original_target"]
        ):
            errors.append(
                f"Non-finite perturbed probability for {pair}"
            )

        if record["noise_std"] <= 0:
            errors.append(
                f"Invalid noise std for {pair}"
            )

    records_per_method = {
        method: sum(
            record["method"] == method
            for record in records
        )
        for method in expected_methods
    }

    if any(
        count != expected_num_samples
        for count in records_per_method.values()
    ):
        errors.append(
            f"Unexpected records per method: "
            f"{records_per_method}"
        )

    return {
        "valid": len(errors) == 0,
        "record_count": len(records),
        "records_per_method": records_per_method,
        "errors": errors,
    }

def validate_xai_stability_map(
    attribution_map: np.ndarray,
    expected_shape: tuple[int, int] = (224, 224),
) -> dict:
    """
    Validate one perturbed XAI attribution map.
    """
    errors = []

    array = np.asarray(attribution_map)

    if array.shape != expected_shape:
        errors.append(
            f"Expected map shape {expected_shape}, "
            f"found {array.shape}."
        )

    if array.dtype != np.float32:
        errors.append(
            f"Expected float32 map, found {array.dtype}."
        )

    if not np.all(np.isfinite(array)):
        errors.append("Map contains non-finite values.")

    if array.size > 0:
        map_min = float(array.min())
        map_max = float(array.max())

        if map_min < 0.0 or map_max > 1.0:
            errors.append(
                f"Map values outside [0, 1]: "
                f"min={map_min}, max={map_max}."
            )
    else:
        errors.append("Map is empty.")

    return {
        "valid": len(errors) == 0,
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()) if array.size > 0 else None,
        "max": float(array.max()) if array.size > 0 else None,
        "errors": errors,
    }

def run_xai_stability_evaluation(
    model,
    dataset,
    original_records: list[dict],
    original_map_dirs: dict[str, Path],
    output_map_dirs: dict[str, Path],
    device: torch.device,
    noise_std: float = 0.05,
    random_seed: int = 42,
) -> list[dict]:
    """
    Run the controlled XAI stability evaluation.

    Existing original attribution maps are reused. For each test image,
    one deterministic noisy input is generated and reused across all
    XAI methods.
    """
    import time

    model.eval()

    records_by_key = {
        (
            record["dataset_index"],
            record["method"],
        ): record
        for record in original_records
    }

    methods = ("gradcam", "lime", "shap")
    stability_records = []

    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]

        image = sample["image"].unsqueeze(0).to(device)
        true_label = int(sample["label"])

        original_record = next(
            record
            for record in original_records
            if record["dataset_index"] == dataset_index
            and record["method"] == "gradcam"
        )

        original_predicted_class = int(
            original_record["predicted_class"]
        )
        original_probability = float(
            original_record["predicted_probability"]
        )

        perturbed_image = generate_stability_noise(
            image_tensor=image,
            noise_std=noise_std,
            seed=random_seed + dataset_index,
        )

        with torch.no_grad():
            perturbed_logits = model(perturbed_image)
            perturbed_probabilities = torch.softmax(
                perturbed_logits,
                dim=1,
            )

        perturbed_predicted_class = int(
            torch.argmax(
                perturbed_probabilities,
                dim=1,
            ).item()
        )

        perturbed_probability = float(
            perturbed_probabilities[
                0,
                original_predicted_class,
            ].item()
        )

        for method in methods:
            original_map_path = (
                original_map_dirs[method]
                / f"dataset_index_{dataset_index:03d}.npy"
            )

            original_map = np.load(original_map_path)

            start_time = time.perf_counter()

            if method == "gradcam":
                perturbed_map, runtime_seconds = generate_gradcam(
                    model=model,
                    image=perturbed_image,
                    target_class=original_predicted_class,
                    device=device
                )

            elif method == "lime":
                perturbed_map, runtime_seconds = generate_lime(
                    model=model,
                    image_tensor=perturbed_image,
                    device=device,
                    target_class=original_predicted_class,
                    random_seed=random_seed,
                )

            elif method == "shap":
                perturbed_map, runtime_seconds = generate_shap(
                    model=model,
                    image_tensor=perturbed_image,
                    device=device,
                    target_class=original_predicted_class,
                    random_seed=random_seed,
                )

            else:
                raise ValueError(
                    f"Unsupported XAI method: {method}"
                )

            runtime_seconds = (
                time.perf_counter() - start_time
            )

            perturbed_map = np.asarray(
                perturbed_map,
                dtype=np.float32,
            )

            validation = validate_xai_stability_map(
                perturbed_map
            )

            if not validation["valid"]:
                raise ValueError(
                    f"Invalid perturbed map for "
                    f"{method}, dataset_index="
                    f"{dataset_index}: "
                    f"{validation['errors']}"
                )

            spearman_correlation = (
                calculate_map_spearman_correlation(
                    original_map,
                    perturbed_map,
                )
            )

            mae = calculate_map_mae(
                original_map,
                perturbed_map,
            )

            output_map_path = (
                output_map_dirs[method]
                / f"dataset_index_{dataset_index:03d}.npy"
            )

            np.save(
                output_map_path,
                perturbed_map,
            )

            record = build_xai_stability_record(
                image_id=dataset_index,
                dataset_index=dataset_index,
                true_label=true_label,
                original_predicted_class=(
                    original_predicted_class
                ),
                perturbed_predicted_class=(
                    perturbed_predicted_class
                ),
                original_probability=original_probability,
                perturbed_probability=(
                    perturbed_probability
                ),
                method=method,
                spearman_correlation=(
                    spearman_correlation
                ),
                mae=mae,
                noise_std=noise_std,
                random_seed=(
                    random_seed + dataset_index
                ),
            )

            record["runtime_seconds"] = float(
                runtime_seconds
            )

            stability_records.append(record)

        print(
            f"Completed dataset_index={dataset_index}"
        )

    return stability_records

def calculate_pairwise_explanation_spearman(map_a, map_b):
    """
    Calculate Spearman rank correlation between two XAI maps.

    The maps must have identical shapes. The maps are flattened before
    calculating the correlation.
    """
    import numpy as np
    from scipy.stats import spearmanr

    a = np.asarray(map_a, dtype=np.float64).ravel()
    b = np.asarray(map_b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError(
            f"Map shape mismatch: {a.shape} vs {b.shape}"
        )

    result = spearmanr(a, b)

    if not np.isfinite(result.statistic):
        raise ValueError(
            "Pairwise explanation Spearman correlation is non-finite."
        )

    return float(result.statistic)


def build_pairwise_explanation_agreement_record(
    dataset_index,
    method_a,
    method_b,
    spearman,
):
    """
    Build one pairwise explanation-agreement record.
    """
    return {
        "dataset_index": int(dataset_index),
        "method_a": str(method_a),
        "method_b": str(method_b),
        "spearman": float(spearman),
    }


def validate_pairwise_explanation_agreement_records(
    records,
    expected_dataset_indices,
    expected_method_pairs,
):
    """
    Validate pairwise explanation-agreement records.
    """
    import math

    errors = []

    expected_indices = set(expected_dataset_indices)
    expected_pairs = {
        tuple(pair) for pair in expected_method_pairs
    }

    expected_count = (
        len(expected_indices) * len(expected_pairs)
    )

    if len(records) != expected_count:
        errors.append(
            f"Expected {expected_count} records, "
            f"found {len(records)}."
        )

    seen = set()

    for record in records:
        required_keys = {
            "dataset_index",
            "method_a",
            "method_b",
            "spearman",
        }

        missing = required_keys - set(record.keys())

        if missing:
            errors.append(
                f"Missing keys {sorted(missing)} "
                f"in record."
            )
            continue

        key = (
            int(record["dataset_index"]),
            str(record["method_a"]),
            str(record["method_b"]),
        )

        if key in seen:
            errors.append(
                f"Duplicate record: {key}"
            )

        seen.add(key)

        if int(record["dataset_index"]) not in expected_indices:
            errors.append(
                f"Unexpected dataset index: "
                f"{record['dataset_index']}"
            )

        pair = (
            str(record["method_a"]),
            str(record["method_b"]),
        )

        if pair not in expected_pairs:
            errors.append(
                f"Unexpected method pair: {pair}"
            )

        spearman = float(record["spearman"])

        if not math.isfinite(spearman):
            errors.append(
                f"Non-finite Spearman value for {key}"
            )

        if not -1.0 <= spearman <= 1.0:
            errors.append(
                f"Spearman outside [-1, 1] for {key}: "
                f"{spearman}"
            )

    if len(seen) != expected_count:
        errors.append(
            f"Expected {expected_count} unique records, "
            f"found {len(seen)}."
        )

    return errors

def calculate_xai_runtime_summary(records, methods):
    """
    Calculate descriptive runtime statistics for original XAI records.
    """
    import numpy as np

    summary = {}

    for method in methods:
        runtimes = np.array(
            [
                float(record["runtime_seconds"])
                for record in records
                if record["method"] == method
            ],
            dtype=np.float64,
        )

        if len(runtimes) == 0:
            raise ValueError(
                f"No runtime records found for method: {method}"
            )

        if not np.all(np.isfinite(runtimes)):
            raise ValueError(
                f"Non-finite runtime detected for method: {method}"
            )

        if np.any(runtimes < 0):
            raise ValueError(
                f"Negative runtime detected for method: {method}"
            )

        summary[method] = {
            "n": int(len(runtimes)),
            "runtime_mean_seconds": float(runtimes.mean()),
            "runtime_median_seconds": float(np.median(runtimes)),
            "runtime_std_seconds": float(runtimes.std()),
            "runtime_min_seconds": float(runtimes.min()),
            "runtime_max_seconds": float(runtimes.max()),
        }

    return summary