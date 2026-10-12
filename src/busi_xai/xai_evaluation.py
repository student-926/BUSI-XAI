from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
import shap
import json
from scipy.stats import spearmanr
import time
import math

from busi_xai.xai import generate_gradcam, attribution_to_top_percentile_mask, generate_lime, generate_shap

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