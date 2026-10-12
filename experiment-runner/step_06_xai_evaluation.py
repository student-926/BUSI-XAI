import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import statistics
from busi_xai.config import (
    XAI_OUTPUT_DIR,
    XAI_METHODS,
    GRADCAM_OUTPUT_DIR,
    LIME_OUTPUT_DIR,
    SHAP_OUTPUT_DIR,
    METRICS_DIR,
    XAI_STABILITY_OUTPUT_DIR,
    XAI_STABILITY_NOISE_STD, 
    XAI_STABILITY_RANDOM_SEED,
    SPLITS_DIR, 
    MODELS_DIR,
    STABILITY_GRADCAM_OUTPUT_DIR,
    STABILITY_LIME_OUTPUT_DIR,
    STABILITY_SHAP_OUTPUT_DIR
)
from busi_xai.xai_evaluation import (
    evaluate_faithfulness_records,
    calculate_xai_runtime_summary,
    calculate_pairwise_explanation_spearman,
    build_pairwise_explanation_agreement_record
)
from busi_xai.data import BUSIDataset, build_image_transform
from busi_xai.xai import explain_single_prediction
from busi_xai.model import build_resnet50, load_checkpoint
from busi_xai.xai_evaluation import (
    validate_xai_stability_records,
    evaluate_faithfulness_records,
    run_xai_stability_evaluation
)

print("=" * 60, flush=True)
print("Start loading test data", flush=True)
print("=" * 60, flush=True)
test_df = pd.read_csv(SPLITS_DIR / "test.csv")
test_dataset = BUSIDataset(
    dataframe=test_df,
    image_column="processed_image",
    class_column="class_name",
    image_transform=build_image_transform(train=False),
    mask_column="processed_mask",
)
print("=" * 60, flush=True)
print("Finish loading test data", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start preparing pretrained classification model", flush=True)
print("=" * 60, flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_resnet50(num_classes=2)
checkpoint_path = MODELS_DIR / "resnet50_best.pt"
checkpoint = load_checkpoint(
    model=model,
    checkpoint_path=checkpoint_path,
    device=device,
)
model = model.to(device)
print("=" * 60, flush=True)
print("Finish preparing pretrained classification model", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("1- Faithfulness", flush=True)
print("=" * 60, flush=True)
print("=" * 60, flush=True)
print("Start Faithfulness evaluation", flush=True)
print("=" * 60, flush=True)
XAI_RECORDS_PATH = XAI_OUTPUT_DIR / "xai_records.json"
with open(XAI_RECORDS_PATH, "r") as f:
    xai_records = json.load(f)
faithfulness_records = evaluate_faithfulness_records(
    model=model,
    test_dataset=test_dataset,
    records=xai_records,
    xai_output_dir=XAI_OUTPUT_DIR,
    device=device,
    percentile=90.0,
    baseline_value=0.0,
)
print("=" * 60, flush=True)
print("Finish Faithfulness evaluation", flush=True)
print("=" * 60, flush=True)

FAITHFULNESS_OUTPUT_PATH = (
    XAI_OUTPUT_DIR
    / "xai_faithfulness_records.json"
)
FAITHFULNESS_OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
with open(FAITHFULNESS_OUTPUT_PATH, "w") as f:
    json.dump(
        faithfulness_records,
        f,
        indent=2,
    )
print("Faithfulness records saved.")
print("Path:", FAITHFULNESS_OUTPUT_PATH)

print("=" * 60, flush=True)
print("Start Faithfulness aggregation", flush=True)
print("=" * 60, flush=True)
faithfulness_aggregate = {}
for method in XAI_METHODS:
    method_records = [
        record
        for record in faithfulness_records
        if record["method"] == method
    ]

    probability_drops = [
        float(record["probability_drop"])
        for record in method_records
    ]

    original_probabilities = [
        float(record["original_probability"])
        for record in method_records
    ]

    perturbed_probabilities = [
        float(record["perturbed_probability"])
        for record in method_records
    ]

    deleted_fractions = [
        float(record["deleted_fraction"])
        for record in method_records
    ]

    faithfulness_aggregate[method] = {
        "n": len(method_records),
        "original_probability_mean": statistics.mean(
            original_probabilities
        ),
        "perturbed_probability_mean": statistics.mean(
            perturbed_probabilities
        ),
        "probability_drop_mean": statistics.mean(
            probability_drops
        ),
        "probability_drop_median": statistics.median(
            probability_drops
        ),
        "probability_drop_std": statistics.stdev(
            probability_drops
        ),
        "probability_drop_min": min(
            probability_drops
        ),
        "probability_drop_max": max(
            probability_drops
        ),
        "deleted_fraction_mean": statistics.mean(
            deleted_fractions
        ),
    }

print("=" * 60, flush=True)
print("Finish Faithfulness aggregation", flush=True)
print("=" * 60, flush=True)

print("=== Aggregate faithfulness metrics ===")

for method, metrics in faithfulness_aggregate.items():
    print(f"\n{method.upper()}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

FAITHFULNESS_AGGREGATE_PATH = (
    METRICS_DIR
    / "xai_faithfulness_aggregate_metrics.json"
)
with open(FAITHFULNESS_AGGREGATE_PATH, "w") as f:
    json.dump(
        faithfulness_aggregate,
        f,
        indent=2,
    )

print("Faithfulness aggregation records saved.")
print("Path:", FAITHFULNESS_AGGREGATE_PATH)


print("=" * 60, flush=True)
print("2- Stability", flush=True)
print("=" * 60, flush=True)
xai_records_path = XAI_OUTPUT_DIR / "xai_records.json"
with open(xai_records_path, "r") as f:
    xai_records = json.load(f)
method_map_dirs = {
    "gradcam": GRADCAM_OUTPUT_DIR,
    "lime": LIME_OUTPUT_DIR,
    "shap": SHAP_OUTPUT_DIR,
}
output_map_dirs = {
    "gradcam": STABILITY_GRADCAM_OUTPUT_DIR,
    "lime": STABILITY_LIME_OUTPUT_DIR,
    "shap": STABILITY_SHAP_OUTPUT_DIR,
}
for method, map_dir in method_map_dirs.items():
    map_files = sorted(
        map_dir.glob("dataset_index_*.npy")
    )

print("=" * 60, flush=True)
print("Start Stability calculation", flush=True)
print("=" * 60, flush=True)
stability_records = run_xai_stability_evaluation(
    model=model,
    dataset=test_dataset,
    original_records=xai_records,
    original_map_dirs=method_map_dirs,
    output_map_dirs=output_map_dirs,
    device=device,
    noise_std=XAI_STABILITY_NOISE_STD,
    random_seed=XAI_STABILITY_RANDOM_SEED,
)
print("=" * 60, flush=True)
print("Finish Stability calculation", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start Stability validation", flush=True)
print("=" * 60, flush=True)
stability_validation = validate_xai_stability_records(
    stability_records,
    expected_num_samples=97,
    expected_methods=("gradcam", "lime", "shap"),
)
print("=" * 60, flush=True)
print("Finish Stability validation", flush=True)
print("=" * 60, flush=True)

stability_records_path = (
    XAI_STABILITY_OUTPUT_DIR
    / "xai_stability_records.json"
)

with open(stability_records_path, "w") as f:
    json.dump(
        stability_records,
        f,
        indent=2,
    )
print("=== Stability records saved ===")
print(f"Path: {stability_records_path}")

print("=" * 60, flush=True)
print("Start Stability aggregations", flush=True)
print("=" * 60, flush=True)
stability_aggregate = {}
for method in XAI_METHODS:
    method_records = [
        record
        for record in stability_records
        if record["method"] == method
    ]

    spearman_values = np.array([
        record["spearman_correlation"]
        for record in method_records
    ])

    mae_values = np.array([
        record["mae"]
        for record in method_records
    ])

    runtime_values = np.array([
        record["runtime_seconds"]
        for record in method_records
    ])

    prediction_changed_values = np.array([
        record["prediction_changed"]
        for record in method_records
    ], dtype=np.float32)

    stability_aggregate[method] = {
        "n": len(method_records),
        "spearman_mean": float(np.mean(spearman_values)),
        "spearman_median": float(np.median(spearman_values)),
        "spearman_std": float(np.std(spearman_values)),
        "spearman_min": float(np.min(spearman_values)),
        "spearman_max": float(np.max(spearman_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_median": float(np.median(mae_values)),
        "mae_std": float(np.std(mae_values)),
        "mae_min": float(np.min(mae_values)),
        "mae_max": float(np.max(mae_values)),
        "runtime_mean_seconds": float(np.mean(runtime_values)),
        "runtime_median_seconds": float(np.median(runtime_values)),
        "prediction_changed_count": int(
            np.sum(prediction_changed_values)
        ),
        "prediction_changed_rate": float(
            np.mean(prediction_changed_values)
        ),
    }
print("=" * 60, flush=True)
print("Finish Stability aggregations", flush=True)
print("=" * 60, flush=True)

print("=== Aggregate stability metrics ===")

for method, metrics in stability_aggregate.items():
    print(f"\n{method.upper()}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

stability_aggregate_path = (
    METRICS_DIR
    / "xai_stability_aggregate_metrics.json"
)

with open(stability_aggregate_path, "w") as f:
    json.dump(
        stability_aggregate,
        f,
        indent=2,
    )

print("=== Aggregate stability metrics saved ===")
print(f"Path: {stability_aggregate_path}")

print("=" * 60, flush=True)
print("3- Pairwise agreement", flush=True)
print("=" * 60, flush=True)
expected_n = 97
EXPECTED_DATASET_INDICES = list(range(expected_n))
map_dirs = {
    method: XAI_OUTPUT_DIR / method
    for method in XAI_METHODS
}
def load_original_xai_map(method, dataset_index):
    path = map_dirs[method] / f"dataset_index_{dataset_index:03d}.npy"
    return np.load(path)
pairwise_agreement_records = []
method_pairs = [
    ("gradcam", "lime"),
    ("gradcam", "shap"),
    ("lime", "shap"),
]

print("=" * 60, flush=True)
print("Start pairwise calculation", flush=True)
print("=" * 60, flush=True)

for dataset_index in EXPECTED_DATASET_INDICES:
    maps = {
        method: load_original_xai_map(method, dataset_index)
        for method in XAI_METHODS
    }
    for method_a, method_b in method_pairs:

        spearman = calculate_pairwise_explanation_spearman(
            maps[method_a],
            maps[method_b],
        )
        record = build_pairwise_explanation_agreement_record(
            dataset_index=dataset_index,
            method_a=method_a,
            method_b=method_b,
            spearman=spearman,
        )
        pairwise_agreement_records.append(record)

print("=" * 60, flush=True)
print("Finish pairwise calculation", flush=True)
print("=" * 60, flush=True)

print("=== Pairwise explanation agreement ===")
for method_a, method_b in method_pairs:
    values = np.array([
        r["spearman"]
        for r in pairwise_agreement_records
        if r["method_a"] == method_a
        and r["method_b"] == method_b
    ])

    print(
        f"{method_a.upper()} vs {method_b.upper()}: "
        f"n={len(values)}, "
        f"mean={values.mean():.6f}, "
        f"median={np.median(values):.6f}, "
        f"std={values.std():.6f}, "
        f"min={values.min():.6f}, "
        f"max={values.max():.6f}"
    )

print("=" * 60, flush=True)
print("Start pairwise aggregations", flush=True)
print("=" * 60, flush=True)
pairwise_aggregate = {}
for method_a, method_b in method_pairs:

    values = np.array([
        record["spearman"]
        for record in pairwise_agreement_records
        if record["method_a"] == method_a
        and record["method_b"] == method_b
    ], dtype=np.float64)

    pair_key = f"{method_a}_vs_{method_b}"

    pairwise_aggregate[pair_key] = {
        "n": int(len(values)),
        "spearman_mean": float(values.mean()),
        "spearman_median": float(np.median(values)),
        "spearman_std": float(values.std()),
        "spearman_min": float(values.min()),
        "spearman_max": float(values.max()),
    }
print("=" * 60, flush=True)
print("Finish pairwise aggregations", flush=True)
print("=" * 60, flush=True)

PAIRWISE_RECORDS_PATH = (
    XAI_OUTPUT_DIR
    / "xai_pairwise_agreement_records.json"
)
PAIRWISE_AGGREGATE_PATH = (
    XAI_OUTPUT_DIR
    / "xai_pairwise_agreement_aggregate_metrics.json"
)
with open(PAIRWISE_RECORDS_PATH, "w") as f:
    json.dump(pairwise_agreement_records, f, indent=2)

with open(PAIRWISE_AGGREGATE_PATH, "w") as f:
    json.dump(pairwise_aggregate, f, indent=2)
print("=== Pairwise agreement artifacts saved ===")
print(f"Records: {PAIRWISE_RECORDS_PATH}")
print(f"Aggregate: {PAIRWISE_AGGREGATE_PATH}")

print("=" * 60, flush=True)
print("4- Computational efficiency", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start runtime summary", flush=True)
print("=" * 60, flush=True)
runtime_summary = calculate_xai_runtime_summary(
    records=xai_records,
    methods=XAI_METHODS,
)
print("=" * 60, flush=True)
print("Finish runtime summary", flush=True)
print("=" * 60, flush=True)

print("=== XAI Runtime Aggregates ===")
for method in XAI_METHODS:
    result = runtime_summary[method]
    print(
        f"{method.upper()}: "
        f"n={result['n']}, "
        f"mean={result['runtime_mean_seconds']:.6f}s, "
        f"median={result['runtime_median_seconds']:.6f}s, "
        f"std={result['runtime_std_seconds']:.6f}s, "
        f"min={result['runtime_min_seconds']:.6f}s, "
        f"max={result['runtime_max_seconds']:.6f}s"
    )
XAI_RUNTIME_AGGREGATE_PATH = (
    METRICS_DIR
    / "xai_runtime_aggregate_metrics.json"
)
with open(XAI_RUNTIME_AGGREGATE_PATH, "w") as f:
    json.dump(runtime_summary, f, indent=2)
print("=== XAI runtime artifact saved ===")
print(f"Path: {XAI_RUNTIME_AGGREGATE_PATH}")