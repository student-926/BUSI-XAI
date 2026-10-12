import json
from pathlib import Path
import pandas as pd
import torch
from busi_xai.model import (
    build_resnet50, 
    load_checkpoint
)
from busi_xai.config import (
    XAI_RANDOM_SEED,
    XAI_EVALUATION_SPLIT,
    XAI_NUM_TEST_SAMPLES,
    XAI_IMAGE_SIZE,
    XAI_MAP_DTYPE,
    XAI_MAP_MIN,
    XAI_MAP_MAX,
    XAI_TARGET,
    XAI_MASK_RESIZE_INTERPOLATION,
    XAI_MASK_BINARY,
    XAI_METHODS,
    XAI_RECORD_RUNTIME,
    XAI_OUTPUT_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    METRICS_DIR
)
from busi_xai.data import (
    BUSIDataset, 
    build_image_transform
)
from busi_xai.xai import (
    evaluate_xai_methods,
    save_xai_evaluation_outputs,
    aggregate_xai_results
)

print("=" * 60, flush=True)
print("Start preparing XAI protocol", flush=True)
print("=" * 60, flush=True)
protocol = {
    "experiment": "BUSI-XAI",
    "evaluation_split": XAI_EVALUATION_SPLIT,
    "num_test_samples": XAI_NUM_TEST_SAMPLES,
    "xai_methods": list(XAI_METHODS),
    "target": XAI_TARGET,
    "image_size": list(XAI_IMAGE_SIZE),
    "common_map": {
        "dtype": XAI_MAP_DTYPE,
        "minimum": XAI_MAP_MIN,
        "maximum": XAI_MAP_MAX,
        "non_negative": True,
    },
    "ground_truth_masks": {
        "resize_interpolation": XAI_MASK_RESIZE_INTERPOLATION,
        "binary": XAI_MASK_BINARY,
    },
    "random_seed": XAI_RANDOM_SEED,
    "record_runtime": XAI_RECORD_RUNTIME,
    "model_selection": "validation_loss_only",
    "test_set_used_for_model_selection": False,
    "model_retraining_for_xai": False,
}
print("=" * 60, flush=True)
print("Finish preparing XAI protocol", flush=True)
print("=" * 60, flush=True)

protocol_path = XAI_OUTPUT_DIR / "xai_protocol.json"
protocol_path.parent.mkdir(parents=True, exist_ok=True)
with protocol_path.open("w") as f:
    json.dump(protocol, f, indent=2)
print(f"Protocol saved to: {protocol_path}")

print("=" * 60, flush=True)
print("Start Loading test data", flush=True)
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
print("Finish Loading test data", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start loading and preparing pretrained classification model", flush=True)
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
print("Finish loading and preparing pretrained classification model", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start XAI evaluation process", flush=True)
print("=" * 60, flush=True)
records, attribution_maps = evaluate_xai_methods(
    model=model,
    dataset=test_dataset,
    device=device,
    methods=("gradcam", "lime", "shap"),
    random_seed=42,
    lime_num_samples=1000,
    shap_num_samples=100,
    localisation_percentile=90.0,
    max_samples=None,
    output_dir=XAI_OUTPUT_DIR,
)
print("=" * 60, flush=True)
print("Finish XAI evaluation process", flush=True)
print("=" * 60, flush=True)

records_path = save_xai_evaluation_outputs(
    records=records,
    attribution_maps=attribution_maps,
    output_dir=XAI_OUTPUT_DIR,
)
print("XAI Records saved to:")
print(records_path)

print("=" * 60, flush=True)
print("Start aggregation", flush=True)
print("=" * 60, flush=True)
aggregate_results = aggregate_xai_results(records)
print("=" * 60, flush=True)
print("Finish aggregation", flush=True)
print("=" * 60, flush=True)

print("\n=== Aggregation Report ===", list(aggregate_results.keys()))
for method, result in aggregate_results.items():
    print(f"\n=== {method.upper()} ===")

    for group_name, summary in result.items():
        print(f"{group_name}: n={summary['n']}")

print("\nTotal records:", len(records))
for method, result in aggregate_results.items():
    print(f"\n{'=' * 60}")
    print(f"{method.upper()}")
    print(f"{'=' * 60}")

    for group_name, summary in result.items():
        print(f"\n--- {group_name} (n={summary['n']}) ---")

        print(
            f"IoU:     mean={summary['iou_mean']:.6f}, "
            f"median={summary['iou_median']:.6f}, "
            f"std={summary['iou_std']:.6f}"
        )

        print(
            f"Dice:    mean={summary['dice_mean']:.6f}, "
            f"median={summary['dice_median']:.6f}, "
            f"std={summary['dice_std']:.6f}"
        )

        print(
            f"Runtime: mean={summary['runtime_mean']:.6f}s, "
            f"median={summary['runtime_median']:.6f}s, "
            f"std={summary['runtime_std']:.6f}s"
        )

AGGREGATE_FILE = METRICS_DIR / "xai_aggregate_metrics.json"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
with AGGREGATE_FILE.open("w") as f:
    json.dump(aggregate_results, f, indent=2)

print("Saved:", AGGREGATE_FILE)