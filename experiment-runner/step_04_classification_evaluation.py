import pandas as pd
import torch
import json
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from busi_xai.config import (
    BATCH_SIZE,
    SPLITS_DIR,
    MODELS_DIR,
    BEST_MODEL_FILENAME,
    METRICS_DIR
)
from busi_xai.data import (
    BUSIDataset,
    build_image_transform
)
from busi_xai.model import (
    build_resnet50,
    load_checkpoint,
    evaluate_model
)

print("=" * 60, flush=True)
print("Start Loading testing data", flush=True)
print("=" * 60, flush=True)
test_df = pd.read_csv(SPLITS_DIR / "test.csv")
test_dataset = BUSIDataset(
    dataframe=test_df,
    image_column="processed_image",
    class_column="class_name",
    mask_column="processed_mask",
    image_transform=build_image_transform(train=False),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print("=" * 60, flush=True)
print("Finish Loading testing data", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start evaluating classification model", flush=True)
print("=" * 60, flush=True)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
checkpoint_path = MODELS_DIR / BEST_MODEL_FILENAME

model = build_resnet50().to(device)

checkpoint = load_checkpoint(
    model=model,
    checkpoint_path=checkpoint_path,
    device=device,
)

test_results = evaluate_model(
    model=model,
    loader=test_loader,
    device=device,
)
print("=" * 60, flush=True)
print("Finish evaluating classification model", flush=True)
print("=" * 60, flush=True)

print("=" * 60, flush=True)
print("Start evaluation Report", flush=True)
print("=" * 60, flush=True)
labels = test_results["labels"].numpy()
predictions = test_results["predictions"].numpy()

# Confusion matrix.
cm = confusion_matrix(
    labels,
    predictions,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

# Overall metrics.
accuracy = accuracy_score(labels, predictions)

precision = precision_score(
    labels,
    predictions,
    zero_division=0,
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0,
)

specificity = tn / (tn + fp)

# Per-class metrics.
report = classification_report(
    labels,
    predictions,
    target_names=["benign", "malignant"],
    output_dict=True,
    zero_division=0,
)

metrics = {
    "model": "ResNet-50",
    "dataset": "BUSI",
    "test_samples": int(len(labels)),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "validation_loss_at_checkpoint": float(
        checkpoint["validation_loss"]
    ),
    "accuracy": float(accuracy),
    "precision_malignant": float(precision),
    "recall_malignant": float(recall),
    "f1_malignant": float(f1),
    "specificity_benign": float(specificity),
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "classification_report": report,
}

metrics_path = METRICS_DIR / "test_classification_metrics.json"

with metrics_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=2,
    )

print("Final classification metrics saved to:")
print(metrics_path)
print("Confusion matrix:")
print(cm)

print()
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

print()
print("Classification report:")
print(report)
print("=" * 60, flush=True)
print("End evaluation Report", flush=True)
print("=" * 60, flush=True)