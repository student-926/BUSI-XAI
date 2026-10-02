import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
import copy
import random
import numpy as np
from pathlib import Path
import math
import csv
import time

NUM_CLASSES = 2


def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build an ImageNet-pretrained ResNet-50 for binary classification.
    """

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace the ImageNet 1000-class classifier.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def build_classification_loss(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    """
    Create the weighted cross-entropy loss used for model training.
    """
    return nn.CrossEntropyLoss(weight=class_weights)

def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Adam:
    """
    Create the optimizer used for ResNet-50 training.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train the model for one epoch.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Evaluate the model on the validation set for one epoch.
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def save_best_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    path,
):
    """
    Save the current model state as the best validation checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "validation_loss": validation_loss,
    }

    torch.save(checkpoint, path)

def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible model training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(
    train_loader,
    validation_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_path: str | Path,
    history_path: str | Path,
    seed: int,
):
    """
    Train ResNet-50 using the fixed experimental configuration.

    Model selection is based only on validation loss.

    The test set is intentionally not accepted by this function
    to prevent accidental test-set leakage.
    """
    set_random_seed(seed)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_resnet50().to(device)

    class_weights = class_weights.to(device)

    criterion = build_classification_loss(
        class_weights=class_weights,
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    best_validation_loss = math.inf
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = validate_one_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.perf_counter() - epoch_start

        is_best = validation_loss < best_validation_loss

        if is_best:
            best_validation_loss = validation_loss
            best_epoch = epoch

            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                path=checkpoint_path,
            )
        
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
            f"epoch time: {epoch_time:.4f}"
            f"is best epoch: {is_best}"
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_time_seconds": epoch_time,
            "best_checkpoint": is_best,
        }

        history.append(epoch_metrics)

        print("  → Best model checkpoint updated.")

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "validation_loss",
                    "validation_accuracy",
                    "epoch_time_seconds",
                    "best_checkpoint",
                ],
            )

            writer.writeheader()
            writer.writerows(history)
    return (
        history,
        best_epoch,
        best_validation_loss,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    """
    Load a saved model checkpoint into the supplied model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
):
    """
    Generate predictions and probabilities for a dataset.

    This function does not calculate training or validation loss.
    It is intended for final held-out evaluation.
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probabilities)

    accuracy = (
        (predictions == labels).float().mean().item()
    )

    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
    }