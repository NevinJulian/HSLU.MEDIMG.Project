"""Training loop, early stopping, and experiment runner."""

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.evaluate import compute_metrics
from src.utils import ExperimentLogger


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for a single epoch."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / n_batches


def validate(model, dataloader, criterion, device):
    """Run validation and return loss + metrics."""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels_float = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels_float)

            running_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())

    avg_loss = running_loss / n_batches
    y_true = np.array(all_labels)
    y_proba = np.array(all_probs)
    metrics = compute_metrics(y_true, y_proba)

    return avg_loss, metrics


def train_model(
    model,
    dataloaders: dict,
    criterion,
    optimizer,
    device,
    num_epochs: int = 15,
    scheduler=None,
    patience: int = 5,
    experiment_name: str = "experiment",
    save_dir: str = "results/checkpoints",
    verbose: bool = True,
) -> dict:
    """Full training loop with early stopping and checkpointing.

    Returns a dictionary with training history, best model state, and final metrics.
    """
    logger = ExperimentLogger(experiment_name)
    best_model_state = copy.deepcopy(model.state_dict())
    best_auroc = 0.0
    epochs_no_improve = 0

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_metrics = validate(model, dataloaders["val"], criterion, device)

        if scheduler is not None:
            scheduler.step()

        logger.log_epoch(epoch, train_loss, val_loss, val_metrics)

        current_auroc = val_metrics["auroc"]
        improved = current_auroc > best_auroc

        if improved:
            best_auroc = current_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_state, save_path / f"{experiment_name}_best.pt")
        else:
            epochs_no_improve += 1

        if verbose:
            status = "*" if improved else ""
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {current_auroc:.4f} | "
                f"F1: {val_metrics['f1_macro']:.4f} {status}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    model.load_state_dict(best_model_state)

    summary = logger.get_summary()
    summary["training_time_seconds"] = elapsed
    summary["final_model_path"] = str(save_path / f"{experiment_name}_best.pt")

    return summary


def run_experiment(
    model_name: str,
    model,
    dataloaders: dict,
    device,
    config: dict,
    experiment_name: str = None,
) -> dict:
    """Convenience wrapper that sets up criterion, optimizer, scheduler and trains."""
    from src.models import get_optimizer, FocalLoss

    if experiment_name is None:
        experiment_name = model_name

    use_focal = config.get("model", {}).get("use_focal_loss", False) and model_name == "densenet_attention"

    if use_focal:
        gamma = config["model"].get("focal_loss_gamma", 2.0)
        alpha = config["model"].get("focal_loss_alpha", 0.6)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        train_info = dataloaders["info"]["train_class_dist"]
        n_pos = train_info.get(1, 1)
        n_neg = train_info.get(0, 1)
        pos_weight = torch.tensor([n_neg / n_pos]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]
    optimizer = get_optimizer(model, model_name, lr=lr, weight_decay=wd)

    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])

    model = model.to(device)

    results = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config["training"]["num_epochs"],
        scheduler=scheduler,
        patience=config["training"]["early_stopping_patience"],
        experiment_name=experiment_name,
    )

    return results
