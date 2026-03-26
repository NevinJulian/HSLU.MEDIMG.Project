"""Utility functions for reproducibility, configuration, and logging."""

import os
import random
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: dict, experiment_name: str, output_dir: str = "results"):
    """Save experiment results as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = output_dir / filename

    serializable = _make_serializable(results)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    return filepath


def _make_serializable(obj):
    """Convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


def get_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset."""
    targets = []
    for _, label in dataset:
        targets.append(label)

    targets = np.array(targets)
    counts = np.bincount(targets)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class ExperimentLogger:
    """Simple logger that tracks metrics per epoch."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        self.best_metric = 0.0
        self.best_epoch = 0

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_metrics"].append(val_metrics)

        auroc = val_metrics.get("auroc", 0.0)
        if auroc > self.best_metric:
            self.best_metric = auroc
            self.best_epoch = epoch

    def get_summary(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "best_val_auroc": self.best_metric,
            "best_epoch": self.best_epoch,
            "num_epochs_trained": len(self.history["train_loss"]),
            "history": self.history,
        }
