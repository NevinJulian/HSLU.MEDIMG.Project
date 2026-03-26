"""Evaluation metrics, Grad-CAM visualization, and analysis utilities."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from pathlib import Path


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all quantitative metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "auroc": roc_auc_score(y_true, y_pred_proba),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_normal": f1_score(y_true, y_pred, pos_label=0),
        "f1_pneumonia": f1_score(y_true, y_pred, pos_label=1),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Find threshold maximizing Youden's J statistic and the sensitivity-focused threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    youden_threshold = thresholds[optimal_idx]

    sensitivity_mask = tpr >= 0.95
    if sensitivity_mask.any():
        valid_thresholds = thresholds[sensitivity_mask]
        sensitivity_threshold = valid_thresholds[-1]
    else:
        sensitivity_threshold = thresholds[0]

    return {
        "youden_threshold": float(youden_threshold),
        "youden_sensitivity": float(tpr[optimal_idx]),
        "youden_specificity": float(1 - fpr[optimal_idx]),
        "sensitivity95_threshold": float(sensitivity_threshold),
    }


def evaluate_model(model, dataloader, device, threshold: float = 0.5) -> dict:
    """Run model evaluation on a dataloader and return metrics + predictions."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())

    y_true = np.array(all_labels)
    y_proba = np.array(all_probs)

    metrics = compute_metrics(y_true, y_proba, threshold)
    threshold_analysis = find_optimal_threshold(y_true, y_proba)

    return {
        "metrics": metrics,
        "threshold_analysis": threshold_analysis,
        "y_true": y_true,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5, title="Confusion Matrix", ax=None):
    """Plot a confusion matrix heatmap."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc_curve(results_dict: dict, ax=None):
    """Plot ROC curves for multiple models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        auroc = res["metrics"]["auroc"]
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_training_history(history: dict, title: str = "Training History"):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    val_aurocs = [m["auroc"] for m in history["val_metrics"]]
    axes[1].plot(epochs, val_aurocs, label="Val AUROC", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title(f"{title} - AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_metrics_comparison(all_results: dict):
    """Create a bar chart comparing key metrics across all models."""
    metrics_to_plot = ["auroc", "f1_macro", "sensitivity", "specificity", "npv"]
    model_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)

    for i, name in enumerate(model_names):
        values = [all_results[name]["metrics"][m] for m in metrics_to_plot]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def generate_gradcam(model, images, target_layer, device):
    """Generate Grad-CAM heatmaps for a batch of images.

    Returns numpy array of heatmaps with shape (B, H, W).
    """
    model.eval()
    images = images.to(device)
    images.requires_grad_(True)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    output = model(images).squeeze()
    if output.dim() == 0:
        output = output.unsqueeze(0)

    model.zero_grad()
    target = torch.sigmoid(output)
    target.sum().backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0]
    grad = gradients[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
    cam = torch.relu(cam)

    cam_min = cam.flatten(1).min(dim=1).values.view(-1, 1, 1)
    cam_max = cam.flatten(1).max(dim=1).values.view(-1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam.cpu().numpy()


def plot_gradcam_grid(images, heatmaps, labels, predictions, n_samples=8):
    """Plot a grid of images with Grad-CAM overlays."""
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    class_names = ["Normal", "Pneumonia"]

    for i in range(n):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        heatmap = heatmaps[i]
        import cv2
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"True: {class_names[labels[i]]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(img)
        axes[1, i].imshow(heatmap_resized, cmap="jet", alpha=0.4)
        pred_label = class_names[int(predictions[i] >= 0.5)]
        axes[1, i].set_title(f"Pred: {pred_label} ({predictions[i]:.2f})", fontsize=9)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)

    plt.tight_layout()
    return fig
