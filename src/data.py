"""Dataset loading, augmentation pipelines, and data split management."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class ChestXrayDataset(Dataset):
    """Chest X-ray dataset for pneumonia detection."""

    def __init__(self, image_paths: list, labels: list, transform=None, grayscale: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L" if self.grayscale else "RGB")

        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                image = self.transform(image)

        return image, label


def _collect_image_paths_and_labels(data_dir: str):
    """Walk a directory with NORMAL/ and PNEUMONIA/ subdirectories."""
    paths, labels = [], []
    data_dir = Path(data_dir)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpeg", ".jpg", ".png"):
                paths.append(str(img_file))
                labels.append(class_idx)

    return paths, labels


def get_transforms(augmentation_level: str = "standard", image_size: int = 224, use_albumentations: bool = True):
    """Return train and test transform pipelines based on augmentation level.

    Levels:
        - "none": resize and normalize only
        - "basic": horizontal flip and small rotation
        - "standard": flip, rotation, brightness/contrast, random crop
        - "heavy": standard + elastic transform, grid distortion, coarse dropout
    """
    if use_albumentations:
        return _get_albumentations_transforms(augmentation_level, image_size)
    return _get_torchvision_transforms(augmentation_level, image_size)


def _get_albumentations_transforms(level: str, size: int):
    """Build albumentations pipelines."""
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    test_transform = A.Compose([
        A.Resize(size, size),
        normalize,
        ToTensorV2(),
    ])

    if level == "none":
        train_transform = test_transform
    elif level == "basic":
        train_transform = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            normalize,
            ToTensorV2(),
        ])
    elif level == "standard":
        train_transform = A.Compose([
            A.RandomResizedCrop(size, size, scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            normalize,
            ToTensorV2(),
        ])
    elif level == "heavy":
        train_transform = A.Compose([
            A.RandomResizedCrop(size, size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ElasticTransform(alpha=40, sigma=40 * 0.05, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.CoarseDropout(max_holes=4, max_height=30, max_width=30, p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            normalize,
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {level}")

    return train_transform, test_transform


def _get_torchvision_transforms(level: str, size: int):
    """Fallback torchvision pipelines."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])

    if level == "none":
        train_transform = test_transform
    elif level == "basic":
        train_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    return train_transform, test_transform


def get_dataloaders(
    root_dir: str,
    augmentation: str = "standard",
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    train_fraction: float = 1.0,
    use_albumentations: bool = True,
) -> dict:
    """Create train, validation, and test dataloaders.

    The original val set is merged into train and a new stratified val split is created.
    """
    root = Path(root_dir)

    train_paths, train_labels = _collect_image_paths_and_labels(root / "train")
    orig_val_paths, orig_val_labels = _collect_image_paths_and_labels(root / "val")
    test_paths, test_labels = _collect_image_paths_and_labels(root / "test")

    all_train_paths = train_paths + orig_val_paths
    all_train_labels = train_labels + orig_val_labels

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        all_train_paths, all_train_labels,
        test_size=val_split,
        stratify=all_train_labels,
        random_state=seed,
    )

    if train_fraction < 1.0:
        n_keep = max(1, int(len(tr_paths) * train_fraction))
        tr_paths, _, tr_labels, _ = train_test_split(
            tr_paths, tr_labels,
            train_size=n_keep,
            stratify=tr_labels,
            random_state=seed,
        )

    train_transform, test_transform = get_transforms(augmentation, image_size, use_albumentations)

    train_dataset = ChestXrayDataset(tr_paths, tr_labels, transform=train_transform)
    val_dataset = ChestXrayDataset(val_paths, val_labels, transform=test_transform)
    test_dataset = ChestXrayDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    dataset_info = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_class_dist": dict(zip(*np.unique(tr_labels, return_counts=True))),
        "val_class_dist": dict(zip(*np.unique(val_labels, return_counts=True))),
        "test_class_dist": dict(zip(*np.unique(test_labels, return_counts=True))),
        "class_names": CLASS_NAMES,
    }

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "info": dataset_info,
    }


def get_flat_features(root_dir: str, image_size: int = 64, val_split: float = 0.15, seed: int = 42):
    """Load images as flattened numpy arrays for the logistic regression baseline.

    Images are converted to grayscale and resized to image_size x image_size.
    """
    root = Path(root_dir)

    train_paths, train_labels = _collect_image_paths_and_labels(root / "train")
    orig_val_paths, orig_val_labels = _collect_image_paths_and_labels(root / "val")
    test_paths, test_labels = _collect_image_paths_and_labels(root / "test")

    all_train_paths = train_paths + orig_val_paths
    all_train_labels = train_labels + orig_val_labels

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        all_train_paths, all_train_labels,
        test_size=val_split,
        stratify=all_train_labels,
        random_state=seed,
    )

    def load_flat(paths):
        features = []
        for p in paths:
            img = Image.open(p).convert("L").resize((image_size, image_size))
            arr = np.array(img, dtype=np.float32).flatten() / 255.0
            features.append(arr)
        return np.stack(features)

    X_train = load_flat(tr_paths)
    X_val = load_flat(val_paths)
    X_test = load_flat(test_paths)

    return {
        "X_train": X_train, "y_train": np.array(tr_labels),
        "X_val": X_val, "y_val": np.array(val_labels),
        "X_test": X_test, "y_test": np.array(test_labels),
    }
