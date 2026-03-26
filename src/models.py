"""Model architectures for pneumonia classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ShallowCNN(nn.Module):
    """A simple 3-layer CNN baseline."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention block."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseNetAttention(nn.Module):
    """DenseNet-121 backbone with channel attention and a custom classifier head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        self.backbone = densenet.features
        self.feature_dim = 1024

        if use_attention:
            self.attention = ChannelAttention(self.feature_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = F.relu(features, inplace=True)

        if self.use_attention:
            features = self.attention(features)

        x = self.pool(features).view(features.size(0), -1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """Return feature maps before pooling (for Grad-CAM)."""
        features = self.backbone(x)
        features = F.relu(features, inplace=True)
        if self.use_attention:
            features = self.attention(features)
        return features


class ResNet18Finetune(nn.Module):
    """Fine-tuned ResNet-18 with replaced classifier head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.resnet(x)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.6, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function for model creation."""
    models_map = {
        "shallow_cnn": ShallowCNN,
        "resnet18_finetune": ResNet18Finetune,
        "densenet_attention": DenseNetAttention,
    }

    if model_name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_map.keys())}")

    return models_map[model_name](**kwargs)


def get_optimizer(model: nn.Module, model_name: str, lr: float = 0.001, weight_decay: float = 0.0001):
    """Create optimizer with differential learning rates for pretrained models."""
    if model_name in ("resnet18_finetune", "densenet_attention"):
        if model_name == "resnet18_finetune":
            backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
            head_params = [p for n, p in model.named_parameters() if "fc" in n]
        else:
            backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
            head_params = [p for n, p in model.named_parameters()
                          if "backbone" not in n]

        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ]
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
