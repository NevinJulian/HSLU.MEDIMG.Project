# Methods

## Overview

We implement four approaches spanning a range of complexity:
two baselines to establish lower bounds, and two deep learning models for comparison.

## Baseline 1: Logistic Regression

A simple linear classifier operating on flattened pixel values.

- Images resized to 64x64 and converted to grayscale
- Pixel values flattened into a 4096-dimensional feature vector
- L2-regularized logistic regression (scikit-learn)
- Serves as a sanity check: any reasonable deep model should exceed this

This baseline establishes what is achievable without learned feature representations.

## Baseline 2: Shallow CNN

A lightweight convolutional neural network trained from scratch.

Architecture:
- 3 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
- Channel progression: 1 -> 32 -> 64 -> 128
- Global average pooling
- Single fully connected layer with dropout
- Sigmoid output

Training:
- Binary cross-entropy loss with class weights
- Adam optimizer, lr=0.001
- 15 epochs with early stopping

This baseline tests whether a simple learned feature extractor suffices.

## Model 1: Fine-tuned ResNet-18 (Reference Implementation)

Based on He et al. (2016), using the torchvision pretrained implementation.

- ResNet-18 pretrained on ImageNet
- Final fully connected layer replaced (512 -> 1)
- All layers fine-tuned with a lower learning rate for pretrained layers (0.0001)
  and a higher rate for the new classifier head (0.001)
- Input images converted to 3-channel (grayscale replicated) to match ImageNet format

This represents a well-established transfer learning approach commonly used in
medical imaging literature.

## Model 2: DenseNet-Attention (Student-Developed)

A custom architecture combining a DenseNet-121 backbone with a channel attention
mechanism, trained with focal loss.

### Architecture
- **Backbone**: DenseNet-121 pretrained on ImageNet (feature extractor)
- **Channel Attention Module**: Squeeze-and-Excitation style block that learns to
  weight feature channels based on their relevance. This encourages the model to
  focus on the most discriminative features for pneumonia detection.
- **Classifier Head**: Dropout -> FC(1024, 256) -> ReLU -> Dropout -> FC(256, 1)
- **Output**: Sigmoid activation for binary probability

### Attention Mechanism
The channel attention module applies:
1. Global average pooling to produce a channel descriptor
2. A bottleneck MLP (reduction ratio = 16) to model channel interdependencies
3. Sigmoid gating to re-weight the feature map channels

This is motivated by the observation that not all feature channels contribute
equally to the classification -- some may encode texture patterns relevant to
lung consolidation, while others capture irrelevant background features.

### Focal Loss
Standard cross-entropy treats all examples equally, but in our imbalanced
setting, easy-to-classify normal cases dominate the gradient. Focal loss
(Lin et al., 2017) down-weights well-classified examples:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

We use gamma=2.0 and alpha=0.6 (higher weight for pneumonia class).

### Training Strategy
- Differential learning rates: backbone (0.0001), attention + head (0.001)
- Cosine annealing learning rate schedule
- Early stopping on validation AUROC with patience=5
- Weight decay 0.0001

### Rationale
The combination of DenseNet (strong feature reuse), attention (adaptive feature
selection), and focal loss (handling class imbalance) is designed to address
the specific challenges of this task: limited data, class imbalance, and the
need for interpretable attention patterns.
