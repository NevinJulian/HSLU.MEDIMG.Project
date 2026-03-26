# Pneumonia Detection from Chest X-Rays

A medical image analysis project implementing and comparing multiple deep learning approaches for binary classification of chest X-ray images (Normal vs. Pneumonia), framed as a clinical decision support system.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methods](#methods)
- [Evaluation Protocol](#evaluation-protocol)
- [Ablation Studies](#ablation-studies)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)

## Project Overview

This project addresses the task of automated pneumonia detection from posteroanterior (PA) chest X-ray images. It is designed as a **clinical decision support system** that assists radiologists in triaging and flagging potential pneumonia cases -- not as a standalone diagnostic tool.

We implement and compare four approaches:
1. **Baseline 1**: Logistic Regression on flattened pixel features
2. **Baseline 2**: A shallow CNN trained from scratch
3. **Model 1**: Fine-tuned ResNet-18 (published reference architecture)
4. **Model 2**: Custom DenseNet-based architecture with attention (student-developed)

## Dataset

**Chest X-Ray Images (Pneumonia)** from Kaggle, originally curated by Kermany et al. (2018).

- Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- 5,863 JPEG images across train / val / test splits
- Binary labels: NORMAL vs. PNEUMONIA
- Significant class imbalance (~74% pneumonia in train)
- Images are grayscale-compatible PA chest X-rays of pediatric patients (ages 1-5)

**Important**: The original validation set is very small (16 images). We create a proper validation split from the training data (see `src/data.py`).

### Download Instructions

1. Download the dataset from Kaggle:
   ```bash
   # Option A: Kaggle CLI
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/

   # Option B: Manual download from the Kaggle link above and unzip to data/
   ```

2. Expected directory layout after extraction:
   ```
   data/
   └── chest_xray/
       ├── train/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       ├── val/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── test/
           ├── NORMAL/
           └── PNEUMONIA/
   ```

## Setup & Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.10+
- uv (`pip install uv` or see https://docs.astral.sh/uv/getting-started/installation/)
- CUDA-capable GPU recommended (CPU training is possible but slow)

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd pneumonia-detection

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install all dependencies
uv pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
pneumonia-detection/
├── README.md                   # This file
├── pyproject.toml              # Project metadata
├── requirements.txt            # Pinned dependencies
├── configs/
│   └── default.yaml            # Default training configuration
├── src/
│   ├── __init__.py
│   ├── data.py                 # Dataset loading, splits, augmentation
│   ├── models.py               # All model architectures
│   ├── train.py                # Training loop and evaluation
│   ├── evaluate.py             # Metrics computation and analysis
│   └── utils.py                # Seed setting, logging, helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_model_experiments.ipynb
│   ├── 04_ablation_studies.ipynb
│   └── 05_results_analysis.ipynb
├── scripts/
│   ├── run_all_experiments.sh  # Reproduce all experiments
│   └── run_ablation.sh         # Reproduce ablation studies
├── docs/
│   ├── problem_definition.md   # Clinical context and motivation
│   ├── evaluation_protocol.md  # Metrics and evaluation methodology
│   └── methods.md              # Description of all methods
└── results/                    # Generated experiment outputs
```

## Usage

### Run All Experiments

```bash
# Full reproduction of all experiments
bash scripts/run_all_experiments.sh

# Or run individual notebooks in order:
jupyter notebook notebooks/
```

### Quick Single Training Run

```python
from src.train import train_model
from src.models import get_model
from src.data import get_dataloaders

dataloaders = get_dataloaders("data/chest_xray", augmentation="standard")
model = get_model("resnet18_finetune")
results = train_model(model, dataloaders, num_epochs=15)
```

### Configuration

All hyperparameters are centralized in `configs/default.yaml`. Override via CLI or notebook.

## Methods

See [docs/methods.md](docs/methods.md) for detailed descriptions. Summary:

| Method | Type | Description |
|--------|------|-------------|
| Logistic Regression | Baseline | Linear model on resized+flattened images |
| Shallow CNN | Baseline | 3-layer CNN trained from scratch |
| ResNet-18 Fine-tune | Published | ImageNet-pretrained ResNet-18 with replaced classifier |
| DenseNet-Attention | Student | Custom DenseNet-121 backbone + channel attention + focal loss |

## Evaluation Protocol

See [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for full details.

### Quantitative Metrics
- **Primary**: AUROC, F1-Score (macro and per-class)
- **Secondary**: Sensitivity (Recall), Specificity, Precision, Accuracy
- **Clinical**: Negative Predictive Value (NPV) -- critical for ruling out pneumonia

### Qualitative Analysis
- Grad-CAM visualizations to verify model focus on lung regions
- Confusion matrix analysis
- Error case analysis (false negatives are clinically more dangerous)

## Ablation Studies

Conducted on the student-developed DenseNet-Attention model:

1. **Data Augmentation**: None vs. basic (flip/rotate) vs. full (color jitter, elastic transforms)
2. **Training Set Size**: 25%, 50%, 75%, 100% of training data
3. **Model Components**: With/without channel attention module, with/without focal loss
4. **Multiple Runs**: 5 runs with different random seeds per configuration

## Ethical Considerations

- This system is designed as a **decision support tool**, not a replacement for clinical judgment
- The dataset consists of pediatric patients; generalization to adult populations is not validated
- Class imbalance and the high cost of false negatives must be carefully managed
- Patient privacy: all images are de-identified in the source dataset
- Potential bias: single-center data from a specific demographic group

## References

- Kermany, D. et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." Cell, 172(5), 1122-1131.
- He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
- Huang, G. et al. (2017). "Densely Connected Convolutional Networks." CVPR.
- Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

## License

This project is for educational purposes as part of the Medical Image Analysis module.
