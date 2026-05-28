# Pneumonia Detection from Chest X-Rays

A medical image analysis project implementing and comparing multiple deep learning approaches for binary classification of chest X-ray images (Normal vs. Pneumonia), framed as a clinical decision support system.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methods](#methods)
- [Evaluation Protocol](#evaluation-protocol)
- [Ablation Studies](#ablation-studies)
- [Threshold Analysis](#threshold-analysis)
- [Bias Investigation](#bias-investigation)
- [Ethical Considerations](#ethical-considerations)

## Key Results

Best model on the held-out test set (n = 624, threshold = 0.5):

| Metric      | DenseNet-Attention | ResNet-18 |
|-------------|-------------------:|----------:|
| AUROC       | 0.986              | 0.988     |
| F1 macro    | 0.870              | 0.885     |
| Accuracy    | 0.886              | 0.899     |
| Sensitivity | 0.992              | 0.995     |
| Specificity | 0.709              | 0.739     |
| NPV         | 0.982              | 0.989     |

At the Youden-optimal threshold (0.88) the same DenseNet-Attention model reaches **sensitivity 0.951 and specificity 0.953** — a much more balanced operating point that we recommend over the default 0.5 (see [Threshold Analysis](#threshold-analysis)).

**Finding from the component ablation:** a plain DenseNet-121 (without the attention module and without focal loss) reaches F1 = 0.937 and specificity = 0.876 at threshold 0.5 — outperforming the full student-developed model on F1, specificity and accuracy with statistically tied AUROC. We report this rather than over-claim improvement.

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
│   ├── 01_data_exploration.ipynb     # Class distribution, sample images, bias check
│   ├── 02_baseline_experiments.ipynb # Logistic regression + shallow CNN + 3 random baselines
│   ├── 03_model_experiments.ipynb    # ResNet-18 and DenseNet-Attention training
│   ├── 04_ablation_studies.ipynb     # 6 ablation dimensions + 5-seed stability
│   ├── 05_results_analysis.ipynb     # Final test-set evaluation, threshold sweep
│   └── 06_error_analysis.ipynb       # Calibration, FP/FN cases, Grad-CAM
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

Six ablation dimensions on the student-developed DenseNet-Attention model. Full numbers in `results/ablation_results_*.json`; headline takeaways:

| # | Dimension                | Best configuration       | Headline finding |
|---|--------------------------|--------------------------|------------------|
| 1 | Data augmentation        | `heavy` (or `standard`)  | None / basic collapse to majority-class prediction; heavy edges standard on F1 and specificity |
| 2 | Training-set size        | 75%                      | Monotonic gain 25% → 75%; the dip at 100% is most likely single-seed noise |
| 3 | Model components         | **Plain DenseNet** (no attention, no focal) | Wins on F1 (0.937) and specificity (0.876); the "full" student model loses ~6 F1 points |
| 4 | Image size               | 224×224                  | Matches ImageNet pretraining; 256×256 buys nothing and slightly hurts specificity |
| 5 | Optimizer                | **AdamW**                | AUROC 0.988, F1 0.896 vs Adam 0.978 / 0.697 — free win, recommend switching the default |
| 6 | Random-seed stability    | 5 seeds                  | AUROC 0.987 ± 0.005 — sensitivity is rock-solid (± 0.001), specificity is the noisiest |

## Threshold Analysis

The same trained DenseNet-Attention model evaluated at five thresholds (`notebooks/05_results_analysis.ipynb`):

| Threshold | Sensitivity | Specificity | F1 macro | NPV   | Clinical use case            |
|-----------|------------:|------------:|---------:|------:|------------------------------|
| 0.30      | 0.997       | 0.564       | 0.801    | 0.992 | Triage / screening           |
| 0.50      | 0.992       | 0.709       | 0.870    | 0.982 | Default                      |
| 0.70      | 0.985       | 0.833       | 0.921    | 0.970 | Good trade-off               |
| **0.85**  | **0.964**   | **0.910**   | **0.940**| 0.938 | **Best F1 in the sweep**     |
| 0.95      | 0.818       | 0.983       | 0.877    | 0.764 | Very conservative            |

The Youden-optimal threshold computed analytically is **0.88** (sensitivity 0.951, specificity 0.953). Headline numbers in [Key Results](#key-results) are at the default 0.5; for clinically meaningful comparison the Youden point is more representative.

## Bias Investigation

Following the midterm-review feedback we investigate whether the model can shortcut on data-collection artifacts rather than pathology. See `notebooks/01_data_exploration.ipynb` (sections 5 and 6) and `docs/problem_definition.md`. Two checks:

1. **Class-stratified image dimensions** — width / height / aspect-ratio distributions split by class. Cohen's d is reported. Mitigation: all images are resized to 224×224 before training, so absolute dimensions are not visible to the model.
2. **Filename pattern leakage** — NORMAL files use `IM-XXXX-XXXX.jpeg` while PNEUMONIA files use `personN_bacteria_M.jpeg` / `personN_virus_M.jpeg`. The model never sees filenames so this is not direct leakage, but the distinct naming conventions indicate the two classes were processed through different pipelines, meaning any residual collection-process artifact is a potential shortcut.

## Ethical Considerations

- This is a **decision-support tool**, not a replacement for clinical judgment.
- The dataset is single-center pediatric (Guangzhou, ages 1–5). Generalization to adults or other geographies is **not** validated.
- False negatives carry serious clinical harm; we therefore prefer thresholds biased toward sensitivity (~0.30–0.50 for triage) and report Youden-optimal numbers for balanced reporting.
- All source images are de-identified; no PII is handled at any stage.
- Class-correlated collection process (see [Bias Investigation](#bias-investigation)) is a known limitation. We mitigate via resize + normalization but cannot fully eliminate residual shortcut risk.

## References

- Kermany, D. et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." Cell, 172(5), 1122-1131.
- He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
- Huang, G. et al. (2017). "Densely Connected Convolutional Networks." CVPR.
- Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

## License

This project is for educational purposes as part of the Medical Image Analysis module.
