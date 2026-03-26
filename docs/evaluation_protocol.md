# Evaluation Protocol

## Learning Task

**Task**: Binary classification of chest X-ray images into NORMAL vs. PNEUMONIA.

**Clinical framing**: This is a screening/triage task. The system outputs a probability
score indicating the likelihood of pneumonia. In a clinical workflow, cases above a
certain threshold would be flagged for priority radiologist review.

## Data Splits

The original Kaggle dataset has a problematic split (only 16 validation images).
We address this by:

1. Merging the original train and val sets
2. Creating a stratified 85/15 train/val split from the merged data
3. Keeping the original test set untouched for final evaluation

This ensures:
- Sufficient validation data for model selection and hyperparameter tuning
- An untouched held-out test set for unbiased final evaluation
- Stratified splitting to preserve class ratios

## Quantitative Metrics

### Primary Metrics

- **AUROC (Area Under the ROC Curve)**: Threshold-independent measure of discrimination.
  Appropriate because it evaluates the full range of sensitivity/specificity trade-offs.

- **F1-Score (per-class and macro)**: Harmonic mean of precision and recall.
  Important given the class imbalance -- accuracy alone would be misleading.

### Secondary Metrics

- **Sensitivity (Recall for PNEUMONIA)**: Proportion of actual pneumonia cases correctly
  identified. Clinically the most important metric -- missing a pneumonia case (false
  negative) has serious consequences.

- **Specificity (Recall for NORMAL)**: Proportion of healthy patients correctly identified.
  Important for reducing unnecessary follow-up procedures.

- **Precision**: Proportion of flagged cases that are true positives. Relevant for
  managing the radiologist's workload.

- **Accuracy**: Overall correctness. Reported but interpreted with caution due to
  class imbalance.

### Clinical Metric

- **NPV (Negative Predictive Value)**: Probability that a negative prediction is
  truly negative. Critical for a screening system -- a high NPV means patients
  predicted as normal can be safely de-prioritized.

### Threshold Selection

For the clinical decision support system, we analyze performance across multiple
thresholds and report:
- Metrics at the default 0.5 threshold
- Metrics at the threshold that maximizes the Youden index (sensitivity + specificity - 1)
- The sensitivity-focused operating point (>= 95% sensitivity)

## Qualitative Analysis

### Grad-CAM Visualizations
We use Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which
regions of the X-ray the model focuses on. Clinically valid models should attend to:
- Lung parenchyma (the functional tissue of the lungs)
- Areas of consolidation or opacity in pneumonia cases
- Should NOT focus primarily on image borders, labels, or artifacts

### Error Analysis
We conduct systematic error analysis:
- Examine false negatives (missed pneumonia) -- most clinically dangerous
- Examine false positives (false alarms) -- impact on workflow
- Look for patterns in misclassified images (image quality, atypical presentations)

### Confusion Matrix Analysis
Visual confusion matrices for each model to compare error patterns and trade-offs.

## Robustness Evaluation

### Ablation Studies (>= 3 aspects)
1. **Data augmentation impact**: none / basic / standard / heavy
2. **Training set size**: 25%, 50%, 75%, 100%
3. **Model components**: attention module and focal loss (with/without)
4. **Learning rate sensitivity**: tested across orders of magnitude

### Multiple Runs
Each configuration is run with 5 different random seeds to assess:
- Mean and standard deviation of all metrics
- Stability of training convergence
- Consistency of Grad-CAM attention patterns

## Comparison Protocol

All models are compared on the same test set with the same preprocessing pipeline.
Statistical significance is assessed via confidence intervals from multiple runs.
