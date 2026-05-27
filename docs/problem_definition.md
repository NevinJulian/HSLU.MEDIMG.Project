# Problem Definition: Pneumonia Detection from Chest X-Rays

## Clinical Context

Pneumonia is one of the leading causes of morbidity and mortality worldwide, particularly
among children under 5 and adults over 65. The World Health Organization estimates that
pneumonia kills approximately 2.5 million people annually. Early and accurate diagnosis is
critical for timely treatment.

Chest X-ray (CXR) is the most common first-line imaging modality for pneumonia diagnosis.
It is widely available, relatively inexpensive, and involves lower radiation exposure compared
to CT scans. However, CXR interpretation requires trained radiologists, and in many healthcare
settings -- especially in low-resource environments -- there is a significant shortage of
qualified readers.

## Motivation

An automated decision support system for pneumonia detection from CXRs can:

- **Triage assistance**: Flag suspicious cases for priority review by radiologists
- **Screening support**: Aid in large-scale screening programs, particularly in underserved areas
- **Consistency**: Reduce inter-reader variability in CXR interpretation
- **Workload reduction**: Help manage increasing imaging volumes

This system is explicitly designed as a **support tool** -- it does not replace clinical
judgment but provides a second opinion to assist the decision-making process.

## Dataset: Chest X-Ray Images (Pneumonia)

### Source
Kermany, D. et al. (2018), hosted on Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Description
- 5,863 JPEG images of pediatric chest X-rays (anterior-posterior view)
- Collected from Guangzhou Women and Children's Medical Center
- Patient age group: 1-5 years
- Binary classification: NORMAL (1,583 images) vs. PNEUMONIA (4,273 images)
- Pneumonia cases include both bacterial and viral subtypes

### Dataset Relevance
This dataset is relevant because:
1. It comes from a published, peer-reviewed study (Cell, 2018)
2. Labels were assigned by expert physicians and verified by a separate review panel
3. The binary classification task is well-defined and clinically meaningful
4. The class imbalance (~73% pneumonia) reflects real-world clinical prevalence patterns
5. It has been widely used as a benchmark in medical imaging literature

### Known Limitations
- **Single center**: All images from one hospital, limiting generalizability
- **Pediatric only**: Results may not transfer to adult populations
- **Small validation set**: Original split has only 16 validation images (we re-split)
- **No severity grading**: Binary labels do not capture disease severity
- **Mixed pneumonia types**: Bacterial and viral pneumonia are grouped together
- **Image quality variation**: Some images have artifacts or suboptimal exposure

## Bias Investigation

In response to midterm feedback, we explicitly investigate whether the data
collection process introduces shortcuts that the model could exploit instead
of learning real pathology features. The full analysis is in
`notebooks/01_data_exploration.ipynb` (sections 5 and 6); the headline checks
are summarized here.

### 1. Class-stratified image dimensions

The dataset combines images from potentially different scanners and acquisition
settings. If NORMAL and PNEUMONIA images have systematically different sizes
or aspect ratios, the model can shortcut on dimensions instead of lung
anatomy. We therefore plot width / height / aspect-ratio distributions
**stratified by class** (not just overall, as in the original midterm version)
and report Cohen's d to quantify how separable the two classes are by
dimension alone.

- **Mitigation in place**: `src/data.py` resizes every image to a fixed shape
  (224×224 by default) before training. Image dimensions are therefore not
  directly visible to any model.
- **Residual risk**: pixel-level artifacts that survive a resize (border
  markings, exposure differences, sensor noise patterns) can still leak the
  acquisition source. Cohen's d values reported in notebook 01 should be
  cited in the final presentation's limitations slide.

### 2. Filename pattern leakage

NORMAL and PNEUMONIA images follow distinct filename conventions:

- NORMAL files: `IM-XXXX-XXXX.jpeg`
- PNEUMONIA files: `personN_bacteria_M.jpeg` or `personN_virus_M.jpeg`

The model never sees filenames, so this is not direct leakage. It is, however,
a clear signal that the two classes were collected through different pipelines.
We compute a "filename-only classifier" accuracy in notebook 01: if filenames
alone reach near-perfect accuracy, *something* in the acquisition process
correlates with the label, and any artifact that survives the resize is a
potential shortcut.

### 3. Random / trivial baselines

We compare every learned model against three non-learning baselines (majority
class, stratified random, uniform random) in `notebooks/02_baseline_experiments.ipynb`
section 6. Any deep model that fails to beat these on F1-macro or NPV is not
actually learning anything useful — this is the floor against which the
sensitivity-saturated outputs of the imbalanced training regime should be
judged.
