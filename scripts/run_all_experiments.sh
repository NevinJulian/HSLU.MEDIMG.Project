#!/bin/bash
# Run all experiments for the pneumonia detection project.
# This script reproduces all results from scratch.

set -e

echo "=============================================="
echo "Pneumonia Detection - Full Experiment Pipeline"
echo "=============================================="

# Check that the dataset exists
if [ ! -d "data/chest_xray" ]; then
    echo "ERROR: Dataset not found at data/chest_xray/"
    echo "Please download it from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
    echo "and extract it to data/chest_xray/"
    exit 1
fi

# Create results directory
mkdir -p results/checkpoints

echo ""
echo "[1/5] Running data exploration..."
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb \
    --output 01_data_exploration_executed.ipynb \
    --ExecutePreprocessor.timeout=600

echo ""
echo "[2/5] Running baseline experiments..."
jupyter nbconvert --to notebook --execute notebooks/02_baseline_experiments.ipynb \
    --output 02_baseline_experiments_executed.ipynb \
    --ExecutePreprocessor.timeout=3600

echo ""
echo "[3/5] Running model experiments..."
jupyter nbconvert --to notebook --execute notebooks/03_model_experiments.ipynb \
    --output 03_model_experiments_executed.ipynb \
    --ExecutePreprocessor.timeout=7200

echo ""
echo "[4/5] Running ablation studies..."
jupyter nbconvert --to notebook --execute notebooks/04_ablation_studies.ipynb \
    --output 04_ablation_studies_executed.ipynb \
    --ExecutePreprocessor.timeout=14400

echo ""
echo "[5/5] Running results analysis..."
jupyter nbconvert --to notebook --execute notebooks/05_results_analysis.ipynb \
    --output 05_results_analysis_executed.ipynb \
    --ExecutePreprocessor.timeout=600

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results saved in results/"
echo "=============================================="
