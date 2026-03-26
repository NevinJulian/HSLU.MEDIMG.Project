#!/bin/bash
# Run only the ablation studies.
# Assumes the dataset is already in data/chest_xray/

set -e

echo "=============================================="
echo "Pneumonia Detection - Ablation Studies"
echo "=============================================="

if [ ! -d "data/chest_xray" ]; then
    echo "ERROR: Dataset not found at data/chest_xray/"
    exit 1
fi

mkdir -p results/checkpoints

echo "Running ablation studies notebook..."
jupyter nbconvert --to notebook --execute notebooks/04_ablation_studies.ipynb \
    --output 04_ablation_studies_executed.ipynb \
    --ExecutePreprocessor.timeout=14400

echo ""
echo "Ablation studies complete! Results in results/"
