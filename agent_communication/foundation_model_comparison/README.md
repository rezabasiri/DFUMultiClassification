# Foundation Model Comparison for DFU Healing Phase Classification

Compares DINOv2 and BiomedCLIP foundation models against FUSE4DFU (DenseNet121 + RF fusion) on the same dataset and CV splits.

## Approach

Both models use **frozen feature extraction** (no fine-tuning of the foundation model). Features are extracted once per image, then a lightweight classifier (logistic regression or MLP) is trained on the concatenated features.

This is the standard evaluation protocol for foundation models on small datasets, testing the quality of their pretrained representations.

## Experiments

| # | Model | Modalities | Metadata | Classifier | Notes |
|---|-------|-----------|----------|------------|-------|
| 1 | DINOv2-B | RGB+Depth+Thermal | No | LogReg | Baseline |
| 2 | DINOv2-B | RGB+Depth+Thermal | Numeric | LogReg | + metadata features |
| 3 | DINOv2-B | RGB only | No | LogReg | Single modality |
| 4 | DINOv2-B | RGB+Depth+Thermal | No | MLP | Nonlinear head |
| 5 | DINOv2-B | RGB+Depth+Thermal | Numeric | MLP | Best config |
| 6 | BiomedCLIP | RGB+Depth+Thermal | No | LogReg | Baseline |
| 7 | BiomedCLIP | RGB+Depth+Thermal | Text | LogReg | Metadata as text |
| 8 | BiomedCLIP | RGB+Depth+Thermal | Numeric | LogReg | + metadata features |
| 9 | BiomedCLIP | RGB only | No | LogReg | Single modality |
| 10 | BiomedCLIP | RGB+Depth+Thermal | Numeric | MLP | Best config |

## FUSE4DFU Reference (our model)

| Configuration | Kappa | Accuracy |
|--------------|-------|----------|
| Best combo (Meta+RGB+Thermal) | 0.58 | 73.3% |
| Optimized ensemble (4 combos) | 0.65 | 80.6% |

## Setup

```bash
pip install torch torchvision open_clip_torch transformers
```

## Running

```bash
cd agent_communication/foundation_model_comparison/scripts

# Run all experiments
bash run_all.sh

# Run only DINOv2 or BiomedCLIP
bash run_all.sh dinov2
bash run_all.sh biomedclip

# Individual experiment
python run_dinov2.py --modalities depth_rgb --classifier logreg
python run_biomedclip.py --modalities depth_rgb thermal_map --include_metadata_numeric
```

## Directory Structure

```
foundation_model_comparison/
  README.md
  scripts/
    data_loader.py       Shared data loading (reuses same CV splits)
    run_dinov2.py        DINOv2 experiments
    run_biomedclip.py    BiomedCLIP experiments
    run_all.sh           Run all experiments
  results/               Per-experiment CSV and JSON results
  logs/                  Execution logs
```

## Key Differences from FUSE4DFU

| Aspect | FUSE4DFU | Foundation Models |
|--------|----------|------------------|
| Image backbone | DenseNet121 (fine-tuned) | DINOv2/BiomedCLIP (frozen) |
| Training | Two-stage (500+30 epochs) | Feature extraction only |
| Metadata | RF OOF probabilities | Raw features or text encoding |
| Fusion | Feature concatenation + FC | Simple concatenation + classifier |
| Parameters trained | ~8M (backbone + heads) | ~10K (classifier only) |
