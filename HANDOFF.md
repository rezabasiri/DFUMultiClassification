# Project Handoff: DFU Multi-Classification

## What This Project Does
Multimodal deep learning system for Diabetic Foot Ulcer (DFU) healing phase classification. Predicts 3 classes:
- Class 0 (I): Inflammatory
- Class 1 (P): Proliferative
- Class 2 (R): Remodeling

Combines 4 image modalities + clinical metadata through attention-based fusion, with an ensemble gating network.

## Architecture Overview

```
5 modalities → individual branches → attention fusion → classification
  metadata (RF probs)  ─┐
  depth_rgb (EfficientNet) ─┤
  depth_map (EfficientNet) ─┼─→ concat fusion → Dense(3, softmax) → predictions
  thermal_rgb (EfficientNet)─┤
  thermal_map (EfficientNet)─┘

Block A: Train all 31 modality combinations independently (3-fold CV)
Block B: Gating network ensembles Block A outputs (meta-learner)
```

### Training Flow (per modality combination)
1. **Pre-train**: Image backbone + projection head on image-only data (200 epochs, lr=1e-3)
2. **Stage 1**: Freeze image backbone, train fusion layers (50 epochs, lr=1e-4)
3. **Stage 2**: Unfreeze top 20% of backbone, fine-tune all (50 epochs, lr=1e-5)

### Metadata Branch
- Random Forest trained on clinical features produces calibrated class probabilities
- RF probabilities passed through as-is (no Dense layers on top)
- Fusion learns to weight RF vs image predictions

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point, orchestration, CLI args |
| `src/utils/production_config.py` | **ALL hyperparameters** (single source of truth) |
| `src/models/builders.py` | Model architecture construction |
| `src/data/dataset_utils.py` | Data pipeline, TF datasets, caching |
| `src/training/training_utils.py` | Training loops, CV, checkpointing |
| `src/models/losses.py` | Focal ordinal loss, custom metrics |
| `src/data/preprocessing.py` | Data normalization |
| `src/data/image_processing.py` | Image loading, bounding boxes |
| `src/utils/gpu_config.py` | GPU setup and distribution strategy |

## Per-Modality Configs (Validated by Hparam Search)

In `src/utils/production_config.py`, `MODALITY_CONFIGS` holds per-modality overrides:

| Modality | Backbone | Head | L2 | Finetune Epochs |
|----------|----------|------|-----|-----------------|
| depth_rgb | EfficientNetB2 | [256, 64] | 0.0 | 30 |
| depth_map | EfficientNetB2 | 64 | 0.001 | 30 |
| thermal_map | EfficientNetB0 | 128 | 0.0 | 30 |
| thermal_rgb | EfficientNetB2 | [256, 64] | 0.0 | 30 |

Search results are in `agent_communication/*/` subdirectories. depth_rgb and thermal_map were validated on full data (USE_CORE_DATA=False). **depth_map search was NOT re-run on full data** (machine CUDA failure interrupted it).

## How to Run

```bash
# Production run
python src/main.py --data_percentage 100 --device-mode multi --verbosity 2 --resume_mode fresh --cv_folds 3

# Quick test
python src/main.py --data_percentage 10 --cv_folds 1 --device-mode single --verbosity 2 --resume_mode fresh
```

Do NOT pass `--fold` manually. The orchestrator auto-spawns fold subprocesses when `cv_folds > 1`.

### Key CLI Args
- `--data_percentage`: Subset of data (10 for quick test, 100 for production)
- `--device-mode`: `cpu`, `single`, `multi`, `custom`
- `--resume_mode`: `fresh` (clean), `auto` (resume), `from_data`, `from_models`, `from_predictions`
- `--cv_folds`: Number of cross-validation folds (default 3)
- `--verbosity`: 0=minimal, 1=normal, 2=detailed, 3=full+progress

## Data

~600 patients, ~3100 total image sets (multiple visits per patient). Patient-level CV splitting prevents data leakage.

```
data/raw/
  DataMaster_Processed_V12_WithMissing.csv   # Master metadata
  bounding_box_depth.csv                      # Depth bounding boxes
  bounding_box_thermal.csv                    # Thermal bounding boxes
  Depth_RGB/                                  # Depth RGB images
  Depth_Map_IMG/                              # Depth map images
  Thermal_RGB/                                # Thermal RGB images
  Thermal_Map_IMG/                            # Thermal map images
```

## Uncommitted Changes (IMPORTANT)

The current branch `claude/fix-trainable-parameters-UehxS` has uncommitted changes in:

1. **`src/models/builders.py`**: Added two-layer projection head support. `head_units` now accepts a list (e.g., `[256, 64]`) for multi-layer heads, not just a single int.

2. **`src/utils/production_config.py`**: Updated `MODALITY_CONFIGS` with search-validated hyperparameters for each modality (backbone, head_units, head_l2, finetune_epochs).

These changes are critical. Commit them before doing anything else on the new machine.

## Pending Work

### 1. depth_map Hparam Search (Re-run Needed)
The depth_map search was run with `USE_CORE_DATA=True` (filtered to 2072 samples) but production uses `USE_CORE_DATA=False` (full 3108 samples). It needs to be re-run on full data to validate the current config.

```bash
python agent_communication/depth_map_pipeline_audit/depth_map_hparam_search.py --fresh
```

The current depth_map config (`EfficientNetB2, head=64, l2=0.001, finetune_epochs=30`) may or may not change after the full-data search.

### 2. Modality Error Correlation Analysis
Plan exists at `agent_communication/modality_error_correlation/TODO.md`. After standalone runs complete, create `scripts/modality_error_correlation.py` to measure whether modalities make complementary or redundant errors (predicts fusion benefit).

### 3. Production Run
After confirming all per-modality configs, run the full production pipeline with updated `MODALITY_CONFIGS`.

## Agent Communication Directory

Each subdirectory documents a completed or in-progress investigation:

| Directory | Status | Summary |
|-----------|--------|---------|
| `depth_rgb_pipeline_audit/` | Done | Hparam search validated EfficientNetB2, [256,64] head |
| `depth_map_pipeline_audit/` | Partial | Search done with core data, needs re-run on full data |
| `thermal_map_pipeline_audit/` | Done | BASELINE (B0, 128 head) validated on full data |
| `fusion_fix/` | Done | Fixed multi-stage training, backbone weight visibility |
| `generative_augmentation/` | Done | SDXL v3 synthetic data pipeline (currently disabled) |
| `metadata_diagnosis/` | Done | RF calibration, no-Dense-layer policy |
| `outlier_detection/` | Done | Isolation Forest, 25% contamination |
| `modality_error_correlation/` | Planned | Measure cross-modality error complementarity |

## Critical Design Decisions

1. **No Dense on RF probabilities**: RF outputs calibrated probabilities. Adding Dense layers on top degrades kappa from 0.20 to 0.109.

2. **Backbone called directly, not via Lambda**: Lambda wrapping hid backbone weights from the optimizer. Direct call exposes all weights for training.

3. **Patient-level CV splitting**: Prevents data leakage across folds. Same patient never appears in both train and validation.

4. **Per-fold subprocess isolation**: Each CV fold runs in a separate subprocess to prevent CUDA memory leaks between folds.

5. **Two-layer heads**: depth_rgb and thermal_rgb use `[256, 64]` projection heads (validated by search). builders.py loops over the list.

## Environment Notes

- Python 3.11.14
- TensorFlow 2.18.1 (Keras 3.13.2)
- PyTorch 2.9.1+cu128 (used only for generative augmentation, not main training)
- CUDA 12.8 (torch compiled), driver supports up to CUDA 13.0
- See `ENVIRONMENT_SETUP.md` for full setup instructions
