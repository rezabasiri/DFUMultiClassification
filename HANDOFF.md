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
data/raw/                                      (11,650 files total, 8.6 GB)
  DataMaster_Processed_V12_WithMissing.csv     # Master metadata
  bounding_box_depth.csv                        # Depth bounding boxes
  bounding_box_thermal.csv                      # Thermal bounding boxes
  Depth_RGB/                                    # 2,884 files
  Depth_Map_IMG/                                # 3,025 files
  Thermal_RGB/                                  # 2,884 files
  Thermal_Map_IMG/                              # 2,854 files
```

## File Transfer Verification

Use these counts to verify all files transferred correctly:

| Directory | Files | Size | Notes |
|-----------|-------|------|-------|
| `data/raw/` | 11,650 | 8.6 GB | 3 CSVs + 4 image dirs |
| `data/raw/Depth_RGB/` | 2,884 | | RGB depth images |
| `data/raw/Depth_Map_IMG/` | 3,025 | | Depth map images |
| `data/raw/Thermal_RGB/` | 2,884 | | RGB thermal images |
| `data/raw/Thermal_Map_IMG/` | 2,854 | | Thermal map images |
| `results/GenerativeAug_Models/` | 3 | 9.6 GB | SDXL checkpoint (required for generative aug) |
| `src/models/sdxl_checkpoints/` | 4 | 20 GB | SDXL training checkpoints (best, epoch30, epoch35, latest) |
| `src/` (Python files) | 31 | | All source code |
| `scripts/` | 10 | | Utility scripts |
| `agent_communication/` | 362 | | Investigation docs and search scripts |

**SDXL model duplication note**: There are SDXL checkpoints in two locations:
- `results/GenerativeAug_Models/sdxl_full/` (9.6 GB) — the production checkpoint (`checkpoint_epoch_0035.pt`), referenced by `GENERATIVE_AUG_SDXL_MODEL_PATH` in production_config.py
- `src/models/sdxl_checkpoints/` (20 GB) — training checkpoints (best, epoch30, epoch35, latest)
- `agent_communication/generative_augmentation/` also contains ~20 GB of nested checkpoint copies

Only `results/GenerativeAug_Models/sdxl_full/checkpoint_epoch_0035.pt` (9.6 GB) is needed for production. The others are training artifacts that can be skipped to save ~40 GB of transfer.

Quick verification command:
```bash
echo "=== File counts ===" && \
echo "data/raw total: $(find data/raw -type f | wc -l) (expect 11650)" && \
echo "Depth_RGB: $(find data/raw/Depth_RGB -type f | wc -l) (expect 2884)" && \
echo "Depth_Map_IMG: $(find data/raw/Depth_Map_IMG -type f | wc -l) (expect 3025)" && \
echo "Thermal_RGB: $(find data/raw/Thermal_RGB -type f | wc -l) (expect 2884)" && \
echo "Thermal_Map_IMG: $(find data/raw/Thermal_Map_IMG -type f | wc -l) (expect 2854)" && \
echo "GenerativeAug_Models: $(find results/GenerativeAug_Models -type f | wc -l) (expect 3)" && \
echo "src .py files: $(find src -name '*.py' -type f | wc -l) (expect 31)" && \
echo "scripts: $(find scripts -type f | wc -l) (expect 10)" && \
echo "agent_communication: $(find agent_communication -type f | wc -l) (expect 362)"
```

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
