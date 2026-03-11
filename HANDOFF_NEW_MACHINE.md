# DFU Multi-Classification — New Machine Handoff Document

## Project Overview

This project classifies **Diabetic Foot Ulcer (DFU) healing phases** into 3 classes:
- **I** (Inflammatory), **P** (Proliferative), **R** (Remodeling)

It uses a **multimodal fusion** approach combining:
- **Metadata** (clinical features → Random Forest)
- **depth_rgb** (depth camera RGB images → CNN/pretrained backbone)
- **thermal_map** (thermal camera heat map images → CNN/pretrained backbone)

Primary metric: **Cohen's Kappa (quadratic weighted)**

---

## Current Task: Standalone Hyperparameter Search Re-run

We are running standalone hyperparameter searches for `depth_rgb` and `thermal_map` modalities independently, BEFORE plugging their best configs into the fusion pipeline. The search uses a **Top-K R1 carry-forward** design:

1. **R1**: Test 8 backbone×freeze combos → pick top-3 winners
2. **R2–R6**: Each R1 winner gets an independent branch running through head architecture, loss/regularization, training dynamics, augmentation/image-size, and fine-tuning strategy
3. **TOP3**: Pool all results cross-branch, pick top 3, run 5-fold CV on each
4. **BASELINE**: Run default EfficientNetB0 frozen on all 5 folds for comparison

### Scripts to Run

```bash
# Run both sequentially (--fresh backs up old CSV and deletes stale cache):
bash run_standalone_searches.sh

# Or individually:
python agent_communication/depth_rgb_pipeline_audit/depth_rgb_hparam_search.py --fresh
python agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py --fresh
```

### After Standalones Complete

1. Take the best config JSON from each standalone audit
2. Update `_STANDALONE_CONFIGS` in `agent_communication/fusion_pipeline_audit/fusion_hparam_search.py`
3. Run fusion search and verify pre-training kappas match standalone fold 0

---

## CRITICAL BUG: `post_eval_kappa=0.0` on Multi-GPU

### Symptom

On the new 2-GPU machine, the first depth_rgb result shows:
```
trainable_weights=6, best_s1_kappa=0.208, best_s2_kappa=3.576e-07, post_eval_kappa=0.0
```

Stage 1 learns (kappa=0.208), but Stage 2 fine-tuning destroys the model (kappa≈0), and since `EarlyStopping(restore_best_weights=True)` restores the best **Stage 2** weights (which are terrible), the final post-eval kappa is 0.0.

### Root Cause Analysis

There are likely **two interacting issues**:

#### Issue 1: Mixed Precision + Softmax Output

`src/utils/gpu_config.py` line 278 enables `mixed_float16` globally:
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

The standalone scripts define the output layer as:
```python
output = Dense(3, activation='softmax', name='output')(x)
```

With `mixed_float16`, this softmax output becomes float16, which can cause **numerical instability** in loss computation and metrics. The correct approach is to either:
- Use `dtype='float32'` on the output Dense layer, OR
- Use a separate `Activation('softmax', dtype='float32')` layer, OR
- Cast predictions back to float32 before loss computation

**Fix option A** (in `build_model()` in both standalone scripts):
```python
output = Dense(3, activation='softmax', name='output', dtype='float32')(x)
```

**Fix option B** (in `build_model()`):
```python
x = Dense(3, name='output')(x)
output = tf.keras.layers.Activation('softmax', dtype='float32')(x)
```

#### Issue 2: BatchNorm + Multi-GPU in Stage 2

For `depth_rgb`, `freeze_bn_in_stage2=False` (BatchNorm layers become trainable during fine-tuning). With `MirroredStrategy`, each GPU computes batch statistics independently, causing BatchNorm mean/variance to diverge across replicas. This can destabilize fine-tuning catastrophically.

For `thermal_map`, `freeze_bn_in_stage2=True` (BatchNorm frozen in Stage 2), so this is less of an issue.

**Fix**: Either set `freeze_bn_in_stage2=True` for depth_rgb too, or add `tf.keras.layers.experimental.SyncBatchNormalization` — but the simpler fix is just freezing BN.

#### Issue 3: Stage 2 Destroys Stage 1 Weights

Even if Stage 2 fails, we should preserve Stage 1's learned weights. Currently:
- Stage 1: EarlyStopping restores best S1 weights ✓
- Stage 2: EarlyStopping restores best S2 weights (which may be worse than S1) ✗
- Post-eval uses whatever weights the model currently has

**Fix** (in `train_single_config()` in both scripts): Save Stage 1 weights before starting Stage 2, and restore them if Stage 2 doesn't improve:

```python
# Before Stage 2 begins:
s1_weights = model.get_weights()

# After Stage 2:
if best_s2_kappa < best_s1_kappa:
    print(f"  Stage 2 did NOT improve (s2={best_s2_kappa:.4f} < s1={best_s1_kappa:.4f}). Restoring S1 weights.")
    model.set_weights(s1_weights)
```

### `trainable_weights=6` is NOT a Bug

6 weight **tensors** for frozen EfficientNetB0 + `head_units=[128]` is correct:
- `head_dense_0` kernel + bias = 2 tensors
- `head_bn_0` gamma + beta = 2 tensors
- `output` kernel + bias = 2 tensors
- Total = 6 tensors (representing ~400+ actual parameters)

The backbone's ~4M weights are frozen (not counted as trainable). This count is correct.

---

## Key File Locations

### Standalone Search Scripts
| File | Purpose |
|------|---------|
| `agent_communication/depth_rgb_pipeline_audit/depth_rgb_hparam_search.py` | depth_rgb modality hyperparameter search |
| `agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py` | thermal_map modality hyperparameter search |
| `run_standalone_searches.sh` | Runs both searches sequentially with `--fresh` |

### Fusion Pipeline
| File | Purpose |
|------|---------|
| `agent_communication/fusion_pipeline_audit/fusion_hparam_search.py` | Fusion hyperparameter search (uses `_STANDALONE_CONFIGS`) |

### Core Source Code
| File | Purpose |
|------|---------|
| `src/main.py` | Main training entry point (2855 lines) |
| `src/models/builders.py` | Model construction (supports EfficientNetB0/B2, DenseNet121, ResNet50V2) |
| `src/models/losses.py` | Loss functions (focal ordinal, custom metrics) |
| `src/data/dataset_utils.py` | Data pipeline, TF dataset creation, patient-fold splitting |
| `src/data/image_processing.py` | Image loading, bounding box cropping |
| `src/utils/production_config.py` | Master config (596 lines, all hyperparameters) |
| `src/utils/gpu_config.py` | GPU detection, MirroredStrategy setup, **mixed_float16 is enabled here** |
| `src/utils/config.py` | Project paths (`get_project_paths()`, `get_data_paths()`) |

### Results & Caches
| Location | Purpose |
|----------|---------|
| `agent_communication/depth_rgb_pipeline_audit/depth_rgb_search_results.csv` | depth_rgb search results |
| `agent_communication/thermal_map_pipeline_audit/thermal_map_search_results.csv` | thermal_map search results |
| `agent_communication/*/logs/` | Per-run log files (timestamped) |
| `results/search_cache/depth_rgb_*` | Cached TF datasets for depth_rgb |
| `results/search_cache/thermal_map_*` | Cached TF datasets for thermal_map |
| `agent_communication/*_pipeline_audit/*_best_config.json` | Best config output |

### Configuration Files
| File | Purpose |
|------|---------|
| `claude.md` | Claude AI instructions (tracker maintenance, project structure) |
| `environment_multimodal.yml` | Conda environment spec |
| `requirements_multimodal.txt` | pip requirements |

---

## Environment

- **Python**: Use the project's virtual environment
- **TensorFlow**: 2.15.x (with CUDA support)
- **Key dependencies**: scikit-learn, scipy, numpy, pandas
- **GPU setup**: `src/utils/gpu_config.py` auto-detects GPUs, uses `MirroredStrategy` for multi-GPU
  - Compute cap < 9.0: Uses NCCL
  - Compute cap >= 9.0: Uses `ReductionToOneDevice` (for RTX 5090+)
- **Mixed precision**: `mixed_float16` enabled globally (line 278 of `gpu_config.py`)
- **Deterministic ops**: Enabled via environment variables in standalone scripts

---

## Dataset

- **3108 samples** (after filtering) of DFU wound images
- **3 classes**: I (Inflammatory), P (Proliferative), R (Remodeling) — stored in `Healing Phase Abs` column
- **Patient-wise splitting**: `StratifiedGroupKFold` ensures all images from one patient stay in the same fold
- **5-fold CV**: `n_folds=5` throughout (matches fusion pipeline)
- **Data paths**: Configured in `src/utils/config.py` via `get_data_paths()`
- **Modalities available**: depth_rgb, depth_map, thermal_rgb, thermal_map (currently only depth_rgb and thermal_map used in fusion)

---

## Architecture of Search Scripts

Both standalone scripts (`depth_rgb_hparam_search.py` and `thermal_map_hparam_search.py`) have identical structure:

1. **SearchConfig dataclass** — all hyperparameters with defaults
2. **Model building** — `build_simple_cnn()` and `build_pretrained_backbone()` → `build_model()`
3. **Loss functions** — `make_focal_loss()` and `make_cce_loss()`
4. **CohenKappaMetric** — custom Keras metric
5. **Data pipeline** — `load_data()` and `prepare_datasets()` reuse project's `prepare_dataset` and `prepare_cached_datasets`
6. **Training** — `train_single_config()` handles 2-stage training:
   - Stage 1: Frozen backbone + head training
   - Stage 2: Partial unfreeze + fine-tuning (only if `freeze=='frozen'` and `finetune_epochs>0`)
   - Post-eval: Iterate validation set, compute kappa/acc/f1
7. **Search rounds** — `round1_backbone_freeze()` through `round6_finetuning()`
8. **Results I/O** — CSV append, resume support, `pick_best()`, `pick_topk()`
9. **Main orchestration** — R1 → pick_topk(k=3) → 3 branches × R2-R6 → TOP3 cross-branch → BASELINE → comparison

### Key Differences Between depth_rgb and thermal_map Scripts

| Parameter | depth_rgb | thermal_map |
|-----------|-----------|-------------|
| `freeze_bn_in_stage2` default | `False` | `True` |
| SimpleCNN architecture | 4 layers (64→128→64→32) | 3 layers (128→64→32) |
| R2 head options | includes `[256, 64]` two-layer | includes `[128, 32]` two-layer |
| R5 image sizes | 224, 256 | 128, 256 |
| Input preprocessing | Overrides `RGB_BACKBONE` | Overrides `MAP_BACKBONE` |
| Data pipeline variable | `USE_GENERAL_AUGMENTATION` | Same |

---

## Applying the Fix

The fix needs to address 3 things in **both** standalone scripts:

### Fix 1: Mixed Precision Output Layer

In `build_model()` function, change:
```python
output = Dense(3, activation='softmax', name='output')(x)
```
to:
```python
output = Dense(3, activation='softmax', name='output', dtype='float32')(x)
```

**depth_rgb**: around line 223
**thermal_map**: around line 211

### Fix 2: Preserve Stage 1 Weights

In `train_single_config()`, before the Stage 2 block:
```python
# Save Stage 1 weights in case Stage 2 doesn't improve
s1_weights = model.get_weights()
```

After Stage 2 completes (after `ran_stage2 = True`):
```python
# If Stage 2 didn't improve, restore Stage 1 weights
if best_s2_kappa < best_s1_kappa * 0.95:  # 5% tolerance
    print(f"  Stage 2 degraded (s2={best_s2_kappa:.4f} < s1={best_s1_kappa:.4f}). Restoring S1 weights.")
    model.set_weights(s1_weights)
```

**depth_rgb**: Stage 2 block starts around line 668
**thermal_map**: Stage 2 block starts around line 618

### Fix 3 (Optional): Freeze BN in Stage 2 for depth_rgb

Change `freeze_bn_in_stage2` default in `SearchConfig` from `False` to `True` in depth_rgb. Or just leave it and let the search find the best setting (R6 tests both `True` and `False`).

---

## Verification Checklist

After applying fixes and running `--fresh`:

- [ ] `trainable_weights` for frozen backbones should be 6 (correct, this is tensor count)
- [ ] `best_s1_kappa` should be > 0.15 for most frozen backbone configs
- [ ] `best_s2_kappa` should be > 0 (not ~1e-7)
- [ ] `post_eval_kappa` should be close to `max(best_s1_kappa, best_s2_kappa)`
- [ ] `[FOLD]` debug prints should show `n_folds=5` throughout
- [ ] `n_val_samples` should be consistent (553 for fold 0 of 5-fold on 3108 samples)
- [ ] TOP3 selection picks 3 configs (not 5)
- [ ] Both scripts complete without errors
- [ ] Best config JSON saved to `*_best_config.json`

---

## Git Branch

Current branch: `claude/fix-trainable-parameters-UehxS`
Base branch: `main`

The standalone scripts with Top-K R1 carry-forward are already committed. The mixed-precision fix and Stage 1 weight preservation fix still need to be applied.
