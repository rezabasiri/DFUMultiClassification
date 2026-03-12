# TODO: Investigate Why Image Modalities Underperform Metadata

**Status: INVESTIGATION COMPLETE. Fix implemented. Awaiting fusion audit re-run.**

See: `agent_communication/INVESTIGATION_image_modality_underperformance.md` for full findings.

## Objective

The desired modality performance ranking (best to worst):
1. metadata + depth_rgb + thermal_map
2. metadata + thermal_map
3. metadata (alone)
4. thermal_map (alone)
5. depth_rgb (alone)

**Current state:** metadata alone outperforms all image-containing combinations.

---

## Environment

- GPU: NVIDIA RTX A5000 (24GB)
- Framework: TensorFlow (mixed_float16 enabled)
- Dataset: 3,108 samples (DFU wound images), 3 classes (I/P/R)
- Image size: 256x256
- Batch size: 64
- Generative augmentation: OFF (`USE_GENERATIVE_AUGMENTATION = False`)
- General augmentation: ON (`USE_GENERAL_AUGMENTATION = True`)
- Do NOT modify `main_original.py`
- Ignore `depth_map_pipeline_audit/` folder (no depth_map audit was performed)

---

## Polish v2 Phase 1 Results

Source: `results/misclassifications_saved/phase1_baseline.json`
Settings: 3 runs x 4 combos x 5 folds, `track_misclass='both'`

| Modality | Accuracy | Accuracy Std | Macro F1 | Kappa | Kappa Std |
|----------|----------|-------------|----------|-------|-----------|
| **metadata** | 58.06% | +/-7.55% | 53.96% | **0.3331** | +/-0.0757 |
| metadata+depth_rgb+thermal_map | 48.81% | +/-3.52% | 45.72% | 0.2797 | +/-0.0775 |
| metadata+thermal_map | 40.85% | +/-4.29% | 38.43% | 0.2825 | +/-0.0939 |
| depth_rgb+thermal_map | 46.00% | +/-4.30% | 41.65% | 0.2385 | +/-0.0401 |

Per-class F1 scores:

| Modality | F1-I | F1-P | F1-R |
|----------|------|------|------|
| metadata | 0.463 | 0.651 | 0.504 |
| metadata+depth_rgb+thermal_map | 0.453 | 0.540 | 0.379 |
| metadata+thermal_map | 0.486 | 0.323 | 0.344 |
| depth_rgb+thermal_map | 0.443 | 0.498 | 0.308 |

---

## Phase 1 Training Log Observations

Source: `results/misclassifications/phase1_detailed.log`

### First run, metadata+depth_rgb+thermal_map combo (lines ~4038-4145):

**depth_rgb pre-training:**
- Stage 1 (frozen backbone, head only): best val_kappa = 0.1508
- Stage 2 (unfreeze top 20%, LR=1e-5, 30 epochs): best val_kappa = 0.1404
- Stage 2 did NOT improve. Restored Stage 1 weights.
- Final depth_rgb kappa: 0.1508

**thermal_map pre-training:**
- Stage 1 (frozen backbone, head only): best val_kappa = 0.3217
- Stage 2 (unfreeze top 50%, LR=1e-5, 50 epochs): best val_kappa = 0.3186
- Stage 2 did NOT improve. Restored Stage 1 weights.
- Final thermal_map kappa: 0.3217

**Fusion (after pre-training):**
- Stage 1 (LR=0.001): best val_kappa = 0.2578 at epoch 9/29
- Stage 2 (unfreeze top 20%, LR=5e-6, 50 epochs): best val_kappa = 0.2619
- Fusion kappa (0.2619) is worse than thermal_map alone (0.3217)

**metadata (same run):**
- Epoch 1/200 -> val_kappa = 0.1413 (some folds), 0.3244-0.4588 (other folds)
- Converges by epoch 20 consistently (early stopping triggers at epoch 21)
- Training uses RF classifier (no image backbone)

---

## Audit Results (Hyperparameter Searches)

### depth_rgb Audit

Script: `agent_communication/depth_rgb_pipeline_audit/depth_rgb_hparam_search.py`
Best config: `agent_communication/depth_rgb_pipeline_audit/depth_rgb_best_config.json`
Results CSV: `agent_communication/depth_rgb_pipeline_audit/depth_rgb_search_results.csv`

**Winner: BASELINE config** (no improvement found over default settings)

```json
{
  "backbone": "EfficientNetB0",
  "freeze": "frozen",
  "head_units": [128],
  "head_dropout": 0.3, "head_l2": 0.0,
  "learning_rate": 0.001,
  "finetune_lr": 1e-5, "finetune_epochs": 30, "unfreeze_pct": 0.2,
  "freeze_bn_in_stage2": false,
  "label_smoothing": 0.0,
  "use_mixup": false,
  "image_size": 256,
  "loss_type": "focal", "focal_gamma": 2.0
}
```

Top audit results (single fold, post_eval_kappa):
- BASELINE fold4: 0.364 (EfficientNetB0)
- DenseNet121 fold4: 0.384 (best single fold, but not selected as winner)
- BASELINE fold3: 0.331
- BASELINE fold2: 0.322

### thermal_map Audit

Script: `agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py`
Best config: `agent_communication/thermal_map_pipeline_audit/thermal_map_best_config.json`
Results CSV: `agent_communication/thermal_map_pipeline_audit/thermal_map_search_results.csv`

**Winner: B3_R6_ft_top50_50ep**

```json
{
  "backbone": "EfficientNetB2",
  "freeze": "frozen",
  "head_units": [128, 32],
  "head_dropout": 0.3, "head_l2": 0.0,
  "learning_rate": 0.003,
  "finetune_lr": 1e-5, "finetune_epochs": 50, "unfreeze_pct": 0.5,
  "freeze_bn_in_stage2": true,
  "label_smoothing": 0.0,
  "use_mixup": true, "mixup_alpha": 0.2,
  "image_size": 256,
  "loss_type": "focal", "focal_gamma": 2.0
}
```

Top audit results (single fold, post_eval_kappa):
- B3_R6_ft_top50_50ep: 0.369 (EfficientNetB2)
- DenseNet121 variant: 0.357
- B3_R5_aug_on_256: 0.346

### Fusion Audit

Script: `agent_communication/fusion_pipeline_audit/fusion_hparam_search.py`
Best config: `agent_communication/fusion_pipeline_audit/fusion_best_config.json`
Results CSV: `agent_communication/fusion_pipeline_audit/fusion_search_results.csv`

**Winner: B3_R7_ft_top20_50ep_lr**

```json
{
  "image_modalities": ["depth_rgb", "thermal_map"],
  "fusion_strategy": "feature_concat",
  "fusion_head_units": [],
  "stage1_lr": 0.001, "stage1_epochs": 100,
  "stage2_lr": 5e-6, "stage2_epochs": 50, "stage2_unfreeze_pct": 0.2,
  "label_smoothing": 0.1,
  "use_mixup": false,
  "rf_n_estimators": 300, "rf_max_depth": 10
}
```

Top audit results (single fold, post_eval_kappa):
- TOP3_2 fold4: 0.357
- TOP3_3 fold4: 0.332
- TOP3_1 fold4: 0.327

**NOTE:** The fusion audit has been updated with Round 8 (image projection dimension search). Re-run with `--fresh` to test the new projection feature.

---

## Current Production Config

Source: `src/utils/production_config.py` lines 77-142

```python
MODALITY_CONFIGS = {
    'depth_rgb': {
        'backbone': 'EfficientNetB0',
        'head_units': [128],
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.001,
        'finetune_lr': 1e-5,
        'finetune_epochs': 30,
        'unfreeze_pct': 0.2,
        'freeze_bn_stage2': False,
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
    'thermal_map': {
        'backbone': 'EfficientNetB2',
        'head_units': [128, 32],
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.003,
        'finetune_lr': 1e-5,
        'finetune_epochs': 50,
        'unfreeze_pct': 0.5,
        'freeze_bn_stage2': True,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
    'depth_map': {
        'backbone': 'EfficientNetB2',
        'head_units': [128, 32],
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.003,
        'finetune_lr': 1e-5,
        'finetune_epochs': 50,
        'unfreeze_pct': 0.5,
        'freeze_bn_stage2': True,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
    'thermal_rgb': {
        'backbone': 'EfficientNetB0',
        'head_units': [128],
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.001,
        'finetune_lr': 1e-5,
        'finetune_epochs': 30,
        'unfreeze_pct': 0.2,
        'freeze_bn_stage2': False,
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
}
```

Other relevant settings:
- `IMAGE_SIZE = 256` (line 30)
- `N_EPOCHS = 200` (line 33)
- `GLOBAL_BATCH_SIZE = 64` (line 31)
- `USE_GENERAL_AUGMENTATION = True` (line 233)
- `USE_GENERATIVE_AUGMENTATION = False` (line 239)
- **NEW:** `FUSION_IMAGE_PROJECTION_DIM = 8` (line 172) — projects image features to 8-dim before fusion

---

## Key Code Paths

### Image Loading (Main Training Path)
- `src/data/dataset_utils.py` lines 191-317: `process_single_sample()`
  - Uses `tf.io.decode_jpeg` -> float32 [0, 255]
  - NOTE: `decode_jpeg` correctly handles PNG files via auto-detection (verified empirically)
  - Crops to bounding box, resizes with aspect ratio, pads
  - Normalization at lines 285-311: checks `RGB_BACKBONE`/`MAP_BACKBONE`
    - EfficientNet: keeps [0, 255] (backbone has built-in Rescaling(1/255))
    - SimpleCNN: divides by 255 -> [0, 1]
  - For maps: max-normalizes then scales (SimpleCNN: [0,1], EfficientNet: [0,255])

### Older Image Loading (NOT used in main training path)
- `src/data/image_processing.py` line 354: `load_and_preprocess_image()`
  - Always divides RGB by 255 (line 466) — does NOT check backbone type
  - Imported in `dataset_utils.py` (line 22) but never called during training
  - Used by: `src/utils/outlier_detection.py`, `src/data/caching.py`

### Model Building
- `src/models/builders.py` lines 42-116: `create_efficientnet_branch()`
  - Loads EfficientNet with ImageNet weights
  - EfficientNet has built-in `Rescaling(scale=1/255, offset=0)` as first layer
  - Then `Normalization(mean=0, variance=1)` as second layer
  - Sets `base_model.trainable = True`
- `src/models/builders.py` lines 181-221: `create_image_branch()`
  - Adds projection head: Dense -> BatchNorm -> Dropout(0.3) per head layer
- **NEW:** `src/models/builders.py` lines 458-475: Image feature projection
  - When `FUSION_IMAGE_PROJECTION_DIM > 0` and metadata present:
  - Concatenated image features (160-dim) -> Dense(8)+BN+Dropout -> (8-dim)
  - Reduces image:metadata ratio from 53:1 to 2.7:1

### Training Loop (2-Stage)
- `src/training/training_utils.py`:
  - Stage 1: frozen backbone, train head only, early stopping
  - Stage 2: unfreeze top X% of backbone, lower LR, fine-tune
  - If Stage 2 doesn't improve over Stage 1: restore Stage 1 weights
  - Projection heads are SEPARATE from EfficientNet sub-models -> remain trainable when backbone is frozen

### Augmentation
- `src/data/generative_augmentation_v3.py`: `create_enhanced_augmentation_fn()`
  - Applied at batch time on training data
  - General augmentation: brightness (+/-10% x 255), contrast (0.8-1.2x), gaussian noise
  - Operates in [0, 255] range, clips to [0, 255]
  - Generative augmentation: OFF (controlled by `USE_GENERATIVE_AUGMENTATION`)

### Audit Script Normalization Handling
- Audit scripts override `RGB_BACKBONE`/`MAP_BACKBONE` in both `production_config` and `dataset_utils` modules
- For EfficientNet: sets to `'EfficientNetB0'` -> normalization keeps [0, 255]
- For SimpleCNN: sets to `'SimpleCNN'` -> normalization divides by 255

---

## Investigation Findings (Summary)

### Root Causes Identified (6 confirmed, 1 ruled out)

| # | Severity | Issue | Actionable? |
|---|----------|-------|-------------|
| ~~1~~ | ~~CRITICAL~~ | ~~PNG decoded as JPEG~~ | ~~RULED OUT~~ — TF handles it correctly |
| 2 | HIGH | Extreme upsampling of 70px crops to 256x256 | No — already tested 128/256/384 in audits |
| 3 | HIGH | Only 335 R-class samples (5.6:1 imbalance) | No — fundamental data limitation |
| 4 | HIGH | 53:1 image:metadata dimensionality mismatch | **YES — FIX IMPLEMENTED** |
| 5 | MEDIUM | Stage 2 fine-tuning consistently degrades | No — already tested in R7 of all audits |
| 6 | MEDIUM | ImageNet domain mismatch for medical images | No — already tested 6 backbones in audits |
| 7 | LOW-MED | Metadata branch has 0 trainable params | Partially — projection fix helps indirectly |

### What the Audits Already Exhaustively Tested (400+ configs)
- Backbones: EfficientNetB0, B2, B3, DenseNet121, ResNet50V2, MobileNetV3Large
- Image sizes: 128, 256, 384
- Fusion strategies: prob_concat, feature_concat, feature_concat_attn, gated, hybrid
- Fusion heads: [], [32], [64], [64, 32], [128, 64]
- Various LRs, augmentation, mixup, label smoothing, RF params, Stage 2 fine-tuning

### What the Audits Did NOT Test (NEW in Round 8)
- **Image feature projection before fusion concatenation** — the only genuinely untested architectural idea

---

## Observations (Factual)

1. **Stage 2 never improved** over Stage 1 in the Phase 1 log (both depth_rgb and thermal_map reverted to Stage 1 weights).

2. **depth_rgb kappa (0.15)** in Phase 1 is much lower than its best audit result (0.36 on fold 4). Audit used same BASELINE config.

3. **thermal_map kappa (0.32)** in Phase 1 is close to its best audit result (0.37 on single fold).

4. **Fusion kappa (0.26)** in Phase 1 is lower than standalone thermal_map (0.32). Adding depth_rgb to thermal_map reduced performance.

5. **Metadata kappa varies 0.14-0.46** across folds within a single run (high variance). Average across all folds/runs: 0.333.

6. **Training epoch count**: metadata converges by epoch 20 (early stopping). Image modalities show similar early stopping patterns.

7. **Audit scripts and polish v2** use the same underlying source code (`src/`). The audit configs were written back into `production_config.py` as `MODALITY_CONFIGS`.

8. The audit winner for depth_rgb was the BASELINE config — the audit found no improvement over defaults.

9. **Audit single-fold kappas** (0.33-0.38) vs **Phase 1 5-fold average kappas** (0.15-0.33): single fold results have high variance. Best single folds in audits may not represent average performance.

10. **NEW:** `tf.io.decode_jpeg` correctly handles PNG files via auto-detection. Verified empirically — pixel values are identical to `tf.io.decode_png`. This is NOT a bug.

---

## Phase 2 Results (for reference)

Source: `results/misclassifications_saved/best_thresholds.json`

Best result (eval 9): I=50, P=47, R=60 thresholds
- Kappa: 0.504, Accuracy: 62.0%, Macro F1: 60.1%
- Modality: metadata+depth_rgb+thermal_map
- Filtered dataset: 473/648 samples (73% retention)

Best kappa observed (eval 12): I=42, P=48, R=60
- Kappa: 0.534, Accuracy: 64.8%

22 of 30 planned evaluations completed.

---

## Next Action

```bash
# Re-run fusion audit with --fresh to test image projection (Round 8)
python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --fresh
```

After audit completes:
1. Check if any `proj_dim > 0` config outperforms `proj_dim=0`
2. If yes: update `FUSION_IMAGE_PROJECTION_DIM` in production_config.py to winning value
3. Re-run Phase 1 baseline to validate on full 5-fold CV
4. If no: root causes #2/#3/#6 are fundamental — consider metadata-only for production

---

## Files Referenced

| File | Purpose |
|------|---------|
| `src/data/dataset_utils.py` | Main data pipeline, image loading, normalization |
| `src/data/image_processing.py` | Older image loading (not used in training path) |
| `src/models/builders.py` | EfficientNet/backbone model creation, **image projection (NEW)** |
| `src/training/training_utils.py` | 2-stage training loop |
| `src/utils/production_config.py` | All hyperparameters, **FUSION_IMAGE_PROJECTION_DIM (NEW)** |
| `src/data/generative_augmentation_v3.py` | Augmentation pipeline |
| `scripts/auto_polish_dataset_v2.py` | Polish v2 orchestration script |
| `results/misclassifications_saved/phase1_baseline.json` | Phase 1 baseline metrics |
| `results/misclassifications/phase1_detailed.log` | Full Phase 1 training log |
| `results/misclassifications_saved/phase2_checkpoint.json` | Phase 2 optimization history |
| `agent_communication/depth_rgb_pipeline_audit/` | depth_rgb audit script + results |
| `agent_communication/thermal_map_pipeline_audit/` | thermal_map audit script + results |
| `agent_communication/fusion_pipeline_audit/` | Fusion audit script + results, **Round 8 added (NEW)** |
| `agent_communication/INVESTIGATION_image_modality_underperformance.md` | Full investigation findings |
