# Depth RGB Pipeline Comprehensive Audit

**Objective:** Systematically verify every element of the `depth_rgb` image path — from raw data loading to final metric calculation — ensuring correctness at each stage.

**Scope:** depth_rgb modality ONLY (metadata path confirmed correct).

**How to use:** Check off items as you verify them. Add comments in the `Notes` column.

---

## Phase 1: Data Discovery & Path Resolution

### 1.1 Image File Paths

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 1.1.1 | `config.py:75` — `image_folder` resolves to `data/raw/Depth_RGB` | [ ] | Verify folder exists and contains images |
| 1.1.2 | Depth_RGB folder contains `.jpg`/`.png` files with expected naming `_P{patient}{appt}{dfu}` pattern | [ ] | Count files, check naming consistency |
| 1.1.3 | `best_matching.csv` exists at `results/best_matching.csv` (`config.py:80`) | [ ] | Verify it has `depth_rgb` column with valid filenames |
| 1.1.4 | Every `depth_rgb` filename in `best_matching.csv` corresponds to a real file in `Depth_RGB/` folder | [ ] | Run verification script 1.1 |
| 1.1.5 | `best_matching.csv` has correct bounding box columns: `depth_xmin`, `depth_ymin`, `depth_xmax`, `depth_ymax` | [ ] | Check for NaN/zero/negative values |
| 1.1.6 | Bounding boxes are within image dimensions (not out of bounds) | [ ] | Run verification script 1.2 |

### 1.2 CSV Data Linkage

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 1.2.1 | `extract_info_from_filename()` (`image_processing.py:34`) correctly parses Patient#, Appt#, DFU# from depth_rgb filenames | [ ] | Test with sample filenames |
| 1.2.2 | `create_best_matching_dataset()` (`image_processing.py:111`) correctly maps depth_rgb files to CSV metadata rows via Patient#/Appt#/DFU# join | [ ] | Verify no mismatches |
| 1.2.3 | `Healing Phase Abs` column maps correctly: `I→0, P→1, R→2` | [ ] | Check in `dataset_utils.py:793` and throughout |
| 1.2.4 | No duplicate rows in best_matching.csv (same Patient#/Appt#/DFU# appearing multiple times) | [ ] | Count unique vs total |
| 1.2.5 | Label distribution in `best_matching.csv` matches expected class ratios | [ ] | Compare against DataMaster CSV |

---

## Phase 2: Image Loading & Preprocessing

### 2.1 Image Decoding

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 2.1.1 | `dataset_utils.py:244-246` — Images decoded as JPEG with 3 channels (RGB) | [ ] | Verify no grayscale or RGBA images in folder |
| 2.1.2 | Images decoded as `float32` (`dataset_utils.py:246`) | [ ] | Range should be [0, 255] before normalization |
| 2.1.3 | `process_single_sample()` builds correct path: `data/raw/Depth_RGB/{filename}` via `tf.case` at line 237 | [ ] | Verify the tf.case routing is correct for depth_rgb |

### 2.2 Bounding Box Crop

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 2.2.1 | For `depth_rgb`, bounding box is used AS-IS (no FOV adjustment) — see `tf.case` at line 277-280 | [ ] | `depth_map` gets FOV adjustment, `depth_rgb` does NOT — is this correct? |
| 2.2.2 | Coordinates clamped to image bounds (`dataset_utils.py:283-286`) | [ ] | `xmin >= 0`, `xmax <= width`, etc. |
| 2.2.3 | Crop dimensions are positive (`xmax > xmin`, `ymax > ymin`) after clamping | [ ] | Edge case: what if BB is degenerate? |
| 2.2.4 | `tf.image.crop_to_bounding_box(img, ymin, xmin, crop_height, crop_width)` called with correct argument order | [ ] | TF expects (offset_height, offset_width, target_height, target_width) |

### 2.3 Resize & Padding

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 2.3.1 | Aspect ratio preserved during resize (`dataset_utils.py:295-311`) | [ ] | Check that image isn't stretched |
| 2.3.2 | Resize uses `lanczos3` interpolation (`dataset_utils.py:315`) | [ ] | Appropriate for medical images |
| 2.3.3 | Target size = `IMAGE_SIZE` = 256 from `production_config.py:28` | [ ] | Confirm 256x256 is used, not 128 from config.py:87 |
| 2.3.4 | Padding is zero-valued (black) and centered (`dataset_utils.py:325`) | [ ] | Verify pad amounts are symmetric |
| 2.3.5 | Final shape is exactly `(256, 256, 3)` with `set_shape` enforcement (`dataset_utils.py:356-357`) | [ ] | |

### 2.4 Normalization

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 2.4.1 | **CRITICAL:** For EfficientNet backbone, depth_rgb images are NOT divided by 255 (`dataset_utils.py:332-338`) | [ ] | EfficientNetB3 has built-in `Rescaling(1/255)` layer — input must be [0, 255] |
| 2.4.2 | `_rgb_has_builtin_rescaling` is `True` when `RGB_BACKBONE='EfficientNetB3'` | [ ] | Verify at line 332 |
| 2.4.3 | `_normalize_rgb()` returns image unchanged (keeps [0, 255]) when EfficientNet backbone is used | [ ] | Double-normalization would squish values to [0, 0.004] |
| 2.4.4 | If SimpleCNN backbone is ever used, images ARE divided by 255 → [0, 1] | [ ] | Alternate path at line 338 |
| 2.4.5 | Verify EfficientNetB3's built-in rescaling layer is present and operating correctly | [ ] | Check model.layers for Rescaling |

---

## Phase 3: Augmentation Pipeline

### 3.1 General Augmentation

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 3.1.1 | `USE_GENERAL_AUGMENTATION = True` (`production_config.py:127`) | [ ] | Enabled by default |
| 3.1.2 | Augmentation applied ONLY to training data, NOT validation | [ ] | Check `is_training` flag at `dataset_utils.py:533-539` |
| 3.1.3 | For `depth_rgb`, augmentation settings: brightness ±60%, contrast 0.6-1.4x, saturation 0.6-1.4x, noise σ=0.15 | [ ] | Check in generative_augmentation_v3.py modality_settings |
| 3.1.4 | Augmentation probability = 60% (not applied to every sample) | [ ] | |
| 3.1.5 | Augmentation values make sense for [0, 255] range images (not [0, 1]) | [ ] | If brightness ±60% on 255-scale, max = 408 → clipping needed? |

### 3.2 Generative Augmentation

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 3.2.1 | `USE_GENERATIVE_AUGMENTATION = False` (`production_config.py:133`) — disabled | [ ] | Should NOT be active |
| 3.2.2 | No SDXL model loading when disabled | [ ] | Verify gen_manager is None at `training_utils.py:1039-1040` |

---

## Phase 4: Model Architecture

### 4.1 Backbone Selection & Construction

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 4.1.1 | `RGB_BACKBONE = 'EfficientNetB3'` (`production_config.py:63`) used for depth_rgb | [ ] | |
| 4.1.2 | `create_image_branch()` (`builders.py:146`) routes depth_rgb to `create_efficientnet_branch()` | [ ] | `modality in ['depth_rgb', 'thermal_rgb']` → `backbone = RGB_BACKBONE` |
| 4.1.3 | EfficientNetB3 loaded with `include_top=False, pooling='avg'` (`builders.py:112-115`) | [ ] | Output = 1536D global average pooled features |
| 4.1.4 | Weights loaded: local file if exists, otherwise ImageNet (`builders.py:107-125`) | [ ] | Check if `local_weights/efficientnetb3_notop.h5` exists |
| 4.1.5 | `base_model.trainable = True` (`builders.py:132`) — all EfficientNet layers trainable | [ ] | |
| 4.1.6 | Lambda wrapper correctly passes images through backbone (`builders.py:137-140`) | [ ] | Verify no shape issues with the Lambda wrapping |

### 4.2 Projection Head

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 4.2.1 | Projection pipeline: 1536D → Dense(512, relu) + BN + Dropout(0.15) | [ ] | `builders.py:176-178` |
| 4.2.2 | → Dense(256, relu) + BN + Dropout(0.10) | [ ] | `builders.py:179-181` |
| 4.2.3 | → Dense(128, relu) + BN | [ ] | `builders.py:182-183` |
| 4.2.4 | → Dense(64, relu) + BN | [ ] | `builders.py:185-186` |
| 4.2.5 | L2 regularization (0.001) on Dense(512) and Dense(256) only | [ ] | Lines 176, 179 have `kernel_regularizer`, lines 182, 185 do not |
| 4.2.6 | `he_normal` kernel initializer on all Dense layers | [ ] | Good for ReLU activations |
| 4.2.7 | OptimizedModularAttention applied to 64D output (`builders.py:190-191`) | [ ] | Element-wise attention weighting |

### 4.3 Classification Head

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 4.3.1 | **depth_rgb only:** `Dense(3, activation='softmax', name='output')` (`builders.py:416`) | [ ] | Correct: 3-class softmax |
| 4.3.2 | **metadata + depth_rgb:** `Dense(3, activation='softmax', name='image_classifier')` → LearnableFusionWeights (`builders.py:432-442`) | [ ] | Image branch softmax + weighted fusion |
| 4.3.3 | LearnableFusionWeights init: `sigmoid(logit)` where `logit = log(0.70/0.30) ≈ 0.847` → RF=70%, Image=30% | [ ] | `builders.py:41` |
| 4.3.4 | Fusion formula: `output = rf_weight * rf_probs + (1 - rf_weight) * image_probs` (`builders.py:55`) | [ ] | Weighted average of probability distributions |
| 4.3.5 | Fusion output sums to ~1.0 (both rf_probs and image_probs are softmax, weighted average preserves) | [ ] | |

### 4.4 Model Shapes Verification

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 4.4.1 | Input shape for depth_rgb: `(256, 256, 3)` | [ ] | Set by `ProcessedDataManager.process_all_modalities()` at `training_utils.py:356` |
| 4.4.2 | Total trainable parameters count is reasonable for EfficientNetB3 + projection | [ ] | ~12M for B3 + ~500K for projection ≈ ~12.5M |
| 4.4.3 | `model.summary()` shows correct layer connectivity | [ ] | Run verification script 4.1 |

---

## Phase 5: Training Pipeline

### 5.1 Pre-training (depth_rgb standalone)

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 5.1.1 | Pre-training triggered when fusion model lacks pre-trained weights (`training_utils.py:1331`) | [ ] | |
| 5.1.2 | Pre-training uses `PRETRAIN_LR = 1e-3` (`production_config.py:67`) | [ ] | Compiled at `training_utils.py:1361` |
| 5.1.3 | Pre-training uses `focal_ordinal_loss` with same alpha as fusion | [ ] | `training_utils.py:1355-1356` |
| 5.1.4 | Pre-training monitors `val_cohen_kappa` (not `val_weighted_f1_score`) for early stopping | [ ] | `training_utils.py:1384-1392` — fixed from previous bug |
| 5.1.5 | Pre-training epochs = `N_EPOCHS = 200` (`training_utils.py:1431`) | [ ] | Full 200 epochs, early stopping applies |
| 5.1.6 | Pre-trained weights transferred to fusion model layer-by-layer (`training_utils.py:1447-1458`) | [ ] | Only layers with `image_modality` in name |
| 5.1.7 | Pre-trained `output` layer weights transferred to `image_classifier` if shapes match (`training_utils.py:1463-1471`) | [ ] | Critical for Stage 1 to start with good image probs |
| 5.1.8 | Pre-training saves checkpoint to `pretrain_cache` for reuse (`training_utils.py:1476-1479`) | [ ] | |

### 5.2 Fusion Training (metadata + depth_rgb)

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 5.2.1 | Fusion compiled with `STAGE1_LR = 1e-4` (`production_config.py:68`, `training_utils.py:1526`) | [ ] | |
| 5.2.2 | All layers trainable during fusion (no freeze/unfreeze stages) | [ ] | See `training_utils.py:1640-1662` — end-to-end training |
| 5.2.3 | EarlyStopping monitors `val_weighted_f1_score` with `patience=20` | [ ] | `training_utils.py:1533-1540` |
| 5.2.4 | ReduceLROnPlateau on `val_weighted_f1_score` with `patience=10`, factor=0.5 | [ ] | `training_utils.py:1541-1548` |
| 5.2.5 | ModelCheckpoint saves best by `val_weighted_f1_score` | [ ] | `training_utils.py:1549-1555` |
| 5.2.6 | Best weights loaded after training completes (`training_utils.py:1677-1680`) | [ ] | |

### 5.3 Loss Function

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 5.3.1 | `focal_ordinal_loss` defined in `losses.py:123-143` | [ ] | |
| 5.3.2 | **Focal component:** `alpha * (1 - y_pred)^gamma * (-y_true * log(y_pred))` | [ ] | Class-reweighted focal loss |
| 5.3.3 | `alpha` values = frequency-based weights from training data (normalized to sum=3) | [ ] | NOT hardcoded [0.598, 0.315, 1.597] — computed per fold |
| 5.3.4 | `gamma = 2.0` (default in `get_focal_ordinal_loss` at `losses.py:145`) | [ ] | Focuses on hard examples |
| 5.3.5 | **Ordinal component:** `(true_class - pred_class)^2` penalty | [ ] | `ordinal_weight = 0.05` (from config dict, NOT 0.5 default) |
| 5.3.6 | Predictions clipped to `[epsilon, 1-epsilon]` to prevent `log(0)` | [ ] | `losses.py:126` |
| 5.3.7 | Loss operates on one-hot `y_true` and softmax `y_pred` | [ ] | Both shape `(batch, 3)` |

### 5.4 Class Weights

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 5.4.1 | `USE_FREQUENCY_BASED_WEIGHTS = True` (`production_config.py:87`) | [ ] | Alpha computed from inverse class frequency |
| 5.4.2 | Alpha computation at `dataset_utils.py:1006-1010`: `alpha = 1/freq`, normalized to sum=3 | [ ] | |
| 5.4.3 | `TRAINING_CLASS_WEIGHT_MODE = 'frequency'` — class_weight dict uses alpha values | [ ] | `training_utils.py:1185-1187` |
| 5.4.4 | `class_weight` passed to `model.fit()` is... | [ ] | **CHECK: Is class_weight actually passed to model.fit()?** Search for `class_weight=` in .fit() calls |
| 5.4.5 | Alpha values are consistent between WeightedF1Score metric and focal loss alpha | [ ] | Both use `alpha_value` from `prepare_cached_datasets` |

---

## Phase 6: Evaluation & Metrics

### 6.1 Prediction Generation

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 6.1.1 | Predictions generated via `model.predict(model_inputs, verbose=0)` (`training_utils.py:1695`) | [ ] | |
| 6.1.2 | `model_inputs` excludes `sample_id` key (Keras 3 compatibility) (`training_utils.py:1694`) | [ ] | |
| 6.1.3 | Predictions are raw softmax outputs — no additional activation applied | [ ] | shape `(N, 3)`, values in [0,1], sum≈1 |
| 6.1.4 | `y_pred = np.argmax(batch_pred, axis=1)` for class labels (`training_utils.py:1697`) | [ ] | argmax of softmax = predicted class |
| 6.1.5 | `y_true = np.argmax(batch_labels, axis=1)` from one-hot labels (`training_utils.py:1696`) | [ ] | Reverses the one_hot encoding from `dataset_utils.py:416` |

### 6.2 True Label ↔ Prediction Alignment

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 6.2.1 | Labels one-hot encoded at `dataset_utils.py:415-416`: `label = tf.one_hot(label, depth=3)` | [ ] | Class 0→[1,0,0], Class 1→[0,1,0], Class 2→[0,0,1] |
| 6.2.2 | Label mapping: I=0, P=1, R=2 — consistent between CSV encoding and model output | [ ] | `dataset_utils.py:793`: `{'I': 0, 'P': 1, 'R': 2}` |
| 6.2.3 | `CLASS_LABELS = ['I', 'P', 'R']` (`config.py:94`) — index 0=I, 1=P, 2=R | [ ] | Used in classification_report at `training_utils.py:1784-1787` |
| 6.2.4 | Confusion matrix uses `labels=[0, 1, 2]` explicitly (`training_utils.py:1762`) | [ ] | Prevents sklearn from reordering |
| 6.2.5 | **CRITICAL:** `y_true` and `y_pred` iterate in same batch order — no shuffling between them | [ ] | Both extracted from same `batch` in the for-loop |
| 6.2.6 | Training predictions use `pre_aug_train_dataset` (unaugmented) for evaluation (`training_utils.py:1689`) | [ ] | Not polluted by augmentation artifacts |
| 6.2.7 | Validation predictions iterate over `validation_steps` batches only (`training_utils.py:1727`) | [ ] | Not re-reading data beyond one epoch's worth |

### 6.3 Metrics Calculation

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 6.3.1 | Cohen's Kappa: `cohen_kappa_score(y_true, y_pred, weights='quadratic')` (`training_utils.py:1759`) | [ ] | Quadratic weights correct for ordinal I→P→R |
| 6.3.2 | F1 Macro: `f1_score(y_true, y_pred, average='macro', zero_division=0)` | [ ] | Equal weight per class |
| 6.3.3 | F1 Weighted: `f1_score(y_true, y_pred, average='weighted', zero_division=0)` | [ ] | Weighted by support |
| 6.3.4 | Accuracy: `accuracy_score(y_true, y_pred)` | [ ] | Simple % correct |
| 6.3.5 | **In-training CohenKappa metric** (`training_utils.py:362-416`): uses quadratic weights matching sklearn | [ ] | Weight matrix: `w_ij = (i-j)^2 / (k-1)^2` |
| 6.3.6 | **In-training WeightedF1Score** (`losses.py:9-70`): alpha-weighted F1, NOT same as sklearn weighted F1 | [ ] | **Potential discrepancy:** training metric uses alpha weights, eval uses support weights |
| 6.3.7 | **In-training MacroF1** (`training_utils.py:447-494`): unweighted macro F1 matches sklearn macro | [ ] | |

### 6.4 Saved Results Format

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 6.4.1 | Predictions saved as `.npy` with shape `(N, 3)` float32 (`training_utils.py:1706`) | [ ] | Via `save_run_predictions()` |
| 6.4.2 | True labels saved alongside predictions (`training_utils.py:1706`) | [ ] | Same function saves both |
| 6.4.3 | Sample IDs saved for traceability (`training_utils.py:1706`) | [ ] | Patient#, Appt#, DFU# array |
| 6.4.4 | Model weights saved as `.weights.h5` (Keras 3 format) | [ ] | `training_utils.py:204` |

---

## Phase 7: Cross-Cutting Concerns

### 7.1 Data Leakage Prevention

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 7.1.1 | Patient-level split: no patient appears in both train and validation | [ ] | `create_patient_folds()` at `dataset_utils.py:39` splits by Patient# |
| 7.1.2 | Same patient split used for pre-training and fusion training | [ ] | Fold splits passed from cross_validation loop |
| 7.1.3 | Augmentation applied ONLY to training set | [ ] | `is_training` check at `dataset_utils.py:533` |
| 7.1.4 | `best_matching.csv` mapping is deterministic (not recomputed per fold) | [ ] | Static file in results/ |

### 7.2 Reproducibility

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 7.2.1 | Seeds set per fold: `42 + run * (run + 3)` (`training_utils.py:909-912`) | [ ] | Python, numpy, TF, hash |
| 7.2.2 | Deterministic TF ops enabled (`production_config.py:454-455`) | [ ] | `TF_DETERMINISTIC_OPS = "1"` |
| 7.2.3 | Cache per fold prevents cross-fold contamination (`dataset_utils.py:484`) | [ ] | Cache key includes fold_id and seed |

### 7.3 Potential Issues & Red Flags

| # | Check Item | Status | Notes |
|---|-----------|--------|-------|
| 7.3.1 | **`config.py:87` has `IMAGE_SIZE = 128`** but `production_config.py:28` has `IMAGE_SIZE = 256` — which is actually used? | [ ] | Training uses production_config.py value (256). config.py value is legacy/unused. Confirm. |
| 7.3.2 | `PRETRAIN_LR = 1e-3` is 10x higher than `STAGE1_LR = 1e-4` — appropriate for learning from random projection head | [ ] | But EfficientNet backbone is also trainable → risk of forgetting ImageNet features |
| 7.3.3 | `class_weight` dict — is it actually passed to `model.fit()`? | [ ] | **Search all `.fit()` calls for `class_weight=` parameter** |
| 7.3.4 | EfficientNet Lambda wrapping may prevent proper gradient flow to backbone | [ ] | Lambda layers sometimes break graph optimization |
| 7.3.5 | `SAMPLING_STRATEGY = 'none'` with `USE_FREQUENCY_BASED_WEIGHTS = True` — no resampling, only loss weighting | [ ] | Is this sufficient for class imbalance? |
| 7.3.6 | Validation uses `drop_remainder=True` (`dataset_utils.py:527`) — some samples may be dropped | [ ] | Could lose up to `batch_size - 1` validation samples |
| 7.3.7 | Pre-training and fusion both use same alpha values, but computed from training split only | [ ] | Correct — no data leakage from validation |
| 7.3.8 | `FOCAL_ORDINAL_WEIGHT = 0.5` in production_config but code uses `config.get('ordinal_weight', 0.05)` | [ ] | The default in the code is 0.05, NOT 0.5 from config — which actually applies? |

---

## Verification Scripts

See `verification_scripts.py` in this directory for automated checks.

### Quick Run Commands

```bash
# Activate environment
source /opt/miniforge3/bin/activate multimodal

# Run all verification checks
python agent_communication/depth_rgb_pipeline_audit/verification_scripts.py

# Run specific phase
python agent_communication/depth_rgb_pipeline_audit/verification_scripts.py --phase 1
python agent_communication/depth_rgb_pipeline_audit/verification_scripts.py --phase 2
python agent_communication/depth_rgb_pipeline_audit/verification_scripts.py --phase 4
```

---

## Summary of High-Priority Items

These items have the highest risk of being incorrect and should be verified first:

1. **2.4.1** — Normalization: EfficientNet expects [0,255], SimpleCNN expects [0,1]. Verify the correct path is taken.
2. **3.1.5** — Augmentation operates on [0,255] range images. Brightness ±60% could push values to 408. Check clipping.
3. **5.3.5** — Ordinal weight: `FOCAL_ORDINAL_WEIGHT=0.5` vs code default `0.05`. Which is used?
4. **5.4.4** — `class_weight` parameter may NOT be passed to `model.fit()`. Check all .fit() calls.
5. **6.3.6** — In-training weighted F1 (alpha-weighted) differs from eval weighted F1 (support-weighted). Monitor choices may be misleading.
6. **7.3.1** — Two different `IMAGE_SIZE` values (128 vs 256). Confirm 256 is used everywhere.
7. **7.3.4** — Lambda-wrapped EfficientNet gradient flow. May silently fail to train backbone.
8. **2.2.1** — depth_rgb uses depth bounding box AS-IS. depth_map gets FOV adjustment. Investigate if depth_rgb also needs adjustment.
