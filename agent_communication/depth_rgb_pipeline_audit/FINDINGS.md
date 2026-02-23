# Depth RGB Pipeline Audit — Findings Log

Use this file to record findings as you work through the AUDIT_PLAN.md checklist.

---

## Critical Findings

| # | Finding | Severity | Action Needed |
|---|---------|----------|---------------|
| 3.1.5 | Gaussian noise clips to [0,1] on [0,255] images at generative_augmentation_v3.py:341,388 — destroys ~18% of training samples | CRITICAL | Change to `tf.clip_by_value(image + noise, 0.0, 255.0)` and scale stddev to `0.15 * 255 = 38.25` |
| 3.1.5b | Brightness max_delta=0.6 is negligible on [0,255] scale + no clipping after brightness/contrast/saturation | MEDIUM | Scale max_delta to `0.6 * 255 = 153.0`, add `tf.clip_by_value(image, 0.0, 255.0)` after each op |
| 7.3.4 | **Lambda-wrapped EfficientNetB3 backbone is NOT training.** model.trainable_weights=19, backbone has 338 tensors invisible to optimizer. EfficientNet is a frozen ImageNet feature extractor. No domain adaptation occurs. | **CRITICAL** | Replace `Lambda(lambda img: base_model(img), ...)` with direct `base_model(image_input)` call in builders.py:137-140 |
| 5.3.5 | `FOCAL_ORDINAL_WEIGHT=0.5` is dead code in production — actual ordinal_weight is 0.05 (10x lower) from `.get()` fallback | MEDIUM | Inject `ordinal_weight` into single-config dict, or update `.get()` default to 0.5 |
| 5.4.4 | `class_weight` dict computed/logged but NEVER passed to `model.fit()` | LOW | Not a real bug — focal loss alpha handles class weighting. But dead code is misleading. |

---

## Phase 1 Notes

**Status: COMPLETE — ALL PASS (1 minor warning)**

### 1.1 Image File Paths
- 1.1.1 PASS: `image_folder` resolves to `data/raw/Depth_RGB`, folder exists with 2,884 PNGs
- 1.1.2 PASS: All 2,884 files match `_P{patient}{appt}{dfu}` naming pattern
- 1.1.3 PASS: `best_matching.csv` has `depth_rgb` column, 3,108 rows, 0 NaN
- 1.1.4 PASS: 0 missing files; 54 orphan disk files not in CSV (unused, not error)
- 1.1.5 PASS: BB columns present, 0 NaN/negative, 1 valid zero `ymin` (edge case)
- 1.1.6 PASS: All bounding boxes within 1280x720 image dimensions

### 1.2 CSV Data Linkage
- 1.2.1 PASS: `extract_info_from_filename()` regex correct; DFU# intentionally discarded (comes from BB file)
- 1.2.2 PASS: 3-key join (Patient#/Appt#/DFU#) in `create_best_matching_dataset()` is correct
- 1.2.3 PASS: Label mapping I→0, P→1, R→2 consistent across all 6 codebase occurrences
- 1.2.4 WARNING: 6 rows (Patient 222) have duplicate BB annotations — 0.19% of data, negligible impact
- 1.2.5 PASS: Label distribution I=28.7%, P=60.5%, R=10.8% — clinically reasonable

---

## Phase 2 Notes

**Status: COMPLETE — ALL PASS (1 code quality warning)**

### 2.1 Image Decoding
- 2.1.1 PASS (WARNING): `tf.io.decode_jpeg` used on PNG files — works per TF docs but misleading name
- 2.1.2 PASS: `tf.cast(img, tf.float32)` produces [0.0, 255.0] range
- 2.1.3 PASS: `tf.case` correctly routes depth_rgb to image_folder

### 2.2 Bounding Box Crop
- 2.2.1 PASS: depth_rgb BB used as-is (no FOV adjustment) — correct, same sensor as annotations
- 2.2.2 PASS: Coordinates clamped to image bounds at lines 283-286
- 2.2.3 PASS: Crop dimensions guaranteed positive (`xmax >= xmin+1`, `ymax >= ymin+1`)
- 2.2.4 PASS: `crop_to_bounding_box(img, ymin, xmin, height, width)` matches TF API order

### 2.3 Resize & Padding
- 2.3.1 PASS: Aspect ratio preserved via conditional resize logic (lines 295-312)
- 2.3.2 PASS: Uses `lanczos3` interpolation
- 2.3.3 PASS: IMAGE_SIZE=256 from production_config flows through; config.py:128 is legacy/unused
- 2.3.4 PASS: Zero-padding centered (pad_top/bottom, pad_left/right split evenly)
- 2.3.5 PASS: Shape enforced to (256, 256, 3) via reshape + set_shape

### 2.4 Normalization
- 2.4.1-2.4.3 PASS: RGB_BACKBONE='EfficientNetB3', `_rgb_has_builtin_rescaling=True`, images kept at [0, 255]
- 2.4.5 PASS: EfficientNetB3 confirmed to have built-in Rescaling(1/255) as layer 1

---

## Phase 3 Notes

**Status: COMPLETE — 2 BUGS FOUND**

### 3.1 General Augmentation
- 3.1.1 PASS: `USE_GENERAL_AUGMENTATION = True`
- 3.1.2 PASS: Augmentation gated behind `is_training` at dataset_utils.py:533
- 3.1.3 PASS: depth_rgb settings present (brightness ±0.6, contrast 0.6-1.4, sat 0.6-1.4, noise σ=0.15)
- 3.1.4 PASS: Overall prob 60%; per-op: brightness 60%, contrast 60%, saturation 40%, noise 30%
- 3.1.5 **BUG (CRITICAL)**: `tf.clip_by_value(image + noise, 0.0, 1.0)` at line 341 crushes [0,255] images to [0,1]. Affects ~18% of training samples. Same bug at line 388 for maps.
- 3.1.5b **BUG (MEDIUM)**: No clipping after brightness/contrast/saturation. Also `max_delta=0.6` is negligible on [0,255] scale (0.24% brightness change instead of intended 60%).
- Root cause: Augmentation code written for [0,1] range, never updated when preprocessing changed to [0,255] for EfficientNet.

### 3.2 Generative Augmentation
- 3.2.1 PASS: `USE_GENERATIVE_AUGMENTATION = False` — disabled in production

---

## Phase 4 Notes

**Status: COMPLETE — ALL PASS (1 WARNING on trainable weights)**

### 4.1 Backbone
- 4.1.1-4.1.6 ALL PASS: EfficientNetB3, include_top=False, pooling='avg', 1536D, trainable=True, Lambda wrapper

### 4.2 Projection Head
- 4.2.1-4.2.7 ALL PASS: 1536→512→256→128→64 with correct BN/Dropout/L2/he_normal/Attention

### 4.3 Classification Head
- 4.3.1-4.3.5 ALL PASS: Standalone softmax, fusion LearnableFusionWeights, init=0.70 RF weight, convex combination

### 4.4 Model Shapes
- 4.4.1 PASS: Input shape (256, 256, 3)
- 4.4.2 WARNING: Model reports 19 trainable weight tensors but EfficientNetB3 has 338. Lambda wrapping may hide backbone weights from model.trainable_weights — backbone might not be training. Needs investigation.

---

## Phase 5 Notes

**Status: COMPLETE — 2 FAILs (1 design issue, 1 dead code), 3 WARNINGs**

### 5.1 Pre-training
- 5.1.1-5.1.8 ALL PASS: Trigger logic, LR=1e-3, focal loss, val_cohen_kappa monitoring, 200 epochs, weight transfer, cache — all correct.

### 5.2 Fusion Training
- 5.2.1-5.2.6 ALL PASS: LR=1e-4, all layers trainable, EarlyStopping(patience=20), ReduceLR(patience=10, factor=0.5), ModelCheckpoint, best weights loaded.

### 5.3 Loss Function
- 5.3.1-5.3.2 PASS: focal_ordinal_loss formula correct
- 5.3.3 WARNING: Alpha normalizes to sum=10 in production (comment says sum=3)
- 5.3.4 PASS: gamma=2.0 via wrapper
- 5.3.5 **FAIL**: `FOCAL_ORDINAL_WEIGHT=0.5` is dead code — only injected into config when `SEARCH_MULTIPLE_CONFIGS=True`. In production, `config.get('ordinal_weight', 0.05)` returns 0.05.
- 5.3.6-5.3.7 PASS

### 5.4 Class Weights
- 5.4.1, 5.4.3 PASS
- 5.4.2 WARNING: Code normalizes alpha to sum=10 but comment says sum=3
- 5.4.4 **FAIL**: `class_weight` dict computed/logged but NEVER passed to `model.fit()`. Not a true bug — focal loss alpha handles weighting — but misleading dead code.
- 5.4.5 WARNING: WeightedF1Score re-normalizes alpha to sum=3; focal loss uses raw sum=10

---

## Phase 6 Notes

**Status: COMPLETE — ALL PASS (1 WARNING)**

### 6.1 Prediction Generation
- 6.1.1-6.1.5 ALL PASS: model.predict() correct, sample_id excluded, softmax outputs, argmax for classes

### 6.2 True Label ↔ Prediction Alignment
- 6.2.1-6.2.7 ALL PASS: One-hot encoding, I=0/P=1/R=2 consistent, CLASS_LABELS correct, confusion matrix labels explicit, same batch order, unaugmented train predictions, validation_steps limiting

### 6.3 Metrics Calculation
- 6.3.1-6.3.5, 6.3.7 ALL PASS: Cohen's Kappa (quadratic), F1 macro/weighted, accuracy, in-training metrics match sklearn
- 6.3.6 WARNING: In-training WeightedF1Score uses alpha weights (favors rare classes) but reported sklearn f1_weighted uses support weights (favors frequent classes). Model optimizes for different metric than reported. Arguably beneficial for minority class protection, but should be documented.

### 6.4 Saved Results
- 6.4.1-6.4.4 ALL PASS: Predictions, labels, sample IDs as .npy; model weights as .weights.h5

---

## Phase 7 Notes

**Status: COMPLETE — 1 CRITICAL FAIL, 1 FAIL, 4 WARNINGs**

### 7.1 Data Leakage Prevention
- 7.1.1-7.1.4 ALL PASS: Patient-level splits, same split for pre-train/fusion, augmentation train-only, deterministic CSV

### 7.2 Reproducibility
- 7.2.1-7.2.3 ALL PASS: Per-fold seeding, deterministic TF ops, cache per fold

### 7.3 Potential Issues
- 7.3.1 WARNING: Legacy IMAGE_SIZE=128 in config.py (unused in production)
- 7.3.2 WARNING: PRETRAIN_LR 10x higher than STAGE1_LR — mitigated by Lambda bug (only head trains)
- 7.3.3 FAIL: class_weight not passed to model.fit() (same as 5.4.4)
- 7.3.4 **CRITICAL FAIL**: Lambda wrapping hides EfficientNetB3 backbone from optimizer. Only 19/357 weight tensors train. Backbone is frozen ImageNet feature extractor — no domain adaptation to DFU images.
- 7.3.5 PASS: Sampling=none + frequency weights intentional
- 7.3.6 WARNING: Validation drop_remainder=True drops up to batch_size-1 samples
- 7.3.7 PASS: Alpha from training split only
- 7.3.8 WARNING: FOCAL_ORDINAL_WEIGHT=0.5 vs 0.05 fallback (same as 5.3.5)

---

## Resolution Tracking

| Finding # | Fix Applied | Verified | Date |
|-----------|------------|----------|------|
| | | | |
