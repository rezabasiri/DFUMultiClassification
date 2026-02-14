# Investigation: RF Metadata Predictions Ruining Fusion Performance

## Date: 2025-02-13
## Status: Theory proposed, needs independent verification

---

## System Environment

### Hardware
- **GPUs**: 2x NVIDIA RTX A5000 (24 GB VRAM each)
- **OS**: Ubuntu 20.04, Linux 5.4.0-216-generic, x86_64

### Software
- **Python environment**: `/venv/multimodal/bin/python` (Python 3.11.14)
- **TensorFlow**: 2.18.1 (Keras 3.13.2)
- **Key libraries**: numpy 1.26.4, pandas 3.0.0, scikit-learn 1.8.0, imbalanced-learn 0.14.1, pillow 12.1.0
- **Multi-GPU**: TensorFlow MirroredStrategy for distributed training
- **tfdf (TensorFlow Decision Forests)**: NOT installed — falls back to sklearn RandomForestClassifier

### How to run
```bash
# All commands use the multimodal venv Python:
/venv/multimodal/bin/python src/main.py --data_percentage 100 --device-mode multi --verbosity 2 --resume_mode fresh

# Key CLI arguments:
#   --data_percentage 100     Use 100% of data (lower for quick tests, e.g. 40)
#   --device-mode multi       Use MirroredStrategy (2 GPUs). Other: 'single'
#   --verbosity 2             Full debug output (0=silent, 1=summary, 2=detailed)
#   --resume_mode fresh       Ignore existing checkpoints, train from scratch
#   --cv_folds 3              Number of patient-level CV folds (default: 3)
#   --fold N                  Run only fold N (1-indexed), load others from disk

# Which modality combinations to run is controlled in src/utils/production_config.py:
#   INCLUDED_COMBINATIONS = [('metadata', 'depth_rgb'), ('metadata',)]
# See "Modalities & Combinations" section below for all options.
```

### Project location
- Working directory: `/workspace/DFUMultiClassification`
- Git branch: `claude/fix-trainable-parameters-UehxS`

---

## Project Overview

This is a **Diabetic Foot Ulcer (DFU) multi-classification** project that classifies wound images into 3 healing phases:
- **I (Inflammation)**: ~1880 samples (most common)
- **P (Proliferation)**: ~893 samples
- **R (Remodeling)**: ~335 samples (least common)

Class ratio: P:I:R = 5.6:2.7:1 (imbalanced)

### Training pipeline
1. **Confidence-based filtering** — Preliminary RF/NN pass to identify unreliable samples
2. **Patient-level stratified K-fold CV** — Default 3 folds, split at patient level (no patient in both train+val)
3. **Oversampling** — Training data balanced via RandomOverSampler/SMOTE (before RF and NN training)
4. **Per-modality pre-training** — Each image modality trained standalone, weights transferred to fusion model
5. **Fusion training** — End-to-end training with pre-trained image weights + RF metadata

### Cross-validation
- Default: 3-fold patient-level CV (`--cv_folds 3`)
- Split ensures no data leakage: all images from a patient are in the same fold
- Configurable via CLI: `--cv_folds N` or `--fold N` for single fold

---

## Modalities & Combinations

### Available modalities (5 total)
| Modality | Type | Input Shape | Backbone | Description |
|----------|------|-------------|----------|-------------|
| `metadata` | Clinical data | (3,) | None (RF probabilities) | RF predictions from clinical features |
| `depth_rgb` | RGB image | (256, 256, 3) | EfficientNetB3 | Depth camera RGB photograph |
| `depth_map` | Map image | (256, 256, 3) | EfficientNetB1 | Depth map visualization |
| `thermal_rgb` | RGB image | (256, 256, 3) | EfficientNetB3 | Thermal camera RGB photograph |
| `thermal_map` | Map image | (256, 256, 3) | EfficientNetB1 | Thermal map visualization |

### Combination system
- Controlled via `INCLUDED_COMBINATIONS` in `src/utils/production_config.py` (line 387)
- `MODALITY_SEARCH_MODE = 'all'` tests all 31 non-empty subsets of the 5 modalities
- `MODALITY_SEARCH_MODE = 'custom'` uses only `INCLUDED_COMBINATIONS`
- Examples:
  ```python
  INCLUDED_COMBINATIONS = [
      ('metadata',),                                    # Metadata only (1 modality)
      ('depth_rgb',),                                   # Single image (1 modality)
      ('metadata', 'depth_rgb'),                        # Metadata + 1 image (2 modalities)
      ('metadata', 'depth_rgb', 'thermal_rgb'),         # Metadata + 2 images (3 modalities)
      ('depth_rgb', 'thermal_map'),                     # 2 images, no metadata (2 modalities)
      ('metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map'),  # All 5
  ]
  ```

### Architecture varies by number of modalities

The model architecture in `src/models/builders.py` (lines 359-557) adapts based on how many modalities are selected and whether metadata is included:

**1 modality:**
- Metadata only: `Activation('softmax')(rf_probs)` — no trainable params
- Image only: `Dense(3, softmax)(image_features)` — standard classifier

**2 modalities (metadata + 1 image):**
- `image_classifier = Dense(3, softmax)(image_features)` → image_probs
- `LearnableFusionWeights([rf_probs, image_probs])` → weighted average
- Only 1 fusion param (sigmoid-constrained blend weight)

**3 modalities (metadata + 2 images):**
- Images concatenated → `Dense(32) → BN → Dropout → Dense(3, softmax)` → image_probs
- `LearnableFusionWeights([rf_probs, image_probs])`
- More trainable fusion params (Dense layers between concat and classifier)

**4 modalities (metadata + 3 images):**
- Images concatenated → `Dense(64) → Dense(32) → Dense(3, softmax)` → image_probs
- `LearnableFusionWeights([rf_probs, image_probs])`

**5 modalities (metadata + 4 images):**
- Images concatenated → `Dense(128) → Dense(64) → Dense(32) → Dense(3, softmax)` → image_probs
- `LearnableFusionWeights([rf_probs, image_probs])`

**Any N modalities without metadata:**
- All image branches concatenated → Dense layers → `Dense(3, softmax)(output)`
- No fusion weights, standard classification head

### LearnableFusionWeights (line 26-60 in builders.py)
```python
# Single learnable parameter: fusion_logit
# output = sigmoid(fusion_logit) * rf_probs + (1 - sigmoid(fusion_logit)) * image_probs
# Initialized: sigmoid(0.847) ≈ 0.70 RF / 0.30 Image
# FUSION_INIT_RF_WEIGHT = 0.70 (configurable in production_config.py)
```

---

## The Problem

When running fusion (metadata + depth_rgb), we observe:

### Metadata-only run (0 trainable parameters):
```
Epoch 1: loss: 0.1828 - acc: 0.9622 - val_loss: 0.4935 - val_acc: 0.4071 - val_kappa: 0.0525
Final Kappa: 0.1852
```

### Fusion run (metadata + depth_rgb):
```
Pre-training depth_rgb alone: Best val kappa: 0.1261
Fusion Epoch 1: loss: 0.1641 - acc: 0.9633 - val_loss: 0.5529 - val_acc: 0.4502 - val_kappa: 0.0765
Final Kappa: 0.1967
```

### Critical observation:
- **Training accuracy: 96%** with metadata alone (0 trainable params!)
- **Validation accuracy: 40%** with metadata alone
- The 96% → 40% gap means RF predictions are wildly different between train and val
- The neural network sees near-perfect RF predictions during training, then at validation RF predictions are much weaker
- The user reports: "I know the predictions from the RF models for the 3 classes is much more accurate, but when those probabilities go into the dense layers etc to get processed they are ruined somehow"

---

## My Theory: RF In-Sample vs Out-of-Sample Prediction Mismatch

### How RF predictions are generated (code flow):

**File: `src/data/dataset_utils.py`**

1. **Line 1096**: Training data is oversampled (RandomOverSampler/SMOTE) BEFORE RF training:
   ```python
   train_data, alpha_values = apply_mixed_sampling_to_df(train_data, apply_sampling=True, mix=False)
   ```

2. **Lines 1262-1359** (`is_training=True`): RF models are trained on this oversampled training data:
   ```python
   # Two binary RF classifiers (ordinal decomposition):
   rf_model1: I vs (P+R)  — binary classifier
   rf_model2: R vs (I+P)  — binary classifier
   # Hyperparameters: num_trees=646, max_depth=14, min_samples_split=19
   # Note: tfdf is NOT installed, so the sklearn path (lines 1326-1359) is used
   ```

3. **Lines 1360-1399** (runs for BOTH train and val): RF predicts on `metadata_df`:
   ```python
   # For training: metadata_df = the oversampled training data (SAME data RF was trained on)
   # For validation: metadata_df = original validation data (never seen by RF)
   # sklearn path (lines 1377-1381):
   prob1 = rf_model1.predict_proba(dataset)[:, 1]
   prob2 = rf_model2.predict_proba(dataset)[:, 1]
   ```

4. **Lines 1383-1399**: Combine into 3-class probabilities:
   ```python
   prob_I_unnorm = 1 - prob1           # P(class <= 0) = P(I)
   prob_P_unnorm = prob1 * (1 - prob2) # P(class > 0) * P(class <= 1) = P(P)
   prob_R_unnorm = prob2               # P(class > 1) = P(R)
   # Normalize to sum to 1.0:
   total = prob_I_unnorm + prob_P_unnorm + prob_R_unnorm
   prob_I = prob_I_unnorm / total
   prob_P = prob_P_unnorm / total
   prob_R = prob_R_unnorm / total
   # Stored as rf_prob_I, rf_prob_P, rf_prob_R in the DataFrame
   ```

5. **Lines 1413-1418**: Both splits preprocessed:
   ```python
   source_data = train_data.copy()  # OVERSAMPLED train data
   train_data, rf_model1, rf_model2 = preprocess_split(source_data, is_training=True, ...)
   valid_data, _, _ = preprocess_split(valid_data, is_training=False, rf_model1=rf_model1, rf_model2=rf_model2, ...)
   ```

### The alleged problem:
- RF trains on oversampled training data
- RF then predicts on the SAME oversampled training data → **in-sample predictions → ~96% accuracy**
  - Oversampled data contains exact duplicates (RandomOverSampler) that RF trivially memorizes
  - Deep trees (max_depth=14) with 646 trees can memorize patterns
- RF predicts on unseen validation data → **out-of-sample predictions → ~40% accuracy**
- This 96% vs 40% gap causes the neural network to learn wrong fusion weights

### Why this ruins fusion:
1. During training, the neural network sees near-perfect RF predictions
2. The loss landscape strongly favors relying on RF (because it's "96% accurate" in training)
3. The image branch receives minimal gradient signal (RF already explains the labels)
4. The fusion weight learns to trust RF heavily
5. At validation, RF accuracy drops to ~40%, and the model has no fallback from image features
6. Result: Fusion performs WORSE than either modality alone would suggest

### Proposed fix (not yet implemented):
Use **cross-validated (out-of-fold) RF predictions** for the training set. This gives each training sample an RF prediction from a model that never saw that sample, making train RF accuracy match validation (~40-50%).

---

## What Needs Independent Verification

### 1. Is the theory correct?
- Does the in-sample RF prediction actually cause the train/val accuracy gap?
- Or is there another reason for 96% train / 40% val (e.g., the loss function, data augmentation, label encoding)?
- The user's view: "the two RF models are used to get trained from the metadata in an ordinal way. then the predictions (probabilities) of the metadata path (joined RF models) on the training data is used along with the probabilities from the image branch(s) to do a joint training. so the RF models during their internal training dont need train and vali, because the RF parameters are fixed. but the RF models are used once on the same training data to provide probabilities for the training along with image paths (if they are active in the modalities list) and once again on the valid set to do the validation along with those image path or by itself if no image path is called."

### 2. Check the metadata-only model behavior:
- With 0 trainable parameters and just a softmax activation on RF inputs, loss = 0.1828 and acc = 96.2%
- Is this consistent with near-perfect RF in-sample predictions? Or could something else explain this?
- Note: The metadata-only model is `Activation('softmax', name='output')(branches[0])` — it applies softmax to the RF probabilities. Since RF probabilities are already valid probabilities (sum to 1), softmax would distort them. **Check if this matters.**

### 3. How oversampled is the training data?
- Original class distribution: I:~1880, P:~893, R:~335 (full dataset, before train/val split)
- After train/val split (~80/20): train I:~1500, P:~714, R:~268
- After oversampling (combined strategy): balanced to middle class count (~1500 each?)
- How many duplicate rows exist? Does this affect RF memorization?

### 4. What do the RF predictions actually look like?
- Sample RF predictions from the debug output (training data):
  ```
  [[0.877 0.086 0.037],  # Very confident I → label [1,0,0] correct
   [0.159 0.696 0.145],  # Moderately confident P → label [0,1,0] correct
   [0.857 0.115 0.028],  # Very confident I → label [1,0,0] correct
   [0.885 0.094 0.021],  # Very confident I → label [1,0,0] correct
   [0.825 0.130 0.045]]  # Very confident I → label [1,0,0] correct
  ```
- These are training RF predictions — they're highly confident and correct
- What do validation RF predictions look like? (Not logged currently)

### 5. Alternative explanations to investigate:
- Could the softmax on already-valid RF probabilities be distorting them?
- Could the focal ordinal loss function be interacting badly with RF probabilities?
- Could the oversampling itself be causing issues beyond RF (duplicate images in training)?
- Is the 40% validation accuracy for RF consistent with what RF achieves in standalone evaluation?
- Could the metadata feature engineering / feature selection (Mutual Information, top 40 features) be contributing?
- Are the class weights / alpha values configured correctly across all paths?

---

## Key Code Locations for Investigation

| What | File | Lines |
|------|------|-------|
| Model architecture (all cases) | `src/models/builders.py` | 359-557 |
| Metadata-only model | `src/models/builders.py` | 406-412 |
| Fusion model (2 modalities) | `src/models/builders.py` | 417-441 |
| Fusion model (3 modalities) | `src/models/builders.py` | 447-468 |
| Fusion model (4 modalities) | `src/models/builders.py` | 477-501 |
| Fusion model (5 modalities) | `src/models/builders.py` | 513-540 |
| Image-only models (no metadata) | `src/models/builders.py` | 413-415, 442-445, 469-475, 502-511, 541-553 |
| LearnableFusionWeights | `src/models/builders.py` | 26-60 |
| Image branch (EfficientNet) | `src/models/builders.py` | 79-144 |
| Image branch (projection layers) | `src/models/builders.py` | 146-194 |
| Metadata branch | `src/models/builders.py` | 243-257 |
| Oversampling | `src/data/dataset_utils.py` | 769-929, called at 1096 |
| Feature engineering | `src/data/dataset_utils.py` | 1121-1150 |
| Feature selection (MI) | `src/data/dataset_utils.py` | 1199-1259 |
| RF training (sklearn path) | `src/data/dataset_utils.py` | 1326-1359 |
| RF training (tfdf path, unused) | `src/data/dataset_utils.py` | 1262-1325 |
| RF prediction (both paths) | `src/data/dataset_utils.py` | 1360-1399 |
| Train/val patient split | `src/data/dataset_utils.py` | 641-676 |
| preprocess_split calls | `src/data/dataset_utils.py` | 1413-1418 |
| Fusion training loop | `src/training/training_utils.py` | 1650-1673 |
| Pre-training loop | `src/training/training_utils.py` | 1327-1497 |
| Weight transfer (3 locations) | `src/training/training_utils.py` | 1247-1276, 1296-1320, 1455-1484 |
| Loss function (focal ordinal) | `src/models/losses.py` | 123-148 |
| Production config | `src/utils/production_config.py` | Full file |
| Modality combinations config | `src/utils/production_config.py` | 375-399 |
| Training hyperparams | `src/utils/production_config.py` | 29-91 |

---

## Training Pipeline Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| IMAGE_SIZE | 256 | production_config.py |
| GLOBAL_BATCH_SIZE | 32 | production_config.py |
| N_EPOCHS | 200 | production_config.py |
| PRETRAIN_LR | 1e-3 | production_config.py |
| STAGE1_LR | 1e-4 | production_config.py (used for fusion training) |
| STAGE2_LR | 1e-5 | production_config.py |
| FUSION_INIT_RF_WEIGHT | 0.70 | production_config.py |
| EARLY_STOP_PATIENCE | 20 | production_config.py |
| REDUCE_LR_PATIENCE | 10 | production_config.py |
| EPOCH_PRINT_INTERVAL | 20 | production_config.py |
| RGB_BACKBONE | EfficientNetB3 | production_config.py |
| MAP_BACKBONE | EfficientNetB1 | production_config.py |
| CV folds (default) | 3 | CLI --cv_folds |

---

## Changes Already Made in This Session

### 1. Weight transfer fix (training_utils.py)
- Pre-trained standalone model's `output` Dense(3) weights now transfer to fusion model's `image_classifier` Dense(3)
- Both layers have identical shape (64 input dims → 3 classes), so weights are directly compatible
- Applied at 3 locations: standalone checkpoint loading (~line 1263), cache loading (~line 1310), fresh pre-training (~line 1471)

### 2. Removed two-stage training (training_utils.py)
- Removed Stage 1 (frozen image branch, 20 epochs) + Stage 2 (unfrozen, LR=1e-5)
- Replaced with single end-to-end training at LR=1e-4 using pre-trained weights as initialization
- Stage 1 was harmful: only ~196 trainable params (image_classifier kernel+bias + fusion_logit), couldn't learn meaningful cross-modal interaction, actively degraded performance vs either modality alone
- Removed the freezing step entirely — pre-trained weights stay loaded but all layers remain trainable

### 3. Prior session fixes (still in code):
- **Double normalization bug fix**: EfficientNet (B0-B3) has built-in `Rescaling(scale=1/255)` layer. Pipeline was ALSO dividing by 255, squishing pixels to [0, 0.004]. Fixed with backbone-aware normalization in `dataset_utils.py`
- **GLOBAL_BATCH_SIZE**: 600 → 32 (was creating only 1 step/epoch)
- **PRETRAIN_LR**: 1e-5 → 1e-3 (was too low for pre-training from random head weights)
- **sys.executable fix**: `confidence_based_filtering.py` was using `'python'` instead of `sys.executable`
- **Validation steps**: `ceil` → `floor` for `drop_remainder=True`
- **Diagnostic sample saving**: Added `_save_diagnostic_samples()` function

---

## Raw Experiment Outputs

### Metadata-only confusion matrix (val):
```
        Predicted: I    P    R
Actual Inflam:  176   91   24
Actual Prolif:  371  270   75
Actual Remodl:   42   99   36
```
Kappa: 0.1852, Accuracy: 0.4071

### Fusion (metadata + depth_rgb) confusion matrix (val):
```
        Predicted: I    P    R
Actual Inflam:  158  113   20
Actual Prolif:  293  360   63
Actual Remodl:   36  109   32
```
Kappa: 0.1967, Accuracy: 0.4645

### Previous baseline (before all fixes):
Kappa: 0.2404 +/- 0.027

---

## Data Files

| File | Description |
|------|-------------|
| `data/best_matching/best_matching.csv` | Main dataset CSV with all patient metadata + image paths |
| `data/best_matching/depth_rgb/` | Depth RGB images |
| `data/best_matching/depth_map/` | Depth map images |
| `data/best_matching/thermal_rgb/` | Thermal RGB images |
| `data/best_matching/thermal_map/` | Thermal map images |
| `results/models/` | Saved model checkpoints |
| `results/logs/` | Training logs |
| `results/visualizations/` | Diagnostic samples, training plots |
