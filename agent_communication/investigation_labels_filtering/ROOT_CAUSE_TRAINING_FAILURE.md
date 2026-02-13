# Root Cause Analysis: Training Restores From Epoch 1

**Date**: 2026-02-13
**Branch**: `claude/fix-trainable-parameters-UehxS`
**Status**: ROOT CAUSES IDENTIFIED, FIXES IN PROGRESS

## Symptom
All training runs restore weights from epoch 1. Models don't improve after first epoch.
Final metrics: Accuracy ~0.37, Kappa ~0.23 (should be much higher for 3-class).

## Evidence From training_fold1.log

### Pre-training (image-only depth_rgb):
```
Epoch 1/200: train_acc=0.3333, val_acc=0.1496, train_kappa=0.0001, val_kappa=0.0000
Epoch 20/200: train_acc=0.4400, val_acc=0.1496, val_kappa=0.0000  ← STILL ZERO
Epoch 21: early stopping. Restoring model weights from epoch 1.
depth_rgb pre-training completed! Best val kappa: 0.0000  ← COMPLETE FAILURE
```

### Stage 1 (frozen image, only image_classifier trainable):
```
Total trainable parameters across all layers: 2 (tensors, ~195 actual params)
Epoch 11: early stopping. Restoring model weights from epoch 1.
Stage 1 completed. Best val kappa: 0.0991
```

### Stage 2 (unfrozen, LR=1e-6):
```
Epoch 11: early stopping. Restoring model weights from epoch 1.
Stage 2: Kappa 0.0991  ← No improvement from Stage 1
```

---

## 6 Root Causes (priority order)

### RC1: Pre-training completely fails (val_kappa=0.0000)
**File**: `src/training/training_utils.py:1340-1425`

EfficientNetB3 has ~12M parameters. Training data: ~471 samples after resampling.
This is massive overfitting. The model memorizes training data (acc 44%) but
produces degenerate predictions on validation (all class R → acc 14.96%).

val_acc=14.96% matches exactly the proportion of class R in validation set:
`Valid dist: {0: 0.299, 1: 0.552, 2: 0.149}` → class 2 = 14.9%.

The pre-trained image weights are USELESS. They get transferred to the fusion
model and frozen. Stage 1 then tries to learn from garbage features.

### RC2: Alpha weights cause degenerate predictions
**File**: `src/models/losses.py:123-143`

Alpha values from original distribution: [0.599, 0.311, 2.09] (I, P, R).
With focal loss gamma=2: `focal_weight = alpha * (1-y_pred)^gamma`

For class R: alpha=2.09 (7x higher than class P at 0.311).
The loss heavily penalizes missing R samples, so the model learns to predict
ALL samples as class R to minimize the penalty. This explains the all-R
validation predictions during pre-training.

### RC3: Stage 1 has only ~195 trainable params
**File**: `src/models/builders.py:394-395`

With frozen image branch, only `image_classifier = Dense(3, softmax)` trains.
Input: 64-dim features → Output: 3 classes = 64*3 + 3 = 195 params.
Converges in 1 epoch (linear classifier on fixed features).

### RC4: Stage 2 LR too low (1e-6)
**File**: `src/training/training_utils.py:1700`

Stage 2 unfreezes image branch and uses LR=1e-6 (100x lower than Stage 1).
With bad pre-trained weights, the model needs HIGHER LR to relearn, not lower.
Result: no meaningful gradient updates, early stopping fires after 10 epochs.

### RC5: Fixed fusion weights (0.70 RF / 0.30 Image) reduce gradient signal
**File**: `src/models/builders.py:397-412`

`output = 0.70*RF + 0.30*Image`. The 0.30 multiplier on image predictions
means gradient flowing to image_classifier is scaled by 0.30.
Combined with RC3 (few params) and RC4 (low LR), the image branch barely learns.

### RC6: steps_per_epoch mismatch for pre-training
**File**: `src/training/training_utils.py:1183,1416`

Master dataset has 717 samples → steps_per_epoch=2 (ceil(717/600)).
Pre-training dataset has 471 samples but uses SAME steps_per_epoch=2.
With batch_size=600 > 471 samples, each epoch cycles the data multiple times
via `repeat()`. Warning: "Your input ran out of data" appears in logs.

---

## Fixes APPLIED (all on branch claude/fix-trainable-parameters-UehxS)

1. **Pre-training uses equal alpha [1,1,1] + lower LR (1e-5)** (RC1, RC2)
   - File: `src/training/training_utils.py:1338-1354`
   - Prevents degenerate all-R predictions during image pre-training

2. **Alpha weight ratio capped at 3x** (RC2)
   - File: `src/data/dataset_utils.py:745-760, 787-800`
   - MAX_ALPHA_RATIO=3.0 prevents extreme class weighting

3. **Stage 2 LR increased from 1e-6 to 1e-5** (RC4)
   - File: `src/training/training_utils.py:1700`
   - 10x more gradient signal for fine-tuning

4. **Fusion weights now LEARNABLE** (initialized 0.70/0.30) (RC5)
   - File: `src/models/builders.py:26-60` (LearnableFusionWeights class)
   - Applied to all 2/3/4/5 modality fusion architectures
   - Adds 1 trainable param (sigmoid-constrained logit) + gradient now flows unscaled

5. **Pre-training monitors val_cohen_kappa instead of val_weighted_f1_score** (RC1)
   - File: `src/training/training_utils.py:1373-1397`
   - Kappa=0 for degenerate predictions, so early stopping won't save bad models

6. **Weight transfer excludes 'output' layer** (RC5)
   - File: `src/training/training_utils.py:1248-1260, 1285-1295, 1435-1447`
   - Standalone model has Dense(3) output; fusion has LearnableFusionWeights

---

## LOCAL AGENT: After fixes are pushed, run:
```bash
cd /home/user/DFUMultiClassification
git pull origin claude/fix-trainable-parameters-UehxS
source /opt/miniforge3/bin/activate multimodal
python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 --resume_mode fresh 2>&1 | tee debug_after_fixes.log
```

### What to look for:
1. Pre-training val_kappa > 0 (was 0.0000 before)
2. Stage 1 best epoch > 1
3. Stage 2 shows improvement over Stage 1
4. Final Kappa > 0.25 (was 0.20)
5. No "Restoring model weights from epoch 1" for all stages
