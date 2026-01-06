# Task: Enable General Augmentation

**Goal:** Run training with general (non-generative) augmentation enabled for both RGB and map images, then compare with baseline (no augmentation).

**Context:** General augmentation is already implemented in `src/data/generative_augmentation_v2.py` with different settings for RGB vs maps:
- RGB (depth_rgb, thermal_rgb): brightness, contrast, saturation, gaussian noise
- Maps (depth_map, thermal_map): brightness, contrast, gaussian noise (no saturation)

---

## Step 1: Code Changes Required

### Fix 1: Import augmentation in image_processing.py

**File:** `src/data/image_processing.py`
**Line:** After line 13 (after other imports), add:
```python
from src.data.generative_augmentation_v2 import augment_image, AugmentationConfig
```

### Fix 2: Create module-level augmentation config

**File:** `src/data/image_processing.py`
**Line:** After line 22 (after data_paths = ...), add:
```python

# Create augmentation config for general augmentation
_augmentation_config = Augmentation Config()
```

### Fix 3: Fix augment_image call signature

**File:** `src/data/image_processing.py`
**Line:** 397-398

**Change from:**
```python
img_tensor = augment_image(img_tensor, modality,
                        tf.random.uniform([], maxval=1000000, dtype=tf.int32))
```

**Change to:**
```python
# Add batch dimension for augmentation (expects 4D: [batch, height, width, channels])
img_tensor_batched = tf.expand_dims(img_tensor, 0)
img_tensor_batched = augment_image(img_tensor_batched, modality,
                                  tf.random.uniform([], maxval=1000000, dtype=tf.int32),
                                  _augmentation_config)
# Remove batch dimension
img_tensor = tf.squeeze(img_tensor_batched, 0)
```

### Fix 4: Enable augmentation for training data

**File:** `src/data/dataset_utils.py`
**Line:** 229

**Change from:**
```python
augment=False
```

**Change to:**
```python
augment=is_training
```

**Then:** Find where `is_training` is defined (should be earlier in the function around line 187). Make sure it exists and is True for training, False for validation.

---

## Step 2: Add Configuration Flag

**File:** `src/utils/production_config.py`
**After line 56** (after OUTLIER_BATCH_SIZE), add:
```python

# General augmentation (brightness, contrast, saturation for RGB; brightness, contrast for maps)
USE_GENERAL_AUGMENTATION = True  # Enable/disable general augmentation
```

---

## Step 3: Run Experiments

### Experiment 1: Baseline (Already Done) ✓
- Configuration: `USE_GENERAL_AUGMENTATION = False` (or augment=False hardcoded)
- Result: Kappa 0.2976
- Location: Already completed

### Experiment 2: With General Augmentation (NEW)

**Steps:**
1. Apply all code changes above
2. Verify configuration:
   ```python
   # src/utils/production_config.py
   OUTLIER_REMOVAL = True
   OUTLIER_CONTAMINATION = 0.15
   USE_GENERAL_AUGMENTATION = True  # NEW
   INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
   ```

3. Clean previous results:
   ```bash
   python src/main.py --mode search --device-mode single --resume_mode fresh
   ```

4. Monitor for:
   - Augmentation being applied (no errors about undefined augment_image)
   - Training completes successfully
   - Final Kappa score

---

## Expected Behavior

**With augmentation enabled:**
- RGB images: Random brightness ±0.6, contrast 0.6-1.4x, saturation 0.6-1.4x, noise σ=0.15
- Map images: Random brightness ±0.4, contrast 0.6-1.4x, noise σ=0.1
- Applied with 60% probability during training only
- Validation data NOT augmented

**Performance expectation:**
- Augmentation typically improves generalization
- Expected Kappa: 0.28-0.32 (slight improvement or similar to baseline)
- If Kappa drops > 5%, report as potential issue

---

## Step 4: Report Results

Create: `agent_communication/outlier_detection/AUGMENTATION_RESULTS.md`

```markdown
# General Augmentation Experiment Results

**Date:** YYYY-MM-DD

## Configuration

| Setting | Value |
|---------|-------|
| Outlier Removal | True (15%) |
| General Augmentation | True / False |
| Modality Combination | metadata + thermal_map |
| Image Size | 32x32 |

## Results

### Baseline (No Augmentation)
- Kappa: 0.2976 ± 0.08
- Accuracy: 0.5561
- F1 Macro: 0.4937
- Fold 1: 0.3667, Fold 2: 0.3196, Fold 3: 0.2066

### With General Augmentation
- Kappa: [FILL]
- Accuracy: [FILL]
- F1 Macro: [FILL]
- Fold 1: [FILL], Fold 2: [FILL], Fold 3: [FILL]

## Comparison

| Metric | Baseline | With Augmentation | Change |
|--------|----------|-------------------|--------|
| Kappa | 0.2976 | [FILL] | [FILL]% |
| Accuracy | 0.5561 | [FILL] | [FILL]% |
| F1 Macro | 0.4937 | [FILL] | [FILL]% |

## Conclusion

[Analysis of whether augmentation helped, hurt, or had no effect]
```

---

## Troubleshooting

### Error: NameError: name 'augment_image' is not defined
**Fix:** Add import at top of image_processing.py (Fix 1)

### Error: augment_image() takes 4 positional arguments but 3 were given
**Fix:** Add config parameter to call (Fix 3)

### Error: Input must be 4-dimensional
**Fix:** Add/remove batch dimension with expand_dims/squeeze (Fix 3)

### Error: NameError: name 'is_training' is not defined
**Check:** In dataset_utils.py, find where is_training is defined. It should be a parameter to create_cached_dataset function around line 187. If missing, use `augment=True` for training dataset only.

---

## Timeline

| Step | Time | Cumulative |
|------|------|------------|
| Code changes | 10 min | 10 min |
| Test run | 5 min | 15 min |
| Full training (3 folds) | 15 min | 30 min |
| Results analysis | 5 min | 35 min |

**Total:** ~35 minutes

---

**Questions or major issues?** Create bug report and push to branch.
