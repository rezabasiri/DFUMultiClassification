# General Augmentation Experiment Results

**Date:** 2026-01-06
**Branch:** claude/run-dataset-polishing-X1NHe
**Status:** COMPLETED - Augmentation working, minimal impact

## Summary

This experiment tested whether general augmentation (brightness, contrast, saturation, Gaussian noise) improves model performance. After fixing two bugs, the augmentation runs correctly but provides **negligible improvement** over baseline.

## Bugs Fixed

### Bug 1: Image Size Mismatch
**Root Cause:** Hardcoded 64x64 in `AugmentationConfig` instead of using `IMAGE_SIZE` from production_config.

**Fix Applied:**
```python
# In generative_augmentation_v2.py
from src.utils.production_config import IMAGE_SIZE

# Changed from:
'output_size': {'height': 64, 'width': 64}
# To:
'output_size': {'height': IMAGE_SIZE, 'width': IMAGE_SIZE}
```

### Bug 2: GPU Determinism Error
**Root Cause:** `tf.image.random_contrast` doesn't have a deterministic GPU implementation, causing failures when `TF_DETERMINISTIC_OPS=1`.

**Fix Applied:**
```python
# In augment_image() - wrap augmentation in CPU context:
with tf.device('/CPU:0'):
    # ... augmentation operations ...
```

## Results (Valid - Augmentation Applied)

| Metric | With Augmentation | Baseline | Change | % Change |
|--------|------------------|----------|--------|----------|
| Cohen's Kappa | 0.2991 | 0.2976 | +0.0015 | +0.50% |
| Accuracy | 0.5587 | 0.5561 | +0.0026 | +0.47% |
| Macro F1 | 0.4930 | 0.4937 | -0.0007 | -0.14% |
| Weighted F1 | 0.5610 | 0.5568 | +0.0042 | +0.75% |

## Per-Class F1 Scores

| Class | With Augmentation | Baseline | Change |
|-------|------------------|----------|--------|
| I (Infection) | 0.4719 | 0.4668 | +0.51% |
| P (Peripheral) | 0.6375 | 0.6420 | -0.70% |
| R (Registered) | 0.3697 | 0.3766 | -1.83% |

## Per-Fold Kappa

| Fold | Kappa |
|------|-------|
| 1 | 0.3672 |
| 2 | 0.3225 |
| 3 | 0.2077 |
| **Mean** | **0.2991** |
| Std | 0.0672 |

## Analysis

1. **Overall Impact:** Negligible - changes are within noise range
2. **Per-Class Impact:**
   - Slight improvement for Infection class (+0.51%)
   - Slight degradation for Peripheral (-0.70%) and Registered (-1.83%)
3. **Variance:** High variance between folds (Kappa: 0.21 to 0.37)

## Recommendation

**Keep augmentation:** NO - disable `USE_GENERAL_AUGMENTATION`

The augmentation adds computational overhead (CPU-based to avoid GPU determinism issues) without meaningful improvement. The ~0.5% Kappa improvement is within the noise range given the high variance (std = 0.067).

## Configuration for Disabling

To disable augmentation, set in `production_config.py`:
```python
USE_GENERAL_AUGMENTATION = False
```

## Warnings (Non-Critical)

| Warning | Severity | Notes |
|---------|----------|-------|
| sklearn FutureWarning | Low | BaseEstimator deprecation, cosmetic |
| TensorFlow cache warning | Low | Dataset caching optimization |
| NodeDef attribute warning | Low | TensorFlow version compatibility |
| Keras input structure warning | Low | Multi-input model structure |

## Conclusion

**Experiment Status:** COMPLETED

General augmentation (brightness, contrast, saturation, Gaussian noise) does not meaningfully improve performance for this dataset. The high variance between folds suggests the model is sensitive to data splits, not augmentation. Consider focusing on other improvements like model architecture or feature engineering.
