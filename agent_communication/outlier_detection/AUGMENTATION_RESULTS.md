# General Augmentation Experiment Results

**Date:** 2026-01-06
**Branch:** claude/run-dataset-polishing-X1NHe
**Status:** EXPERIMENT INVALID - Bug discovered

## Summary

This experiment intended to test whether general augmentation (brightness, contrast, saturation, Gaussian noise) improves model performance. **However, a bug caused the augmentation to fail for all images**, making the results equivalent to baseline.

## Bug Description

**Root Cause:** Shape mismatch between config and actual image size.

| Component | Value |
|-----------|-------|
| Expected size (config) | 64x64 |
| Actual image size | 32x32 |

**Technical Details:**
- `src/data/image_processing.py:26` creates a global `_augmentation_config = AugmentationConfig()` with default 64x64 output size
- `src/training/training_utils.py:946-947` updates a **local** `aug_config` to 32x32, but this never updates the global config used by `image_processing.py`
- When `augment_image()` is called, `tf.ensure_shape()` at `generative_augmentation_v2.py:171-176` fails because it expects `[None, 64, 64, 3]` but gets `[1, 32, 32, 3]`
- Exceptions are caught and images fall back to non-augmented versions

**Error Messages Observed:**
```
Error in augment_image: Dimension 1 in both shapes must be equal, but are 64 and 32.
A deterministic GPU implementation of AdjustContrastv2 is not currently available.
```

## Results (Invalid - No Augmentation Applied)

| Metric | With Augmentation | Baseline | Change | % Change |
|--------|------------------|----------|--------|----------|
| Cohen's Kappa | 0.2977 | 0.2976 | +0.0001 | +0.03% |
| Accuracy | 0.5616 | 0.5561 | +0.0055 | +0.99% |
| Macro F1 | 0.4951 | 0.4937 | +0.0014 | +0.28% |
| Weighted F1 | 0.5629 | 0.5568 | +0.0061 | +1.10% |

**Note:** Results are virtually identical because augmentation failed silently.

## Per-Class F1 Scores

| Class | With Augmentation | Baseline |
|-------|------------------|----------|
| I (Infection) | 0.4668 | - |
| P (Peripheral) | 0.6420 | - |
| R (Registered) | 0.3766 | - |

## Required Fix

To enable augmentation, update `image_processing.py` to pass the correct image size to `AugmentationConfig`, or expose a function to update the global config:

**Option 1:** Update global config during initialization
```python
# In image_processing.py, add a function:
def set_augmentation_config(image_size):
    global _augmentation_config
    _augmentation_config.generative_settings['output_size']['width'] = image_size
    _augmentation_config.generative_settings['output_size']['height'] = image_size
```

**Option 2:** Remove the hardcoded `tf.ensure_shape()` check in `augment_image()` that assumes 64x64

## Recommendation

1. **Fix the bug** before re-running the augmentation experiment
2. Do not merge augmentation changes until the fix is verified
3. Re-run the experiment after the fix to get valid results

## Warnings and Other Issues (Non-Critical)

| Warning | Severity | Notes |
|---------|----------|-------|
| sklearn FutureWarning | Low | BaseEstimator deprecation, cosmetic |
| TensorFlow cache warning | Low | Dataset caching optimization |
| cuDNN/cuBLAS registration | Low | Normal TensorFlow startup |
| GPU determinism (AdjustContrastv2) | Medium | Related to augmentation failure |

## Conclusion

**Experiment Status:** INVALID

The augmentation was not applied due to a configuration bug. Results are baseline equivalent. The bug must be fixed before a valid comparison can be made.

**Keep augmentation:** UNDETERMINED - retest required after bug fix
