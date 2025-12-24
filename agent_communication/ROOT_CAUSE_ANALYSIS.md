# ROOT CAUSE ANALYSIS: Min F1=0.0 Issue

## Problem
Training produces Min F1=0.0 because model predicts ONLY class P (majority class) 100% of the time.

## Investigation Results

### ✅ What Works
1. **Data Loading** (Phase 1): Data loads correctly with 3107 samples, 3 classes
2. **Focal Loss Math** (Phase 5): Alpha weights applied correctly (9.27x for rare class R)
3. **Alpha Calculation**: Inverse frequency values computed correctly [I=0.725, P=0.344, R=1.931]

### ❌ What Doesn't Work
**Phase 6 - CRITICAL FINDING**: Focal loss + alpha weights ALONE are insufficient!

- Model trained with focal loss (gamma=2.0) + inverse frequency alpha
- Training immediately plateaued (loss=6.73, acc=60.5%)
- Model predicts ONLY class P for 100% of samples
- F1 scores: [0.0, 0.754, 0.0] - identical to plain cross-entropy
- **Conclusion**: Alpha=[0.725, 0.344, 1.931] is TOO WEAK for 6:1 class imbalance

## Root Cause Identified

**File**: `src/data/dataset_utils.py:714`

```python
train_data, alpha_values = apply_mixed_sampling_to_df(train_data, apply_sampling=False, mix=False)
```

**THE BUG**: `apply_sampling=False` - **Oversampling is DISABLED!**

The codebase HAS a working RandomOverSampler implementation (lines 662-676) that balances classes, but it's turned OFF.

## Class Imbalance Severity

```
Original distribution:
- Class I: 892 samples (28.7%)
- Class P: 1880 samples (60.5%) ← Majority class
- Class R: 335 samples (10.8%) ← 5.6x fewer than P!

Imbalance ratios:
- P:I = 2.1:1
- P:R = 5.6:1 ← SEVERE
```

With 5.6:1 imbalance, focal loss alone cannot force the model to learn minority classes.

## The Fix

**Change line 714** from:
```python
train_data, alpha_values = apply_mixed_sampling_to_df(train_data, apply_sampling=False, mix=False)
```

**To**:
```python
train_data, alpha_values = apply_mixed_sampling_to_df(train_data, apply_sampling=True, mix=False)
```

This will:
1. Oversample minority classes (I, R) to match majority class (P)
2. Balanced training data → model sees equal samples of each class
3. Combined with focal loss + alpha → model learns all 3 classes
4. Min F1 should be > 0

## Why This Combination Works

1. **Oversampling**: Ensures model sees enough minority class examples during training
2. **Focal Loss**: Focuses learning on hard-to-classify examples
3. **Alpha Weights**: Further prioritizes minority class errors in loss calculation

All three together handle severe class imbalance effectively.

## Expected Result After Fix

- Training data balanced: ~1880 samples per class (from oversampling)
- Model should predict all 3 classes
- Min F1 > 0.3 (reasonable performance on all classes)
- Macro F1 > 0.5

## Verification Plan

1. Enable oversampling: `apply_sampling=True`
2. Run quick test (10 epochs) to verify model predicts all 3 classes
3. Check prediction distribution is not 100% class P
4. Confirm Min F1 > 0
5. Run full training if test passes
