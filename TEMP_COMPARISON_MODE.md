# ⚠️ TEMPORARY COMPARISON MODE ACTIVE

## Current Status

**Smart alpha capping is DISABLED** for comparison testing.

### What Changed

**File**: `src/data/dataset_utils.py` (lines 611-652)

**Before** (smart capping):
```python
# Normalize to sum=1.0 first
alpha_sum = sum(alpha_values)
alpha_values = [alpha/alpha_sum for alpha in alpha_values]

# Cap and redistribute at MAX_ALPHA=0.5
# ... (smart redistribution algorithm)
```

**Now** (original normalization):
```python
# TEMPORARY: Use original normalization for comparison testing
alpha_sum = sum(alpha_values)
alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]
```

This matches `main_original.py` exactly.

## Why

To do an **apples-to-apples** comparison:
- Same alpha calculation
- Same model architecture
- Same data preprocessing

Any metric differences will be due to **code refactoring only**, not algorithmic improvements.

## Expected Results

With identical alpha values, the refactored code should produce:
- ✅ **Identical or near-identical metrics** (< 0.01% difference)
- ✅ **Same alpha values** (e.g., `[0.748, 0.282, 1.971]`)
- ✅ **Same training behavior**

## After Testing

**IMPORTANT**: Restore smart alpha capping after comparison is complete!

```bash
# Revert to restore smart capping
git revert 5e9ec5c

# Or manually uncomment the smart capping code in:
# src/data/dataset_utils.py lines 616-652
```

## Comparison Command

```bash
# Run comparison with identical alpha values
python scripts/simple_comparison_test.py metadata

# Should see matching alpha values in both versions:
# Alpha values (ordered) [I, P, R]: [0.748, 0.282, 1.971]

# Keep files for inspection (skip automatic cleanup)
python scripts/simple_comparison_test.py metadata --no-cleanup

# Test with more data
python scripts/simple_comparison_test.py metadata --data_pct 50
```

## File Management

**Automatic Cleanup** (default):
- Comparison files are automatically deleted after test completes
- Keeps repository clean
- No manual cleanup needed

**Manual Control**:
```bash
# Keep files after test
python scripts/simple_comparison_test.py metadata --no-cleanup

# Manual cleanup later
python -c "from src.utils.config import cleanup_for_resume_mode; cleanup_for_resume_mode('fresh')"
```

**What Gets Created**:
- Model weights: `results/models/*.h5`
- Predictions: `results/checkpoints/*pred*.npy`
- Patient splits: `results/checkpoints/patient_split*.npz`
- CSV results: `results/csv/*.csv`
- TF cache: `results/tf_records/`

**Gitignore Coverage**: ✅ All comparison files are automatically ignored by `.gitignore`

---

**Created**: 2025-12-21
**Purpose**: Controlled comparison testing
**Status**: ⚠️ TEMPORARY - REVERT AFTER TESTING
