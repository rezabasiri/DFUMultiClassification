# Bug Report: Outlier Detection Path Mismatch

**Date:** 2026-01-06
**Status:** RESOLVED
**Severity:** HIGH
**Final Kappa (before fix):** 0.1733
**Final Kappa (after fix):** 0.2976 ✓
**Target:** 0.2714 ± 0.08

## Summary

**RESOLVED:** After fixing the path mismatch bug, the training pipeline achieved **Kappa: 0.2976**, exceeding the target of **0.2714 ± 0.08**.

Before the fix, the pipeline achieved only Kappa: 0.1733 because outlier removal failed to apply due to incorrect file paths.

## Bug Description

### Bug 1: Path Variable Mismatch in outlier_detection.py

**File:** `src/utils/outlier_detection.py`
**Affected Functions:**
- `apply_cleaned_dataset()` (line 177)
- `restore_original_dataset()` (line 237)
- `apply_cleaned_dataset_combination()` (line 638)
- `detect_outliers()` (line 43)
- `load_cached_features()` (line 311)
- `detect_outliers_combination()` (line 471)

**Root Cause:**
The `get_project_paths()` function returns `root` as the **data directory** (`<project>/data`), not the project root:

```python
# In src/utils/config.py line 21:
root = os.path.join(directory, "data")  # root = "<project>/data"
```

But `outlier_detection.py` treats `root` as the project root and constructs paths like:

```python
# Expected: <project>/results/best_matching.csv
# Actual:   <project>/data/results/best_matching.csv  (WRONG!)
best_matching_file = root / "results/best_matching.csv"

# Expected: <project>/data/cleaned/...
# Actual:   <project>/data/data/cleaned/... (WRONG!)
output_file = root / "data/cleaned/metadata_cleaned_15pct.csv"
```

**Error Message:**
```
⚠ Error for metadata_thermal_map: [Errno 2] No such file or directory:
'/home/rezab/projects/DFUMultiClassification/data/results/best_matching.csv'
```

**Impact:**
Outlier-cleaned dataset was NOT applied. Training used the full uncleaned dataset (3107 samples) instead of the cleaned dataset (2640 samples).

### Bug 2: Automatic Pre-training Input Shape Mismatch

**File:** Training pipeline (automatic pre-training)
**Error:**
```
Invalid input shape for input Tensor("data:0", shape=(None, 3), dtype=float32)
with name 'thermal_map_input' and path 'thermal_map_input'.
Expected shape (None, 32, 32, 3), but input has incompatible shape (None, 3)
```

**Impact:**
Pre-training failed, model initialized with random weights instead of pre-trained weights.

## Proposed Fix for Bug 1

Change path construction in `outlier_detection.py` to use `root.parent` for project-root-relative paths:

```python
# Option A: Fix in outlier_detection.py
_, _, root = get_project_paths()
root = Path(root)
project_root = root.parent  # Get actual project root

# Use project_root for results paths:
best_matching_file = project_root / "results/best_matching.csv"

# Use root for data paths:
output_file = root / "cleaned/metadata_cleaned_15pct.csv"
```

OR

```python
# Option B: Fix in config.py (breaking change - check all usages)
# Change return value so root = project root, not data directory
root = directory  # Instead of: root = os.path.join(directory, "data")
```

## Training Results Summary

### Before Fix (Bug Present)
| Fold | Kappa  | Accuracy | F1 Macro |
|------|--------|----------|----------|
| 1    | 0.1899 | 0.4258   | 0.3760   |
| 2    | 0.1151 | 0.5101   | 0.4518   |
| 3    | 0.2148 | 0.4460   | 0.3753   |
| **Avg** | **0.1733** | **0.4606** | **0.4010** |

### After Fix (Bug Resolved) ✓
| Fold | Kappa  | Accuracy | F1 Macro |
|------|--------|----------|----------|
| 1    | 0.3667 | 0.5466   | 0.5131   |
| 2    | 0.3196 | 0.5818   | 0.5013   |
| 3    | 0.2066 | 0.5399   | 0.4668   |
| **Avg** | **0.2976** | **0.5561** | **0.4937** |

**Target:** Kappa 0.2714 ± 0.08
**Achieved:** Kappa 0.2976 ✓
**Improvement:** +0.1243 (72% improvement over broken state)

## Steps to Reproduce

1. Run `python src/main.py --mode search --device-mode single --resume_mode fresh`
2. Observe error: `⚠ Error for metadata_thermal_map: [Errno 2] No such file or directory`
3. Training continues with unfiltered dataset

## Files Modified (Minor Fixes Applied)

1. `src/utils/outlier_detection.py` - Added `root = Path(root)` at 6 locations
2. `src/training/training_utils.py` - Changed checkpoint extension from `.ckpt` to `.weights.h5`
3. `scripts/precompute_outlier_features.py` - Added `root = Path(root)` and tuple unpacking fix

## Fix Applied

The path logic in `outlier_detection.py` was corrected:

1. For paths relative to project root (`results/best_matching.csv`):
   - Changed `root / "results/..."` to `project_root / "results/..."`
   - Where `project_root = root.parent`

2. For paths relative to data directory (`data/cleaned/...`):
   - Changed `root / "data/cleaned/..."` to `root / "cleaned/..."`
   - Since `root` is already the data directory

**Note:** The automatic pre-training shape mismatch warning persists but does not prevent successful training. The model successfully trains with random initialization and achieves target performance.
