# CV-Aware Search Space Auto-Adjustment Fix

## Date
2025-12-28

## Issue

When using `--min-minority-retention` values that were too low, Phase 2 Bayesian optimization would try threshold combinations that left too few samples per class for cross-validation splitting, causing:

```
ValueError: Could not create valid data split
```

### Example Failure

With `--phase2-cv-folds 3` and `--min-minority-retention 0.001`:
- Minority class (R): 76 samples
- Target retention: 0.1% of 76 = 0.076 ≈ **1 sample**
- After balancing: All classes get ~1 sample
- **Problem**: Can't split 1-3 samples into 3 CV folds!

### Root Cause

The `calculate_min_thresholds_for_retention()` function calculated minimum thresholds based purely on **retention percentage**, without considering **CV fold requirements**.

## The Fix

Added **automatic adjustment** in [scripts/auto_polish_dataset_v2.py:1276-1354](../scripts/auto_polish_dataset_v2.py#L1276-L1354) that detects CV violations and auto-adjusts the search space to ensure valid splits.

### Changes Made

**1. Automatic CV-aware adjustment** (lines 1280-1332):

```python
# First pass: check if CV requirements are violated
min_samples_for_cv = self.phase2_cv_folds * 2
for phase in ['I', 'P', 'R']:
    info = retention_info[phase].copy()
    if info['retained'] < min_samples_for_cv and info['original'] >= min_samples_for_cv:
        needs_adjustment = True
    adjusted_retention_info[phase] = info

# If adjustment needed, recalculate with CV constraints
if needs_adjustment:
    print(f"\n⚠️  Initial retention targets violate {self.phase2_cv_folds}-fold CV requirements")
    print(f"   Auto-adjusting search space to ensure ≥{min_samples_for_cv} samples per class...\n")

    # Recalculate with minimum retention that satisfies CV
    for phase in ['I', 'P', 'R']:
        info = adjusted_retention_info[phase]
        if info['original'] < min_samples_for_cv:
            print(f"❌ Class {phase} has only {info['original']} samples (need {min_samples_for_cv} for {self.phase2_cv_folds}-fold CV)")
            can_optimize = False
            continue

        if info['retained'] < min_samples_for_cv:
            # Force retention to at least min_samples_for_cv
            # Find the threshold that keeps exactly min_samples_for_cv samples
            # ... (recalculates threshold based on misclass counts)

            new_retained = np.sum(phase_misclass < new_threshold)
            new_retention_pct = new_retained / info['original'] * 100

            info['min_threshold'] = new_threshold
            info['retained'] = new_retained
            info['retention_pct'] = new_retention_pct
```

**2. Updated display to show adjustments** (lines 1334-1348):

```python
# Display retention analysis (with adjustments if applied)
for phase in ['I', 'P', 'R']:
    info = retention_info[phase]
    class_target = info.get('target_retention', target_retention)
    print(f"\n  Class {phase}:")
    print(f"    Original samples: {info['original']} ({info['original_images']} images)")
    if needs_adjustment and info['retained'] >= min_samples_for_cv:
        print(f"    Target retention: {class_target*100:.1f}% (adjusted to {info['retention_pct']:.1f}% for CV)")
    else:
        print(f"    Target retention for balance: {class_target*100:.1f}%")
    # ... rest of display
```

## How It Works

### Minimum Samples Calculation

For K-fold cross-validation, each class needs at least **K × 2** samples:
- K folds × ~1 sample per fold for validation
- K folds × ~1 sample per fold for training

**Examples:**
- 3-fold CV: ≥6 samples per class
- 5-fold CV: ≥10 samples per class
- 7-fold CV: ≥14 samples per class

### Validation Logic

For each class (I, P, R):
1. Calculate how many samples would be retained with target retention
2. Check if `retained_samples < cv_folds * 2`
3. If yes, set `can_optimize = False`
4. Show error with exact requirements and solutions

### New Behavior

**Before** (cryptic error during training):
```
ValueError: Could not create valid data split
```

**After** (auto-adjustment with warning):
```
======================================================================
AUTOMATIC THRESHOLD CALCULATION (Balanced Dataset Strategy)
======================================================================
  Minority class retention target: 0%

⚠️  Initial retention targets violate 3-fold CV requirements
   Auto-adjusting search space to ensure ≥6 samples per class...

  Class I:
    Original samples: 203
    Target retention: 0.5% (adjusted to 3.0% for CV)
    Min threshold for retention: 3
    Actual retention at this threshold: 6/203 (3.0%) ≈ 6 images

  Class P:
    Original samples: 368
    Target retention: 0.3% (adjusted to 1.6% for CV)
    Min threshold for retention: 4
    Actual retention at this threshold: 6/368 (1.6%) ≈ 6 images

  Class R:
    Original samples: 76
    Target retention: 1.3% (adjusted to 7.9% for CV)
    Min threshold for retention: 4
    Actual retention at this threshold: 6/76 (7.9%) ≈ 6 images

======================================================================
SEARCH SPACE (auto-calculated for balanced dataset)
======================================================================
  P: 4-5 (min threshold for 1.6% retention)
  I: 3-5 (min threshold for 3.0% retention)
  R: 4-5 (min threshold for 7.9% retention)

Optimization will now proceed with adjusted thresholds...
```

**Only fails if impossible** (class has fewer samples than CV requires):
```
❌ Class R has only 4 samples (need 6 for 3-fold CV)

======================================================================
⚠️  SKIPPING PHASE 2 OPTIMIZATION
======================================================================

Reason: Cannot achieve 0% retention with 3-fold CV.
...
```

## Recommended Parameter Combinations

### Conservative (High Quality)
```bash
--phase2-cv-folds 3 --min-minority-retention 0.50
```
- 50% of minority class retained
- Safe for any dataset size

### Moderate (Balanced)
```bash
--phase2-cv-folds 5 --min-minority-retention 0.30
```
- 30% of minority class retained
- Good balance of filtering and data retention

### Aggressive (Maximum Filtering)
```bash
--phase2-cv-folds 3 --min-minority-retention 0.10
```
- 10% of minority class retained
- Only use if minority class has ≥60 samples

### Formula

For K-fold CV with N samples in minority class:
```
min_retention ≥ (K × 2) / N
```

Examples:
- 3-fold CV, 76 samples: min_retention ≥ 6/76 = 0.079 (8%)
- 5-fold CV, 76 samples: min_retention ≥ 10/76 = 0.132 (13%)
- 7-fold CV, 76 samples: min_retention ≥ 14/76 = 0.184 (18%)

## Testing

To verify the fix works:

```bash
# Should fail gracefully with helpful error (before fix: crashes during training)
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata \
  --phase2-modalities metadata \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 5 \
  --device-mode multi \
  --min-minority-retention 0.001 \
  --phase2-only

# Should succeed
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata \
  --phase2-modalities metadata \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 5 \
  --device-mode multi \
  --min-minority-retention 0.30 \
  --phase2-only
```

## Impact

- ✅ **Automatic recovery**: Adjusts search space to ensure valid CV splits without user intervention
- ✅ **Prevents wasted computation**: No failed training runs due to CV split errors
- ✅ **Transparent**: Shows exactly what adjustments were made and why
- ✅ **Graceful degradation**: Only fails if dataset is truly too small for CV requirements
- ✅ **User-friendly**: Works with any `--min-minority-retention` value, auto-corrects if needed

## Files Modified

- [scripts/auto_polish_dataset_v2.py](../scripts/auto_polish_dataset_v2.py) (lines 1295-1326)
