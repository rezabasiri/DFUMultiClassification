# Misclassification Tracking Bug Fix

## Date
2025-12-27

## Issue
When running auto_polish with `--phase1-n-runs=1` and `--phase1-cv-folds=3`, some samples had misclassification counts > 1, which should be impossible since each sample appears in validation exactly once per run.

## Root Cause

### The Problem
The `track_misclassifications()` function in [src/evaluation/metrics.py](../src/evaluation/metrics.py) was tracking at **IMAGE LEVEL** instead of **SAMPLE LEVEL**.

### Why This Caused Incorrect Counts

1. **Samples can have multiple images** (e.g., left/middle/right camera views of same wound)
2. **Different images can be misclassified differently** (e.g., left view predicted as "I", right view as "R")
3. **Original deduplication logic** used `['Sample_ID', 'True_Label', 'Predicted_Label']`
4. **Result**: Same sample appeared multiple times in CSV with different predicted labels

### Example of the Bug

Sample P005A03D1 (True Label = "P") with 3 images in one fold:
- Image 1 (left): Predicted "I" → Row 1: (P005A03D1, P, I)
- Image 2 (middle): Predicted "I" → Row 2: (P005A03D1, P, I) [deduplicated with Row 1]
- Image 3 (right): Predicted "R" → Row 3: (P005A03D1, P, R) [kept as separate row!]

After deduplication on `['Sample_ID', 'True_Label', 'Predicted_Label']`:
- (P005A03D1, P, I, count=1)
- (P005A03D1, P, R, count=1)

**Misclass_Count increases by 2 for this fold instead of 1!**

Over 3 folds with similar patterns → counts of 2, 3, or higher even with just 1 run and 3 CV folds.

## The Fix

### Changes to `track_misclassifications()` in [src/evaluation/metrics.py:9-116](../src/evaluation/metrics.py#L9-L116)

**1. Removed `Predicted_Label` from DataFrame (lines 31-36)**
```python
# OLD - Tracked what the prediction was
current_misclass = pd.DataFrame({
    'Patient': ...,
    'Appointment': ...,
    'DFU': ...,
    'True_Label': ...,
    'Predicted_Label': [CLASS_LABELS[int(y)] for y in y_pred[misclassified_mask]]  # REMOVED
})

# NEW - Only track that it was wrong, not how
current_misclass = pd.DataFrame({
    'Patient': sample_ids[misclassified_mask, 0].astype(int),
    'Appointment': sample_ids[misclassified_mask, 1].astype(int),
    'DFU': sample_ids[misclassified_mask, 2].astype(int),
    'True_Label': [CLASS_LABELS[int(y)] for y in y_true[misclassified_mask]]
})
```

**2. Changed deduplication to sample-level (lines 49-51)**
```python
# OLD - Multiple entries per sample if predictions differed
current_misclass = current_misclass.drop_duplicates(
    subset=['Sample_ID', 'True_Label', 'Predicted_Label']
)

# NEW - One entry per sample per fold, regardless of predictions
current_misclass = current_misclass.drop_duplicates(
    subset=['Sample_ID', 'True_Label']
)
```

**3. Updated matching logic when updating counts (lines 59-62 and 92-95)**
```python
# OLD - Tried to match on Predicted_Label (would fail now)
mask = (
    (existing_misclass['Sample_ID'] == row['Sample_ID']) &
    (existing_misclass['True_Label'] == row['True_Label']) &
    (existing_misclass['Predicted_Label'] == row['Predicted_Label'])  # REMOVED
)

# NEW - Only match on Sample_ID and True_Label
mask = (
    (existing_misclass['Sample_ID'] == row['Sample_ID']) &
    (existing_misclass['True_Label'] == row['True_Label'])
)
```

**4. Added documentation (lines 13-16)**
```python
"""
FIXED: Now counts at SAMPLE LEVEL (how many folds was sample misclassified in),
not image level. Each sample counted once per fold regardless of:
- How many images it has
- What the predicted label was
"""
```

## Expected Behavior After Fix

### With `--phase1-n-runs=1 --phase1-cv-folds=3`:
- **Max Misclass_Count**: 3 (misclassified in all 3 folds)
- Each sample counted **once per fold** regardless of:
  - Number of images (1, 2, 3, or more)
  - What the predicted label was (I, P, or R)

### With `--phase1-n-runs=2 --phase1-cv-folds=5`:
- **Max Misclass_Count**: 10 (2 runs × 5 folds = 10 evaluations)
- Each sample evaluated in 5 different folds per run

### CSV Format After Fix
```csv
Patient,Appointment,DFU,True_Label,Sample_ID,Misclass_Count
5,3,1,P,P005A03D1,3
2,0,1,I,P002A00D1,2
...
```

**Note**: No more `Predicted_Label` column - we only care **if** sample was misclassified, not **how**.

## Why This Fix is Correct

1. **Sample-level filtering**: The goal is to identify problematic **samples** (unique wounds), not images
2. **Fold-based counting**: Misclass_Count should represent "in how many folds was this sample wrong?"
3. **Prediction irrelevance**: Whether images were predicted as I, P, or R doesn't matter - we want to know if the sample is consistently difficult

## Impact on Dataset Polishing

After this fix:
- **Phase 1** correctly identifies samples that are consistently misclassified across folds
- **Thresholds** (e.g., `I:12, P:9, R:12`) now properly represent "exclude samples misclassified in ≥N folds"
- **Retention calculations** work as intended (sample-level filtering)

## Testing

To verify the fix works:
1. Delete existing misclassification CSV files
2. Run auto_polish with `--phase1-n-runs=1 --phase1-cv-folds=3`
3. Check `frequent_misclassifications_metadata.csv`
4. **Expected**: All Misclass_Count values should be ≤ 3

## Files Modified
- [src/evaluation/metrics.py](../src/evaluation/metrics.py) (lines 31-116)
