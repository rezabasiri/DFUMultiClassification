# Diagnostic Test Results

## Test Results Summary

| Test | Status | Key Findings |
|------|--------|--------------|
| 1. Data Loading | **PASS** | 890 rows, 72 columns. Labels are 'I', 'P', 'R' (not numeric) |
| 2. Feature Engineering | **PASS** | Created BMI, Age bins, etc. 1580 NaN values after encoding |
| 3. Preprocessing | **PASS** | KNN imputation + normalization working correctly |
| 4. RF Training | **PASS** | Acc=52.8%, Kappa=0.0809 ⚠️ **POOR PERFORMANCE** |
| 5. End-to-End | **PASS** | ⚠️ **CRITICAL BUG FOUND** - metadata shape wrong! |

## Critical Metrics

**Test 4 - RF Training (MOST IMPORTANT):**
- Validation Accuracy: **52.8%** ⚠️ (above 30% threshold but still poor)
- Cohen's Kappa: **0.0809** ⚠️ (barely above 0.1 threshold)
- Per-class F1 scores:
  - I: **0.28** (poor)
  - P: **0.67** (acceptable)
  - R: **0.00** ⚠️ **COMPLETE FAILURE** - Cannot predict class R at all!
- Binary classifiers:
  - RF1 (I vs P+R): Train=99.9%, Valid=70.2%
  - RF2 (I+P vs R): Train=99.9%, Valid=84.8%

**Test 5 - Integration:**
- prepare_cached_datasets source: **caching.py**
- Batch metadata_input shape: **(32, 3)** ⚠️ **WRONG! Should be (32, 73)**
- Batch y shape: **(32, 3)** ⚠️ One-hot encoded (expected)
- Label distribution in batch: [12, 13, 7] (I, P, R)

## Issues Found

### Small Issues (Fixed by Local Agent)
- [x] Hardcoded path `/home/user/DFUMultiClassification` → Fixed to use relative path
- [x] `data_paths['metadata']` → Fixed to `data_paths['csv_file']`
- [x] Labels expected as numeric [0,1,2] → Fixed to accept ['I','P','R']
- [x] Test 4 needed label encoding I→0, P→1, R→2 → Added label_map
- [x] Test 5 had wrong number of return values (expected 6, got 4) → Fixed unpacking
- [x] Test 5 called caching.py with `run=0` parameter → Removed for caching.py

### Major Issues (Report to Main Agent)

#### 🚨 CRITICAL BUG #1: Metadata Features Lost in Pipeline
**Location:** Test 5 - End-to-End Integration

**Problem:**
- Expected metadata_input shape: `(32, 73)` where 73 is number of features
- Actual metadata_input shape: `(32, 3)`
- The metadata features are being **collapsed/corrupted** to only 3 values!

**Evidence:**
```
✓ Batch X is dict with keys: ['metadata_input', 'sample_id']
  metadata_input: shape=(32, 3), dtype=<dtype: 'float32'>  ← WRONG!
  sample_id: shape=(32, 3), dtype=<dtype: 'int32'>
✓ Batch y shape: (32, 3), dtype=<dtype: 'float32'>
```

**Impact:**
The model is trying to learn from only 3 features instead of 73. This explains the catastrophic failure!

#### 🚨 CRITICAL BUG #2: RF Ordinal Classifier Poor Performance
**Location:** Test 4 - RF Training

**Problem:**
The RF ordinal classifier (which is used for metadata in `caching.py`) shows poor performance even on simple train/validation split:
- Class R F1-score: **0.00** - Cannot predict class R at all
- Overall Kappa: **0.0809** - Barely above random
- Validation accuracy: **52.8%** - Poor for 3-class problem

**Classification Report:**
```
              precision    recall  f1-score   support

           I       0.65      0.18      0.28        62
           P       0.52      0.92      0.67        90
           R       0.00      0.00      0.00        26

    accuracy                           0.53       178
   macro avg       0.39      0.37      0.32       178
weighted avg       0.49      0.53      0.43       178
```

**Evidence:**
- Class R is completely ignored (0% recall, 0% precision)
- Model heavily biases toward class P (92% recall)
- The ordinal approach is broken for the minority class

## Error Messages

All tests passed after fixing small bugs. No fatal errors.

## Root Cause Analysis

### PRIMARY ROOT CAUSE: Feature Dimension Collapse
The metadata features (73 dimensions) are being reduced to 3 dimensions somewhere in the `caching.py` pipeline. This is likely happening in one of these locations:

1. **Hypothesis 1:** The RF ordinal classifier predictions (`(32, 2)` for two binary classifiers) are being used as features instead of the original metadata
2. **Hypothesis 2:** The data is being incorrectly reshaped/sliced in the TensorFlow dataset creation
3. **Hypothesis 3:** The metadata is being confused with labels (which are also 3 classes)

### SECONDARY ROOT CAUSE: RF Ordinal Classifier Failure
Even if we had all 73 features, the RF classifier itself is performing poorly:
- Cannot predict class R (minority class with 13.3% of data)
- This suggests class imbalance handling is insufficient
- The ordinal binary decomposition (I vs P+R, I+P vs R) may be fundamentally flawed

## Next Steps

### Immediate Action Required:
1. **Inspect `caching.py` metadata handling:**
   - Search for where metadata features are processed
   - Find where the shape goes from 73 → 3
   - Check if RF predictions are being used as features instead of metadata

2. **Inspect RF ordinal implementation:**
   - File: `src/data/caching.py` - search for "RandomForestClassifier"
   - Check how predictions from binary classifiers are combined
   - Verify if original features are preserved or only predictions are used

3. **Recommended fixes:**
   - Option A: Use original 73 metadata features, not RF predictions
   - Option B: Concatenate RF predictions WITH original features (73+2=75 features)
   - Option C: Fix RF training with better class balancing (SMOTE for minority class)

### Investigation Path:
```
Test 3 ✓ → Features shape: (712, 73)
           ↓
Test 5 ✗ → Features shape: (32, 3)  ← BUG HERE!
```

The bug occurs between test 3 (preprocessing) and test 5 (dataset creation), specifically in the `prepare_cached_datasets()` function in `caching.py`.

## Test 6: Production Code Path Results 🚨

**Test 6 - Production (dataset_utils.py):**
- Source: **dataset_utils.py** (actual production code)
- Batch metadata_input shape: **(32, 3)** ✓ (correct - intentional design)
- Batch y shape: **(32, 3)** ✓ (one-hot encoded)
- Label distribution: [6, 16, 10] (I, P, R) ✓

**⚠️ CRITICAL BUG: Probabilities are Standardized!**
```
Metadata probability stats:
  Mean: [-0.28,  0.35, -0.04]  ← Should be ~0.33 each!
  Min:  [-0.83, -0.84, -0.79]  ← Should be 0.0!
  Max:  [ 1.59,  1.66,  1.45]  ← Should be 1.0!
  Row sums: -0.22 to 0.16      ← Should be 1.0!
```

**Root Cause Identified:**
- Location: [dataset_utils.py:983-1002](src/data/dataset_utils.py#L983-L1002)
- Bug: StandardScaler normalizes **ALL** numeric columns including RF probabilities
- The `exclude_cols` list is missing: `'rf_prob_I', 'rf_prob_P', 'rf_prob_R'`
- This converts probabilities to z-scores (mean=0, std=1), destroying their meaning!

**Code causing the bug:**
```python
exclude_cols = [
    'Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'
]
# ❌ MISSING: 'rf_prob_I', 'rf_prob_P', 'rf_prob_R'

cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
scaler = StandardScaler()
train_data[cols_to_normalize] = scaler.fit_transform(train_data[cols_to_normalize])
```

## Conclusion

**Tests 1-3 PASS:** Data loading, feature engineering, and preprocessing work correctly.

**Test 4 MARGINAL PASS:** RF training shows poor performance (52.8% accuracy, Kappa=0.08, class R F1=0.00).

**Test 5 (caching.py):** Shape (32, 3) is intentional design - not a bug. However, caching.py is unused in production.

**Test 6 REVEALS THE ACTUAL BUG:** 🚨
- The RF probabilities are being **standardized** instead of kept as probabilities
- StandardScaler converts `[0.2, 0.5, 0.3]` → `[-0.8, 1.6, -0.4]` (z-scores)
- The model receives meaningless z-scores instead of class probabilities
- This explains the catastrophic 10% accuracy!

**FIX Required:**
Add RF probability columns to `exclude_cols` in [dataset_utils.py:983](src/data/dataset_utils.py#L983):
```python
exclude_cols = [
    'Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax',
    'rf_prob_I', 'rf_prob_P', 'rf_prob_R'  # ← ADD THIS LINE
]
```

**PRIORITY:** This is the root cause of the 10% accuracy. Fix this first before addressing RF classifier performance.
