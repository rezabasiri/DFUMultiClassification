# RF Improvement - Phase 2: Advanced Optimization

## Overview

Phase 2 tests additional strategies to improve metadata classifier performance beyond the current baseline (Kappa 0.220 from solution_3_tune_rf.py).

## Testing Strategy

All tests use:
- **100% of dataset** (no data percentage limitation)
- **Patient-level 5-fold CV** (strict, no data leakage)
- **Imputer/scaler fit on train only**
- **Class weights from train only**
- **Same ordinal RF setup** (unless testing alternative decompositions)

## Solutions to Test

### Solution 6: Bayesian Hyperparameter Optimization
**File:** `solution_6_bayesian_optimization.py`

Systematically searches for optimal hyperparameters using Bayesian optimization (scikit-optimize). Optimizes RF1 and RF2 independently with 10 iterations on fold 1, then validates across all 5 folds.

**Search space:**
- n_estimators: [200, 1000]
- max_depth: [5, 30]
- min_samples_split: [5, 30]
- min_samples_leaf: [1, 10]
- max_features: ['sqrt', 'log2', None]
- max_samples: [0.5, 1.0]

**Expected benefit:** May find better hyperparameter combinations than manual tuning.

### Solution 7: Feature Selection
**File:** `solution_7_feature_selection.py`

Tests different numbers of features (k=20, 30, 40, 50, 60, 73) selected by Mutual Information. Reduces noise from irrelevant features.

**Expected benefit:** Removing noisy features may improve generalization.

### Solution 8: Imputation Strategy Comparison
**File:** `solution_8_imputation_strategies.py`

Compares 6 imputation methods:
- KNN (k=3, 5, 10)
- Median
- Mean
- Iterative (MICE)

**Expected benefit:** Current KNN(k=5) might not be optimal for this dataset.

### Solution 9: Alternative Ordinal Decomposition
**File:** `solution_9_alternative_decomposition.py`

Tests 4 decomposition strategies:
- **Current:** RF1=(I vs P+R), RF2=(I+P vs R)
- **Strategy A:** RF1=(I+P vs R), RF2=(I vs P) [reverse order]
- **Strategy B:** RF1=(I vs P), RF2=(P vs R) [sequential]
- **Strategy C:** One-vs-Rest (3 binary classifiers)

**Expected benefit:** Different splits may better capture class boundaries.

### Solution 10: Hybrid Best-of-All
**File:** `solution_10_hybrid_best.py`

Combines best elements from solutions 6-9. **Run this LAST** after updating configuration based on solutions 6-9 results.

## Dependencies

Ensure these packages are installed:
```bash
pip install scikit-optimize  # For Bayesian optimization
```

All other dependencies (sklearn, pandas, numpy) should already be available.

## Execution Order

1. Run solutions 6-9 in parallel (independent)
2. Analyze results and identify best:
   - Imputation method (solution 8)
   - Feature selection k (solution 7)
   - RF hyperparameters (solution 6)
   - Decomposition strategy (solution 9)
3. Update solution_10_hybrid_best.py configuration
4. Run solution 10 to validate combined approach

## Expected Runtime

- Solution 6 (Bayesian): ~15-20 minutes (10 iterations × 3 inner CV × 5 folds)
- Solution 7 (Feature selection): ~10-15 minutes (6 k values × 5 folds)
- Solution 8 (Imputation): ~10-15 minutes (6 methods × 5 folds)
- Solution 9 (Decomposition): ~10-15 minutes (4 strategies × 5 folds)
- Solution 10 (Hybrid): ~3-5 minutes (1 config × 5 folds)

**Total:** ~50-70 minutes for complete phase 2 testing

## Interpreting Results

Each solution reports:
- **Kappa** (primary metric): Target >0.22 for improvement
- **Accuracy**: Secondary metric
- **F1 Macro**: Balance across classes
- **Per-class F1**: Check minority class (R) performance

**Success criteria:**
- Kappa improvement >0.02 (e.g., 0.220 → 0.240)
- All classes have F1 >0.20
- Low variance across folds (robust)

## Troubleshooting

**If solution fails with import errors:**
```bash
pip install scikit-optimize
```

**If Bayesian search is too slow:**
- Reduce n_iter from 10 to 5 in solution 6
- Or run on fewer folds (change n_folds=3)

**If memory issues occur:**
- Solutions run sequentially per fold (low memory)
- If still issues, reduce n_estimators to 300
