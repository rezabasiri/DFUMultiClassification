# Phase 4 Results - Mystery Investigation

**Date:** 2026-01-05 13:10-14:10 UTC
**Goal:** Resolve "50% beats 100%" and "0 trainable params" mysteries

---

## Test 1: combined vs combined_smote (Fusion @ 100%)

| Strategy | Fold 1 | Fold 2 | Fold 3 | **Avg Kappa** | SYN_* Errors |
|----------|--------|--------|--------|---------------|--------------|
| combined_smote | 0.1373 | 0.1878 | 0.1687 | **0.1646** | ~399/fold |
| combined | 0.1857 | 0.1091 | 0.2043 | **0.1664** | 0 |

**Winner:** `combined` (+1.1% improvement)
**Conclusion:** `combined` is marginally better and cleaner (no missing image errors)

---

## Test 2: RF Hyperparameter Tuning

| Configuration | n_estimators | max_depth | Training Kappa |
|---------------|--------------|-----------|----------------|
| Baseline (Current) | 300 | None | 0.9982 |
| Reduced Complexity | 100 | 10 | 0.9312 |
| Minimal Complexity | 50 | 5 | 0.5072 |

**Note:** Test only measured TRAINING kappa (same data for train/predict), so cannot detect overfitting.

**Conclusion:** Test inconclusive for overfitting detection. Near-perfect training kappa (0.9982) might indicate overfitting, but we need validation kappa to confirm.

---

## Test 3: Trainable Fusion Weights

| Fusion Type | Stage 1 Trainable Params | Fold 1 | Fold 2 | Fold 3 | **Avg Kappa** |
|-------------|--------------------------|--------|--------|--------|---------------|
| Fixed 70/30 | 0 | 0.1857 | 0.1091 | 0.2043 | **0.1664** |
| Trainable Dense | 21 (6*3+3) | 0.1438 | 0.0000 | 0.1620 | **0.1019** |

**Result:** Trainable fusion is **38.7% WORSE** than fixed fusion!

**Why it failed:**
1. Small initialization (stddev=0.01) starts with near-zero weights
2. Model struggles to learn good RF/image balance from scratch
3. Fold 2 completely collapsed to 0.0000 kappa
4. Fixed 70/30 was already close to optimal for this problem

**Conclusion:** Fixed 70/30 fusion is better. Trainable fusion NOT recommended.

---

## Overall Findings

### Mystery 1: "50% beats 100%" - PARTIALLY SOLVED

**Status:** Partially understood, not fully solved

**Root cause analysis:**
1. `combined` sampling improves 100% data performance (0.029 → 0.1664)
2. But 50% data still beats 100% (0.22 vs 0.1664)
3. RF tuning test inconclusive (only measured training kappa)
4. Remaining gap (~25%) suggests:
   - Data quality issues in full dataset
   - Different effective sample distribution
   - Possible overfitting not captured by RF tuning test

**Recommendation:**
- Use `combined` sampling for 100% data
- Consider using 50% data for best results
- Further investigation needed (e.g., cross-validation RF test, data quality audit)

### Mystery 2: "0 trainable params" - PARTIALLY SOLVED

**Status:** Issue identified, fix NOT recommended

**Root cause:**
- Stage 1 freezes image branch → only fusion weights trainable
- Fixed Lambda weights have 0 trainable params
- This is BY DESIGN to prevent RF quality degradation

**Attempted fix:**
- Trainable Dense fusion layer adds 21 parameters
- Result: 38.7% WORSE performance
- Model fails to learn better weighting than fixed 70/30

**Conclusion:**
- "0 trainable params" in Stage 1 is NOT the real problem
- The fixed 70/30 weighting is actually beneficial
- Learning fusion weights hurts performance

---

## Best Configuration Found

| Parameter | Value |
|-----------|-------|
| Sampling | `combined` |
| Fusion | Fixed 70/30 (NOT trainable) |
| Image size | 32x32 |
| Best Kappa @ 100% | **0.1664** |
| Best Kappa @ 50% | **0.22** (from Phase 2) |

---

## Recommendations for Cloud Agent

1. **Use `combined` sampling** - Cleaner than `combined_smote`, no SYN_* errors
2. **Keep fixed 70/30 fusion** - Trainable fusion made things worse
3. **50% data is still optimal** - Consider using 50% for best performance
4. **Further investigation needed:**
   - Cross-validation RF tuning test (to detect overfitting)
   - Data quality audit of 100% vs 50% samples
   - Different 50% sample seeds to check consistency

---

## Files Generated

- `run_fusion_32x32_100pct_combined.txt` - Test 1 results
- `run_fusion_32x32_100pct_trainable.txt` - Test 3 results
- `run_rf_tuning.txt` - Test 2 results
- `PHASE4_RESULTS.md` - This report
