# Sampling Strategy Comparison - metadata @ 100%

**Date:** 2026-01-05
**Goal:** Find best sampling strategy for metadata RF at 100% data

---

## Results Summary

| Strategy | Fold 1 | Fold 2 | Fold 3 | **Avg Kappa** | vs Baseline |
|----------|--------|--------|--------|---------------|-------------|
| random   | 0.0136 | 0.0670 | 0.1884 | **0.0897**    | baseline    |
| smote    | 0.0255 | 0.1450 | 0.2095 | **0.1267**    | +41%        |
| combined | 0.1868 | 0.1151 | 0.1913 | **0.1644**    | +83%        |
| combined_smote | 0.1701 | 0.1308 | 0.1944 | **0.1651** | **+84%**    |

---

## Test 1: Random Oversampling (Baseline)

**Console:** "Using simple random oversampling..."
**Log:** `run_metadata_100pct_random.txt`

| Fold | Kappa |
|------|-------|
| 1    | 0.0136 |
| 2    | 0.0670 |
| 3    | 0.1884 |
| **Avg** | **0.0897** |

**Conclusion:** Confirms baseline failure at 100% data (~0.09)

---

## Test 2: SMOTE (Synthetic Oversampling)

**Console:** "Using SMOTE (synthetic oversampling)..."
**Log:** `run_metadata_100pct_smote.txt`

| Fold | Kappa |
|------|-------|
| 1    | 0.0255 |
| 2    | 0.1450 |
| 3    | 0.2095 |
| **Avg** | **0.1267** |

**Improvement:** +41% over baseline
**Conclusion:** SMOTE helps but not as much as combined strategies

---

## Test 3: Combined Sampling (Under + Over)

**Console:** "Using combined sampling (under + over)..."
**Log:** `run_metadata_100pct_combined.txt`

| Fold | Kappa |
|------|-------|
| 1    | 0.1868 |
| 2    | 0.1151 |
| 3    | 0.1913 |
| **Avg** | **0.1644** |

**Improvement:** +83% over baseline
**Conclusion:** Undersampling majority class is key improvement!

---

## Test 4: Combined + SMOTE (Best of Both)

**Console:** "Using combined + SMOTE (under + synthetic over)..."
**Log:** `run_metadata_100pct_combined_smote.txt`

| Fold | Kappa |
|------|-------|
| 1    | 0.1701 |
| 2    | 0.1308 |
| 3    | 0.1944 |
| **Avg** | **0.1651** |

**Improvement:** +84% over baseline
**Conclusion:** Best performer (slightly)

---

## Winner: `combined_smote`

**Kappa: 0.1651 (+84% over baseline)**

**Reasons:**
1. Balances to MIDDLE class (fewer samples) - reduces overfitting
2. Uses SMOTE (synthetic samples) - no exact duplicates
3. Most consistent fold performance

**Close second:** `combined` (0.1644) - nearly identical

---

## Key Insights

1. **Random oversampling HURTS performance** at 100% data
   - Creates too many duplicates (R class: 7.4x duplication)
   - RF overfits to duplicated samples

2. **Undersampling majority is KEY**
   - Combined strategies undersample P class to match I class
   - This reduces dataset size but improves quality

3. **SMOTE alone not enough**
   - SMOTE without undersampling still oversamples to MAX class
   - Better than random but worse than combined

4. **Combined + SMOTE is best**
   - Gets benefits of both: smaller dataset + synthetic samples
   - 84% improvement over baseline!

---

## Next Steps

1. Use `combined_smote` for fusion testing at 100% data
2. Expected fusion Kappa: 0.20-0.25 (vs 0.029 baseline)
3. Stage 1 should have ALL positive kappa values (not negative!)

**Config update:**
```python
SAMPLING_STRATEGY = 'combined_smote'
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```
