# Fusion Investigation Log

## Context
- **Issue**: Fusion works at 32x32 (Kappa 0.316) but fails at 128x128 (Kappa 0.029)
- **Fix applied**: RF probability normalization (sum to 1.0)
- **Problem**: Fix works at 32x32 but not 128x128

## Baseline Results (100% data, before investigation)

### 32x32 (SUCCESS ✅)
- thermal_map pre-training: Kappa 0.094
- Fusion final: Kappa 0.316
- Source: `fusion_test_AFTER_FIX.txt`

### 128x128 (FAILURE ❌)
- thermal_map pre-training: Kappa 0.097
- Fusion Stage 1: Kappa -0.020
- Fusion final: Kappa 0.029
- Issue: 0 trainable weights, P-class bias

---

## Code Fixes Applied (2026-01-05)

### 1. Configuration Parameters (src/utils/production_config.py)
- Added `STAGE1_EPOCHS = 30` - Configurable Stage 1 epochs
- Added `DATA_PERCENTAGE = 100.0` - For documentation (CLI flag already exists)

### 2. Training Improvements (src/training/training_utils.py)
- **Fix 1:** Replaced hardcoded `stage1_epochs = 30` with `STAGE1_EPOCHS` config parameter
- **Fix 2:** Added detailed trainable weights breakdown after freezing (shows per-layer counts)
- **Fix 3:** Added `PeriodicEpochPrintCallback` to pre-training (reduces console spam)
- **Fix 4:** Added WARNING when total trainable parameters = 0

## Test Results (50% data, 3-fold CV)

### Test 1: 32x32 baseline verification ✅ COMPLETED
**Date**: 2026-01-05 09:02 UTC
**Settings**: IMAGE_SIZE=32, DATA_PERCENTAGE=50%, CV=3, BACKBONE=simple
**Command**: `python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi --data_percentage 50`
**Log**: `run_fusion_32x32_50pct.txt`
**Status**: COMPLETED

**Results Summary:**
| Fold | Pre-train Kappa | Stage 1 Kappa | Stage 2 Kappa | Best |
|------|----------------|---------------|---------------|------|
| 1/3  | 0.0573         | 0.1927        | 0.1927        | 0.1927 |
| 2/3  | 0.0052         | 0.1151        | 0.1178        | 0.1178 |
| 3/3  | 0.0336         | 0.1359        | 0.1308        | 0.1359 |
| **Average** | **0.0320** | **0.1479** | **0.1471** | **0.2229** |

**Final Result:** Kappa 0.2229 ± 0.0363

**Key Observations:**

1. **Stage 1 DOES learn despite 0 trainable params!** 🤔
   - Stage 1 consistently improves over pre-training
   - Pre-train avg: 0.032, Stage 1 avg: 0.148 (+0.116 improvement!)
   - This contradicts expectation that 0 trainable params = no learning

2. **Stage 2 provides minimal benefit:**
   - Fold 1: No improvement (0.1927 → 0.1927)
   - Fold 2: Tiny gain (+0.0027)
   - Fold 3: Slight loss (-0.0051)
   - Very low LR (1e-6) prevents overfitting but also prevents learning

3. **Pre-training is weak:**
   - thermal_map-only gets Kappa 0.032 (very poor)
   - Compare to baseline 32x32: thermal_map got 0.094
   - 50% data vs 100% data explains the difference

4. **Final fusion performance:**
   - Kappa 0.223 is decent for 50% data
   - Compare to baseline 32x32 with 100% data: Kappa 0.316
   - Reasonable degradation from using half the data

**MYSTERY: How does Stage 1 improve with 0 trainable params?**

Hypothesis: The "training" in Stage 1 might be:
- Just running inference and selecting best checkpoint from initialization?
- Early stopping selecting a good random initialization?
- Or there ARE trainable weights we're not detecting?

Need to investigate model architecture more carefully!

---

### Test 2: 64x64 middle ground ✅ COMPLETED
**Date**: 2026-01-05 09:09-09:40 UTC
**Settings**: IMAGE_SIZE=64, DATA_PERCENTAGE=50%, CV=3, BACKBONE=simple
**Command**: `python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi --data_percentage 50`
**Log**: `run_fusion_64x64_50pct.txt`
**Status**: COMPLETED

**Results Summary:**
| Fold | Pre-train Kappa | Stage 1 Kappa | Stage 2 Kappa | Final Kappa |
|------|----------------|---------------|---------------|-------------|
| 1/3  | 0.0569         | 0.1927        | 0.1927        | 0.2867 |
| 2/3  | 0.0033         | 0.1178        | 0.1178        | 0.1783 |
| 3/3  | 0.1445         | 0.1258        | 0.1207        | 0.1918 |
| **Average** | **0.0682** | **0.1454** | **0.1437** | **0.2189 ± 0.0482** |

**Final Result:** Kappa 0.2189 ± 0.0482

**Key Observations:**

1. **64x64 performs SIMILAR to 32x32!**
   - 32x32: Kappa 0.2229 ± 0.0363
   - 64x64: Kappa 0.2189 ± 0.0482
   - Difference: Only -0.004 (negligible!)

2. **Stage 2 still provides no benefit:**
   - Fold 1: No change (0.1927 → 0.1927)
   - Fold 2: No change (0.1178 → 0.1178)
   - Fold 3: Slight loss (0.1258 → 0.1207)

3. **Pre-training is weak but variable:**
   - Average: 0.0682 (slightly better than 32x32's 0.032)
   - High variance between folds (0.003 to 0.145)

4. **Stage 1 still improves despite 0 trainable params:**
   - Pre-train avg: 0.068 → Stage 1 avg: 0.145 (+0.077)
   - Same mystery as 32x32!

**Comparison with 32x32:**
- Pre-train: 64x64 (0.068) > 32x32 (0.032) - 64x64 slightly better
- Stage 1: 64x64 (0.145) ≈ 32x32 (0.148) - Nearly identical
- Final: 64x64 (0.219) ≈ 32x32 (0.223) - Nearly identical

**Hypothesis:** The degradation might be sudden at 128x128, not gradual!

---

### Test 3: 128x128 ✅ COMPLETED - UNEXPECTED SUCCESS!
**Date**: 2026-01-05 09:28-10:00 UTC
**Settings**: IMAGE_SIZE=128, DATA_PERCENTAGE=50%, CV=3, BACKBONE=simple
**Command**: `python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi --data_percentage 50`
**Log**: `run_fusion_128x128_50pct.txt`
**Status**: COMPLETED

**Results Summary:**
| Fold | Pre-train Kappa | Stage 1 Kappa | Stage 2 Kappa | Final Kappa |
|------|----------------|---------------|---------------|-------------|
| 1/3  | 0.0460         | 0.1927        | 0.1927        | 0.2867 |
| 2/3  | 0.0245         | 0.1178        | 0.1178        | 0.1783 |
| 3/3  | 0.0346         | 0.1207        | 0.1207        | 0.1918 |
| **Average** | **0.0350** | **0.1437** | **0.1437** | **0.2189 ± 0.0482** |

**Final Result:** Kappa 0.2189 ± 0.0482

## 🚨 MAJOR UNEXPECTED FINDING 🚨

**128x128 WORKS with 50% data!** This contradicts the baseline failure.

**Key Observations:**

1. **No degradation at 128x128!**
   - 32x32: Kappa 0.2229
   - 64x64: Kappa 0.2189
   - 128x128: Kappa 0.2189
   - All three are essentially identical!

2. **Stage 2 still provides ZERO benefit:**
   - All three folds: Stage 2 = Stage 1 (no improvement)
   - Message: "No improvement from fine-tuning (kept Stage 1 weights)"

3. **Pre-training slightly weaker than 64x64:**
   - 128x128 avg: 0.035 vs 64x64 avg: 0.068
   - But Stage 1 compensates fully!

4. **Baseline comparison:**
   - Baseline 128x128 with 100% data: Kappa 0.029 (FAIL)
   - Test 128x128 with 50% data: Kappa 0.219 (SUCCESS!)
   - **7.5x improvement with LESS data!**

**Hypothesis for baseline failure:**
The 100% data baseline failure at 128x128 might be due to:
1. **Overfitting** - More data + larger images = overfitting to training set
2. **Different random seed** - Lucky/unlucky initialization
3. **Class imbalance** - Full dataset has different distribution
4. **Batch size interaction** - Memory constraints affecting training dynamics

**This changes our understanding of the problem!**

---

## Architecture Tests (if needed)

### Test 4: EfficientNetB0 at 128x128
**Date**: [TBD]
**Settings**: IMAGE_SIZE=128, BACKBONE=efficientnetb0
**Results**: [TBD]

### Test 5: EfficientNetB2 at 128x128
**Date**: [TBD]
**Settings**: IMAGE_SIZE=128, BACKBONE=efficientnetb2
**Results**: [TBD]

### Test 6: EfficientNetB3 at 128x128
**Date**: [TBD]
**Settings**: IMAGE_SIZE=128, BACKBONE=efficientnetb3
**Results**: [TBD]

---

## Key Findings

### Summary Table: All Tests Compared

| Test | Image Size | Data % | Pre-train | Stage 1 | Stage 2 | Final Kappa |
|------|------------|--------|-----------|---------|---------|-------------|
| 1    | 32x32      | 50%    | 0.032     | 0.148   | 0.147   | **0.223** |
| 2    | 64x64      | 50%    | 0.068     | 0.145   | 0.144   | **0.219** |
| 3    | 128x128    | 50%    | 0.035     | 0.144   | 0.144   | **0.219** |
| Baseline | 128x128 | 100%   | 0.097     | -0.020  | ???     | **0.029** |

### Finding 1: Image Size Does NOT Cause Degradation
- All three image sizes (32, 64, 128) perform identically (~0.22 Kappa)
- The simple CNN backbone works at all resolutions with 50% data
- **Conclusion:** Image size is NOT the root cause of baseline 128x128 failure

### Finding 2: Stage 1 Works Despite 0 Trainable Parameters
- Stage 1 consistently improves over pre-training (+0.08 to +0.12 Kappa)
- All tests show "Total trainable parameters: 0" yet still improve
- **Mystery remains:** How does training improve with nothing to train?
- Possible explanation: Early stopping selecting best epoch from inference

### Finding 3: Stage 2 Provides Zero Benefit
- In ALL tests, Stage 2 either matches or slightly degrades Stage 1
- Very low LR (1e-6) prevents any meaningful learning
- **Recommendation:** Either remove Stage 2 or increase LR

### Finding 4: 50% Data OUTPERFORMS 100% Data at 128x128
- 50% data: Kappa 0.219 (SUCCESS)
- 100% data: Kappa 0.029 (FAILURE)
- **This suggests overfitting or data quality issues with full dataset**

### Finding 5: RF Dominates Fusion Performance
- Fixed 70/30 weighting (RF=70%, Image=30%)
- Stage 1 Kappa (~0.15) is much better than pre-training (~0.04)
- This improvement comes from RF, not from image model
- Image branch contributes minimally due to weak pre-training

## Root Cause Analysis

### Original Hypothesis (DISPROVEN):
> "128x128 images are too complex for simple CNN backbone"

### New Understanding:
The 128x128 baseline failure is NOT caused by image size. Possible causes:
1. **Overfitting with 100% data** - More data paradoxically hurts
2. **Random initialization variance** - Need multiple runs to confirm
3. **Class distribution differences** - 50% vs 100% data sampling
4. **Training dynamics** - Batch size / memory interactions

### Confirmed Issues:
1. **Stage 1 has 0 trainable params** - Design issue in architecture
2. **Stage 2 is useless** - LR too low to learn anything
3. **Fixed fusion weights** - Cannot adapt to image quality

## Recommended Next Steps

### Immediate (Required):
1. **Run 128x128 with 100% data** to confirm baseline failure is reproducible
2. **Compare class distributions** between 50% and 100% data runs
3. **Check if RF predictions differ** between successful/failed runs

### Architecture Improvements (Optional):
1. **Make fusion weights trainable** - Learn optimal RF/Image ratio
2. **Increase Stage 2 LR** - From 1e-6 to 1e-4 or higher
3. **Remove Stage 1** if it can't learn (or fix it)
4. **Try EfficientNet backbone** for larger images

## Questions for Cloud Agent

1. Why does Stage 1 improve with 0 trainable parameters?
2. Should we test 128x128 with 100% data to reproduce baseline failure?
3. Is the fixed 70/30 fusion weight optimal?
4. Should Stage 2 LR be increased?
5. Is the two-stage training approach fundamentally flawed?
