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

## Test Results (50% data for speed)

### Test 1: 32x32 baseline verification
**Date**: [TBD]
**Settings**: IMAGE_SIZE=32, DATA_PERCENTAGE=50%, CV=3, BACKBONE=simple
**Command**: `python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi`
**Log**: `run_fusion_32x32_50pct.txt`

**Results**:
- thermal_map pre-training: Kappa [TBD]
- Fusion Stage 1 (frozen): Kappa [TBD]
- Fusion final: Kappa [TBD]

**Trainable weights**:
- Pre-training: [TBD]
- Stage 1: [TBD]
- Stage 2: [TBD]

**Observations**:
- [TBD]

**Conclusion**: [TBD]

---

### Test 2: 64x64 middle ground
**Date**: [TBD]
**Settings**: IMAGE_SIZE=64, DATA_PERCENTAGE=50%, CV=3, BACKBONE=simple
**Log**: `run_fusion_64x64_50pct.txt`

**Results**: [TBD]

---

### Test 3: 128x128 current failure
**Date**: [TBD]
**Settings**: IMAGE_SIZE=128, DATA_PERCENTAGE=50%, CV=3, BACKBONE=simple
**Log**: `run_fusion_128x128_50pct.txt`

**Results**: [TBD]

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

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Root Cause

[To be determined after investigation]

## Recommended Solution

[To be determined after investigation]
