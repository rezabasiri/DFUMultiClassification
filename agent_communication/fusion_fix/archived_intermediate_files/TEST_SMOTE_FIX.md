# Test SMOTE Fix - Instructions for Local Agent

**Date:** 2026-01-05
**Fix:** Replace random oversampling with SMOTE to fix RF overfitting

---

## What Changed?

### 1. All Augmentations Disabled ✅
- **Generative augmentation**: Disabled (`gen_manager = None` in `training_utils.py:950`)
- **Regular augmentation**: Already commented out (was never active)
- **Result**: Pure, deterministic training with NO augmentations

### 2. SMOTE Implemented ✅
- **File**: `src/data/dataset_utils.py`
- **Change**: Lines 685-694 now use SMOTE when `USE_SMOTE=True`
- **Config**: New parameter `USE_SMOTE` in `src/utils/production_config.py:39`
- **Default**: `USE_SMOTE = True` (enabled by default)

---

## Expected Results

### Baseline (Before SMOTE)
```
metadata-only @ 100% data: Kappa 0.09 ± 0.07
metadata-only @ 50% data:  Kappa 0.22
```

### After SMOTE Fix
```
metadata-only @ 100% data: Kappa 0.15-0.20 (expected)
                            Should be closer to 50% data performance
```

**Why?** SMOTE generates synthetic samples instead of duplicating, preventing RF overfitting to repeated patterns.

---

## Test Plan

### Test 1: Verify SMOTE Works (metadata-only @ 100%)
**Purpose:** Confirm SMOTE improves RF quality

**Command:**
```bash
# Edit config
# src/utils/production_config.py:
#   INCLUDED_COMBINATIONS = [('metadata',)]
#   USE_SMOTE = True

source /opt/miniforge3/bin/activate multimodal
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_smote.txt
```

**Expected Output:**
- Console should print: `"Using SMOTE (synthetic oversampling)..."`
- Final Kappa: **0.15-0.20** (vs baseline 0.09)

**Success Criteria:**
- Kappa > 0.15
- Improvement of at least 60% over baseline (0.09)

---

### Test 2: Verify Fusion Works @ 100% (with SMOTE)
**Purpose:** Confirm fusion no longer fails with 100% data

**Command:**
```bash
# Edit config
# src/utils/production_config.py:
#   INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
#   IMAGE_SIZE = 128
#   USE_SMOTE = True

source /opt/miniforge3/bin/activate multimodal
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_128x128_100pct_smote.txt
```

**Expected Output:**
- Console should print: `"Using SMOTE (synthetic oversampling)..."`
- Stage 1 should be POSITIVE (vs negative with baseline)
- Final Kappa: **0.20-0.25** (vs baseline 0.09)

**Success Criteria:**
- Stage 1 Kappa > 0 (no negative values!)
- Final Kappa > 0.20
- Fusion beats thermal_map alone

---

### Test 3: Compare SMOTE vs Random (Optional)
**Purpose:** Directly compare SMOTE vs random oversampling

**Step 1 - With SMOTE:**
```bash
# USE_SMOTE = True (default)
# Run metadata-only @ 100%
```

**Step 2 - Without SMOTE:**
```bash
# Edit src/utils/production_config.py:
#   USE_SMOTE = False

# Run metadata-only @ 100%
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_random.txt
```

**Compare:**
- SMOTE should outperform random by 60-100%
- Random: Kappa ~0.09
- SMOTE: Kappa ~0.15-0.20

---

## Config Updates Needed

### Before Testing

**File:** `src/utils/production_config.py`

For **Test 1** (metadata-only @ 100%):
```python
IMAGE_SIZE = 128  # Doesn't matter for metadata-only
USE_SMOTE = True  # Already set
INCLUDED_COMBINATIONS = [('metadata',)]  # Change this!
```

For **Test 2** (fusion @ 100%):
```python
IMAGE_SIZE = 128
USE_SMOTE = True
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]  # Change this!
```

---

## Results Format

**Create:** `agent_communication/fusion_fix/SMOTE_TEST_RESULTS.md`

**Template:**
```markdown
# SMOTE Test Results

## Test 1: metadata-only @ 100% with SMOTE
**Date:** [DATE]
**Config:** USE_SMOTE=True, CV=3, metadata-only

**Results:**
| Fold | Kappa |
|------|-------|
| 1    | [X]   |
| 2    | [X]   |
| 3    | [X]   |
| **Avg** | **[X]** |

**Baseline Comparison:**
- Baseline (random): Kappa 0.09 ± 0.07
- With SMOTE: Kappa [X]
- Improvement: [X]%

**Conclusion:** [PASS/FAIL]

---

## Test 2: Fusion @ 100% with SMOTE
**Date:** [DATE]
**Config:** USE_SMOTE=True, CV=3, 128x128 fusion

**Results:**
| Fold | Pre-train | Stage 1 | Final |
|------|-----------|---------|-------|
| 1    | [X]       | [X]     | [X]   |
| 2    | [X]       | [X]     | [X]   |
| 3    | [X]       | [X]     | [X]   |
| **Avg** | **[X]** | **[X]** | **[X]** |

**Baseline Comparison:**
- Baseline (random): Kappa 0.09, Stage 1 negative in 2/3 folds
- With SMOTE: Kappa [X], Stage 1 [positive/negative]
- Improvement: [X]%

**Conclusion:** [PASS/FAIL]
```

---

## Troubleshooting

### If SMOTE fails with "n_neighbors too large"
**Error:** `ValueError: Expected n_neighbors <= n_samples, but n_neighbors = 6, n_samples = 80`

**Solution:** Reduce k_neighbors in `dataset_utils.py:689`:
```python
# Change from:
oversampler = SMOTE(random_state=42 + run * (run + 3), k_neighbors=5)

# To:
oversampler = SMOTE(random_state=42 + run * (run + 3), k_neighbors=3)
```

### If results don't improve
**Check:**
1. Console prints "Using SMOTE..." (confirms SMOTE is active)
2. No other code changes affecting RF training
3. Same random seed as baseline
4. Same CV folds as baseline

---

## Success Metrics

### Minimum Acceptable Results:
- **Test 1:** metadata @ 100% Kappa > 0.15 (vs 0.09 baseline)
- **Test 2:** fusion @ 100% Kappa > 0.20, Stage 1 > 0 (vs 0.09, Stage 1 negative)

### Ideal Results:
- **Test 1:** metadata @ 100% Kappa ≈ 0.20-0.22 (close to 50% data)
- **Test 2:** fusion @ 100% Kappa ≈ 0.25-0.30 (beating all individual modalities)

---

## Next Steps After Testing

### If SMOTE fixes the issue:
1. Report results to cloud agent
2. Cloud agent will implement Fix #2 (trainable fusion)
3. Test fusion @ 100% with both SMOTE + trainable fusion
4. Expected: Kappa 0.25-0.30 (optimal performance)

### If SMOTE doesn't help enough:
1. Try different k_neighbors values (3, 7, 10)
2. Try ADASYN instead of SMOTE
3. Try reducing oversampling ratio (don't balance perfectly)
4. Report back for alternative strategies

---

## Questions?

Contact cloud agent with:
- Full console output from failed test
- Error messages (if any)
- Actual vs expected results
- Any unexpected behavior

Good luck! The SMOTE fix should resolve the RF overfitting issue! 🔧
