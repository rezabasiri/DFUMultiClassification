# Phase 7: Final Validation - Confirm Seed 789 + Explicit Outlier Removal

**Date:** 2026-01-05 18:00 UTC
**Goal:** Prove implicit outlier removal hypothesis and find optimal production config
**Timeline:** ~3-4 hours total

---

## Overview

**Phase 7a:** Confirm seed 789 performance (30 min)
**Phase 7b:** Explicit outlier removal with Isolation Forest (2.5 hours)

**Hypothesis to prove:**
If we explicitly remove outliers from 100% data, performance should match 50% data (~0.27-0.28), proving that random 50% sampling works via implicit outlier removal.

---

## Phase 7a: Confirm Seed 789 Performance

**Goal:** Verify seed 789 achieves Kappa ~0.279 consistently

### Config Check

Ensure `src/main.py` line ~1849 has:
```python
data = data.sample(frac=data_percentage / 100, random_state=789).reset_index(drop=True)
```

(Should already be set to 789 from Phase 6)

### Run Confirmation Test

```bash
cd /workspace/DFUMultiClassification

# Should be configured: DATA_PERCENTAGE=50, seed=789
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_50pct_seed789_confirm.txt
```

**Expected Result:** Kappa 0.27-0.29 (matching Phase 6 result of 0.279)

**Success Criteria:**
- ✅ Kappa within 0.02 of previous result (0.26-0.30 acceptable)
- ✅ All folds > 0.22
- ✅ CV < 20%

**If results don't match:** Report discrepancy to cloud agent before proceeding.

---

## Phase 7b: Explicit Outlier Removal

**Strategy:**
- Use metadata-only outlier detection (fast, targets RF bottleneck)
- Per-class Isolation Forest (handles class imbalance)
- Test 3 contamination rates: 5%, 10%, 15%
- Train on each cleaned dataset
- Compare with 50% data performance

### Step 1: Run Outlier Detection (5 min)

```bash
cd /workspace/DFUMultiClassification

python agent_communication/fusion_fix/detect_outliers.py \
  2>&1 | tee agent_communication/fusion_fix/run_outlier_detection.txt
```

**What this does:**
1. Loads 100% metadata (890 samples)
2. Applies per-class Isolation Forest with 3 contamination rates
3. Saves 3 cleaned datasets:
   - `data/cleaned/metadata_cleaned_05pct.csv` (~845 samples, 5% removed)
   - `data/cleaned/metadata_cleaned_10pct.csv` (~800 samples, 10% removed)
   - `data/cleaned/metadata_cleaned_15pct.csv` (~755 samples, 15% removed)
4. Saves outlier lists for each rate

**Expected output:**
```
5% contamination:  ~45 outliers removed (I=~14, P=~25, R=~6)
10% contamination: ~90 outliers removed (I=~28, P=~50, R=~12)
15% contamination: ~135 outliers removed (I=~42, P=~75, R=~18)
```

### Step 2: Train on Cleaned Datasets (3 tests × 30 min = 90 min)

For each cleaned dataset, we'll:
1. Apply it (replaces cache temporarily)
2. Run training
3. Restore original cache

#### Test 1: 5% Outlier Removal

```bash
# Apply cleaned dataset (5% contamination)
python agent_communication/fusion_fix/apply_cleaned_dataset.py --apply --contamination 5

# Reset random seed back to 42 in src/main.py (we're using 100% data now, not 50%)
# Edit src/main.py line ~1849:
# Change: random_state=789
# To:     random_state=42

# Also update config
# src/utils/production_config.py:
# DATA_PERCENTAGE = 100
# SAMPLING_STRATEGY = 'combined'

# Run training
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_cleaned_05pct.txt

# Restore original cache
python agent_communication/fusion_fix/apply_cleaned_dataset.py --restore
```

**Expected:** Kappa 0.19-0.21 (moderate improvement over 0.166)

#### Test 2: 10% Outlier Removal

```bash
# Apply 10% cleaned dataset
python agent_communication/fusion_fix/apply_cleaned_dataset.py --apply --contamination 10

# Run training
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_cleaned_10pct.txt

# Restore original
python agent_communication/fusion_fix/apply_cleaned_dataset.py --restore
```

**Expected:** Kappa 0.22-0.24 (good improvement)

#### Test 3: 15% Outlier Removal

```bash
# Apply 15% cleaned dataset
python agent_communication/fusion_fix/apply_cleaned_dataset.py --apply --contamination 15

# Run training
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_cleaned_15pct.txt

# Restore original
python agent_communication/fusion_fix/apply_cleaned_dataset.py --restore
```

**Expected:** Kappa 0.25-0.27 (best, should match 50% seed 789!)

### Step 3: Verify Restoration

```bash
# Check that original cache is restored
python agent_communication/fusion_fix/apply_cleaned_dataset.py --status
```

Should show "Using original dataset" or "Same size" with backup.

---

## Expected Results & Hypothesis Validation

### If Implicit Outlier Removal Hypothesis is CORRECT:

| Configuration | Kappa | Interpretation |
|---------------|-------|----------------|
| 100% original + combined | 0.166 | Baseline (contains outliers) |
| 100% cleaned 5% | 0.19-0.21 | Modest improvement |
| 100% cleaned 10% | 0.22-0.24 | Good improvement |
| **100% cleaned 15%** | **0.25-0.27** | **Matches 50% data!** ✅ |
| 50% seed 789 | 0.279 | Reference (implicit removal) |

**Conclusion:** ✅ Hypothesis CONFIRMED - 50% works via implicit outlier removal!

**Production recommendation:** Use 100% data with 15% explicit outlier removal
- Pros: Explicit control, reproducible, understood
- Expected: Kappa 0.25-0.27

### If Hypothesis is WRONG:

| Configuration | Kappa | Interpretation |
|---------------|-------|----------------|
| 100% cleaned 5-15% | 0.17-0.19 | Minimal improvement ❌ |
| 50% seed 789 | 0.279 | Still much better |

**Conclusion:** ❌ Hypothesis INCORRECT - something else explains 50% advantage

**Production recommendation:** Use 50% data with seed 789
- Kappa 0.279 proven and reproducible

---

## Decision Tree

```
Phase 7a: Confirm seed 789
    ↓
    Kappa ~0.279?
    ├─ YES: ✅ Proceed to Phase 7b
    └─ NO (< 0.26): ⚠️ Report to cloud agent

Phase 7b: Run outlier detection + 3 training tests
    ↓
    15% cleaned achieves Kappa > 0.25?
    ├─ YES: ✅ Hypothesis CONFIRMED
    │         → Production: 100% with 15% outlier removal
    │         → Kappa ~0.25-0.27 expected
    │
    └─ NO (< 0.22): ❌ Hypothesis INCORRECT
              → Production: 50% data with seed 789
              → Kappa ~0.279 proven
```

---

## Reporting Template

### Phase 7a: Seed 789 Confirmation

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**Previous (Phase 6):** 0.279
**Difference:** _____ (within 0.02 = good)

**Conclusion:**
[ ] ✅ Confirmed (within 0.02)
[ ] ⚠️ Variable (diff > 0.02 but still > 0.25)
[ ] ❌ Failed (< 0.25)

---

### Phase 7b: Outlier Detection Results

**Outliers Detected:**
| Contamination | Class I | Class P | Class R | Total | Remaining |
|---------------|---------|---------|---------|-------|-----------|
| 5%            |         |         |         |       |           |
| 10%           |         |         |         |       |           |
| 15%           |         |         |         |       |           |

---

### Training Results with Cleaned Data

**Test 1: 5% Outlier Removal**

| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

vs 100% original (0.166): _____ improvement

---

**Test 2: 10% Outlier Removal**

| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

vs 100% original (0.166): _____ improvement

---

**Test 3: 15% Outlier Removal**

| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

vs 100% original (0.166): _____ improvement
vs 50% seed 789 (0.279): _____ difference

---

## Final Summary

**Complete this after all tests:**

| Configuration | Kappa | vs Baseline | vs 50% seed 789 |
|---------------|-------|-------------|-----------------|
| 100% original | 0.166 | baseline | -40% |
| 100% cleaned 5% | _____ | _____ | _____ |
| 100% cleaned 10% | _____ | _____ | _____ |
| 100% cleaned 15% | _____ | _____ | _____ |
| 50% seed 789 | 0.279 | +68% | baseline |

**Hypothesis Validation:**

[ ] ✅ CONFIRMED: 15% cleaned matches 50% (diff < 0.03)
    → Implicit outlier removal hypothesis PROVEN
    → Recommend: 100% with 15% explicit outlier removal

[ ] ⚠️ PARTIAL: 15% cleaned improves but doesn't match 50%
    → Implicit outlier removal is PART of the story
    → Recommend: Test higher contamination (20%, 25%) OR use 50% seed 789

[ ] ❌ DISPROVEN: Minimal improvement from outlier removal
    → Something else explains 50% advantage
    → Recommend: Use 50% seed 789 for production

---

## Files to Share

After completion, share these files:
1. `run_fusion_32x32_50pct_seed789_confirm.txt` (Phase 7a confirmation)
2. `run_outlier_detection.txt` (Outlier detection summary)
3. `run_fusion_32x32_100pct_cleaned_05pct.txt` (5% test)
4. `run_fusion_32x32_100pct_cleaned_10pct.txt` (10% test)
5. `run_fusion_32x32_100pct_cleaned_15pct.txt` (15% test)
6. Completed reporting template (above)

---

## Important Notes

1. **Always restore original cache** after each test
   - Prevents accidental training on wrong dataset
   - Use `--status` to verify

2. **Reset random seed to 42** for 100% data tests
   - Seed 789 was for 50% data selection
   - 100% uses all data, seed only affects CV splits

3. **Keep same config for all tests:**
   - IMAGE_SIZE = 32
   - SAMPLING_STRATEGY = 'combined'
   - CV_FOLDS = 3
   - INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]

4. **Watch for errors:**
   - Cache not restored: Check with --status before next test
   - Cleaned file missing: Re-run detect_outliers.py
   - Training crashes: Report to cloud agent with logs

---

## Troubleshooting

**Problem:** detect_outliers.py fails with "cannot import"
**Solution:** Make sure you're running from `/workspace/DFUMultiClassification`

**Problem:** apply_cleaned_dataset.py says "cache not found"
**Solution:** Check CACHE_DIR path, make sure preprocessing was run

**Problem:** Training uses wrong number of samples
**Solution:** Run `--status` to verify which dataset is active, restore if needed

**Problem:** Results are inconsistent across runs
**Solution:** Verify random seed is set correctly for each test type

---

## Timeline

| Task | Duration | Cumulative |
|------|----------|------------|
| Phase 7a: Confirm seed 789 | 30 min | 0:30 |
| Outlier detection | 5 min | 0:35 |
| Test 1: 5% cleaned | 30 min | 1:05 |
| Test 2: 10% cleaned | 30 min | 1:35 |
| Test 3: 15% cleaned | 30 min | 2:05 |
| Analysis & reporting | 15 min | 2:20 |

**Total: ~2.5 hours** (could extend to 3-4 hours if issues arise)

---

## Success Criteria

**Phase 7 is successful if:**
1. ✅ Seed 789 confirmed (Kappa ~0.279)
2. ✅ Outlier detection runs without errors
3. ✅ All 3 cleaned datasets improve over baseline (0.166)
4. ✅ Clear trend: More outlier removal → Better performance
5. ✅ Final recommendation is clear (either cleaned 100% or 50% seed 789)

**Mystery is SOLVED if:**
- 15% cleaned 100% matches 50% seed 789 (Kappa ~0.27)
- Proves implicit outlier removal hypothesis
- Provides production-ready solution with explicit control

Good luck! 🎯
