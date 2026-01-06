# Phase 7 RETEST - Corrected with Proper Sampling Strategy

**Date:** 2026-01-05 21:00 UTC
**CRITICAL REQUIREMENT:** Use `combined` sampling strategy for ALL tests

---

## What Went Wrong in Original Phase 7

**❌ MISTAKE:** Tests used NO proper sampling (3107 samples → Kappa 0.0996)
**✅ CORRECT:** Should use `combined` sampling (~2181 samples → Kappa 0.1664)

**Impact:** Baseline was 40% worse than it should be, invalidating all comparisons.

---

## Why `combined` Sampling is CRITICAL

**From Phase 3-6 results:**

| Sampling Strategy | Performance | Why |
|------------------|-------------|-----|
| None/random | Kappa 0.09-0.10 | Massive class imbalance (P >> I >> R) |
| **combined** | **Kappa 0.16** | **Balances classes optimally** |

**What `combined` does:**
1. Undersamples P class (from 496 to ~276)
2. Oversamples R class (from 118 to ~276)
3. Results in ~276 samples per class = balanced
4. Total: ~828 patient-level → ~2181 image-level samples

**Without `combined`:** RF overfits to majority class P, ignores minority R

---

## VERIFICATION CHECKLIST (MUST DO BEFORE EACH TEST)

**Before running ANY test, verify:**

```bash
# 1. Check production_config.py
grep "SAMPLING_STRATEGY" /workspace/DFUMultiClassification/src/utils/production_config.py

Expected output:
  SAMPLING_STRATEGY = 'combined'  # NOT 'random'!

# 2. If wrong, fix it:
# Edit src/utils/production_config.py line ~38
# Change to: SAMPLING_STRATEGY = 'combined'
```

**During training, verify in logs:**
```bash
# After training starts, check sample count
grep "After.*sampling\|Final data:" [log_file].txt

Expected: ~2181 samples (with combined)
Wrong: 3107 samples (no proper sampling)
```

**If you see 3107 samples, STOP immediately - sampling is not applied!**

---

## Phase 7 Retest Plan

### Test 0: Baseline with `combined` Sampling (VERIFY)

**Purpose:** Confirm baseline matches Phase 6 (Kappa 0.1664)

**Critical config check:**
```python
# src/utils/production_config.py
DATA_PERCENTAGE = 100
SAMPLING_STRATEGY = 'combined'  # ← MUST BE 'combined'!
IMAGE_SIZE = 32
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Run:**
```bash
cd /workspace/DFUMultiClassification

# VERIFY config first
grep "SAMPLING_STRATEGY" src/utils/production_config.py
# Must show: SAMPLING_STRATEGY = 'combined'

python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_100pct_combined_verify.txt

# IMMEDIATELY CHECK sample count
grep -i "after.*sampling\|using.*samples" agent_communication/fusion_fix/run_100pct_combined_verify.txt
```

**Expected result:**
- Sample count: ~2181 (NOT 3107!)
- Kappa: 0.16-0.17 (matching Phase 6)

**If wrong:**
- Stop and check SAMPLING_STRATEGY config
- Do NOT proceed to cleaned tests until baseline is correct

---

### Test 1: 5% Cleaned + `combined` Sampling

**Setup:**
```python
# 1. Load cleaned dataset
python agent_communication/fusion_fix/test_cleaned_data.py 05pct

# 2. VERIFY sampling strategy
grep "SAMPLING_STRATEGY" src/utils/production_config.py
# MUST show 'combined'!

# 3. Verify cleaned dataset is loaded
wc -l /workspace/DFUMultiClassification/results/best_matching.csv
# Should show ~2929 lines (845 patients × ~3.5 images + header)
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_cleaned_05pct_combined.txt

# VERIFY sample count immediately
grep -i "after.*sampling\|using.*samples" agent_communication/fusion_fix/run_cleaned_05pct_combined.txt
```

**Expected:**
- Input: 845 patients → ~2929 images
- After `combined` sampling: ~1800 samples (NOT 2929!)
- Kappa: 0.23-0.25 (better than baseline 0.16)

**Restore:**
```bash
python agent_communication/fusion_fix/test_cleaned_data.py restore
```

---

### Test 2: 10% Cleaned + `combined` Sampling

**Setup:**
```bash
python agent_communication/fusion_fix/test_cleaned_data.py 10pct

# VERIFY sampling strategy
grep "SAMPLING_STRATEGY" src/utils/production_config.py
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_cleaned_10pct_combined.txt
```

**Expected:**
- Input: 800 patients → ~2751 images
- After `combined` sampling: ~1700 samples
- Kappa: 0.24-0.26

**Restore:**
```bash
python agent_communication/fusion_fix/test_cleaned_data.py restore
```

---

### Test 3: 15% Cleaned + `combined` Sampling (CRITICAL)

**Setup:**
```bash
python agent_communication/fusion_fix/test_cleaned_data.py 15pct

# VERIFY sampling strategy
grep "SAMPLING_STRATEGY" src/utils/production_config.py
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_cleaned_15pct_combined.txt
```

**Expected:**
- Input: 761 patients → ~2597 images
- After `combined` sampling: ~1600 samples
- **Kappa: 0.26-0.28** ← If matches 50% seed 789 (0.2786), HYPOTHESIS CONFIRMED!

**Restore:**
```bash
python agent_communication/fusion_fix/test_cleaned_data.py restore
```

---

## Expected Results Summary

**IF Hypothesis is CORRECT:**

| Configuration | Samples After Sampling | Kappa | Status |
|--------------|----------------------|-------|--------|
| 100% baseline + combined | ~2181 | 0.16-0.17 | Baseline (Phase 6) |
| 95% (5% removed) + combined | ~1800 | 0.23-0.25 | Better |
| 90% (10% removed) + combined | ~1700 | 0.24-0.26 | Even better |
| **85% (15% removed) + combined** | **~1600** | **0.26-0.28** | **Matches 50% seed 789!** ✅ |
| 50% seed 789 (reference) | ~1550 | 0.2786 | Proven |

**Conclusion:** Outlier removal + proper sampling = 50% performance
→ Hypothesis CONFIRMED!
→ Production: Use 100% with 15% outlier removal + `combined` sampling

---

**IF Hypothesis is WRONG:**

| Configuration | Kappa | Gap to 50% |
|--------------|-------|-----------|
| 15% cleaned + combined | 0.20-0.22 | -0.06 to -0.08 |
| 50% seed 789 | 0.2786 | baseline |

**Conclusion:** Outlier removal helps but doesn't fully explain 50% advantage
→ Hypothesis INCORRECT
→ Production: Use 50% seed 789 (proven best)

---

## Common Mistakes to AVOID

### ❌ MISTAKE 1: Forgetting to set SAMPLING_STRATEGY

**Symptom:** Log shows 3107 or 2929 samples (full dataset)
**Fix:** Edit production_config.py, set SAMPLING_STRATEGY = 'combined'

### ❌ MISTAKE 2: Using wrong baseline for comparison

**Symptom:** Comparing cleaned results to Kappa 0.0996 baseline
**Fix:** Use Phase 6 baseline (Kappa 0.1664) or rerun Test 0

### ❌ MISTAKE 3: Not restoring dataset between tests

**Symptom:** Training on wrong cleaned dataset
**Fix:** Always run `test_cleaned_data.py restore` after each test

---

## Verification After Each Test

```bash
# 1. Check sample count in log
grep -i "after.*sampling" [log_file].txt

Expected pattern:
  After undersampling: Counter({0: XXX, 1: XXX, 2: YYY})
  After oversampling: Counter({0: ZZZ, 1: ZZZ, 2: ZZZ})

# 2. Check total is reasonable
# Should be ~1600-2200 samples (with combined)
# NOT 2500-3100 samples (no sampling)

# 3. Check final kappa
tail -50 [log_file].txt | grep -i "kappa"
```

---

## Quick Reference Card

**BEFORE EACH TEST:**
1. ✅ Check `SAMPLING_STRATEGY = 'combined'` in production_config.py
2. ✅ Load correct cleaned dataset (or restore for baseline)
3. ✅ Verify sample count after training starts

**DURING TEST:**
- Sample count should be ~1600-2200 (with sampling)
- NOT 2500-3100 (no sampling)

**AFTER TEST:**
- Record kappa for each fold
- Compare with 50% seed 789 (0.2786)
- Restore original dataset

**RED FLAGS:**
- 🚨 Sample count > 2500 → NO SAMPLING APPLIED!
- 🚨 Baseline Kappa < 0.14 → WRONG SAMPLING!
- 🚨 All folds < 0.15 → SOMETHING IS WRONG!

---

## Reporting Template

### Test 0: Baseline Verification

**Config verified:**
- [ ] SAMPLING_STRATEGY = 'combined' ✓
- [ ] DATA_PERCENTAGE = 100 ✓
- [ ] Sample count ~2181 (NOT 3107) ✓

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**Matches Phase 6 (0.1664)?** [ ] Yes / [ ] No

---

### Test 1: 5% Cleaned + combined

**Verification:**
- [ ] Cleaned dataset loaded (845 patients)
- [ ] SAMPLING_STRATEGY = 'combined' verified
- [ ] Sample count after sampling: _____ (expect ~1800)

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**vs Baseline (0.166):** +_____ improvement

---

### Test 2: 10% Cleaned + combined

**Verification:**
- [ ] Cleaned dataset loaded (800 patients)
- [ ] SAMPLING_STRATEGY = 'combined' verified
- [ ] Sample count after sampling: _____ (expect ~1700)

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

---

### Test 3: 15% Cleaned + combined ⭐ KEY TEST

**Verification:**
- [ ] Cleaned dataset loaded (761 patients)
- [ ] SAMPLING_STRATEGY = 'combined' verified
- [ ] Sample count after sampling: _____ (expect ~1600)

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**vs 50% seed 789 (0.2786):** _____ (difference)
**Within 0.02?** [ ] YES → Hypothesis CONFIRMED / [ ] NO → Hypothesis INCORRECT

---

## Final Decision

**IF Test 3 Kappa ≥ 0.26 (within 0.02 of 0.2786):**
✅ **Hypothesis CONFIRMED**
- Outlier removal + proper sampling = 50% performance
- **Production config:**
  ```python
  # Use 100% data with 15% outlier removal
  # Apply outlier detection to create cleaned dataset
  # Use cleaned dataset with SAMPLING_STRATEGY = 'combined'
  Expected Kappa: 0.26-0.28
  ```

**IF Test 3 Kappa < 0.24 (gap > 0.04):**
❌ **Hypothesis INCORRECT**
- Outlier removal helps but doesn't fully explain 50% advantage
- **Production config:**
  ```python
  DATA_PERCENTAGE = 50
  RANDOM_SEED = 789
  Expected Kappa: 0.2786 (proven)
  ```

---

## Timeline

| Test | Setup | Run | Total |
|------|-------|-----|-------|
| Test 0: Baseline verify | 2 min | 30 min | 32 min |
| Test 1: 5% cleaned | 2 min | 30 min | 32 min |
| Test 2: 10% cleaned | 2 min | 30 min | 32 min |
| Test 3: 15% cleaned | 2 min | 30 min | 32 min |
| Analysis | - | 10 min | 10 min |
| **TOTAL** | | | **~2.5 hours** |

---

## Success Criteria

✅ **Test is successful if:**
1. Baseline matches Phase 6 (Kappa 0.16-0.17)
2. Sample counts are correct (~1600-2200 with sampling)
3. SAMPLING_STRATEGY = 'combined' verified for all tests
4. Clear trend: More outlier removal → Better performance
5. Final recommendation is data-driven

✅ **Mystery is SOLVED if:**
- Test 3 (15% cleaned + combined) ≈ 0.27-0.28
- Matches 50% seed 789 within 0.02
- Proves implicit outlier removal hypothesis

---

## Notes

**The key difference from original Phase 7:**
- **Original:** No `combined` sampling → Baseline 0.0996 (terrible)
- **Corrected:** WITH `combined` sampling → Baseline 0.1664 (correct)

**This changes everything:**
- Original showed: +119% improvement (0.0996 → 0.2183)
- Corrected will show: +40-60% improvement (0.166 → 0.23-0.27)
- But it's a FAIR comparison because both use proper sampling!

**Remember:** We're testing if (outlier removal + proper sampling) matches 50% seed 789, NOT if outlier removal alone helps (we already know it does).
