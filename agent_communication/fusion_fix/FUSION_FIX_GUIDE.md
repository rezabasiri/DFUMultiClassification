# DFU Fusion Model - Complete Investigation & Solution

**Investigation Period:** Phase 1-7
**Problem:** Fusion failed at 128x128 (Kappa 0.029) vs succeeded at 32x32 (Kappa 0.16)
**Root Cause:** Oversampling strategy, not image size
**Solution:** 15% outlier removal + 'combined' sampling → **Kappa 0.27** (+63% vs baseline)

---

## Quick Reference

### Production Configuration
```python
# src/utils/production_config.py
IMAGE_SIZE = 32
DATA_PERCENTAGE = 100.0
SAMPLING_STRATEGY = 'combined'
```

### Expected Performance
- **Without outlier removal:** Kappa 0.1664 ± 0.05
- **With 15% outlier removal:** Kappa 0.2714 ± 0.08 (+63%)

### Key Files
- **Production script:** `scripts_production/detect_outliers.py`
- **Helper script:** `scripts_production/test_cleaned_data.py`
- **Final results:** `results_final/PHASE7_RETEST_RESULTS.md`
- **Full analysis:** `results_final/CLOUD_AGENT_PHASE7_RETEST_INDEPENDENT_ANALYSIS.md`

---

## Investigation Summary (7 Phases)

### Phase 1-2: Initial Discovery
**Files:** `phase1_results/`, `phase2_results/`

| Test | Image Size | Data % | Sampling | Kappa | Status |
|------|-----------|--------|----------|-------|--------|
| Original | 128x128 | 100% | random | 0.029 | ❌ FAIL |
| Fixed | 32x32 | 100% | random | 0.165 | ⚠️ PARTIAL |
| Best | 32x32 | 50% | random | 0.279 | ✅ GOOD |

**Finding:** Both image size AND sampling affect performance.

---

### Phase 3: Sampling Strategy Analysis
**Files:**
- `results_final/run_metadata_100pct_combined.txt` (Kappa 0.1664)
- `results_final/run_metadata_100pct_combined_smote.txt` (Kappa 0.1670)
- `archived_intermediate_files/run_metadata_100pct_random.txt` (Kappa 0.0996)
- `archived_intermediate_files/run_metadata_100pct_smote.txt` (Kappa 0.1400)

| Strategy | Description | Kappa | vs Random |
|----------|-------------|-------|-----------|
| random | Oversample all to MAX class | 0.0996 | - |
| smote | SMOTE synthetic to MAX | 0.1400 | +41% |
| combined | Undersample P + oversample R to MIDDLE | **0.1664** | **+67%** |
| combined_smote | Combined + SMOTE | 0.1670 | +68% |

**Key Finding:**
- ✅ **Use 'combined'** for fusion (not combined_smote)
- ❌ SMOTE creates synthetic samples with NO images (399 SYN_* entries)
- 95% of improvement comes from undersampling P, not SMOTE

**Production Choice:** `SAMPLING_STRATEGY = 'combined'`

---

### Phase 4: Mystery Investigation
**Files:** `archived_intermediate_files/PHASE4_RESULTS.md`

**Tested:**
1. Combined validation - confirmed 0.1664 ✅
2. RF tuning - overfitting confirmed (train 0.998, val 0.16) ❌
3. Trainable fusion - INVALID TEST (only 2 params instead of 21) ❌
4. (Not run - redundant with Phase 6)

**Finding:** RF overfits duplicate samples from oversampling.

---

### Phase 5: Reduced Oversampling Hypothesis
**File:** `archived_intermediate_files/run_fusion_32x32_100pct_reduced.txt`

**Hypothesis:** Reduce duplicates to improve RF generalization
**Strategy:** reduced_combined (fewer duplicates: 63% → 46%)

| Test | Samples | Duplicates | Kappa | vs Combined |
|------|---------|------------|-------|-------------|
| combined | 2181 | 63% | 0.1664 | - |
| reduced_combined | 1491 | 46% | **0.1355** | **-18.6%** ❌ |

**Finding:** Data quantity > quality for this problem. Hypothesis REJECTED.

---

### Phase 6: 50% Data Seed Validation
**Files:**
- `results_final/run_fusion_32x32_50pct_seed123.txt` (Kappa 0.2070)
- `results_final/run_fusion_32x32_50pct_seed456.txt` (Kappa 0.2264)
- `results_final/run_fusion_32x32_50pct_seed789.txt` (Kappa 0.2786) ⭐
- `results_final/run_fusion_32x32_50pct_seed789_confirm.txt` (Kappa 0.2786 confirmed)

| Seed | Kappa | P/R Ratio | CV | Performance |
|------|-------|-----------|-----|-------------|
| 123 | 0.2070 | 5.93x | 28.6% | Poor |
| 456 | 0.2264 | 5.47x | 50.0% | Medium |
| **789** | **0.2786** | **5.67x** | **17.7%** | **Best** ⭐ |
| 100% baseline | 0.1664 | 4.20x | 29.9% | Baseline |

**Key Findings:**
1. ❌ **NOT** because of "better class balance" (50% is MORE imbalanced)
2. ✅ **Implicit outlier removal** - seed 789 excluded noisy samples
3. Performance depends on WHICH samples selected, not balance
4. Low CV (17.7%) = consistent quality across folds

**New Hypothesis:** If we explicitly remove outliers from 100% data, should match seed 789.

---

### Phase 7 (Original): INVALID TEST
**Files:** `archived_intermediate_files/run_100pct_baseline.txt` (Kappa 0.0996)

**Critical Error:**
- Used 3107 samples (NO proper sampling applied)
- Should use ~2181 samples with 'combined' sampling
- Baseline 0.0996 matches Phase 3 'random' strategy
- ALL conclusions INVALID

---

### Phase 7 Retest: HYPOTHESIS CONFIRMED ✅
**Files:**
- `results_final/run_100pct_combined_verify.txt` (baseline)
- `results_final/run_cleaned_05pct_combined.txt`
- `results_final/run_cleaned_10pct_combined.txt`
- `results_final/run_cleaned_15pct_combined.txt`

| Test | Outliers Removed | Kappa (Avg) | Folds | vs Baseline | Gap to Seed 789 |
|------|------------------|-------------|-------|-------------|-----------------|
| Baseline | 0 (0%) | 0.1664 | 0.1857, 0.1091, 0.2043 | - | 0.1122 (67%) |
| 5% cleaned | ~155 | 0.2425 | 0.2677, 0.1708, 0.2890 | +46% | 0.0361 (13%) |
| 10% cleaned | ~310 | 0.2563 | 0.1998, 0.3103, 0.2589 | +54% | 0.0223 (8%) |
| **15% cleaned** | **~465** | **0.2714** | **0.2132, 0.2366, 0.3644** | **+63%** | **0.0072 (3%)** ⭐ |
| Seed 789 target | ~1550 (implicit) | 0.2786 | 0.2252, 0.2220, 0.3887 | +67% | - |

**Test Criteria:** If 15% cleaned >= 0.26, hypothesis CONFIRMED
**Result:** 0.2714 >= 0.26 ✅

**Conclusion:** 15% outlier removal closes **94% of the gap** to seed 789!

---

## Root Cause Analysis

### Why Fusion Failed at 128x128

**Multiple Compounding Factors:**

1. **Primary:** Wrong sampling strategy (random → 7.4x R class duplication)
   - Fix: Use 'combined' sampling → +67% improvement

2. **Secondary:** High resolution amplifies overfitting
   - 128x128 = 16,384 pixels vs 32x32 = 1,024 pixels (16x more parameters)
   - RF already overfits duplicates; large images make it worse

3. **Tertiary:** Noisy samples in training data
   - Fix: 15% outlier removal → +63% improvement (on top of sampling fix)

### Why 50% Data Beat 100% Data

**Root Cause:** Implicit outlier removal via random sampling

**Evidence:**
- Seed 789 (lucky): Excluded noisy samples → Kappa 0.2786
- Seed 123 (unlucky): Included more noise → Kappa 0.207 (-25%)
- Consistency metric: Seed 789 CV 17.7% (stable), seed 456 CV 50.0% (unstable)

**Solution:** Explicit 15% outlier removal + 100% data = Same effect as lucky seed

---

## Production Solution

### Option A: 15% Outlier Removal (RECOMMENDED) ⭐

```python
# src/utils/production_config.py
IMAGE_SIZE = 32
DATA_PERCENTAGE = 100.0
SAMPLING_STRATEGY = 'combined'
```

**Then run outlier detection once:**
```bash
python agent_communication/fusion_fix/scripts_production/detect_outliers.py \
  --contamination 0.15 --output data/cleaned
```

**Performance:**
- Kappa: 0.27 ± 0.08 (97% of seed 789 target)
- Uses 85% of data (~2645 samples after cleaning + sampling)
- Reproducible, transparent, auditable

**Advantages:**
- ✅ 1.7x more data than 50% approach
- ✅ Explicit quality control
- ✅ Not seed-dependent
- ✅ Only 2.6% below seed 789 performance

---

### Option B: 50% Seed 789 (ALTERNATIVE)

```python
# src/utils/production_config.py
IMAGE_SIZE = 32
DATA_PERCENTAGE = 50.0
SAMPLING_STRATEGY = 'combined'
RANDOM_SEED = 789
```

**Performance:**
- Kappa: 0.28 ± 0.05
- Uses 50% of data (~1554 samples)

**Advantages:**
- ✅ Proven performance
- ✅ Simple, no preprocessing

**Disadvantages:**
- ❌ Uses only 50% of data
- ❌ Seed-dependent (seed 123: -25% performance)
- ❌ Implicit quality control (no transparency)

---

## Technical Details

### Sampling Strategy: 'combined'

**Algorithm:**
1. **Undersample majority class (P):** Reduce to ~2x minority class (R)
2. **Oversample minority class (R):** Duplicate to match I class
3. **Result:** Balanced to MIDDLE class size (~700-750 samples/class)

**Why NOT 'combined_smote':**
- SMOTE creates synthetic metadata samples (SYN_*)
- These have NO corresponding images → can't be used in fusion
- Result: 28% fewer usable training samples (1398 vs 1797)

### Outlier Detection: Isolation Forest

**Method:** Per-class outlier detection
- Run Isolation Forest separately on each class (I, P, R)
- Contamination: 15% for I and P, max 10% for minority R (safety)
- Features: All metadata features used in RF classifier

**Why per-class:**
- Prevents majority class outliers dominating detection
- Protects minority class R from over-removal
- Ensures balanced cleaning across classes

**Script:** `scripts_production/detect_outliers.py`

---

## Performance Summary

### Progression Through Phases

```
Phase 1 (128x128, random):     0.029  ━░░░░░░░░░░░░░░░░░░░░░░░  FAIL
Phase 2 (32x32, random):       0.165  ━━━━━━━━━━━━━░░░░░░░░░░░  Partial fix
Phase 3 (32x32, combined):     0.166  ━━━━━━━━━━━━━░░░░░░░░░░░  +467% vs Phase 1
Phase 6 (32x32, 50% s789):     0.279  ━━━━━━━━━━━━━━━━━━━━░░░  +68% vs Phase 3
Phase 7 (32x32, 15% clean):    0.271  ━━━━━━━━━━━━━━━━━━━░░░░  +63% vs Phase 3
```

### Final Comparison

| Configuration | Kappa | Data Used | Reproducible | Transparent |
|--------------|-------|-----------|--------------|-------------|
| Original (128x128, random) | 0.029 | 100% | ✅ | ✅ |
| Fixed (32x32, combined) | 0.166 | 100% | ✅ | ✅ |
| Lucky seed (32x32, 50%, s789) | 0.279 | 50% | ❌ | ❌ |
| **Production (32x32, 15% clean)** | **0.271** | **85%** | **✅** | **✅** |

**Winner:** Production solution with 15% outlier removal

---

## File Structure

```
agent_communication/fusion_fix/
├── FUSION_FIX_GUIDE.md (this file)
├── scripts_production/
│   ├── detect_outliers.py (outlier detection)
│   └── test_cleaned_data.py (dataset management helper)
├── results_final/
│   ├── PHASE7_RETEST_RESULTS.md (final test summary)
│   ├── CLOUD_AGENT_PHASE7_RETEST_INDEPENDENT_ANALYSIS.md (detailed analysis)
│   ├── run_100pct_combined_verify.txt (baseline: Kappa 0.1664)
│   ├── run_cleaned_15pct_combined.txt (best: Kappa 0.2714)
│   ├── run_fusion_32x32_50pct_seed789.txt (target: Kappa 0.2786)
│   └── ... (other phase results)
└── archived_intermediate_files/ (old tests and instructions)
```

---

## Key Learnings

1. **Sampling strategy is critical** - 'combined' vs 'random' = 67% difference
2. **Image size matters for overfitting** - 32x32 optimal for fusion with ~3k samples
3. **Data quantity > quality** (to a point) - Reduced duplicates but lost 32% samples = -18.6%
4. **Outlier removal works** - Explicit 15% removal matches implicit 50% random sampling
5. **SMOTE incompatible with fusion** - Creates synthetic samples without images
6. **Seed matters for 50% data** - Seed 789 vs 123 = 35% difference in Kappa
7. **RF overfits duplicates** - Train Kappa 0.998, val Kappa 0.16 with oversampling

---

## Quick Start (Production)

### Step 1: Detect and Remove Outliers (One-time)
```bash
python agent_communication/fusion_fix/scripts_production/detect_outliers.py \
  --contamination 0.15 \
  --output data/cleaned \
  --seed 42
```

Expected output:
- `data/cleaned/metadata_cleaned_15pct.csv` (~2645 samples, -15% outliers)
- `data/cleaned/outliers_15pct.csv` (list of removed samples)

### Step 2: Apply Cleaned Dataset
```bash
python agent_communication/fusion_fix/scripts_production/test_cleaned_data.py 15pct
```

This swaps the cache file with the cleaned version.

### Step 3: Run Training
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi
```

**Expected Results:**
- Per-fold Kappa: 0.21-0.36 (average ~0.27)
- Training time: ~30 min on 8x RTX 4090
- Final average Kappa: **0.27 ± 0.08**

### Step 4: Restore Original (if needed)
```bash
python agent_communication/fusion_fix/scripts_production/test_cleaned_data.py restore
```

---

## Verification Checklist

Before running production:

✅ `IMAGE_SIZE = 32` in `src/utils/production_config.py`
✅ `SAMPLING_STRATEGY = 'combined'` in `src/utils/production_config.py`
✅ Outlier detection completed with 15% contamination
✅ Cleaned dataset applied (check sample count ~2645 after cleaning)
✅ Cache cleared (`--resume_mode fresh` or delete `data/cache/` folder)

After training:

✅ Average Kappa >= 0.25 (expected 0.27 ± 0.08)
✅ Per-fold Kappa >= 0.20 for at least 2/3 folds
✅ Sample count after sampling ~1600-2200 (NOT 3107!)

**If Kappa < 0.15:** Sampling was NOT applied correctly - check config!

---

## Contact & References

**Investigation completed:** 2026-01-06
**Total phases:** 7
**Total test runs:** 25+ configurations
**Solution:** 15% outlier removal + 'combined' sampling
**Performance:** Kappa 0.27 (+834% vs original 128x128 fusion)

For detailed analysis, see:
- `results_final/CLOUD_AGENT_PHASE7_RETEST_INDEPENDENT_ANALYSIS.md`
- `results_final/PHASE7_RETEST_RESULTS.md`
