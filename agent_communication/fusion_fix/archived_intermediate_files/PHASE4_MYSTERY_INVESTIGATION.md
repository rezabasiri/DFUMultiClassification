# Phase 4: Mystery Investigation Plan

**Date:** 2026-01-05
**Focus:** Resolve "50% beats 100%" and "0 trainable params" mysteries
**Strategy:** Use 'combined' sampling (simpler, almost identical to combined_smote)

---

## Mysteries to Solve

### Mystery 1: Why 50% Data Beats 100%?
- 50% data: Kappa 0.22
- 100% data with combined: Kappa ~0.16-0.17
- **Gap: ~30%** - NOT explained by sampling strategy

### Mystery 2: Architecture Issues
- Stage 1: 0 trainable parameters (can't learn)
- Stage 2: LR=1e-6 too low (no improvement)
- Fixed 70/30 fusion weights (can't adapt)

---

## Test Plan Overview

| Test | Purpose | Data % | Image Size | Strategy | Time |
|------|---------|--------|------------|----------|------|
| 1 | Validate 'combined' > 'combined_smote' | 100% | 32x32 | combined | 30 min |
| 2 | Test RF hyperparameter tuning | 100% | N/A | combined | 20 min |
| 3 | Multiple 50% seeds (data quality) | 50% | N/A | combined | 60 min |
| 4 | Trainable fusion weights | 100% | 32x32 | combined | 30 min |
| 5 | Check smaller image sizes | 50% | 24x24 | combined | 20 min |

**Total:** ~2.5 hours

---

## Test 1: Fusion @ 100% with 'combined' (Validation)

**Purpose:** Verify 'combined' outperforms 'combined_smote' (no synthetic image errors)

**Hypothesis:** Should get 0.17-0.18 vs 0.1646 with combined_smote (28% more training data)

**Config:**
```python
# src/utils/production_config.py
IMAGE_SIZE = 32
SAMPLING_STRATEGY = 'combined'  # NOT combined_smote
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_combined.txt
```

**Success Criteria:**
- ✅ Kappa > 0.17 (better than combined_smote 0.1646)
- ✅ No "SYN_*" file not found errors
- ✅ All Stage 1 folds positive

**Time:** ~30 min

---

## Test 2: RF Hyperparameter Tuning @ 100%

**Purpose:** Check if RF overfits with 100% data due to hyperparameter mismatch

**Hypothesis:** Current RF (n_estimators=300, max_depth=None) may overfit with 1797 samples

**Implementation:** Create custom test script to tune RF

**File:** `agent_communication/fusion_fix/test_rf_tuning.py`

**What it does:**
- Loads 100% data with 'combined' sampling
- Tests 3 RF configurations:
  1. Baseline: n_estimators=300, max_depth=None
  2. Reduced: n_estimators=100, max_depth=10
  3. Minimal: n_estimators=50, max_depth=5
- Reports Kappa for each

**Run:**
```bash
python agent_communication/fusion_fix/test_rf_tuning.py
```

**Success Criteria:**
- If reduced complexity improves Kappa → confirms hyperparameter issue
- If no improvement → rules out this hypothesis

**Time:** ~20 min

---

## Test 3: Multiple 50% Seeds (Data Quality Test)

**Purpose:** Determine if 50% performance is consistent or lucky

**Hypothesis:** If all seeds get ~0.22, confirms ANY 50% subset is good → other 50% is bad

**Config:**
```python
# Run metadata-only @ 50% with 5 different random seeds
SAMPLING_STRATEGY = 'combined'
INCLUDED_COMBINATIONS = [('metadata',)]
```

**Implementation:** Modify dataset_utils.py to use different seeds

**Run 5 times with seeds:**
```bash
# Seed 1 (current baseline)
export RANDOM_SEED=42
python src/main.py --mode search --cv_folds 3 --verbosity 2 --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed42.txt

# Seed 2
export RANDOM_SEED=123
python src/main.py --mode search --cv_folds 3 --verbosity 2 --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed123.txt

# Seed 3
export RANDOM_SEED=456
python src/main.py --mode search --cv_folds 3 --verbosity 2 --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed456.txt

# Seed 4
export RANDOM_SEED=789
python src/main.py --mode search --cv_folds 3 --verbosity 2 --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed789.txt

# Seed 5
export RANDOM_SEED=999
python src/main.py --mode search --cv_folds 3 --verbosity 2 --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed999.txt
```

**Analysis:**
- If all 5 seeds get 0.20-0.24 → confirms data quality issue
- If results vary widely (0.10-0.30) → suggests random luck

**Time:** ~60 min (12 min each × 5)

---

## Test 4: Trainable Fusion Weights (Fix #2)

**Purpose:** Make fusion adaptive to RF/image quality variations

**Current Issue:**
- Fixed 70/30 weights can't adapt
- Stage 1 has 0 trainable params

**Implementation:** Modify `src/models/builders.py`

**Changes needed:**
```python
# Line ~341, replace:
rf_weight = 0.70
image_weight = 0.30
weighted_rf = Lambda(lambda x: x * rf_weight)(rf_probs)
weighted_image = Lambda(lambda x: x * image_weight)(image_probs)
output = Add()([weighted_rf, weighted_image])

# With:
concatenated = Concatenate(name='concat_rf_image')([rf_probs, image_probs])
fusion = Dense(3, activation='softmax', name='trainable_fusion')(concatenated)
output = fusion
```

**Config:**
```python
IMAGE_SIZE = 32
SAMPLING_STRATEGY = 'combined'
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_trainable.txt
```

**Success Criteria:**
- ✅ Stage 1 shows trainable parameters > 0
- ✅ Kappa improves over fixed fusion (0.17+ → 0.20+)
- ✅ Model learns optimal RF/image weighting

**Time:** ~30 min

---

## Test 5: Smaller Image Sizes (Optional)

**Purpose:** Check if even smaller images work better (isolate image complexity)

**Hypothesis:** If 32x32 works, maybe 24x24 or 16x16 works even better

**Config:**
```python
SAMPLING_STRATEGY = 'combined'
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Run:**
```bash
# Test 24x24 @ 50% (fast)
# Edit config: IMAGE_SIZE = 24
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --data_percentage 50 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_24x24_50pct.txt
```

**Analysis:**
- If 24x24 beats 32x32 → smaller is better (simpler patterns)
- If 24x24 = 32x32 → plateau reached
- If 24x24 < 32x32 → 32 is optimal

**Time:** ~20 min

---

## Expected Outcomes

### Scenario A: RF Hyperparameters (Test 2 finds solution)
- Reduced RF complexity improves 100% → 0.20+
- **Fix:** Update RF hyperparameters for larger datasets
- **Result:** Close gap to 50% performance

### Scenario B: Data Quality (Test 3 confirms issue)
- All 5 seeds get 0.20-0.24
- **Finding:** ANY 50% subset is good
- **Action:** Investigate/remove poor quality samples from full dataset

### Scenario C: Trainable Fusion (Test 4 succeeds)
- Trainable fusion improves to 0.20+
- **Fix:** Keep trainable fusion
- **Result:** Adaptive weighting compensates for data variations

### Scenario D: Combination of Issues
- Multiple factors contribute
- Need all fixes: RF tuning + trainable fusion + data cleaning

---

## Priority Order

**If time limited, run in this order:**

1. **Test 1** (MUST DO) - Validate 'combined' > 'combined_smote'
2. **Test 4** (HIGH IMPACT) - Trainable fusion likely biggest win
3. **Test 3** (DIAGNOSTIC) - Understand data quality issue
4. **Test 2** (QUICK) - Rule out RF hyperparameter issue
5. **Test 5** (OPTIONAL) - Only if curious about image size effects

---

## Success Metrics

**Minimum Acceptable:**
- Fusion @ 100% with 'combined': Kappa > 0.17

**Good Progress:**
- Trainable fusion: Kappa > 0.20
- Understand why 50% beats 100%

**Ideal:**
- Fusion @ 100% with all fixes: Kappa > 0.22 (matches 50%)
- Architecture issues resolved (trainable params > 0)
- Clear path to production deployment

---

## Notes

- All fusion tests use IMAGE_SIZE=32 (fast, proven to work)
- Metadata-only tests can use 100% data (no image overhead)
- Use 'combined' not 'combined_smote' (simpler, no synthetic issues)
- Focus on UNDERSTANDING mysteries, not just getting higher numbers
