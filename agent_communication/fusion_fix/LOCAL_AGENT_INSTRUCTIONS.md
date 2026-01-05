# Local Agent - Sampling Strategy Comparison

**Date:** 2026-01-05
**Task:** Test 3 sampling strategies to find best for metadata RF
**Expected Time:** ~2 hours total (4 tests)

---

## What You Need to Know

**Changes Made:**
- ✅ Three sampling strategies implemented and ready to test
- ✅ All augmentations disabled (fair comparison)
- ✅ Easy to switch between strategies via config

**Strategies Available:**
1. **'random'**: Simple oversampling (baseline) - balances to MAX class
2. **'smote'**: Synthetic oversampling - balances to MAX class (no duplicates)
3. **'combined'**: Under+over sampling - balances to MIDDLE class (fewer samples)
4. **'combined_smote'**: Under+SMOTE - balances to MIDDLE class (synthetic, best!)

---

## Test Plan: Compare All 4 Strategies

**Goal:** Find which strategy gives best RF performance @ 100% data

### Expected Results:
```
Baseline (random):    Kappa 0.09 (known from Phase 2)
SMOTE:                Kappa 0.15-0.20 (synthetic samples)
Combined:             Kappa 0.18-0.22 (fewer duplicates)
Combined + SMOTE:     Kappa 0.20-0.25 (best of both!)
```

---

## Test 1: Random Oversampling (Baseline Verification)

**Purpose:** Verify baseline failure is reproducible

### Config:
Edit `src/utils/production_config.py`:
```python
SAMPLING_STRATEGY = 'random'  # ← CHANGE THIS
INCLUDED_COMBINATIONS = [('metadata',)]
```

### Run:
```bash
source /opt/miniforge3/bin/activate multimodal
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_random.txt
```

**Expected:**
- Console: `"Using simple random oversampling..."`
- Kappa: **~0.09** (should match Phase 2 baseline)
- Runtime: ~20-30 min

---

## Test 2: SMOTE (Synthetic Oversampling)

**Purpose:** Test if synthetic samples reduce overfitting

### Config:
Edit `src/utils/production_config.py`:
```python
SAMPLING_STRATEGY = 'smote'  # ← CHANGE THIS
INCLUDED_COMBINATIONS = [('metadata',)]
```

### Run:
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_smote.txt
```

**Expected:**
- Console: `"Using SMOTE (synthetic oversampling)..."`
- Kappa: **0.15-0.20** (60-120% improvement over baseline)
- Runtime: ~20-30 min

---

## Test 3: Combined Sampling (Under + Over)

**Purpose:** Test if balancing to middle class reduces overfitting

### Config:
Edit `src/utils/production_config.py`:
```python
SAMPLING_STRATEGY = 'combined'  # ← CHANGE THIS
INCLUDED_COMBINATIONS = [('metadata',)]
```

### Run:
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_combined.txt
```

**Expected:**
- Console: `"Using combined sampling (under + over)..."`
- Kappa: **0.18-0.22** (100-150% improvement over baseline)
- Runtime: ~20-30 min

**Why combined might be best:**
- Balances to MIDDLE class (599 samples) not MAX (1164)
- R duplicated only 3.8x (vs 7.4x in random/smote)
- Smaller dataset = less overfitting + faster training

---

## Test 4: Combined + SMOTE (Best of Both!)

**Purpose:** Test if combining undersampling + SMOTE gives best results

### Config:
Edit `src/utils/production_config.py`:
```python
SAMPLING_STRATEGY = 'combined_smote'  # ← CHANGE THIS
INCLUDED_COMBINATIONS = [('metadata',)]
```

### Run:
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_combined_smote.txt
```

**Expected:**
- Console: `"Using combined + SMOTE (under + synthetic over)..."`
- Kappa: **0.20-0.25** (120-180% improvement over baseline)
- Runtime: ~20-30 min

**Why this might be BEST:**
- Balances to MIDDLE class (599 samples) - smaller dataset
- Uses SMOTE (no exact duplicates) - all synthetic
- R only grows 3.8x (vs 7.4x) - less overfitting
- Fast training + best quality samples

---

## Comparison Table

**Fill this in after all tests:**

| Strategy | Final Kappa | vs Baseline | Total Samples | R Duplicates | Best? |
|----------|-------------|-------------|---------------|--------------|-------|
| random   | [X]         | 0%          | 3492          | 7.4x         | [✓/✗] |
| smote    | [X]         | [+X]%       | 3492          | 0 (synthetic)| [✓/✗] |
| combined | [X]         | [+X]%       | 1797          | 3.8x         | [✓/✗] |
| combined_smote | [X]  | [+X]%       | 1797          | 0 (synthetic)| [✓/✗] |

---

## How to Interpret Results

### If SMOTE wins:
- Synthetic samples work best
- Continue with SMOTE for fusion tests

### If Combined wins:
- Fewer duplicates work best
- Consider combined for fusion tests
- OR try: combined undersampling + SMOTE oversampling (best of both!)

### If results are similar:
- Use Combined (faster training, smaller dataset)

---

## After Testing: Choose Best Strategy

Based on results, update config for fusion testing:

```python
# Use the winner
SAMPLING_STRATEGY = '[winner]'  # 'smote' or 'combined'
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

Then run fusion test:
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_128x128_100pct_[winner].txt
```

**Expected fusion results:**
- With best sampling: Kappa **0.20-0.25**
- Stage 1: All folds POSITIVE (not negative!)

---

## Reporting Results

**Create:** `agent_communication/fusion_fix/SAMPLING_COMPARISON_RESULTS.md`

**Template:**

```markdown
# Sampling Strategy Comparison - metadata @ 100%

**Date:** 2026-01-05

## Test 1: Random Oversampling (Baseline)

**Results:**
| Fold | Kappa |
|------|-------|
| 1    | [X]   |
| 2    | [X]   |
| 3    | [X]   |
| **Avg** | **[X]** |

**Console output:** "Using simple random oversampling..."
**Reproducible:** [YES/NO - matches Phase 2 baseline 0.09?]

---

## Test 2: SMOTE

**Results:**
| Fold | Kappa |
|------|-------|
| 1    | [X]   |
| 2    | [X]   |
| 3    | [X]   |
| **Avg** | **[X]** |

**Console output:** "Using SMOTE (synthetic oversampling)..."
**Improvement:** [X]% over baseline

---

## Test 3: Combined Sampling

**Results:**
| Fold | Kappa |
|------|-------|
| 1    | [X]   |
| 2    | [X]   |
| 3    | [X]   |
| **Avg** | **[X]** |

**Console output:** "Using combined sampling (under + over)..."
**Improvement:** [X]% over baseline

---

## Winner: [STRATEGY]

**Reason:** [Explanation]

**Next step:** Test fusion @ 100% with [winner] strategy
**Expected fusion Kappa:** 0.20-0.25
```

---

## Troubleshooting

### Error: "Expected n_neighbors <= n_samples" (SMOTE only)

**Fix:** Edit `src/data/dataset_utils.py` line ~723:
```python
# Change k_neighbors from 5 to 3
oversampler = SMOTE(random_state=42 + run * (run + 3), k_neighbors=3)
```

### Wrong strategy being used

**Check:**
- Console output matches expected message
- Config file saved with correct SAMPLING_STRATEGY value
- No cached old config in memory (restart kernel if needed)

---

## Success Criteria

**Minimum acceptable (any strategy):**
- Kappa > 0.15 (60% improvement over baseline 0.09)

**Ideal:**
- Kappa > 0.20 (120% improvement)
- Clear winner among the three strategies

---

## Commit Results

```bash
git add agent_communication/fusion_fix/
git commit -m "test: Sampling strategy comparison (random vs smote vs combined)

- Random: Kappa [X]
- SMOTE: Kappa [X]
- Combined: Kappa [X]
- Winner: [strategy] with [X]% improvement"

git push -u origin claude/run-dataset-polishing-X1NHe
```

---

## Copy-Paste Quick Commands

**Test 1 (random):**
```bash
# Edit config: SAMPLING_STRATEGY = 'random'
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_random.txt
```

**Test 2 (smote):**
```bash
# Edit config: SAMPLING_STRATEGY = 'smote'
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_smote.txt
```

**Test 3 (combined):**
```bash
# Edit config: SAMPLING_STRATEGY = 'combined'
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_combined.txt
```

**Test 4 (combined_smote):**
```bash
# Edit config: SAMPLING_STRATEGY = 'combined_smote'
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_combined_smote.txt
```

---

**Good luck! The combined strategy should work well based on original code!** 🔬
