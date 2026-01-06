# Phase 4: Mystery Investigation - Local Agent Instructions

**Goal:** Resolve "50% beats 100%" and "0 trainable params" mysteries
**Strategy:** Use 'combined' sampling (simpler, cleaner than combined_smote)
**Time:** ~2.5 hours total

---

## Priority Tests (Run in Order)

### Test 1: Validate 'combined' > 'combined_smote' (30 min) ⭐

**Why:** combined_smote lost ~399 training samples (synthetic without images)

**Config:** Edit `src/utils/production_config.py`
```python
IMAGE_SIZE = 32
SAMPLING_STRATEGY = 'combined'  # NOT combined_smote
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Run:**
```bash
source /opt/miniforge3/bin/activate multimodal
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_combined.txt
```

**Expected:**
- Kappa: **0.17-0.18** (vs 0.1646 with combined_smote)
- No "SYN_*" file errors
- 28% more training data

---

### Test 2: RF Hyperparameter Tuning (20 min)

**Why:** Check if RF overfits with 100% data

**Run:**
```bash
python agent_communication/fusion_fix/test_rf_tuning.py
```

**Look for:**
- If "Reduced Complexity" wins → RF hyperparameters need tuning
- If "Baseline (Current)" wins → RF hyperparameters are fine

**Report:**
- Which configuration won?
- Kappa improvement (if any)

---

### Test 3: Implement Trainable Fusion (30 min) ⭐⭐

**Why:** Fix "0 trainable params" issue, let model learn optimal RF/image weighting

**Implementation:**

**Step 1:** Backup original
```bash
cp src/models/builders.py src/models/builders.py.backup
```

**Step 2:** Edit `src/models/builders.py` line 338-348

Find:
```python
                rf_weight = 0.70
                image_weight = 0.30
                vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)
                weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
                weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)
                output = Add(name='output')([weighted_rf, weighted_image])
```

Replace with:
```python
                vprint("  Using trainable fusion layer (adaptive RF/Image weighting)", level=2)
                concatenated = Concatenate(name='concat_rf_image')([rf_probs, image_probs])
                fusion = Dense(
                    3,
                    activation='softmax',
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                    name='trainable_fusion'
                )(concatenated)
                output = fusion
```

**Step 3:** Test
```python
# Config should already be:
# IMAGE_SIZE = 32
# SAMPLING_STRATEGY = 'combined'
# INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]

python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_trainable.txt
```

**Expected:**
- Console: "Using trainable fusion layer"
- Stage 1: Trainable params > 0
- Kappa: **0.19-0.22** (hopefully matches 50% data!)

---

### Test 4: Multiple 50% Seeds (60 min) - Data Quality Investigation

**Why:** Determine if 50% performance is consistent or lucky

**Run 3 tests with different random seeds:**

**Seed 1 (baseline):**
```python
# Edit src/utils/production_config.py:
SAMPLING_STRATEGY = 'combined'
INCLUDED_COMBINATIONS = [('metadata',)]
```

```bash
# Seed 42 (current baseline - already have this result: 0.22)
# Skip, use Phase 2 result
```

**Seed 2:**
```bash
# Edit src/data/dataset_utils.py line ~464:
# Change: seed = 42 + run * (run + 3)
# To: seed = 123 + run * (run + 3)

python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --data_percentage 50 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed123.txt

# Change back to 42 after test!
```

**Seed 3:**
```bash
# Edit dataset_utils.py line ~464:
# seed = 456 + run * (run + 3)

python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --data_percentage 50 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed456.txt

# Change back to 42 after test!
```

**Analysis:**
- If all 3 seeds get 0.20-0.24 → ANY 50% is good → data quality issue
- If results vary widely → random luck

---

### Test 5: Smaller Image Size (20 min) - Optional

**Why:** Check if even smaller images work better

**Config:**
```python
IMAGE_SIZE = 24  # Try 24x24 (or 16x16)
SAMPLING_STRATEGY = 'combined'
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --data_percentage 50 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_24x24_50pct.txt
```

**Analysis:**
- If 24x24 > 32x32 → smaller is better
- If 24x24 = 32x32 → plateau reached
- If 24x24 < 32x32 → 32 is optimal

---

## Results Template

**Create:** `agent_communication/fusion_fix/PHASE4_RESULTS.md`

```markdown
# Phase 4 Results - Mystery Investigation

## Test 1: combined vs combined_smote (Fusion @ 100%)

| Strategy | Kappa | SYN_* Errors | Training Samples |
|----------|-------|--------------|------------------|
| combined_smote | 0.1646 | 399/fold | ~1398 effective |
| combined | [X] | 0 | 1797 |

**Winner:** [combined/combined_smote]
**Improvement:** [X]%
**Conclusion:** [Explain]

---

## Test 2: RF Hyperparameter Tuning

| Configuration | Kappa | vs Baseline |
|---------------|-------|-------------|
| Baseline (300, None) | [X] | 0% |
| Reduced (100, 10) | [X] | [+X]% |
| Minimal (50, 5) | [X] | [+X]% |

**Winner:** [config]
**Conclusion:** [Does RF need retuning?]

---

## Test 3: Trainable Fusion

| Fusion Type | Stage 1 Params | Kappa | vs Fixed |
|-------------|----------------|-------|----------|
| Fixed 70/30 | 0 | 0.17 | baseline |
| Trainable | [X] | [X] | [+X]% |

**Improvement:** [X]%
**Conclusion:** [Did it help?]

---

## Test 4: Multiple 50% Seeds (Data Quality)

| Seed | Kappa | Note |
|------|-------|------|
| 42 (baseline) | 0.22 | From Phase 2 |
| 123 | [X] | |
| 456 | [X] | |

**Range:** [min] to [max]
**Conclusion:** [Consistent quality OR random luck?]

---

## Test 5: Smaller Images (Optional)

| Image Size | Kappa | vs 32x32 |
|------------|-------|----------|
| 32x32 | 0.22 | baseline |
| 24x24 | [X] | [+X]% |

**Conclusion:** [Optimal size found?]

---

## Overall Findings

**Mystery 1 (50% beats 100%) Status:**
- [SOLVED/PARTIALLY SOLVED/UNSOLVED]
- Root cause: [Explain]
- Fix: [What to do]

**Mystery 2 (0 trainable params) Status:**
- [SOLVED/PARTIALLY SOLVED/UNSOLVED]
- Root cause: [Explain]
- Fix: [What to do]

**Best Configuration Found:**
- Sampling: [strategy]
- Fusion: [trainable/fixed]
- Image size: [X]
- Kappa @ 100%: [X]
```

---

## Commit and Push

```bash
git add agent_communication/fusion_fix/ src/models/builders.py
git commit -m "test: Phase 4 mystery investigation

Test 1 (combined vs combined_smote): Kappa [X]
Test 2 (RF tuning): [winner] wins
Test 3 (trainable fusion): Kappa [X] (+[X]%)
Test 4 (50% seeds): [consistent/varied]

Mysteries:
- 50% beats 100%: [status]
- 0 trainable params: [SOLVED with trainable fusion]"

git push -u origin claude/run-dataset-polishing-X1NHe
```

---

## Quick Reference Commands

**Test 1 (combined validation):**
```bash
# Edit config: SAMPLING_STRATEGY='combined', IMAGE_SIZE=32, fusion
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_combined.txt
```

**Test 2 (RF tuning):**
```bash
python agent_communication/fusion_fix/test_rf_tuning.py
```

**Test 3 (trainable fusion):**
```bash
# After modifying builders.py
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_trainable.txt
```

**Test 4 (50% seeds):**
```bash
# Modify seed in dataset_utils.py, run metadata-only @ 50%
python src/main.py --mode search --cv_folds 3 --verbosity 2 --data_percentage 50 --resume_mode fresh --device-mode multi 2>&1 | tee agent_communication/fusion_fix/run_metadata_50pct_seed[X].txt
```

---

**Focus on Tests 1-3 first! They're the highest impact.** 🎯
