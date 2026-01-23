# Phase 6: Validate 50% Data Consistency

**Date:** 2026-01-05 16:30 UTC
**Goal:** Determine if 50% data performance (Kappa 0.22) is reliable or lucky
**Timeline:** ~90 minutes (3 runs × 30 min each)

---

## Context

**Phase 5 Result:** reduced_combined FAILED (-18.6% worse than combined)
- Root cause: 32% less training data hurt more than duplicate reduction helped
- Conclusion: Data quantity > Data quality for this problem

**Current Best Performance:**
- 50% data @ seed 42: Kappa **0.22** (proven)
- 100% data + combined: Kappa **0.166** (best 100% result)
- Gap: **34% better with 50% data**

**Critical Question:** Is 50% performance consistent, or did we get lucky with seed 42?

---

## Hypothesis to Test

**Hypothesis:** 50% data has better natural class balance (generalizable advantage)

**Alternative:** Seed 42 produced a lucky sample (not reliable)

**Test Method:** Run 50% data with 3 different random seeds

**Expected Results:**

| Scenario | Seed 42 | Seed 123 | Seed 456 | Conclusion |
|----------|---------|----------|----------|------------|
| **Natural balance** | 0.22 | 0.21-0.23 | 0.21-0.23 | ✅ Use 50% for production |
| **Lucky sample** | 0.22 | 0.15-0.19 | 0.16-0.20 | ⚠️ Need smarter selection |
| **Seed 42 anomaly** | 0.22 | 0.17 | 0.18 | ❌ Rethink strategy |

---

## Test Plan

### Test 1: 50% Data with Seed 123 (30 min)

**Configuration:**
```python
# src/utils/production_config.py
DATA_PERCENTAGE = 50
SAMPLING_STRATEGY = 'random'  # Use original, proven strategy
IMAGE_SIZE = 32
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Modify Random Seed:**
```python
# In src/data/dataset_utils.py, line ~150 (wherever train_test_split happens)
# Find: random_state=42
# Change to: random_state=123
```

**OR if there's a config parameter:**
```python
# src/utils/production_config.py
RANDOM_SEED = 123  # Was: 42
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_50pct_seed123.txt
```

**Expected:** Kappa 0.21-0.23 (if natural balance hypothesis is correct)

---

### Test 2: 50% Data with Seed 456 (30 min)

**Same as Test 1, but:**
```python
RANDOM_SEED = 456
```

**Run:**
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi --data_percentage 50 \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_50pct_seed456.txt
```

---

### Test 3: 50% Data with Seed 789 (30 min, optional third seed)

**Same as above:**
```python
RANDOM_SEED = 789
```

---

## Validation Criteria

### Success: Natural Balance Confirmed ✅

**Results:**
```
Seed 42:  Kappa 0.22
Seed 123: Kappa 0.21-0.23
Seed 456: Kappa 0.21-0.23
Seed 789: Kappa 0.21-0.23

Standard deviation < 0.02
Average: 0.21-0.23
```

**Interpretation:**
- 50% data RELIABLY better than 100% (0.22 vs 0.166)
- Natural balance advantage is real and generalizable
- **Recommendation:** Use 50% data for production deployment

**Production Config:**
```python
DATA_PERCENTAGE = 50
SAMPLING_STRATEGY = 'random'
RANDOM_SEED = 42  # Or any seed, all perform well
IMAGE_SIZE = 32
```

---

### Partial Success: Variable Performance ⚠️

**Results:**
```
Seed 42:  Kappa 0.22
Seed 123: Kappa 0.18
Seed 456: Kappa 0.19

Standard deviation: 0.02-0.04
Average: 0.19-0.20
```

**Interpretation:**
- Seed 42 is above average but not anomalous
- 50% data averages 0.19-0.20 (still better than 100% @ 0.166)
- Some variability exists, but overall trend is positive

**Recommendation:**
- Use seed 42 (known good) for production
- Alternatively: Test 5-10 seeds, pick best
- Consider "smart sampling" strategy (next phase)

---

### Failure: Seed 42 Was Lucky ❌

**Results:**
```
Seed 42:  Kappa 0.22
Seed 123: Kappa 0.15
Seed 456: Kappa 0.16

Average: 0.17-0.18 (similar to 100% data)
```

**Interpretation:**
- Seed 42 was an outlier (lucky sample)
- 50% data doesn't have reliable advantage
- Need different strategy

**Next Steps:**
- Option A: Keep seed 42 but acknowledge it's not generalizable
- Option B: Develop smarter 50% selection (stratified, quality-based)
- Option C: Return to 100% data, focus on architecture improvements

---

## Additional Analysis (Optional)

### Compare Class Distributions Before Sampling

**Goal:** Understand WHY 50% might perform better

**Analysis:**
```python
# For each seed, report ORIGINAL class distribution (before any sampling)
print("100% data original: I=?, P=?, R=?")
print("50% seed 42:  I=?, P=?, R=?")
print("50% seed 123: I=?, P=?, R=?")
print("50% seed 456: I=?, P=?, R=?")
```

**Expected Pattern (if natural balance hypothesis is correct):**
```
100% data: I=31%, P=56%, R=13% (imbalanced)
50% seed 42:  I=35%, P=50%, R=15% (better balanced)
50% seed 123: I=33%, P=52%, R=15% (better balanced)
50% seed 456: I=36%, P=48%, R=16% (better balanced)
```

**Interpretation:**
- If 50% samples consistently more balanced: Natural balance confirmed
- If 50% seed 42 uniquely balanced: Lucky sample confirmed

---

## Reporting Template

### Test 1: Seed 123 @ 50%

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**Class Distribution (before sampling):**
- I: _____ (_____%)
- P: _____ (_____%)
- R: _____ (_____%)

**Conclusion:**
[ ] Similar to seed 42 (0.21-0.23)
[ ] Lower than seed 42 (0.18-0.20)
[ ] Much lower (< 0.18)

---

### Test 2: Seed 456 @ 50%

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**Class Distribution (before sampling):**
- I: _____ (_____%)
- P: _____ (_____%)
- R: _____ (_____%)

---

### Test 3: Seed 789 @ 50% (Optional)

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

---

### Overall Summary

**All Seeds Performance:**
| Seed | Kappa | vs 100% (0.166) | vs Seed 42 (0.22) |
|------|-------|-----------------|-------------------|
| 42 | 0.22 | +32% | baseline |
| 123 | _____ | _____ | _____ |
| 456 | _____ | _____ | _____ |
| 789 | _____ | _____ | _____ |

**Statistics:**
- Mean: _____
- Std Dev: _____
- Min: _____
- Max: _____

**Conclusion:**
[ ] ✅ Natural balance confirmed (use 50% data)
[ ] ⚠️ Variable performance (use seed 42 specifically)
[ ] ❌ Seed 42 was lucky (need new strategy)

---

## Decision Tree

```
Run Tests 1-2 (seeds 123, 456)
    ↓
    Both achieve 0.21-0.23?
    ├─ YES: ✅ 50% data validated → Production deployment
    ├─ NO, both 0.18-0.20: ⚠️ Use seed 42 specifically
    └─ NO, both < 0.18: ❌ Phase 7 needed (smart sampling)

If ✅ Natural balance confirmed:
    - Production config: DATA_PERCENTAGE=50%, any seed
    - Investigation complete
    - Mystery solved: Natural balance > More data with duplicates

If ⚠️ Seed 42 is special:
    - Production config: DATA_PERCENTAGE=50%, RANDOM_SEED=42
    - Partial understanding: Some 50% samples better
    - Consider Phase 7: Smart selection strategy

If ❌ Seed 42 was lucky:
    - Reconsider 100% data strategies
    - Phase 7: Alternative approaches (SMOTE metadata-only, etc.)
```

---

## Implementation Notes

### Finding Random Seed Location

**Check these files:**
```bash
# Most likely location:
grep -r "random_state=42" src/data/dataset_utils.py

# Or check config:
grep -r "RANDOM_SEED\|random_seed" src/utils/production_config.py

# Or check main:
grep -r "seed\|random" src/main.py
```

**Common patterns:**
```python
# Option 1: Direct in train_test_split
train, test = train_test_split(..., random_state=42)

# Option 2: From config
from src.utils.production_config import RANDOM_SEED
train, test = train_test_split(..., random_state=RANDOM_SEED)

# Option 3: Multiple places
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

**For this test:** Only need to change the DATA_PERCENTAGE split seed, not all random seeds (CV folds, model init, etc. can stay consistent).

---

## Expected Timeline

| Test | Duration | Priority |
|------|----------|----------|
| Test 1: Seed 123 | 30 min | **REQUIRED** |
| Test 2: Seed 456 | 30 min | **REQUIRED** |
| Test 3: Seed 789 | 30 min | Optional |
| Analysis | 10 min | After all tests |

**Total:** 70-90 minutes for conclusive results

---

## Success Criteria

**Minimum:** Tests 1-2 complete
**Ideal:** Tests 1-3 complete + class distribution analysis

**Deliverables:**
1. Performance across multiple seeds
2. Understanding of variability
3. Clear production recommendation

If all seeds achieve 0.21+, we can confidently deploy 50% data and close this investigation as **SOLVED**.
