# Two-Phase Bayesian Dataset Polishing Guide

## Overview

The new `auto_polish_dataset_v2.py` uses a **smart two-phase approach** to find optimal misclassification filtering thresholds, dramatically reducing total training time while finding better thresholds.

## Comparison: Old vs New Approach

### Old Approach (`auto_polish_dataset.py`)
- **Strategy:** Iterative threshold reduction
- **Total runs:** 10 runs × 5 iterations = **50 training runs** 😱
- **Threshold selection:** Arbitrary 20% reduction per iteration
- **Time:** ~5-10 hours

### New Approach (`auto_polish_dataset_v2.py`) ✨
- **Strategy:** Two-phase Bayesian optimization
- **Total runs:** Phase 1 (10 runs) + Phase 2 (20 evaluations × 3 folds = 20 runs) = **30 training runs**
- **Threshold selection:** Systematic Bayesian search with 20 evaluations
- **Time:** ~2-4 hours
- **Benefits:**
  - 40% fewer total runs
  - Systematically explores threshold space
  - Finds better thresholds through optimization
  - Safety constraint prevents over-filtering

## How It Works

### Phase 1: Misclassification Detection (Run Once)

**Purpose:** Create comprehensive misclassification profile

**Configuration:**
- `n_runs = 10` (different random seeds: 43-52)
- `cv_folds = 1` (single split for speed)
- **Time:** ~30-60 minutes

**What happens:**
```
Run 1 (seed=43): Train metadata → accumulate misclassifications
Run 2 (seed=44): Train metadata → accumulate misclassifications
...
Run 10 (seed=52): Train metadata → accumulate misclassifications
→ Result: misclassification_counts.csv with max count = 10
```

**Output:**
- `frequent_misclassifications_total.csv` - each sample's misclass count (0-10)

---

### Phase 2: Bayesian Threshold Optimization

**Purpose:** Find optimal thresholds using smart search

**Configuration:**
- `n_evaluations = 20` (Bayesian optimization iterations)
- `cv_folds = 3` (robust evaluation per candidate)
- `n_runs = 1` (single run per evaluation)
- **Time:** ~1-2 hours (20 evaluations × ~3-5 min each)

**Search Space** (percentage-based):
- **P class (dominant, 60%):** 30-70% of n_runs → [3, 7] for n_runs=10
- **I class (minority, 30%):** 50-90% of n_runs → [5, 9] for n_runs=10
- **R class (rarest, 10%):** 80-100% of n_runs → [8, 10] for n_runs=10

**Optimization Score:**
```python
score = 0.4 × macro_f1 + 0.4 × min(f1_I, f1_P, f1_R) + 0.2 × kappa
```

Balances:
- Overall performance (macro F1)
- Worst-class performance (min per-class F1)
- Clinical agreement (Cohen's Kappa)

**Safety Constraint:**
- Rejects thresholds that filter > 50% of dataset
- Ensures sufficient data remains for training

**What happens:**
```
Evaluation 1: Try P=3, I=6, R=9
  → Filter dataset (exclude samples with count ≥ threshold)
  → Check: 1850/3107 samples remain (60%) ✓
  → Train metadata with cv_folds=3
  → Score: 0.65

Evaluation 2: Try P=5, I=7, R=10 (Bayesian suggests this)
  → Filter dataset
  → Check: 2100/3107 samples remain (68%) ✓
  → Train metadata with cv_folds=3
  → Score: 0.71 ✓ Better!

...

Evaluation 20: Try P=4, I=7, R=10
  → Score: 0.70

→ Best found: P=5, I=7, R=10 (Score=0.71)
```

**Output:**
- `bayesian_optimization_results.json` - full optimization history

---

## Installation

### Required Package: scikit-optimize

```bash
pip install scikit-optimize
```

**Note:** If scikit-optimize is not available, the script automatically falls back to grid search (slower but works).

---

## Usage

### Basic Usage (Both Phases)

```bash
python scripts/auto_polish_dataset_v2.py \
    --modalities metadata depth_rgb depth_map
```

**What happens:**
1. Phase 1: Runs 10 times to detect misclassifications
2. Phase 2: Runs Bayesian optimization (20 evaluations)
3. Outputs optimal thresholds

**Expected time:** ~2-4 hours total

---

### Advanced Options

#### Phase 1 Only (Detection)

```bash
python scripts/auto_polish_dataset_v2.py \
    --modalities metadata \
    --phase1-only \
    --phase1-n-runs 10
```

Useful if you want to inspect misclassifications before optimizing.

#### Phase 2 Only (If Phase 1 Already Done)

```bash
python scripts/auto_polish_dataset_v2.py \
    --modalities metadata \
    --phase2-only \
    --n-evaluations 20
```

Useful for re-running optimization with different settings.

#### More Thorough Optimization

```bash
python scripts/auto_polish_dataset_v2.py \
    --modalities metadata depth_rgb depth_map \
    --n-evaluations 30  # More evaluations, better results
```

**Trade-off:** +50% time but potentially better thresholds.

#### Custom Safety Constraint

```bash
python scripts/auto_polish_dataset_v2.py \
    --modalities metadata \
    --min-dataset-fraction 0.6  # Keep at least 60% of data
```

**Default:** 0.5 (keep at least 50%)

---

## Using the Results

### Inspect Optimization Results

```bash
cat results/bayesian_optimization_results.json
```

Shows:
- All 20 threshold combinations tried
- Scores for each
- Best threshold found
- Full optimization history

### Train Final Model with Optimal Thresholds

After optimization finds best thresholds (e.g., P=5, I=7, R=10):

```bash
python src/main.py --mode search --cv_folds 5 \
    --threshold_I 7 \
    --threshold_P 5 \
    --threshold_R 10
```

This trains all modalities with the optimized dataset.

---

## Interpreting Results

### Good Optimization Run

```
Evaluation 1: Score=0.65
Evaluation 2: Score=0.71 ✨ New best!
Evaluation 3: Score=0.68
...
Evaluation 20: Score=0.70

Best score: 0.71
Best thresholds: P=5, I=7, R=10
```

**What to look for:**
- Score improves over evaluations (Bayesian is learning!)
- Best score > 0.65 (decent performance)
- Multiple evaluations near the best (confirms optimum)

### Poor Optimization Run

```
Many evaluations rejected (dataset too small)
Best score: 0.45
High variance in scores
```

**Possible issues:**
- Too aggressive filtering (increase min_dataset_fraction)
- Poor baseline performance (check data quality)
- Need more Phase 1 runs (increase phase1_n_runs)

---

## FAQ

### Q: Can I use this with n_runs != 10 in Phase 1?

**A:** Yes! The search space automatically scales:
- `n_runs=5`: P ∈ [2,4], I ∈ [3,5], R ∈ [4,5]
- `n_runs=20`: P ∈ [6,14], I ∈ [10,18], R ∈ [16,20]

### Q: What if scikit-optimize is not installed?

**A:** Script falls back to grid search automatically (slower but works).

### Q: Can I change the score formula?

**A:** Yes, edit `calculate_combined_score()` in the script. Current weights:
- 40% macro F1
- 40% min per-class F1
- 20% Cohen's Kappa

### Q: How many evaluations should I use?

**A:**
- **20:** Good balance (recommended)
- **30:** More thorough, +50% time
- **50:** Very thorough, +150% time, diminishing returns

### Q: Can I run Phase 2 multiple times?

**A:** Yes! If you're not happy with results, just re-run with `--phase2-only`. Phase 1 results are preserved.

---

## Performance Tips

### Speed Up Phase 1
- Use `--phase1-n-runs 5` (faster but less robust)
- Already using `cv_folds=1` (optimal)

### Speed Up Phase 2
- Use `--n-evaluations 15` (fewer evaluations)
- Default `cv_folds=3` is already optimized

### Better Results
- Phase 1: `--phase1-n-runs 15` (more robust misclass detection)
- Phase 2: `--n-evaluations 30` (more thorough search)

---

## Troubleshooting

### "No misclassification file found"

**Cause:** Phase 1 not completed or files deleted

**Fix:** Run Phase 1 first or use `--phase1-only`

### "All evaluations rejected (dataset too small)"

**Cause:** `min_dataset_fraction` too high or data too imbalanced

**Fix:** Lower constraint: `--min-dataset-fraction 0.4`

### "Training failed on run X"

**Cause:** GPU memory, data loading issues

**Fix:** Check GPU availability, data paths, restart and retry

---

## Summary

**Old approach:** 50 runs, arbitrary thresholds, 5-10 hours
**New approach:** 30 runs, optimal thresholds, 2-4 hours ✨

**Key advantages:**
- ✅ 40% fewer total training runs
- ✅ Systematic Bayesian search
- ✅ Safety constraints prevent over-filtering
- ✅ Configurable optimization budget
- ✅ Interpretable results with full history

**Recommended workflow:**
1. Run both phases: `python scripts/auto_polish_dataset_v2.py --modalities metadata depth_rgb depth_map`
2. Check results: `cat results/bayesian_optimization_results.json`
3. Train final model with optimal thresholds
4. Profit! 🎉
