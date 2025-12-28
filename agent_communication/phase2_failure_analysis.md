# Phase 2 Optimization Failure Analysis

## Issue Summary

User ran:
```bash
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 30 \
  --device-mode single \
  --min-minority-retention 0.5 \
  --track-misclass valid \
  --phase2-only
```

**Expected**: 30 Bayesian optimization evaluations testing different threshold combinations

**Actual**: Only 1 evaluation ran (thresholds I=4, P=4, R=4) with terrible results, then stopped

## Evidence

### 1. Training Output Shows Only 1 Run
File: `results/misclassifications_saved/phase2_training_output.tmp` (97 lines total)
- Only contains output from ONE training run
- Thresholds: `{'I': 4, 'P': 4, 'R': 4}`
- Results: Macro F1 = 0.29 (model collapse)

### 2. Model Collapse Symptoms
```
Run 1 Results for metadata+depth_rgb+depth_map+thermal_map:
              precision    recall  f1-score   support

           I       1.00      0.03      0.05       264  ← Model barely predicts I
           P       0.55      0.98      0.71       392  ← Model predicts P 98% of time
           R       0.36      0.07      0.11        59  ← Model barely predicts R

    accuracy                           0.55       715
   macro avg       0.64      0.36      0.29       715
weighted avg       0.70      0.55      0.42       715

Cohen's Kappa: 0.0708
```

**Model is predicting almost exclusively class P** - this is complete model collapse.

### 3. Severe Class Imbalance After Filtering
```
Excluded samples per class:
  Class I: 30 samples
  Class P: 151 samples
  Class R: 32 samples

Total unique samples to exclude: 213 out of 647 (33% removed!)

Class distribution after filtering:
  Class I: 737 rows
  Class P: 1036 rows
  Class R: 173 rows  ← Only 173 R samples left!
```

**P:R ratio = 6:1** - extremely imbalanced dataset

## Root Causes

### Primary Issue: Phase 1 Incomplete

Looking at `results/misclassifications_saved/phase1_modality_results.csv`:
```csv
Modalities,Accuracy (Mean),...
thermal_map,0.4108,0.0333,0.3484,...
```

**Only thermal_map baseline was saved!**

But Phase 2 is trying to optimize for `metadata+depth_rgb+depth_map+thermal_map` (4 modalities).

The baseline should have been from the SAME modality combination being optimized.

### Secondary Issue: Aggressive Threshold = 4

With `--min-minority-retention 0.5`, the auto-calculated thresholds were:
- I: 4
- P: 4
- R: 4

This removed 213/647 samples (33%), leaving only 173 R samples total.

After 3-fold CV split:
- ~58 R samples per training fold
- After oversampling to balance classes, model learns to just predict P

### Why Only 1 Evaluation?

**Missing baseline for 4-modality combination**:
- Phase 1 only tested individual modalities (metadata, depth_rgb, depth_map, thermal_map)
- Phase 2 needs baseline for the COMBINATION (metadata+depth_rgb+depth_map+thermal_map)
- Script likely failed to find proper baseline and stopped

**Check**: Does `phase1_baseline.json` have the 4-modality combination?
```json
{
  "thermal_map": {  ← Only thermal_map!
    "modality": "thermal_map",
    "macro_f1": 0.3484,
    ...
  }
}
```

**NO** - it only has thermal_map baseline, not the 4-modality combination!

## Why Same Thresholds Produce Different Results

User asked: "I don't understand how the I, P and R can have same thresholds (4) but the results are different in the two runs."

**This is EXPECTED behavior**:

1. **Different CV Folds**:
   - Each run uses different train/validation patient splits
   - Different training data → different model weights → different predictions

2. **Different Random Initialization**:
   - Even with same data, different weight initialization → different local minima

3. **TensorFlow Cache Bug Fix**:
   - BEFORE fix: All folds validated on same data (WRONG!)
   - AFTER fix: Each fold validates on different patients (CORRECT!)
   - Previous results were invalidated by the bug fix

4. **Natural Variance**:
   - With threshold=4, we're at a critical point where small changes have big effects
   - Same thresholds but different CV folds can exclude slightly different samples
   - This cascades into very different class distributions per fold

## Solutions

### Option 1: Run Full Pipeline (Recommended)
```bash
# Remove --phase2-only to run both Phase 1 AND Phase 2
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 30 \
  --device-mode single \
  --min-minority-retention 0.5 \
  --track-misclass valid
```

This will:
1. Run Phase 1 to establish ALL baselines (including 4-modality)
2. Run Phase 2 with proper baseline comparison

### Option 2: Less Aggressive Filtering
```bash
# Use higher retention (less filtering)
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 30 \
  --device-mode single \
  --min-minority-retention 0.7  ← Increased from 0.5
  --track-misclass valid
```

Higher retention = more samples kept = less class imbalance = better training

### Option 3: Test Individual Modalities First
```bash
# Start with single modality (easier to debug)
python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata \
  --phase2-modalities metadata \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 10 \
  --device-mode single \
  --min-minority-retention 0.6 \
  --track-misclass valid
```

Once this works, try with more modalities.

## Expected Behavior

With proper baseline and reasonable retention:

**30 Evaluation Runs**:
```
Evaluation 1/30: I=5, P=3, R=6 → Weighted F1: 0.45
Evaluation 2/30: I=4, P=4, R=5 → Weighted F1: 0.48
Evaluation 3/30: I=6, P=3, R=4 → Weighted F1: 0.43
...
Evaluation 30/30: I=5, P=4, R=5 → Weighted F1: 0.51

Best thresholds: I=5, P=4, R=5 (Weighted F1: 0.51)
```

**Not just 1 evaluation and stop!**

## Diagnostic Questions

1. Is there a full log file showing all 30 evaluations, or did it truly stop after 1?
2. What does `bayesian_optimization_results.json` contain?
3. Did Phase 2 actually start the Bayesian loop, or fail during initialization?

## Next Steps

User should:
1. Provide full optimization log (if exists)
2. Run without `--phase2-only` to establish proper baselines
3. Consider increasing `--min-minority-retention` to 0.6-0.7 to avoid model collapse
