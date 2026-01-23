# Expected vs Actual Behavior

## Current Failure

**Observed (from Phase 1 logs):**
- Accuracy: 10%
- Cohen's Kappa: -0.032 (worse than random)
- Class I: 0% precision, 0% recall
- Class P: 16% precision, 3% recall
- Class R: 10% precision, 69% recall

**Analysis:** Model predicting almost everything as class R (minority class with 69% recall).

## Expected Behavior

**Test 4 should show:**
- Validation accuracy: >30% (minimum)
- Cohen's Kappa: >0.1 (positive)
- All classes with non-zero F1
- Per-class F1 scores: >0.15 each

**Historical baseline (thermal_map from logs):**
- Macro F1: 0.36
- Weighted F1: 0.45
- Kappa: 0.08

## Critical Checkpoints

### Data Loading (Test 1)
- ✓ 647 samples total
- ✓ Labels: I=203, P=368, R=76
- ✓ No missing critical columns

### Preprocessing (Test 3)
- ✓ No NaN after imputation
- ✓ Train means ~0, stds ~1
- ✓ No Inf values

### RF Training (Test 4)
- ✓ Binary classifiers should achieve >70% accuracy each
- ✓ Final 3-class accuracy should be >30%
- ⚠️ If this fails → problem in ordinal RF logic

### Integration (Test 5)
- ✓ Dataset creation succeeds
- ✓ Batch retrieval works
- ⚠️ If test 4 passes but test 5 fails → problem in integration

## Possible Root Causes

If tests 1-3 pass but test 4 fails:
1. **Wrong features** - using image columns instead of metadata
2. **Label corruption** - labels not matching data
3. **Class weights wrong** - too extreme, causing over/under-weighting

If test 4 passes but actual training fails:
1. **Multi-GPU issue** - MirroredStrategy not distributing correctly
2. **Batch size problem** - too large/small for GPU memory
3. **Model architecture** - wrong input shape or processing
4. **Different code path** - using caching.py vs dataset_utils.py

If all tests pass:
1. **CV fold bug** - patient-level split causing data leakage/imbalance
2. **Overfitting** - model memorizing train, failing on valid
3. **Different data** - actual run using filtered/corrupted data
