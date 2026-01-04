# Diagnostic Findings Summary

## What Local Agent Discovered

### Tests 1-3: ✅ PASS
- Data loading works
- Feature engineering works (73 features created)
- Preprocessing (imputation + normalization) works

### Test 4: ⚠️ MARGINAL PASS
- **Accuracy: 52.8%** (above 30% threshold but poor)
- **Kappa: 0.0809** (barely positive)
- **Class R F1: 0.00** - Cannot predict minority class at all
- **Conclusion**: RF ordinal classifier is fundamentally weak

### Test 5: ⚠️ MISLEADING RESULT
- Found metadata_input shape=(32, 3) instead of (32, 73)
- **Initially thought this was a bug**
- **Actually intentional design** - verified in original code too

## Design Clarification

**Metadata Modality Pipeline:**
```
Raw data (73 features)
    ↓
Feature engineering
    ↓
Imputation + Normalization
    ↓
RF Ordinal Classifier (2 binary RFs)
    ↓
3 probabilities (prob_I, prob_P, prob_R)  ← THIS becomes "metadata features"
    ↓
BatchNorm
    ↓
Fusion with image CNNs
```

**The 73→3 reduction is intentional!**

## Real Problems Identified

### Problem 1: Weak RF Classifier
- Class R (minority): F1 = 0.00
- Kappa = 0.08 (barely better than random)
- The ordinal approach (2 binary classifiers) fails on imbalanced data

### Problem 2: Production vs Test Performance Gap
- Test 4 (standalone RF): 52.8% accuracy
- Production (Phase 1): 10% accuracy
- **Something else is failing in production!**

### Problem 3: Code Path Confusion
- `caching.py`: Older version, not used in production
- `dataset_utils.py`: Current version, used by training_utils.py
- Test 5 tested wrong version!

## Next Steps

### Immediate: Test Production Code Path
Run `test_6_production.py` to check `dataset_utils.py`:
- Does it produce same (32, 3) shape?
- Do probabilities sum to 1?
- Are there NaN/Inf values?

### Root Cause Hypotheses

**Hypothesis A: Patient-Level CV Corruption**
- Test 4 uses random split (works marginally)
- Production uses patient-level 5-fold CV
- CV might create impossible splits (all R in one fold)

**Hypothesis B: Multi-GPU Batch Distribution**
- Test 4: Single-threaded sklearn
- Production: MirroredStrategy across 8 GPUs
- Batch distribution might corrupt probabilities

**Hypothesis C: RF Training Randomness**
- Test 4: Fixed seed (42)
- Production: Varying seed (42 + run*(run+3))
- Different seeds might produce worse RFs

**Hypothesis D: Data Filtering**
- Test 4: Full dataset
- Production: Might be using filtered/corrupted data

## Recommendations

1. **Run test_6_production.py** - verify dataset_utils.py behavior
2. **Test patient-level CV** - reproduce production CV split logic
3. **Compare RF models** - train RF with production seed and CV
4. **Consider alternative**:
   - Skip RF entirely, use raw 73 features
   - Use better classifier (XGBoost, LightGBM)
   - Improve class balancing (SMOTE, class weights)
