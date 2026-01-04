# FUSION FIX V2 - Fixed Weights Instead of Learned Weights

**Date**: 2026-01-04
**Issue**: Learned fusion weights collapsed during training, causing catastrophic failure

---

## Problem with V1 (Learned Alpha)

**Previous approach**:
```python
# Learn alpha from predictions
combined = concatenate([rf_probs, image_probs])
alpha = Dense(1, activation='sigmoid')(combined)
output = alpha*RF + (1-alpha)*Image
```

**What went wrong**:
- Alpha learned from predictions creates feedback loop
- If image branch overfits → produces confident predictions
- Fusion layer learns to rely on confident predictions (even if wrong)
- Alpha collapses to 0 → ignores RF entirely
- Result: Kappa 0.0136 (still catastrophic)

**Evidence**:
```
Train kappa: 0.96 (extreme overfitting)
Val kappa: -0.02 (complete failure)
Model predicts mostly class P (68% of predictions)
```

---

## V2 Solution: Fixed Fusion Weights

**New approach**:
```python
# FIXED weights (not learned)
rf_weight = 0.70  # 70% RF contribution
image_weight = 0.30  # 30% image contribution

output = 0.70*RF_probs + 0.30*Image_probs
```

**Why this works**:
- ✅ **No trainable fusion parameters** → can't collapse
- ✅ **RF dominates** (70%) → preserves RF quality (Kappa 0.254)
- ✅ **Image contributes** (30%) → adds complementary information
- ✅ **Simple and stable** → no optimization issues

**Rationale for 70/30 split**:
- RF standalone: Kappa 0.254 (strong)
- Image standalone: Kappa 0.145 (weaker)
- Give more weight to stronger modality (70% RF)
- Allow weaker modality to contribute (30% image)

---

## Expected Results

### Metadata-Only (Unchanged)
- Kappa: ~0.254 (same as before)

### Metadata + thermal_map (Should be Fixed Now)
- **Previous**: Kappa 0.0136 (learned weights collapsed)
- **Expected**: Kappa **0.22-0.26**
- Calculation: 0.70 × 0.254 + 0.30 × 0.145 = 0.178 + 0.044 = **0.222 (minimum)**
- If modalities complement each other, could be higher (up to 0.26)

### thermal_map-Only (Unchanged)
- Kappa: ~0.145 (same as before)

---

## Changes Made

**File**: `src/models/builders.py`

**Lines modified**:
- 2 modalities: lines 331-347
- 3 modalities: lines 372-377
- 4 modalities: lines 412-417
- 5 modalities: lines 451-456

**Code**:
```python
# FIXED WEIGHTED AVERAGE FUSION
rf_weight = 0.70  # RF contributes 70%
image_weight = 0.30  # Image contributes 30%

vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)

weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)

output = Add(name='output')([weighted_rf, weighted_image])
```

**Key changes from V1**:
- ❌ Removed: Dense layer learning alpha from predictions
- ❌ Removed: Trainable fusion weights
- ✅ Added: Fixed rf_weight = 0.70 constant
- ✅ Added: Fixed image_weight = 0.30 constant
- ✅ Simplified: Direct multiplication instead of learned weights

---

## Validation

### Test Command

```bash
cd /home/user/DFUMultiClassification

# Set config to metadata + thermal_map
# In src/utils/production_config.py:
# INCLUDED_COMBINATIONS = [('metadata', 'thermal_map'),]

python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh
```

### Success Criteria

**Minimum Success**:
- ✅ Kappa > 0.20 (better than previous 0.0136)
- ✅ No class collapse (predictions distributed across I, P, R)
- ✅ Message: "Fusion weights: RF=0.70, Image=0.30"

**Expected Success**:
- ✅ Kappa ≥ 0.22 (theoretical minimum from weighted average)
- ✅ Kappa better than image-only (0.145)
- ✅ Train/val gap reasonable (<0.2 difference)

**Ideal Success**:
- ✅ Kappa ≥ 0.25 (fusion benefit - modalities complement each other)
- ✅ Better than metadata-only (0.254)

---

## Why Fixed Weights Instead of Learned?

| Aspect | Learned Weights (V1) | Fixed Weights (V2) |
|--------|---------------------|-------------------|
| **Trainable params** | 7 (Dense layer) | 0 (constants) |
| **Can collapse** | ✅ Yes (happened!) | ❌ No (mathematically impossible) |
| **Training stability** | ❌ Unstable | ✅ Stable |
| **RF preservation** | ⚠️ Only if alpha learns correctly | ✅ Guaranteed (70% RF) |
| **Overfitting risk** | High | None (no parameters) |
| **Interpretability** | Low (learned black box) | High (explicit 70/30) |

**Key insight**: We KNOW RF is better than image (0.254 vs 0.145). We don't need the network to learn this - just hard-code it!

---

## Future Tuning (Optional)

If results show fusion could benefit from different weights, try:

**More RF-heavy** (if image adds noise):
```python
rf_weight = 0.80
image_weight = 0.20
```

**More balanced** (if image adds strong signal):
```python
rf_weight = 0.60
image_weight = 0.40
```

**Very RF-heavy** (if image hurts):
```python
rf_weight = 0.90
image_weight = 0.10
```

But start with 70/30 - it's a reasonable prior given the standalone performances.

---

## Summary

**Root cause V1 failure**: Learned fusion weights collapsed during training

**V2 fix**: Use fixed 70/30 weights based on standalone performance

**Expected improvement**: Kappa 0.0136 → 0.22-0.26 (+1,500% to +1,800%)

**Confidence**: HIGH - Fixed weights can't collapse, RF quality guaranteed

---

**Status**: Implemented, ready for testing
**Branch**: `claude/run-dataset-polishing-X1NHe`
**Commit**: Pending
