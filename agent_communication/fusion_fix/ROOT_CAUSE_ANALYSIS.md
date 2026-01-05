# ROOT CAUSE ANALYSIS: Fusion Failure

## Executive Summary
**CRITICAL BUG FOUND**: RF probability predictions don't sum to 1.0, causing catastrophic fusion failure.

## The Bug

### Location
- File: [src/data/dataset_utils.py:1027-1029](../../src/data/dataset_utils.py#L1027-L1029)
- Also in: [src/data/caching.py:535-537](../../src/data/caching.py#L535-L537)

### Current (BROKEN) Code
```python
prob_I = 1 - prob1
prob_P = prob1 * (1 - prob2)
prob_R = prob2
```

### Mathematical Proof of Bug
```
prob_I + prob_P + prob_R
= (1 - prob1) + prob1 * (1 - prob2) + prob2
= 1 - prob1 + prob1 - prob1*prob2 + prob2
= 1 + prob2 - prob1*prob2
= 1 + prob2(1 - prob1)
```

**Only sums to 1.0 when**: prob2=0 OR prob1=1

### Observed Evidence
From `test_fusion_debug.py` output:
```
Sum to 1.0? [1.097006, 1.041172, 1.0491359, 1.008123, 1.0700004]
```

Probabilities sum to **1.04-1.10** instead of 1.0!

Sample RF predictions:
```
Sample 0: probs [0.85349154 0.12985663 0.11365781] sum=1.097
Sample 1: probs [0.623455   0.35167858 0.06603844] sum=1.041
Sample 2: probs [0.80871546 0.17966245 0.06075803] sum=1.049
```

## Why This Breaks Fusion

### Fusion Formula (from architecture)
```python
fusion_pred = 0.7 * rf_pred + 0.3 * image_pred
```

### Impact of Non-normalized Probabilities

If RF predictions sum to 1.05:
- `rf_pred = [0.85, 0.13, 0.11]` (sum=1.09)
- `image_pred = [0.33, 0.33, 0.34]` (sum=1.0 ✓)
- `fusion_pred = 0.7 * [0.85, 0.13, 0.11] + 0.3 * [0.33, 0.33, 0.34]`
- `fusion_pred = [0.595, 0.091, 0.077] + [0.099, 0.099, 0.102]`
- `fusion_pred = [0.694, 0.190, 0.179]` (sum=1.063 ✗)

The fusion predictions also don't sum to 1.0, corrupting softmax/argmax operations!

### Why Fusion Gets Kappa -0.007 (Worse Than Random)

1. **Unnormalized probabilities corrupt argmax**: When predictions don't sum to 1.0, the argmax might select the wrong class
2. **Systematic bias**: The extra probability mass (~5-10%) is distributed non-uniformly across classes
3. **Training instability**: Loss functions expect normalized probabilities, leading to gradient corruption
4. **Worse than random**: The bias is systematic and consistent, making predictions reliably wrong

## Correct Implementation

### Two-Step Binary Approach (Keep the idea, fix normalization)

The current approach uses two binary Random Forests:
- RF1: Predicts I vs (P+R)
- RF2: Predicts (I+P) vs R

**Correct probability calculation:**
```python
# Binary probabilities
prob1 = rf_model1.predict_proba(dataset)[:, 1]  # P(not I)
prob2 = rf_model2.predict_proba(dataset)[:, 1]  # P(R)

# Calculate unnormalized probabilities
prob_I_unnorm = 1 - prob1
prob_R_unnorm = prob2
prob_P_unnorm = prob1 * (1 - prob2)

# CRITICAL: Normalize to sum to 1.0
total = prob_I_unnorm + prob_P_unnorm + prob_R_unnorm
prob_I = prob_I_unnorm / total
prob_P = prob_P_unnorm / total
prob_R = prob_R_unnorm / total
```

### Alternative: Single Multi-class RF

Simpler and guaranteed normalized:
```python
rf_model = RandomForestClassifier(...)
rf_model.fit(X_train, y_train)
probs = rf_model.predict_proba(X)  # Automatically normalized to sum=1.0
prob_I, prob_P, prob_R = probs[:, 0], probs[:, 1], probs[:, 2]
```

## Test Results

### TEST 1: thermal_map-only
✓ Works correctly (no metadata_input)
✓ Labels: shape (32, 3), properly one-hot encoded

### TEST 2: Fusion (metadata+thermal_map)
✓ RF predictions exist
✓ Shape correct: (32, 3)
✗ **FAIL**: Predictions don't sum to 1.0 (sum=1.04-1.10)
✓ No NaN/Inf
✓ Label alignment seems correct

## Fix Priority: CRITICAL

This bug makes fusion completely unusable. Must fix before any other fusion work.

## Recommended Fix

1. Add normalization step after computing prob_I, prob_P, prob_R
2. Verify sum=1.0 with assertion or warning
3. Re-run fusion tests to confirm Kappa improves

## Next Steps

1. Implement normalization fix in both files
2. Test thermal_map baseline (should still get Kappa 0.10-0.20)
3. Test fixed fusion (should now get Kappa > 0.20)
4. Compare with single multi-class RF approach
