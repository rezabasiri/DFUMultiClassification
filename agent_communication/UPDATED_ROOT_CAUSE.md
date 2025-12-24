# UPDATED ROOT CAUSE ANALYSIS

## Issue Evolution

### Phase 7 Results - NEW FINDING!

Oversampling worked, but created **over-correction**:
- **Before**: Model predicts ONLY class P (majority) → Min F1=0.0
- **After**: Model predicts ONLY class R (minority) → Min F1=0.0

**This is progress!** It proves oversampling + focal loss work, but they're **double-correcting**.

## The Double-Correction Bug

**File**: `src/data/dataset_utils.py:650-676`

```python
# Line 650-660: Calculate alpha from ORIGINAL distribution
alpha_values = [1.0/class_frequencies[i] for i in [0, 1, 2]]
alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]
# Result: [I=0.725, P=0.344, R=1.931] ← Based on imbalanced data

# Line 662-663: Apply oversampling
oversampler = RandomOverSampler(random_state=42 + run * (run + 3))
X_resampled, y_resampled = oversampler.fit_resample(X, y)
# Result: All classes now have 1504 samples (BALANCED)

# Line 676: Return ORIGINAL alpha values with BALANCED data
return resampled_df, alpha_values
```

## The Problem

**Double-correction occurs**:

1. **Data-level correction**: Oversampling balances classes (all → 1504 samples)
2. **Loss-level correction**: Alpha weights from ORIGINAL distribution [0.725, 0.344, **1.931**]

Result:
- Class R: 1504 training samples (same as others) + 1.931x weight in loss
- Class P: 1504 training samples (same as others) + 0.344x weight in loss
- Class I: 1504 training samples (same as others) + 0.725x weight in loss

Model learns: "Class R errors are 5.6x more important than class P errors, even though I see them equally often in training"

This over-prioritizes class R → model predicts ONLY class R.

## The Fix

After oversampling, **recalculate alpha from balanced distribution** (which gives [1, 1, 1]):

```python
# AFTER oversampling (line 663)
oversampler = RandomOverSampler(random_state=42 + run * (run + 3))
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Recalculate alpha from BALANCED distribution
final_counts = Counter(y_resampled)
total_resampled = len(y_resampled)
balanced_frequencies = {cls: count/total_resampled for cls, count in final_counts.items()}

# These will be approximately [1, 1, 1] since data is balanced
alpha_values_balanced = [1.0/balanced_frequencies[i] for i in [0, 1, 2]]
alpha_sum = sum(alpha_values_balanced)
alpha_values_balanced = [alpha/alpha_sum * 3.0 for alpha in alpha_values_balanced]

# Return balanced alpha values
return resampled_df, alpha_values_balanced
```

## Recommended Approach

**When using oversampling**: Use equal weights `[1, 1, 1]` for alpha, since data is already balanced.

**When NOT using oversampling**: Use inverse frequency weights for alpha, since data is imbalanced.

**NEVER**: Use inverse frequency alpha WITH oversampling (current bug).

## Expected Result After Fix

With balanced data + equal alpha weights:
- Model sees equal samples of all classes in training
- All classes have equal importance in loss function
- Model should predict all 3 classes reasonably
- Min F1 > 0.3 (all classes learned)

## Implementation

**Simplest fix** (line 676):
```python
# After oversampling, data is balanced, so use equal weights
alpha_values_balanced = [1.0, 1.0, 1.0]
return resampled_df, alpha_values_balanced
```

This removes the double-correction and lets oversampling do its job.
