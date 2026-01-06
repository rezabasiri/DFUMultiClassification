# Combined Sampling Analysis

**Date:** 2026-01-05
**Finding:** Combined sampling (under+over) exists in original_main.py but NOT in current code

---

## What I Found

### In original_main.py (lines 933-985)

**Combined Sampling Strategy (when mix=True):**

```python
# Step 1: UNDERSAMPLE majority class
# Reduce P (proliferative) to match I (inflammatory)
intermediate_target = {
    I_class: count_I,  # Keep I as is
    P_class: count_I,  # Reduce P to match I
    R_class: count_R   # Keep R as is (smallest)
}
undersampler = RandomUnderSampler(sampling_strategy=intermediate_target)
X_under, y_under = undersampler.fit_resample(X, y)

# Step 2: OVERSAMPLE minority class
# Boost R (remodeling) to match others
final_target = {
    I_class: count_I,  # Keep I
    P_class: count_I,  # Keep P (already reduced)
    R_class: count_I   # Boost R to match
}
oversampler = RandomOverSampler(sampling_strategy=final_target)
X_resampled, y_resampled = oversampler.fit_resample(X_under, y_under)
```

**Result:** All classes balanced to MIDDLE class count (I), not MAX class count

---

### In current code (dataset_utils.py)

**Status:** Combined sampling NOT implemented! ❌

- Line 656: `if not mix:` - only simple oversampling branch exists
- Line 718: Returns after simple oversampling
- **NO `else:` block for combined sampling**

The `mix` parameter exists in function signature but the combined logic was never ported!

---

## Comparison of Strategies

### Example Distribution:
```
Original: {I: 599, P: 1164, R: 158}
```

### Strategy 1: Simple Random Oversampling (Current Default)
```
Result: {I: 1164, P: 1164, R: 1164}
- R duplicated: 1164/158 = 7.4x
- Total samples: 3492
- Problem: Massive overfitting to 158 real R samples
```

### Strategy 2: SMOTE (What I Just Implemented)
```
Result: {I: 1164, P: 1164, R: 1164}
- R synthetic: generates new samples via interpolation
- Total samples: 3492
- Benefit: No exact duplicates, less overfitting
```

### Strategy 3: Combined Sampling (Original)
```
Step 1 - Undersample P:
  {I: 599, P: 599, R: 158}

Step 2 - Oversample R:
  {I: 599, P: 599, R: 599}

Final: {I: 599, P: 599, R: 599}
- R duplicated: 599/158 = 3.8x (vs 7.4x!)
- Total samples: 1797 (vs 3492)
- Benefits:
  * 50% fewer duplicates of R class
  * Smaller dataset = faster training
  * Less overfitting
  * Keeps more balanced class distribution
```

---

## Why Combined Sampling Was Better

### Problem with Current Approach:
1. **100% data @ pure oversampling:**
   - R class: 158 → 1164 (7.4x duplication)
   - Creates 1006 duplicate copies
   - RF overfits badly
   - Kappa: 0.09

2. **50% data @ pure oversampling:**
   - R class: ~80 → ~580 (7.25x duplication)
   - Similar ratio BUT fewer total samples
   - Less severe overfitting
   - Kappa: 0.22

### Why 50% Works Better:
It's not about the % - it's about the TOTAL number of duplicates!
- 50% data: ~500 duplicate R samples
- 100% data: ~1000 duplicate R samples
- **Doubling duplicates causes catastrophic overfitting**

### How Combined Sampling Helps:
```
100% data with COMBINED sampling:
- Original R: 158
- After combined: 599 (3.8x vs 7.4x)
- ~600 duplicates vs ~1000 duplicates
- Should perform much better!
```

---

## Recommendation

### Option A: Test SMOTE First (Already Implemented)
**Pros:**
- Already coded and ready to test
- Generates synthetic samples (no exact duplicates)
- Should work well

**Cons:**
- Still balances to MAX class (large dataset)
- Synthetic samples might not capture true distribution

**Expected:** RF Kappa 0.15-0.20 @ 100%

---

### Option B: Implement Combined Sampling
**Pros:**
- Proven to work in original_main.py
- Balances to middle class (smaller dataset)
- Fewer duplicates (3.8x vs 7.4x)
- Faster training
- You said it was "working fine"

**Cons:**
- Need to implement it (port from original_main.py)
- Not as sophisticated as SMOTE

**Expected:** RF Kappa 0.18-0.22 @ 100%

---

### Option C: Combined Sampling + SMOTE
**Best of both worlds:**

```python
# Step 1: Undersample majority to middle
undersampler = RandomUnderSampler(...)  # P: 1164 → 599

# Step 2: SMOTE minority to match
smote = SMOTE(...)  # R: 158 → 599 (synthetic)

# Result: {I: 599, P: 599, R: 599}
# - Only 3.8x growth of R class
# - All synthetic (no duplicates)
# - Smaller, cleaner dataset
```

**Expected:** RF Kappa 0.20-0.25 @ 100% (best!)

---

## What Should We Do?

### My Recommendation:

1. **Test SMOTE alone first** (already implemented)
   - Quick validation
   - See if synthetic samples help
   - Expected: Kappa 0.15-0.20

2. **If SMOTE doesn't reach 0.20:**
   - Implement Option C (Combined + SMOTE)
   - Port combined sampling logic
   - Use SMOTE instead of RandomOverSampler in Step 2

3. **If SMOTE works great (Kappa > 0.20):**
   - Keep it simple, use SMOTE
   - Move to Fix #2 (trainable fusion)

---

## Implementation Plan for Combined Sampling

If we decide to implement it:

**File:** `src/data/dataset_utils.py`
**Location:** After line 718 (add else block)

```python
# Add else block for mix=True
else:
    # Combined sampling: undersample majority, oversample minority
    vprint("Using combined sampling (under + over)...", level=2)

    # Step 1: Undersample P to match I
    intermediate_target = {
        count_items[1][1]: count_items[1][0],  # I stays same
        count_items[2][1]: count_items[1][0],  # P reduced to I
        count_items[0][1]: counts[count_items[0][1]]  # R stays same
    }

    undersampler = RandomUnderSampler(
        sampling_strategy=intermediate_target,
        random_state=42 + run * (run + 3)
    )
    X_under, y_under = undersampler.fit_resample(X, y)

    # Step 2: Oversample R to match others
    if USE_SMOTE:
        # Use SMOTE for minority oversampling
        oversampler = SMOTE(random_state=42 + run * (run + 3), k_neighbors=5)
    else:
        # Use random duplication
        final_target = {i: count_items[1][0] for i in [0, 1, 2]}
        oversampler = RandomOverSampler(
            sampling_strategy=final_target,
            random_state=42 + run * (run + 3)
        )

    X_resampled, y_resampled = oversampler.fit_resample(X_under, y_under)

    # Rest of the code...
```

---

## My Mistake

I should have:
1. ✅ Checked original_main.py for sampling strategy
2. ❌ **Noticed combined sampling was missing**
3. ❌ **Asked you about it before implementing SMOTE**

**What I did:**
- Went straight to SMOTE (common solution for oversampling)
- Missed that you already had a working solution

**What I should have done:**
- Checked if combined sampling existed
- Asked: "Should I port combined sampling or try SMOTE?"

---

## Next Steps

**Your call! Three options:**

### 1. Test SMOTE as-is (Quick)
```bash
# Already implemented, just test
python src/main.py --mode search ...
```
**Time:** 30-40 min
**Expected:** Kappa 0.15-0.20

### 2. Implement Combined Sampling (Medium)
```bash
# I port the logic from original_main.py
# Then you test
```
**Time:** 20 min implement + 30 min test
**Expected:** Kappa 0.18-0.22

### 3. Implement Combined + SMOTE (Best)
```bash
# I combine both strategies
# Then you test
```
**Time:** 30 min implement + 30 min test
**Expected:** Kappa 0.20-0.25 (likely best result)

---

## My Preference

**Try Option 1 first** (test current SMOTE):
- Already done, no extra work
- See if synthetic samples alone solve it
- If Kappa < 0.18, implement Option 3

**Then if needed, do Option 3** (combined + SMOTE):
- Port combined sampling logic
- Use SMOTE for Step 2 oversampling
- Should give best results

What do you prefer? 🤔
