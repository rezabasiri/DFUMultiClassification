# Phase 5: Corrected Investigation - Fixing RF Overfitting

**Date:** 2026-01-05 15:00 UTC
**Based on:** Cloud Agent independent analysis of Phase 4
**Priority:** HIGH - Root cause identified

---

## Critical Findings from Phase 4 Analysis

### ❌ Phase 4 Test 3 Was INVALID
- Local agent implemented trainable fusion with only **2 parameters** (not 21)
- Training logs confirm: "trainable_fusion: 2 trainable weights"
- My recommendation was Dense(3) with Concatenate = 21 params (6×3+3)
- **Conclusion:** Trainable fusion results are INVALID - wrong implementation

### ✅ ROOT CAUSE IDENTIFIED: RF Overfitting Duplicates

**Evidence:**
```
RF Training Kappa:   0.9982 (near perfect)
RF Validation Kappa: 0.16   (poor)
Gap:                 0.84   (SEVERE OVERFITTING)
```

**Why this happens:**
1. `combined` sampling: Oversample R from 118 → 276 samples
2. Creates **158 exact duplicates** (57% of R class, 19% of total dataset)
3. RF memorizes duplicate patterns instead of learning
4. Perfect training performance but poor validation

**This explains why 50% beats 100%:**
- 50% data has better natural balance → less oversampling needed
- Fewer duplicates → RF learns to generalize
- metadata @ 50%: Kappa **0.220**
- metadata @ 100%: Kappa **0.16**

**The RF quality degradation happens BEFORE fusion!**

---

## Phase 5 Strategy: Fix the Data, Not the Model

**Key Insight:** The model architecture is fine. The problem is data quality.

**Options (in priority order):**

### Option A: Reduced Oversampling [RECOMMENDED]
- Accept slight class imbalance to reduce duplicates
- Test with `reduced_combined` sampling strategy

### Option B: Use 50% Data [SIMPLEST]
- Accept that 50% is better (Kappa 0.22 proven)
- Production deployment with 50% data

### Option C: SMOTE for Metadata-Only [COMPLEX]
- Use SMOTE synthetic samples for RF training
- Won't work for fusion (synthetic samples have no images)

---

## Test 1: Diagnostic - Reduced Oversampling Analysis [15 min]

**Goal:** Understand current duplicate problem

**Run:**
```bash
cd /workspace/DFUMultiClassification
python agent_communication/fusion_fix/test_reduced_oversampling.py \
  2>&1 | tee agent_communication/fusion_fix/run_reduced_oversampling_analysis.txt
```

**Expected Output:**
- Current strategy: 158 R duplicates (19% of dataset)
- Reduced strategy: 79 R duplicates (13% of dataset)
- 50% fewer duplicates

**Success Criteria:**
- Confirms duplicate counts
- Shows potential for improvement

---

## Test 2: Implement 'reduced_combined' Sampling [MAIN TEST - 45 min]

**Goal:** Test if reducing duplicates improves performance

### Step 1: Implement New Sampling Strategy

**File:** `src/data/dataset_utils.py`
**Location:** Around line 730 (after `combined_smote` implementation)

**Add this code:**

```python
    elif sampling_strategy == 'reduced_combined':
        """
        REDUCED_COMBINED: Reduce oversampling to minimize duplicates

        Strategy:
        1. Find target = midpoint between R and MIDDLE class
        2. Undersample P and I to target
        3. Oversample R to target (creates fewer duplicates)

        Example: I=276, P=496, R=118
        - Target = (276 + 118) / 2 = 197
        - Result: I=197, P=197, R=197
        - R duplicates: 79 (vs 158 with 'combined')
        """
        vprint("Applying 'reduced_combined' sampling (reduce oversampling, accept imbalance)...", level=1)

        # Find class counts
        class_counts = Counter(split_data['Healing Phase Abs'])
        counts_list = [class_counts['I'], class_counts['P'], class_counts['R']]
        sorted_counts = sorted(counts_list)

        middle_count = sorted_counts[1]
        r_count = class_counts['R']

        # Target: midpoint between R and MIDDLE
        target_count = (middle_count + r_count) // 2

        vprint(f"  Original: I={class_counts['I']}, P={class_counts['P']}, R={class_counts['R']}", level=2)
        vprint(f"  Target count: {target_count} (reduces duplicates by ~50%)", level=2)

        # Split by class
        df_I = split_data[split_data['Healing Phase Abs'] == 'I']
        df_P = split_data[split_data['Healing Phase Abs'] == 'P']
        df_R = split_data[split_data['Healing Phase Abs'] == 'R']

        # Undersample I and P to target
        df_I_sampled = df_I.sample(n=min(target_count, len(df_I)), random_state=42, replace=False)
        df_P_sampled = df_P.sample(n=min(target_count, len(df_P)), random_state=42, replace=False)

        # Oversample R to target (but with fewer duplicates than 'combined')
        df_R_sampled = df_R.sample(n=target_count, random_state=42, replace=True)

        # Combine
        split_data = pd.concat([df_I_sampled, df_P_sampled, df_R_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        final_dist = Counter(split_data['Healing Phase Abs'])
        duplicates_created = target_count - r_count
        vprint(f"  After sampling: I={final_dist['I']}, P={final_dist['P']}, R={final_dist['R']}", level=2)
        vprint(f"  R duplicates: {duplicates_created} ({duplicates_created/target_count*100:.1f}% of R)", level=2)
```

### Step 2: Update Config

**File:** `src/utils/production_config.py`

```python
# Change line ~38:
SAMPLING_STRATEGY = 'reduced_combined'  # Was: 'combined'
```

### Step 3: Run Test

```bash
# Config should be:
# SAMPLING_STRATEGY = 'reduced_combined'
# IMAGE_SIZE = 32
# INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]

python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_reduced.txt
```

**Expected Results:**
- Fewer duplicates (79 vs 158)
- Better RF generalization
- metadata-only: 0.18-0.19 (vs 0.16 with 'combined')
- fusion: 0.19-0.20 (vs 0.166 with 'combined')

**Success Criteria:**
- Kappa > 0.18 for fusion (improvement over 0.166)
- Gap to 50% data (0.22) reduced

---

## Test 3: Metadata-Only with 'reduced_combined' [30 min, Optional]

**Goal:** Isolate RF performance improvement

### Config:
```python
SAMPLING_STRATEGY = 'reduced_combined'
INCLUDED_COMBINATIONS = [('metadata',)]  # Metadata-only
```

### Run:
```bash
python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_metadata_100pct_reduced.txt
```

**Expected:** Kappa 0.18-0.19 (vs 0.16 with 'combined')

---

## Decision Tree

```
Test 1 (Analysis): Run reduced_oversampling.py
    ↓
    Confirms 50% fewer duplicates?
    ↓
Test 2 (Main): Run fusion with 'reduced_combined'
    ↓
    Results?
    ├─ Kappa > 0.19: ✅ Problem solved! Use reduced_combined
    ├─ Kappa 0.17-0.19: ⚠️ Partial improvement, consider 50% data
    └─ Kappa < 0.17: ❌ Didn't help, use 50% data instead

If Test 2 succeeds (Kappa > 0.19):
    - Update production config to use 'reduced_combined'
    - Close investigation - problem solved

If Test 2 fails (Kappa < 0.19):
    - Accept 50% data is optimal (Kappa 0.22)
    - Production deployment with DATA_PERCENTAGE=50%
```

---

## What NOT to Do

❌ **Don't test trainable fusion yet** - Phase 4 implementation was wrong (2 params vs 21)
❌ **Don't change RF hyperparameters** - Already optimized (n_estimators=646, max_depth=14, etc.)
❌ **Don't use 'combined_smote' for fusion** - Synthetic samples have no images

---

## Expected Timeline

| Test | Duration | Priority |
|------|----------|----------|
| Test 1: Reduced oversampling analysis | 15 min | Diagnostic |
| Test 2: Fusion with reduced_combined | 45 min | **HIGH** |
| Test 3: Metadata-only with reduced_combined | 30 min | Optional |

**Total:** 1.5 hours (minimum), 2 hours (with optional test)

---

## Reporting Template

### Test 1: Reduced Oversampling Analysis

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Duplicate Counts:**
- Current 'combined': ___ R duplicates (___% of dataset)
- Proposed 'reduced_combined': ___ R duplicates (___% of dataset)
- Reduction: ___% fewer duplicates

**Conclusion:**
[ ] Confirms hypothesis - proceed to Test 2
[ ] Unexpected result - describe: ___________

---

### Test 2: Fusion with 'reduced_combined'

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**Comparison:**
- combined: 0.1664
- reduced_combined: _____
- Improvement: _____%

**RF Overfitting Check:**
- Training kappa: _____
- Validation kappa: _____
- Gap: _____ (lower is better, currently 0.84)

**Conclusion:**
[ ] Success: Kappa > 0.19 (use reduced_combined)
[ ] Partial: Kappa 0.17-0.19 (marginal improvement)
[ ] Failed: Kappa < 0.17 (use 50% data instead)

---

### Test 3: Metadata-Only (Optional)

**Status:** [ ] Not started / [ ] Running / [ ] Complete

**Results:**
| Fold | Kappa |
|------|-------|
| 1    |       |
| 2    |       |
| 3    |       |
| **Avg** | **_____** |

**Comparison:**
- combined: ~0.16
- reduced_combined: _____
- Improvement: _____%

---

## Questions for Cloud Agent

If any unexpected results occur:

1. **If Test 2 Kappa < 0.17:**
   - Does reduced oversampling help at all?
   - Should we just use 50% data?

2. **If Test 2 Kappa = 0.17-0.19:**
   - Is 8-15% improvement worth added complexity?
   - Should we still consider 50% data (Kappa 0.22)?

3. **If Test 2 Kappa > 0.20:**
   - Should we test even less oversampling?
   - Can we reach 0.22 (matching 50% data)?

---

## Files to Review

After completing tests, share:
- `run_reduced_oversampling_analysis.txt` (Test 1)
- `run_fusion_32x32_100pct_reduced.txt` (Test 2)
- `run_metadata_100pct_reduced.txt` (Test 3, if run)

Cloud agent will:
- Verify results
- Determine if 50% > 100% mystery is solved
- Recommend production configuration
