# Task: Move LOO Influence Filter From Before Split to After Split

## Status: PENDING (do not implement until instructed)

---

## Objective

Move the LOO (Leave-One-Out) influence filtering from its current location (before the train/val patient split) to after the split, so it only operates on `train_data`. Validation data should remain untouched.

---

## Why

Currently the LOO filter runs on the **full dataset** before the train/val split, removing rows from **both** training and validation sets. This has two problems:

1. **Evaluation bias**: Removing validation rows based on training-derived LOO scores cherry-picks what the model is evaluated on, making the test easier rather than the model better.
2. **Data leakage**: The LOO preprocessing (`_preprocess_for_loo`) fits its imputer/scaler on the full dataset, which includes validation patients.

After the move, LOO will only remove harmful patterns from the training set. The validation set stays untouched and provides a fair, unbiased evaluation.

---

## Current Code Location

**File**: `src/data/dataset_utils.py`

**Current LOO block**: Lines ~800-854, inside `prepare_cached_datasets()`, BEFORE the patient split logic (which starts at ~line 855 with "Use pre-computed patient splits").

The block starts with:
```python
# --- LOO Influence Filtering (applied to full dataset before split) ---
from src.utils.production_config import (USE_RF_LOO_FILTERING, RF_LOO_MIN_INFLUENCE,
                                          RF_LOO_MAX_REMOVAL_PCT, RF_LOO_MIN_PATTERNS_PER_CLASS)
if USE_RF_LOO_FILTERING and 'metadata' in selected_modalities and not for_shape_inference:
    vprint("Running LOO influence filtering on full dataset...", level=1)
    ...
```

And ends with:
```python
        data = data[keep_mask].reset_index(drop=True)
    else:
        vprint(f"LOO filter: no harmful patterns found ...", level=2)
```

---

## Target Location

Insert the LOO block AFTER the train/val patient split creates `train_data` and `valid_data`, but BEFORE oversampling. Specifically:

- **After**: The split logic (lines ~855-940) which creates `train_data` and `valid_data`
- **Before**: Line ~1317: `train_data, alpha_values = apply_mixed_sampling_to_df(train_data, ...)`
- Also before the class weight computation block (line ~1280: `if 'metadata' in selected_modalities:`)

---

## Required Changes

### 1. Remove the LOO block from its current location (~lines 800-854)

Delete the entire block from `# --- LOO Influence Filtering ...` through the closing `else` clause.

### 2. Insert the LOO block at the new location

Place it after the split creates `train_data` but before class weight computation (~line 1280).

### 3. Change all `data` references to `train_data`

In the moved block, replace:
- `_preprocess_for_loo(data)` → `_preprocess_for_loo(train_data)`
- `len(data)` → `len(train_data)` (in log messages)
- `data.index.isin(rows_to_remove)` → `train_data.index.isin(rows_to_remove)`
- `data[keep_mask].reset_index(drop=True)` → `train_data = train_data[keep_mask].reset_index(drop=True)`

### 4. Update the log message

Change `"Running LOO influence filtering on full dataset..."` to `"Running LOO influence filtering on training data..."`

### 5. Update the config comment

In `src/utils/production_config.py`, line ~191, change:
```python
# from BOTH training and validation sets (applied before the split).
```
to:
```python
# from the training set only (applied after the split, before oversampling).
```

---

## Do NOT Change

- `_preprocess_for_loo()` function itself (already correct)
- `rf_loo_influence_filter()` function itself (already correct)
- `valid_data` — must remain completely untouched
- Config values in `production_config.py` (thresholds, toggle, etc.)
- Anything inside `preprocess_split()` — the main RF training pipeline is separate

---

## Verification

Run:
```bash
/venv/multimodal/bin/python src/main.py --data_percentage 100 --device-mode multi --verbosity 2 --resume_mode fresh --fold 1 --cv_folds 3
```

Expected:
- LOO message should say "training data" not "full dataset"
- LOO row count should be ~2400 (training portion of 3108), not 3108
- Unique patterns should be ~400-500 (not 648)
- RF validation metrics should be computed on the full, unfiltered validation set
- No errors or shape mismatches

---

## Context: LOO Tuning History

- `RF_LOO_MIN_INFLUENCE=0.001` was too aggressive — flagged noise as harmful, all classes hit the removal cap
- First run removed 193/648 patterns (30%), validation Kappa crashed from ~0.12 to -0.036
- Thresholds were raised to: `MIN_INFLUENCE=0.005`, `MAX_REMOVAL_PCT=15`, `MIN_PATTERNS_PER_CLASS=50`
- Feature selection was removed from `_preprocess_for_loo()` to avoid mismatch with main RF's per-split MI selection
