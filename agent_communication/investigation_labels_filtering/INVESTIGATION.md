# Investigation: Confidence Filtering Not Affecting Metrics

**Date**: 2026-02-13
**Issue**: Per-class confidence filtering percentiles have no effect on metrics.
**Latest code**: commit 883c6c1 (after git pull)

## Cloud Agent Findings

### 1. Label Mapping - VERIFIED CONSISTENT
- `I→0, P→1, R→2` in: `preprocessing.py:153`, `dataset_utils.py:70,564,1150`, `image_processing.py:303`, `caching.py:233`

### 2. Current Config (production_config.py)
```
USE_CONFIDENCE_FILTERING = True
CONFIDENCE_FILTER_PERCENTILE_I = 17
CONFIDENCE_FILTER_PERCENTILE_P = 23
CONFIDENCE_FILTER_PERCENTILE_R = 15
USE_FREQUENCY_BASED_WEIGHTS = True
```

### 3. Code Flow - VERIFIED CORRECT
1. `main.py:2253` - Calls `run_confidence_filtering_pipeline()` if enabled
2. `main.py:2284-2288` - Sets `CONFIDENCE_EXCLUSION_FILE` env var BEFORE main_search()
3. `main.py:2311` - Calls `main_search()` → `prepare_dataset()`
4. `image_processing.py:237-261` - Reads env var and excludes samples

**Order is correct** - env var set before prepare_dataset.

### 4. Current State (after git pull)
- `results/confidence_exclusion_list.txt` - **DELETED** (will regenerate)
- `results/confidence_filter_results.json` - **DELETED** (will regenerate)
- `results/best_matching.csv` - NOT EXISTS
- `results/checkpoints/*.npy` - NO FILES

### 5. Key Concerns

#### A. Sample ID Format Matching
Filter expects: `P{patient:03d}A{appt:02d}D{dfu}` (e.g., P065A00D1)
- Created at: `image_processing.py:248-251`
- Saved at: `training_utils.py:1785,1823`
- **Verify zero-padding matches**

#### B. RF Training Labels
At `dataset_utils.py:1148-1153`:
```python
if y_train_raw.dtype == object:
    y_train_fs = y_train_raw.map({'I': 0, 'P': 1, 'R': 2}).values
else:
    y_train_fs = y_train_raw.values
```
**Check**: Is dtype correct at this point?

#### C. Many try/except
~100 exception handlers in src/. Some may silently swallow errors.

---

## LOCAL AGENT TASKS

### CRITICAL Task 1: Run Training With Debug
**IMPORTANT**: Use `--cv_folds` flag, NOT `CV_N_SPLITS` config parameter!

```bash
cd /home/user/DFUMultiClassification
python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 --resume_mode fresh 2>&1 | tee training_debug.log
```

**Why these flags**:
- `--cv_folds 2`: Controls BOTH preliminary training AND main training folds (was hardcoded to 3, now fixed)
- `--data_percentage 40`: Faster testing with subset of data
- `--device-mode multi`: Use multiple GPUs
- `--verbosity 2`: Show detailed output
- `--resume_mode fresh`: Start from scratch, ignore cached results

**What to watch for**:
1. "CONFIDENCE-BASED FILTERING" header
2. "CV Folds: 2" in preliminary training (not 3!)
3. "Running preliminary training..." vs "Found existing exclusion list"
4. "Confidence filtering: excluded X/Y samples"
5. "DEBUG CONF-FILTER:" lines showing actual filtering

### Task 2: Verify Sample ID Format
After training:
```python
import numpy as np
sid = np.load('results/checkpoints/sample_ids_run1_metadata_valid.npy')
print(f"Shape: {sid.shape}, Sample: {sid[:3]}")
```
Compare to exclusion list:
```bash
head results/confidence_exclusion_list.txt
```

### Task 3: Check RF Predictions
```python
pred = np.load('results/checkpoints/pred_run1_metadata_valid.npy')
print(f"Shape: {pred.shape}")
print(f"Row sums: {pred[:5].sum(axis=1)}")  # Should be ~1.0
```

### Task 4: Add Debug to image_processing.py:243
```python
print(f"DEBUG: CONFIDENCE_EXCLUSION_FILE={confidence_exclusion_file}")
print(f"DEBUG: excluded_ids count={len(excluded_ids)}")
print(f"DEBUG: sample_ids[:5]={sample_ids.head()}")
```

### Task 5: Compare Class Distributions
At checkpoints:
1. Raw CSV (count I, P, R)
2. After best_matching.csv
3. After filtering
4. After resampling

---

## Files to Focus
1. `src/main.py:2214-2288` - confidence filtering
2. `src/data/image_processing.py:237-261` - exclusion application
3. `src/data/dataset_utils.py:1148-1153` - label mapping
4. `scripts/confidence_based_filtering.py:193-310` - collect_predictions

---

## Key Learnings (Local Agent)

### 1. Confidence Filtering Fold Configuration
**PROBLEM**: Preliminary training was hardcoded to use 3 folds, even when main training used different number.

**SOLUTION**: Changed both occurrences in `main.py`:
- Line 2259: `cv_folds=3` → `cv_folds=cv_folds` (inside main() function)
- Line 2672: `cv_folds=3` → `cv_folds=args.cv_folds` (in __main__ section)

Now preliminary training uses same number of folds as main training.

### 2. Modality Testing Configuration
**PROBLEM**: When testing with fewer modalities, need to change `INCLUDED_COMBINATIONS`, NOT `ALL_MODALITIES`.

**WHY**: Because `MODALITY_SEARCH_MODE = 'custom'` in production_config.py, which uses `INCLUDED_COMBINATIONS`.

**Example**:
```python
# production_config.py
MODALITY_SEARCH_MODE = 'custom'
INCLUDED_COMBINATIONS = [
    ('metadata', 'depth_rgb'),  # Only test this combination
]
```

### 3. Command Line Flags vs Config Parameters
**Use `--cv_folds` flag, NOT `CV_N_SPLITS` config!**

- `--cv_folds N`: Controls main training AND preliminary confidence filtering
- `CV_N_SPLITS`: Only used for hierarchical gating network (NOT main training)

**Correct**: `python src/main.py --cv_folds 2`
**Wrong**: Changing `CV_N_SPLITS` in production_config.py

### 4. Fresh Restart Best Practices
When starting fresh test:
```bash
# Clean up old outputs
rm -f debug_run.log /tmp/claude-0/-workspace-DFUMultiClassification/tasks/*.output

# Run with fresh flag
python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 --resume_mode fresh
```

---

## Communication
Update FINDINGS.md with results. Mark tasks DONE/BLOCKED.
