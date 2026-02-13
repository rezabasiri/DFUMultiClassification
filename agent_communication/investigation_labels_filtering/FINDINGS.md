# Investigation Findings

## Cloud Agent Findings (2026-02-13)

### Verified Working
- [x] Label mapping consistent (I→0, P→1, R→2) across all files
- [x] Code flow order correct: env var set BEFORE prepare_dataset() called
- [x] Sample ID format: P{patient:03d}A{appt:02d}D{dfu}

### BUG FOUND AND FIXED: dtype Check Inconsistency

**Location**: `src/data/dataset_utils.py:1148`

**Problem**: Inconsistent dtype check for string labels
- Line 69 (create_patient_folds): Uses robust check
  ```python
  if data['Healing Phase Abs'].dtype in ['object', 'str'] or pd.api.types.is_string_dtype(...)
  ```
- Line 1148 (feature selection): Used incomplete check
  ```python
  if y_train_raw.dtype == object or y_train_raw.dtype.name == 'string'
  ```

**Impact**: If pandas uses StringDtype (nullable string), line 1148 would NOT convert labels, passing string values to RF, causing silent failures or NaN.

**Fix Applied**: Updated line 1148 to use same robust check as line 69.

### Debug Logging Added
Added debug prints to `image_processing.py:237-262`:
- Shows CONFIDENCE_EXCLUSION_FILE env var value
- Shows if file exists
- Shows count of excluded IDs loaded
- Shows sample ID format comparison (first 3 from each)
- Shows actual exclusion count

### Fold Mechanism Documented
Added documentation to `production_config.py:322-346` explaining:
- Difference between `--cv_folds` (CLI) and `CV_N_SPLITS` (config)
- Subprocess isolation mechanism
- Quick test command

---

## Quick Test Command (LOCAL AGENT)

Run this for fast debugging (2 folds, 40% data):
```bash
cd /home/user/DFUMultiClassification
source /opt/miniforge3/bin/activate multimodal
python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 2>&1 | tee debug_run.log
```

**Note**: Cloud agent cannot run this - requires local Python environment with TensorFlow, pandas, etc.

Watch for these debug lines:
```
DEBUG CONF-FILTER: CONFIDENCE_EXCLUSION_FILE env = ...
DEBUG CONF-FILTER: Loaded X excluded IDs from file
DEBUG CONF-FILTER: Sample IDs in data (first 3): [...]
DEBUG CONF-FILTER: Sample IDs in exclusion (first 3): [...]
DEBUG CONF-FILTER: Matched & excluded X samples
```

---

## Local Agent Tasks

### Task 1: Run Debug Test
- [ ] Status: PENDING
- Run the quick test command above
- Check for "DEBUG CONF-FILTER:" lines in output
- Note: First run will generate exclusion list (prelim training)

### Task 2: Verify ID Format Match
- [ ] Status: PENDING
- Compare sample IDs shown in debug output
- Format should be identical (e.g., P065A00D1)

### Task 3: Check Filtering Effect
- [ ] Status: PENDING
- Expected: "Matched & excluded X samples" where X > 0
- If X = 0 but excluded IDs loaded > 0: Format mismatch!

---

## Root Cause Hypotheses

### Hypothesis A: dtype Check Inconsistency - FIXED
The dtype check at line 1148 was incomplete.
**Status**: BUG FIXED (dataset_utils.py updated)

### Hypothesis B: Sample ID Format Mismatch
Sample IDs might not match between exclusion file and data.
**Status**: Debug added to verify

### Hypothesis C: Exclusion File Not Generated
File might not be written after preliminary training.
**Status**: Debug added to verify

### Hypothesis D: Filtering Works But Doesn't Help
User confirmed this is NOT the issue.
**Status**: RULED OUT

---

## Proposed Fix (Applied)
1. Fixed dtype inconsistency at dataset_utils.py:1148
2. Added debug logging to image_processing.py
3. Documented fold mechanism in production_config.py

**Next**: Run debug test and check output for ID format match.
