# Investigation Findings

## Cloud Agent Findings (2026-02-13)

### Verified Working
- [x] Label mapping consistent (I→0, P→1, R→2) across all files
- [x] Code flow order correct: env var set BEFORE prepare_dataset() called
- [x] Sample ID format: P{patient:03d}A{appt:02d}D{dfu}

### After Git Pull (commit 883c6c1)
- Previous exclusion list DELETED - must regenerate
- Previous results JSON DELETED - must regenerate
- No checkpoint .npy files exist
- No best_matching.csv exists

### Suspicious Areas Requiring Local Verification

1. **Sample ID zero-padding mismatch?**
   - Filter creates: `P{int:03d}A{int:02d}D{int}`
   - training_utils saves: `[patient, appt, dfu]` as integers
   - Need to verify conversion format matches

2. **RF Label dtype at training**
   - `dataset_utils.py:1148` checks dtype
   - After resampling, labels might be numeric (0,1,2) instead of strings
   - Map operation would fail silently if labels already numeric

3. **Exclusion file not written to disk?**
   - `confidence_based_filtering.py:904-911` writes file
   - But main.py:2286 assumes file exists
   - **CHECK**: Does the file actually get created?

---

## Local Agent Findings

### Task 1: Confidence Filtering Applied?
- [ ] Status: PENDING
- "CONFIDENCE-BASED FILTERING" header seen?:
- "Confidence filtering: excluded X/Y samples" seen?:
- Notes:

### Task 2: Label Distribution
- [ ] Status: PENDING
- Raw CSV:
  - Class I:
  - Class P:
  - Class R:
- After best_matching:
- After filtering:
- After resampling:
- At model.fit():

### Task 3: RF Model Training
- [ ] Status: PENDING
- y_train_fs dtype before RF:
- RF prediction shape:
- RF prediction sample (first 3 rows):
- Notes:

### Task 4: Fusion Dimensions
- [ ] Status: PENDING
- Metadata input shape:
- Image feature shape:
- Notes:

### Task 5: Silent Failures
- [ ] Status: PENDING
- Errors found:
- Warnings found:
- Exceptions caught:

---

## Root Cause Hypotheses

### Hypothesis A: Exclusion File Not Being Applied
The env var is set, but the file might not exist when prepare_dataset reads it.
**Test**: Add debug print in image_processing.py:237-261

### Hypothesis B: Sample ID Format Mismatch
Sample IDs in exclusion file don't match IDs generated from best_matching.csv.
**Test**: Compare formats side by side

### Hypothesis C: Labels Already Numeric Before RF
If labels are already 0,1,2 after resampling, the dtype check fails.
**Test**: Print dtype at dataset_utils.py:1148

### Hypothesis D: Filtering Works But Doesn't Help
96/648 samples (14.8%) might not be enough to change metrics significantly.
**Test**: Try 30% filtering and compare metrics

---

## Proposed Fix
(To be filled after root cause identified)
