# Metadata Modality Diagnostic Tests

## Problem
Metadata modality achieving 10% accuracy (worse than random) with negative Kappa.

## Instructions for Local Agent

### Run Order
Execute tests in sequence. Stop at first major failure and report back.

```bash
cd agent_communication/metadata_diagnosis

# Test 1: Data loading and basic stats
python test_1_data_loading.py

# Test 2: Feature engineering
python test_2_feature_engineering.py

# Test 3: Preprocessing pipeline
python test_3_preprocessing.py

# Test 4: RF training (quick, 1 run only)
python test_4_rf_training.py

# Test 5: End-to-end minimal test
python test_5_end_to_end.py
```

### What to Report

For each test, report:
1. **PASS/FAIL** status
2. **Key metrics** (shown in output)
3. **Error messages** (if any)
4. **Unexpected values** (flagged with ⚠️)

### Small Bugs
Fix minor issues like:
- Missing imports
- Path corrections
- Small type mismatches

### Major Issues
Report immediately and STOP:
- Wrong shapes/dimensions
- NaN/Inf values in critical data
- Label mismatches
- Accuracy < 20% in test 4
- Negative Kappa in test 4

## Expected Flow
All tests should PASS. If test fails, diagnose and fix small bugs, or report major issue.
