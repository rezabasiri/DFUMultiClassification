# Local Agent Checklist

## Preparation
- [ ] Navigate to: `agent_communication/metadata_diagnosis`
- [ ] Ensure Python environment active with required packages
- [ ] Read README.md and EXPECTED_VS_ACTUAL.md

## Execution
- [ ] Run: `bash run_all_tests.sh` (or run tests individually)
- [ ] Monitor output for failures
- [ ] Note which test fails first

## For Each Test Failure

### Small Bugs (Fix Immediately)
- [ ] Missing import → Add import
- [ ] Wrong path → Correct path
- [ ] Type mismatch → Fix type
- [ ] Column name typo → Fix name

### Major Issues (Stop and Report)
- [ ] Accuracy < 20% in test 4
- [ ] Negative Kappa in test 4
- [ ] NaN/Inf in critical data
- [ ] Shape mismatches
- [ ] Label corruption
- [ ] Complete failure to load data

## Reporting

### If All Tests Pass:
Report: "All diagnostic tests PASSED. Issue likely in:
- Multi-GPU training
- CV fold logic
- Model architecture
- Data filtering in actual run"

### If Test Fails:
Report using REPORT_TEMPLATE.md:
1. Which test failed
2. Error message (full traceback)
3. Key metrics (if test 4)
4. Any warnings from earlier tests

## Critical Metrics to Report

**From Test 4:**
- Validation accuracy
- Cohen's Kappa
- Per-class F1 scores
- Binary classifier accuracies

**From Test 5:**
- Which prepare_cached_datasets used (caching.py vs dataset_utils.py)
- Batch shapes
- Any errors

## Decision Tree

```
Test 1 fails → Data loading/path issue
Test 2 fails → Feature engineering bug
Test 3 fails → Imputation/normalization bug
Test 4 fails → **CRITICAL** - RF training broken
Test 5 fails → Integration/dataset creation bug
All pass → Issue in actual training pipeline
```
