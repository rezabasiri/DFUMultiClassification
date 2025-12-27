# Core Data Flag Verification Guide

## Environment
```bash
source /opt/miniforge3/bin/activate multimodal
cd /home/user/DFUMultiClassification
```

## What Was Changed

### 1. src/main.py (lines 2279-2288, 2450-2483)
**Added**: `--core-data` flag that auto-loads optimized thresholds from `results/bayesian_optimization_results.json`

**Logic**:
```python
if args.core_data:
    # Load bayesian_optimization_results.json
    # Extract best_thresholds (e.g., {'P': 5, 'I': 4, 'R': 3})
    # Pass to filter_frequent_misclassifications()
```

**Override**: Manual `--threshold_I/P/R` args override core-data thresholds if specified

### 2. src/evaluation/metrics.py (lines 172-188)
**Fixed**: `filter_frequent_misclassifications()` now checks multiple locations:
```python
possible_paths = [
    'results/misclassifications_saved/frequent_misclassifications_saved.csv',  # auto_polish v2 location
    'results/misclassifications/frequent_misclassifications_saved.csv',       # legacy location
    'results/frequent_misclassifications_saved.csv',                          # direct path
]
```

## Verification Tasks

### Task 1: Code Review
Check these files exist and contain expected changes:
- [ ] `src/main.py` has `--core-data` argparse flag (line ~2280)
- [ ] `src/main.py` loads JSON and extracts thresholds (line ~2450-2470)
- [ ] `src/evaluation/metrics.py` checks multiple paths (line ~172-188)
- [ ] Sample_ID generation format matches: `P{patient:03d}A{appt:02d}D{dfu}`

### Task 2: Verify File Locations
```bash
# Check Bayesian results exist
ls -l results/bayesian_optimization_results.json

# Check misclassification CSV exists
ls -l results/misclassifications_saved/frequent_misclassifications_saved.csv

# Extract best thresholds
python3 -c "import json; d=json.load(open('results/bayesian_optimization_results.json')); print('Best thresholds:', d['best_thresholds'])"
```

**Expected**: Should show `{'P': 5, 'I': 4, 'R': 3}` (or similar values)

### Task 3: Quick Functional Test

**Test A - Verify flag loads thresholds**:
```bash
# Run with --core-data on small dataset (should load thresholds automatically)
python src/main.py --mode search --data_percentage 5 --cv_folds 1 --verbosity 1 --core-data 2>&1 | tee agent_communication/core_data_verification/test_a_output.txt | grep -A3 "Core data mode"
```

**Expected output**:
```
Core data mode enabled: Using optimized thresholds from Bayesian optimization
  Best thresholds: {'P': 5, 'I': 4, 'R': 3}
  Optimization score: 0.0955
```

**Test B - Verify manual override**:
```bash
# Manual threshold should override core-data
python src/main.py --mode search --data_percentage 5 --cv_folds 1 --verbosity 1 --core-data --threshold_P 10 2>&1 | tee agent_communication/core_data_verification/test_b_output.txt | grep -A2 "override"
```

**Expected**: Should see "Manual thresholds override core-data"

**Test C - Verify filtering actually happens**:
```bash
# Without core-data (baseline)
python src/main.py --mode search --data_percentage 100 --cv_folds 1 --verbosity 1 2>&1 | grep "Using.*% of the data:"

# With core-data (should filter out samples)
python src/main.py --mode search --data_percentage 100 --cv_folds 1 --verbosity 1 --core-data 2>&1 | grep "Using.*% of the data:"
```

**Expected**: Second run should show fewer samples (e.g., 237 → 219 samples after filtering)

### Task 4: Performance Comparison (CRITICAL)

**Goal**: Metrics from `main.py --core-data` should match auto_polish_dataset_v2.py Phase 2 best results

**Step 1 - Get Phase 2 best modality and metrics**:
```bash
python3 -c "
import json
results = json.load(open('results/bayesian_optimization_results.json'))
print('Phase 2 Optimization:')
print(f\"  Best score: {results['best_score']:.4f}\")
print(f\"  Best thresholds: {results['best_thresholds']}\")
print(f\"  Dataset size after filter: {results.get('filtered_dataset_size', 'N/A')}\")
"

# Check what modalities were tested in Phase 2
head -5 results/misclassifications_saved/frequent_misclassifications_depth_map_depth_rgb_metadata_thermal_map.csv
```

**Step 2 - Run main.py with same modalities + core-data**:

Find the modality combination from Phase 2 (look at CSV filename):
- If file is `frequent_misclassifications_depth_map_depth_rgb_metadata_thermal_map.csv`
- Then Phase 2 used: `['depth_map', 'depth_rgb', 'metadata', 'thermal_map']`

Edit `src/utils/production_config.py` temporarily:
```python
MODALITY_SEARCH_MODE = 'custom'
INCLUDED_COMBINATIONS = [
    ('depth_map', 'depth_rgb', 'metadata', 'thermal_map'),  # Use modalities from Phase 2
]
```

Run test:
```bash
python src/main.py --mode search --data_percentage 100 --cv_folds 3 --verbosity 1 --core-data 2>&1 | tee agent_communication/core_data_verification/test_comparison.txt
```

**Step 3 - Compare metrics**:

Extract from `test_comparison.txt`:
- Weighted F1-score
- Macro F1-score
- Accuracy
- Per-class F1 scores (I, P, R)

Compare with Phase 2 results from `results/bayesian_optimization_results.json` or terminal output.

**Expected**: Metrics should be **very similar** (within ±0.02 difference is acceptable due to CV randomness)

### Task 5: Edge Cases

**Test without optimization file**:
```bash
# Temporarily rename file
mv results/bayesian_optimization_results.json results/bayesian_optimization_results.json.bak

# Should warn but continue
python src/main.py --mode search --data_percentage 5 --cv_folds 1 --core-data 2>&1 | grep -i "warning"

# Restore
mv results/bayesian_optimization_results.json.bak results/bayesian_optimization_results.json
```

**Expected**: Warning message about missing file, uses original dataset

## Success Criteria

✅ **Pass**: All tests show expected behavior
✅ **Pass**: Performance comparison shows <2% metric difference
✅ **Pass**: No crashes or errors during filtering
✅ **Pass**: Correct number of samples filtered out

❌ **Fail**: Crashes, wrong thresholds loaded, or >5% metric difference

## Report Format

Save to: `agent_communication/core_data_verification/VERIFICATION_REPORT.md`

```markdown
# Verification Report

## Task 1: Code Review
- [ ] Changes confirmed in src/main.py
- [ ] Changes confirmed in src/evaluation/metrics.py

## Task 2: File Locations
- Best thresholds: {'P': X, 'I': Y, 'R': Z}

## Task 3: Functional Tests
- Test A: [PASS/FAIL] - Reason
- Test B: [PASS/FAIL] - Reason
- Test C: [PASS/FAIL] - Samples: XXX → YYY

## Task 4: Performance Comparison
Phase 2 metrics: Weighted F1=X.XXX, Acc=X.XXX
main.py metrics: Weighted F1=Y.YYY, Acc=Y.YYY
Difference: ±X.XXX
Status: [PASS/FAIL]

## Task 5: Edge Cases
- Missing file test: [PASS/FAIL]

## Overall Result
[PASS/FAIL] - Brief summary

## Issues Found
1. [Issue description if any]
```

## Notes
- Use `--data_percentage 5-10` for quick tests (Tasks 1-3, 5)
- Use `--data_percentage 100` for performance comparison (Task 4)
- All terminal outputs saved to `agent_communication/core_data_verification/`
- DO NOT git commit - user will handle it
