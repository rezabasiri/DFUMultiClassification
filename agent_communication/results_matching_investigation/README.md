# Results Matching Investigation - RESOLVED ✅

## Issue
`main.py` and `auto_polish_dataset_v2.py` produced different results despite using same parameters.

## Root Causes Fixed

### 1. Removed Deprecated `n_runs` Parameter
- Conflicted with `cv_folds` parameter
- Removed from `main.py`, `training_utils.py`

### 2. Fixed Unwanted Default Filtering (CRITICAL)
- **Bug**: When no thresholds specified, main.py still filtered data using hardcoded defaults `{I:12, P:9, R:12}`
- **Fix**: Skip filtering when `thresholds=None`
- **Files**: [src/main.py:1844-1847](../../src/main.py#L1844-L1847), [scripts/auto_polish_dataset_v2.py:877](../../scripts/auto_polish_dataset_v2.py#L877)

### 3. Fixed Dataset Size Calculation
- Added check for `best_matching.csv` in `get_original_dataset_size()`

### 4. Fixed CROSS_VAL_RANDOM_SEED
- Removed env var override from `create_patient_folds()`
- Only set seed for single-split mode, not CV

## Filtering Logic (Now Correct)

```python
# No args → thresholds=None → NO filtering (default)
python src/main.py

# Explicit filtering with manual thresholds
python src/main.py --threshold_I 5 --threshold_P 3 --threshold_R 5

# Use Bayesian-optimized thresholds
python src/main.py --core-data  # Loads from bayesian_optimization_results.json
```

## Test Results
```
Macro F1: 0.298746 (both main.py and auto_polish)
Difference: 0.000000 ✅
```

## Run Test
```bash
cd /workspace/DFUMultiClassification
/venv/multimodal/bin/python agent_communication/results_matching_investigation/test_matching.py
```
