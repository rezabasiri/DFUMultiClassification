# Quick Start for Local Agent

## Setup
```bash
source /opt/miniforge3/bin/activate multimodal
cd /home/user/DFUMultiClassification
```

## What to Verify

The remote agent added `--core-data` flag to automatically use optimized thresholds from auto_polish_dataset_v2.py results.

**Changes**:
1. `src/main.py`: Added flag, loads JSON thresholds
2. `src/evaluation/metrics.py`: Fixed path resolution for misclassification CSV

## Critical Test (5 min)

```bash
# 1. Quick smoke test
python src/main.py --mode search --data_percentage 5 --cv_folds 1 --core-data 2>&1 | grep "Core data mode"
# Should show: "Core data mode enabled... Best thresholds: {'P': 5, 'I': 4, 'R': 3}"

# 2. Verify filtering
python src/main.py --mode search --data_percentage 100 --cv_folds 1 --core-data 2>&1 | grep "Using.*samples"
# Should show FEWER samples than without --core-data

# 3. Check file is found
python src/main.py --mode search --data_percentage 5 --cv_folds 1 --core-data --verbosity 2 2>&1 | grep "Found misclassification"
# Should show: "Found misclassification file: misclassifications_saved/frequent_misclassifications_saved.csv"
```

## Full Verification

See `VERIFICATION_GUIDE.md` for complete test suite (5 tasks)

Most important: **Task 4** - Performance comparison between auto_polish and main.py should match

## Expected Issues to Check

1. ❓ Does it find the CSV in `results/misclassifications_saved/`?
2. ❓ Are thresholds loaded correctly from JSON?
3. ❓ Does filtering actually reduce dataset size?
4. ❓ Do metrics match Phase 2 optimization results?

## Report

Save findings to: `VERIFICATION_REPORT.md`
