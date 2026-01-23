# RF Performance Improvement Tests

## Current Performance
- Accuracy: 30.8%
- Kappa: 0.10
- Class P F1: 4.7% (worst)
- Class R F1: 23.6%

## Goal
Improve to >40% accuracy, Kappa >0.2

## Solutions to Test

1. **Remove dense layers after RF** (hypothesis: overfitting)
2. **SMOTE for class balancing**
3. **Tune RF parameters** (trees, depth, features)
4. **Feature selection** (remove noise)
5. **Ensemble multiple RFs**

## Instructions

```bash
cd agent_communication/rf_improvement
source /Users/rezabasiri/env/multimodal/bin/activate

# Run baseline
python test_baseline.py

# Test solutions (run individually)
python solution_1_remove_dense.py
python solution_2_smote.py
python solution_3_tune_rf.py
python solution_4_feature_selection.py
python solution_5_ensemble.py

# Compare all
python compare_results.py
```

## Report Format

For each solution:
- Accuracy, Kappa, per-class F1
- Better/worse than baseline
- Keep or discard

## Environment
- Activate: `source /Users/rezabasiri/env/multimodal/bin/activate`
- Python 3.9+
- sklearn, numpy, pandas, imblearn
