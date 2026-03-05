# Contamination Search Experiment

Bayesian optimization to find the optimal `OUTLIER_CONTAMINATION` value (range: 0.0-0.4).

## Quick Start

```bash
# Install optuna if needed
pip install optuna

# Run search (default: 15 trials)
python agent_communication/contamination_search/search_contamination.py

# Custom number of trials
python agent_communication/contamination_search/search_contamination.py --n-trials 20
```

## How It Works

1. **Bayesian Optimization**: Uses Optuna's TPE sampler to intelligently explore the contamination range [0.0, 0.4]
2. **Single Fold**: Runs fold 0 only to minimize computation time (~15 trials instead of 45 for 3-fold)
3. **Automatic Config Management**: Temporarily modifies `production_config.py` during each trial, restores original after completion
4. **Objective**: Maximize validation kappa score

## Results

Results are saved to:
- `results/search_results.json` - All trial results and best parameters
- `logs/contamination_X.XXX_fold0.log` - Full training log for each trial

## Typical Timeline

- Per trial: ~15-30 minutes (depends on hardware)
- 15 trials: ~4-8 hours total
- Best value typically found within first 10 trials

## Analysis

After completion, run:
```bash
python agent_communication/contamination_search/analyze_results.py
```

This generates:
- Optimization history plot
- Contamination vs Kappa scatter plot
- Statistical summary
- Recommendation for production config

## Current Baseline

- `OUTLIER_CONTAMINATION = 0.15` (current production value)
- Expected kappa: ~0.27 (from Phase 7 investigation)

The search will explore if values outside this range perform better.
