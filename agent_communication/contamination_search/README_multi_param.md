# Multi-Parameter Hyperparameter Search

Bayesian optimization to find optimal values for multiple hyperparameters simultaneously.

## Parameters Being Optimized

1. **N_EPOCHS** (50-200, step 10): Total training epochs
2. **IMAGE_SIZE** (32-256): Image dimensions - [32, 64, 96, 128, 160, 192, 224, 256]
3. **STAGE1_EPOCHS** (5-20% of N_EPOCHS): Stage 1 fusion training epochs
4. **OUTLIER_CONTAMINATION** (0.05-0.40, step 0.01): Outlier removal rate (informed by contamination search)
5. **OUTLIER_BATCH_SIZE** ([16, 32, 64, 128]): Batch size for outlier detection

## Key Differences from Contamination Search

- **Full 3-fold CV**: Runs all 3 folds per trial (more robust but slower)
- **Multi-parameter**: Optimizes 5 parameters jointly to find best combination
- **Longer runtime**: ~45 min/trial vs ~20 min for single-fold contamination search
- **Metric extraction**: Reads from `results/csv/modality_results_averaged.csv` (Cohen's Kappa Mean column)

## Quick Start

```bash
# Install optuna if needed
pip install optuna

# Start NEW search (20 trials, ~15 hours)
python agent_communication/contamination_search/search_multi_param.py --n-trials 20 --fresh

# Resume EXISTING search (continues from where it left off)
python agent_communication/contamination_search/search_multi_param.py --n-trials 20

# Smaller test run (10 trials, ~7.5 hours)
python agent_communication/contamination_search/search_multi_param.py --n-trials 10 --fresh
```

## Resume Capability

**Automatic Resume:**
- Results saved after **each trial** to `results/multi_param_search_results.json`
- Study state saved to `results/optuna_study.db` (Optuna SQLite database)
- If interrupted, simply re-run the same command to resume

**Start Fresh:**
- Use `--fresh` flag to delete existing study and start from scratch
- Example: `python ... --n-trials 20 --fresh`

**Check Progress:**
- Monitor `results/multi_param_search_results.json` (updated after each trial)
- Shows: completed trials, best params so far, all trial results

## Design Decisions

### Contamination Range: 0.05-0.40 (informed but expanded)
Based on contamination search results:
- Best value from initial search: 0.29 (kappa 0.3809)
- Expanded range: 0.05-0.40 to explore joint effects with other parameters
- Step size: 0.01 (limits decimal places as requested)
- Rationale: Other parameters may shift optimal contamination value

### IMAGE_SIZE: Common sizes only
- Uses standard sizes: 32, 64, 96, 128, 160, 192, 224, 256
- Avoids arbitrary values for GPU efficiency
- Includes current production value (256)

### STAGE1_EPOCHS: Relative to N_EPOCHS
- Defined as ratio (5-20% of N_EPOCHS)
- Ensures valid relationship: STAGE1 < N_EPOCHS
- Minimum 5 epochs to allow proper stage 1 training

### OUTLIER_BATCH_SIZE: Powers of 2
- GPU-efficient sizes: 16, 32, 64, 128
- Current production value: 64

## How It Works

**3-Fold Execution:**
1. Script calls `main.py` without `--fold` argument
2. `main.py` runs all 3 folds in separate subprocesses (prevents resource leaks)
3. Results are aggregated and saved to `results/csv/modality_results_averaged.csv`

**Metric Extraction:**
- Reads CSV file: `results/csv/modality_results_averaged.csv`
- Extracts: `Cohen's Kappa (Mean)` column (average across 3 folds)
- Format: CSV with header, last row contains latest results

## Typical Timeline

- **Per trial**: 30-60 minutes (3 folds × 10-20 min/fold)
  - Depends on N_EPOCHS and IMAGE_SIZE selected
  - Smaller configs (50 epochs, 32px) faster
  - Larger configs (200 epochs, 256px) slower
- **20 trials**: ~10-20 hours total
- **Recommendation**: Run overnight or over weekend

## Results

Results saved to:
- `results/multi_param_search_results.json` - All trial results and best parameters
- `logs_multi/` - Individual training logs per trial

## Log Format

Logs named: `ep{N_EPOCHS}_img{SIZE}_s1_{STAGE1}_cont{CONT}_bs{BATCH}.log`

Example: `ep100_img128_s1_15_cont0.29_bs64.log`
- 100 epochs
- 128×128 images
- 15 stage1 epochs
- 0.29 contamination
- Batch size 64

## After Search Completes

1. Check `results/multi_param_search_results.json` for best parameters
2. Update `src/utils/production_config.py` with optimal values
3. Run final validation with full 3-fold CV
4. Compare against baseline (current config)

## Current Baseline (from production_config.py)

```python
N_EPOCHS = 100
IMAGE_SIZE = 256
STAGE1_EPOCHS = 10
OUTLIER_CONTAMINATION = 0.15
OUTLIER_BATCH_SIZE = 64
```

Search will find if different values improve performance.
