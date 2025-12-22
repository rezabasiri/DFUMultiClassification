# DFU Multimodal Classification - Usage Guide

This guide explains how to run the production pipeline for testing modality combinations.

## Quick Start

### Test All 31 Modality Combinations (Default)

```bash
python src/main.py --mode search
```

This will:
- Test all 31 possible modality combinations
- Use 100% of your data
- 67% train / 33% validation split
- 3 independent runs per combination
- Save results to `results/modality_combination_results.csv`

### Quick Test (10% data, 1 run)

```bash
python src/main.py --mode search --data_percentage 10 --n_runs 1
```

Perfect for verifying everything works before the full run.

### Full Production Run (Recommended)

```bash
python src/main.py --mode search --data_percentage 100 --train_patient_percentage 0.70 --n_runs 5
```

This tests all combinations with:
- 100% of data
- 70% train / 30% validation
- 5 independent runs (more robust statistics)

## Command-Line Options

### `--mode`
**Description**: Mode of operation
**Choices**: `search`, `specialized`, `grid_search`
**Default**: `search`

- `search`: Test modality combinations and save results to CSV
- `specialized`: Run specialized evaluation with gating networks
- `grid_search`: Perform grid search for hyperparameter tuning

**Example**:
```bash
python src/main.py --mode search
```

### `--data_percentage`
**Description**: Percentage of total data to use (1-100)
**Default**: `100.0`

Useful for quick testing with a subset of data.

**Examples**:
- `10` - Quick test with 10% of data
- `50` - Half of the data
- `100` - Full dataset (production)

**Example**:
```bash
python src/main.py --data_percentage 10  # Quick test
```

### `--train_patient_percentage`
**Description**: Percentage of patients to use for training (0.0-1.0)
**Default**: `0.67`

Patient-level split ensures no data leakage between train/validation sets.

**Examples**:
- `0.67` - 67% train / 33% validation
- `0.70` - 70% train / 30% validation
- `0.80` - 80% train / 20% validation

**Example**:
```bash
python src/main.py --train_patient_percentage 0.70
```

### `--n_runs`
**Description**: Number of independent runs with different random patient splits
**Default**: `3`

Results are averaged across runs with standard deviation. More runs = more robust results but longer runtime.

**Examples**:
- `1` - Single run (quick test)
- `3` - Standard (good balance)
- `5` - Robust (publication-quality)

**Example**:
```bash
python src/main.py --n_runs 5
```

## Configuration File

All hyperparameters and modality combinations are configured in:
```
src/utils/production_config.py
```

### Key Configuration Parameters

#### Modality Search Settings

```python
# Search mode: 'all' tests all 31 combinations, 'custom' uses INCLUDED_COMBINATIONS
MODALITY_SEARCH_MODE = 'all'  # Options: 'all', 'custom'

# Combinations to exclude (when mode='all')
EXCLUDED_COMBINATIONS = []  # e.g., [('depth_rgb',), ('thermal_rgb',)]

# Combinations to include (when mode='custom')
INCLUDED_COMBINATIONS = [
    ('metadata',), ('depth_rgb',), ('depth_map',), ('thermal_map',),
    ('metadata','depth_rgb'), ('metadata','depth_map'),
    # ... add your custom combinations
]

# Output filename
RESULTS_CSV_FILENAME = 'modality_combination_results.csv'
```

**To test specific combinations only:**
1. Edit `production_config.py`
2. Set `MODALITY_SEARCH_MODE = 'custom'`
3. List desired combinations in `INCLUDED_COMBINATIONS`

#### Training Hyperparameters

```python
# Core training parameters
IMAGE_SIZE = 64  # Image dimensions (64x64)
GLOBAL_BATCH_SIZE = 30  # Total batch size across all GPUs
N_EPOCHS = 1000  # Maximum epochs (early stopping will stop earlier)

# Learning rates
GATING_LEARNING_RATE = 1e-3
HIERARCHICAL_LEARNING_RATE = 1e-3

# Early stopping patience
GATING_EARLY_STOP_PATIENCE = 20
HIERARCHICAL_PATIENCE = 20
```

## Output

### Results CSV

Results are saved to: `results/modality_combination_results.csv`

**Columns:**
- `Modalities`: Combination tested
- `Accuracy (Mean)`: Average accuracy across runs
- `Accuracy (Std)`: Standard deviation of accuracy
- `Macro Avg F1-score (Mean)`: F1-macro (mean)
- `Macro Avg F1-score (Std)`: F1-macro (std)
- `Weighted Avg F1-score (Mean)`: F1-weighted (mean)
- `Weighted Avg F1-score (Std)`: F1-weighted (std)
- `I F1-score (Mean)`: F1 for Inflammatory phase
- `P F1-score (Mean)`: F1 for Proliferative phase
- `R F1-score (Mean)`: F1 for Remodeling phase
- `Cohen's Kappa (Mean)`: Kappa score (mean)
- `Cohen's Kappa (Std)`: Kappa score (std)

### Progress During Run

The script prints progress information:
- Current modality combination being tested
- Run number (e.g., "Run 1/3")
- Training progress (epochs, loss, accuracy)
- Validation metrics after each run

## Example Workflows

### 1. Quick Verification (5 minutes)
```bash
# Test a few combinations with minimal data
python src/main.py --mode search --data_percentage 10 --n_runs 1
```

### 2. Pilot Study (1-2 hours)
```bash
# Test all combinations with 50% data
python src/main.py --mode search --data_percentage 50 --n_runs 2
```

### 3. Full Production Run (many hours)
```bash
# Test all combinations with full data and robust statistics
python src/main.py --mode search --data_percentage 100 --train_patient_percentage 0.70 --n_runs 5
```

### 4. Test Specific Combinations Only
```bash
# 1. Edit src/utils/production_config.py:
#    - Set MODALITY_SEARCH_MODE = 'custom'
#    - List combinations in INCLUDED_COMBINATIONS
# 2. Run:
python src/main.py --mode search
```

## Getting Help

View all options and examples:
```bash
python src/main.py --help
```

## Runtime Estimates

**Single combination, single run:**
- With early stopping: 10-30 minutes (depends on when it converges)
- Full 1000 epochs: 1-2 hours

**Full production run (all 31 combinations × 5 runs = 155 training sessions):**
- Estimated time: 1-3 days (depending on hardware and early stopping)
- Can be interrupted and resumed (progress is saved)

## Tips

1. **Start small**: Always test with `--data_percentage 10 --n_runs 1` first to verify everything works

2. **Monitor GPU memory**: If you run out of GPU memory, reduce `GLOBAL_BATCH_SIZE` in `production_config.py`

3. **Interrupt safely**: You can stop the run with Ctrl+C. Progress is saved after each combination completes.

4. **Check results incrementally**: The CSV file is updated after each combination, so you can check intermediate results

5. **Reproducibility**: Results are reproducible due to fixed random seeds in `production_config.py`

## Differences from Demo Scripts

| Feature | Demo Scripts | main.py (Production) |
|---------|--------------|---------------------|
| **Purpose** | Quick testing/verification | Research-quality results |
| **Epochs** | 3 | 1000 (with early stopping) |
| **Batch Size** | 4 | 30 |
| **Data** | demo_best_matching.csv | best_matching.csv |
| **Config** | demo_config.py | production_config.py |
| **Runtime** | Minutes | Hours to days |
| **Output** | Text file | CSV with statistics |

## Troubleshooting

**Out of memory error:**
- Reduce `GLOBAL_BATCH_SIZE` in `production_config.py`
- Reduce `IMAGE_SIZE` (e.g., from 64 to 32)

**ImportError:**
- Make sure you're in the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

**CUDA errors:**
- Check GPU availability: `nvidia-smi`
- Reduce batch size or image size

**Results not saving:**
- Check write permissions for `results/` directory
- Verify disk space available
