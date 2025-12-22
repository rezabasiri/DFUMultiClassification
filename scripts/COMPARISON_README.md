# Comparison Testing: Original vs Refactored Code

This directory contains tools to systematically compare the outputs of `src/main_original.py` and `src/main.py` to ensure the refactoring didn't introduce bugs.

## Quick Start

### Test Individual Modalities (Default)

```bash
# Test metadata, depth_rgb, and depth_map individually
python scripts/compare_main_versions.py

# Uses defaults:
# - 10% of data (for quick testing)
# - 80% train / 20% validation split
# - 1 run (deterministic)
```

### Test Specific Modalities

```bash
# Test only metadata
python scripts/compare_main_versions.py --modalities metadata

# Test metadata and depth_rgb
python scripts/compare_main_versions.py --modalities metadata depth_rgb
```

### Test Modality Combinations

```bash
# Test combinations (use quotes with +)
python scripts/compare_main_versions.py --modalities 'metadata+depth_rgb' 'depth_rgb+depth_map'

# Test all: individuals + one combination
python scripts/compare_main_versions.py --modalities metadata depth_rgb depth_map 'metadata+depth_rgb'
```

### Full Data Test

```bash
# Test with 100% of data (slower but more accurate)
python scripts/compare_main_versions.py --data_percentage 100 --modalities metadata depth_rgb
```

## How It Works

The comparison script:

1. **Runs Original Code** (`src/main_original.py`)
   - Creates a wrapper script to run with custom modality configs
   - Uses the legacy `main_with_specialized_evaluation()` function
   - Saves output to `results/comparison/original_output.txt`

2. **Runs Refactored Code** (`src/main.py`)
   - Temporarily modifies `production_config.py` to use custom modalities
   - Runs with `--cv_folds 0` and `--n_runs 1` for single-split comparison
   - Restores original config after completion
   - Saves output to `results/comparison/refactored_output.txt`

3. **Compares Results**
   - Extracts metrics from results CSV files
   - Compares accuracy, F1 scores, kappa, etc.
   - Reports identical vs different results
   - Saves detailed comparison to `results/comparison/comparison_results.json`

## Output Files

All comparison results are saved to `results/comparison/`:

- `original_output.txt` - Full output from original code
- `refactored_output.txt` - Full output from refactored code
- `original_metrics.json` - Extracted metrics from original
- `refactored_metrics.json` - Extracted metrics from refactored
- `comparison_results.json` - Detailed comparison with differences
- `production_config_backup.py` - Backup of config (auto-restored)
- `run_original_wrapper.py` - Generated wrapper for original code

## Exit Codes

- `0` - All results identical (success)
- `1` - Differences found or errors occurred (failure)

## Interpreting Results

### ✓ Identical Results

```
Identical results: 3 modalities
  ✓ metadata
  ✓ depth_rgb
  ✓ depth_map
```

This means the refactored code produces **exactly the same** metrics as the original for these modalities. ✅

### ✗ Different Results

```
Different results: 1 modalities
  ✗ metadata+depth_rgb
      accuracy: diff=1.234e-03, rel_diff=0.12%
      f1_macro: diff=2.456e-04, rel_diff=0.05%
```

This shows differences between original and refactored. The script reports:
- `diff` - Absolute difference in metric values
- `rel_diff` - Relative difference as percentage

**Small differences** (<0.01%) might be due to:
- Floating point precision
- Different random initialization order
- TensorFlow version differences

**Large differences** (>1%) indicate potential bugs that need investigation.

## Systematic Testing Strategy

Start simple and build up:

### Phase 1: Individual Modalities ⭐ START HERE
```bash
# Test each modality individually
python scripts/compare_main_versions.py --modalities metadata
python scripts/compare_main_versions.py --modalities depth_rgb
python scripts/compare_main_versions.py --modalities depth_map
python scripts/compare_main_versions.py --modalities thermal_map

# Or all at once (default)
python scripts/compare_main_versions.py
```

### Phase 2: Pairwise Combinations
```bash
python scripts/compare_main_versions.py --modalities 'metadata+depth_rgb'
python scripts/compare_main_versions.py --modalities 'metadata+depth_map'
python scripts/compare_main_versions.py --modalities 'depth_rgb+depth_map'
```

### Phase 3: Complex Combinations
```bash
python scripts/compare_main_versions.py --modalities 'metadata+depth_rgb+depth_map'
python scripts/compare_main_versions.py --modalities 'metadata+depth_rgb+thermal_map'
```

### Phase 4: Full Data Validation
```bash
# Once quick tests pass, validate with full data
python scripts/compare_main_versions.py --data_percentage 100 --modalities metadata depth_rgb
```

## Troubleshooting

### "Original code failed with return code 1"

Check `results/comparison/original_output.txt` for the error. Common issues:
- Missing dependencies
- CUDA/GPU errors
- Data file not found

### "Refactored code failed with return code 1"

Check `results/comparison/refactored_output.txt` for the error. Common issues:
- Config syntax error
- Import errors
- Memory issues

### "No results CSV found"

The code didn't complete successfully. Check the output files for errors.

### Timeout after 1 hour

The test is taking too long. Try:
- Reduce `--data_percentage` (use 10% or even 5%)
- Test fewer modalities at once
- Check if code is stuck (check output files)

## Notes

- The script **does not modify** `src/main_original.py` (per user requirement)
- The script temporarily modifies `src/production_config.py` but auto-restores it
- All tests use the same random seed (42) for reproducibility
- Single run (`--n_runs 1`) ensures deterministic comparison
- No cross-validation (`--cv_folds 0`) for direct comparison

## Advanced Usage

### Custom Parameters

```bash
# Test with different train/validation split
python scripts/compare_main_versions.py --train_percentage 0.7

# Test with multiple runs (less deterministic)
python scripts/compare_main_versions.py --n_runs 3

# Custom output directory
python scripts/compare_main_versions.py --output_dir results/my_comparison
```

### Integration with CI/CD

```bash
# Use exit code for automated testing
python scripts/compare_main_versions.py --modalities metadata
if [ $? -eq 0 ]; then
    echo "✓ Comparison passed"
else
    echo "✗ Comparison failed"
    exit 1
fi
```

## Expected Results

If the refactoring is correct, you should see:

```
FINAL SUMMARY
============================================================
✓ Identical: 3
✗ Different: 0
⚠ Missing in refactored: 0
⚠ Missing in original: 0

Details saved to: results/comparison
```

This indicates **perfect match** between original and refactored implementations! 🎉
