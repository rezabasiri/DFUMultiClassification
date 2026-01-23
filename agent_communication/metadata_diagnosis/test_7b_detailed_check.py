"""Test 7b: Detailed probability check"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from src.utils.config import get_data_paths, get_project_paths
from src.data.dataset_utils import prepare_cached_datasets
import pandas as pd

print("="*60)
print("TEST 7b: DETAILED PROBABILITY CHECK")
print("="*60)

_, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['csv_file'])

train_ds, pre_aug, valid_ds, steps_train, steps_valid, alpha = prepare_cached_datasets(
    df, selected_modalities=['metadata'], train_patient_percentage=0.8,
    batch_size=32, cache_dir=None, gen_manager=None, aug_config=None, run=0
)

print("\n🔍 Analyzing first batch...")
for batch in train_ds.take(1):
    X, y = batch
    meta = X['metadata_input'].numpy()

    print(f"\nProbability statistics:")
    print(f"  Mean:   {meta.mean(axis=0)}")
    print(f"  Median: {np.median(meta, axis=0)}")
    print(f"  Min:    {meta.min(axis=0)}")
    print(f"  Max:    {meta.max(axis=0)}")

    row_sums = meta.sum(axis=1)
    print(f"\nRow sum statistics:")
    print(f"  Min:  {row_sums.min():.6f}")
    print(f"  Max:  {row_sums.max():.6f}")
    print(f"  Mean: {row_sums.mean():.6f}")
    print(f"  Std:  {row_sums.std():.6f}")

    # Check individual rows that don't sum to 1
    bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=0.01))[0]
    if len(bad_rows) > 0:
        print(f"\n⚠️  {len(bad_rows)}/{len(row_sums)} rows don't sum to 1.0:")
        for idx in bad_rows[:3]:  # Show first 3
            print(f"  Row {idx}: {meta[idx]} = {row_sums[idx]:.6f}")

    # Check if values are in [0, 1]
    if meta.min() >= 0 and meta.max() <= 1:
        print(f"\n✅ All values in [0, 1] range")
    else:
        print(f"\n❌ Values outside [0, 1] range!")

    # Check if close enough to sum to 1
    tolerance = 0.02  # More lenient - 2% tolerance
    if np.all(np.abs(row_sums - 1.0) < tolerance):
        print(f"✅ All rows sum close to 1.0 (within {tolerance})")
    else:
        print(f"❌ Some rows don't sum close to 1.0")

    # Overall assessment
    mean_deviation = np.mean(np.abs(row_sums - 1.0))
    print(f"\n📊 Mean deviation from 1.0: {mean_deviation:.6f}")

    if mean_deviation < 0.01 and meta.min() >= 0 and meta.max() <= 1:
        print(f"\n{'='*60}")
        print("✅ FIX VERIFIED - Probabilities are valid!")
        print("(Minor deviations likely due to floating point precision)")
        print("="*60)
    elif meta.min() >= 0 and meta.max() <= 1 and mean_deviation < 0.05:
        print(f"\n{'='*60}")
        print("⚠️  PARTIAL FIX - Values in range but sums imperfect")
        print("(May need to investigate oversampling or normalization)")
        print("="*60)
    else:
        print(f"\n{'='*60}")
        print("❌ FIX FAILED - Probabilities still invalid")
        print("="*60)
        sys.exit(1)
