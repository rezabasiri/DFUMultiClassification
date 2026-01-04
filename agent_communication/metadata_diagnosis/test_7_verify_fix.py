"""Test 7: Verify RF Probability Fix"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from src.utils.config import get_data_paths, get_project_paths
from src.data.dataset_utils import prepare_cached_datasets
import pandas as pd

print("="*60)
print("TEST 7: VERIFY RF PROBABILITY FIX")
print("="*60)

_, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['csv_file'])

train_ds, pre_aug, valid_ds, steps_train, steps_valid, alpha = prepare_cached_datasets(
    df, selected_modalities=['metadata'], train_patient_percentage=0.8,
    batch_size=32, cache_dir=None, gen_manager=None, aug_config=None, run=0
)

for batch in train_ds.take(1):
    X, y = batch
    meta = X['metadata_input'].numpy()

    print(f"\n✓ Probability statistics:")
    print(f"  Mean: {meta.mean(axis=0)}")
    print(f"  Min:  {meta.min(axis=0)}")
    print(f"  Max:  {meta.max(axis=0)}")

    row_sums = meta.sum(axis=1)
    print(f"\n✓ Row sums: {row_sums.min():.4f} to {row_sums.max():.4f}")

    # Verification checks
    checks_passed = True

    if not np.allclose(row_sums, 1.0, atol=0.01):
        print(f"❌ FAIL: Probabilities don't sum to 1!")
        checks_passed = False
    else:
        print(f"✅ PASS: Probabilities sum to 1.0")

    if meta.min() < 0 or meta.max() > 1:
        print(f"❌ FAIL: Probabilities outside [0,1] range!")
        checks_passed = False
    else:
        print(f"✅ PASS: Probabilities in valid [0,1] range")

    mean_check = np.all(np.abs(meta.mean(axis=0) - 0.33) < 0.2)
    if not mean_check:
        print(f"⚠️  WARNING: Mean distribution skewed")
    else:
        print(f"✅ PASS: Mean distribution reasonable")

    if checks_passed:
        print(f"\n{'='*60}")
        print("TEST 7: PASSED ✓ - FIX VERIFIED!")
        print("="*60)
    else:
        print(f"\n{'='*60}")
        print("TEST 7: FAILED ✗ - FIX DID NOT WORK")
        print("="*60)
        sys.exit(1)
