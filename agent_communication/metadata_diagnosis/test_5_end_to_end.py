"""Test 5: End-to-End using actual caching.py code"""
import sys
sys.path.insert(0, '/home/user/DFUMultiClassification')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from src.utils.config import get_data_paths, get_project_paths

print("="*60)
print("TEST 5: END-TO-END (ACTUAL CACHING.PY)")
print("="*60)

# Check which prepare_cached_datasets is imported
try:
    from src.data.caching import prepare_cached_datasets as prep_cache
    source = "caching.py"
except ImportError:
    from src.data.dataset_utils import prepare_cached_datasets as prep_cache
    source = "dataset_utils.py"

print(f"\n✓ Using prepare_cached_datasets from: {source}")

# Load data
_, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['metadata'])

print(f"✓ Data loaded: {len(df)} samples")

# Quick test with minimal settings
print(f"\n⏳ Running prepare_cached_datasets...")
print(f"  Modalities: ['metadata']")
print(f"  Run: 0")
print(f"  Batch size: 32")

try:
    train_ds, pre_aug, valid_ds, steps_train, steps_valid, alpha = prep_cache(
        df,
        selected_modalities=['metadata'],
        train_patient_percentage=0.8,
        batch_size=32,
        cache_dir=None,
        gen_manager=None,
        aug_config=None,
        run=0
    )
    print(f"\n✓ Dataset creation successful")
    print(f"  Train steps: {steps_train}")
    print(f"  Valid steps: {steps_valid}")

    # Try to get one batch
    print(f"\n⏳ Getting one training batch...")
    for batch in train_ds.take(1):
        X, y = batch
        if isinstance(X, dict):
            print(f"✓ Batch X is dict with keys: {list(X.keys())}")
            for k, v in X.items():
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"✓ Batch X shape: {X.shape}, dtype={X.dtype}")
        print(f"✓ Batch y shape: {y.shape}, dtype={y.dtype}")
        print(f"  Label distribution in batch: {np.bincount(y.numpy().astype(int))}")

    print(f"\n✓ Successfully retrieved batch")

except Exception as e:
    print(f"\n⚠️  FAIL: Error in prepare_cached_datasets!")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*60}")
print("TEST 5: PASSED ✓")
print("="*60)
print(f"\nNOTE: This test only checks dataset creation.")
print(f"If this passes but actual training fails, issue is in:")
print(f"  1. Model architecture")
print(f"  2. Training loop")
print(f"  3. Multi-GPU strategy")
