"""Test 6: Production Code Path (dataset_utils.py)"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from src.utils.config import get_data_paths, get_project_paths
from src.data.dataset_utils import prepare_cached_datasets

print("="*60)
print("TEST 6: PRODUCTION CODE (dataset_utils.py)")
print("="*60)

# Load data
_, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['csv_file'])

print(f"✓ Data loaded: {len(df)} samples")

# Test with production settings
print(f"\n⏳ Running prepare_cached_datasets (dataset_utils.py)...")
print(f"  Modalities: ['metadata']")
print(f"  Run: 0")

try:
    train_ds, pre_aug, valid_ds, steps_train, steps_valid, alpha = prepare_cached_datasets(
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

    # Check batch structure
    print(f"\n⏳ Checking batch structure...")
    for batch in train_ds.take(1):
        X, y = batch
        if isinstance(X, dict):
            print(f"✓ Batch X is dict with keys: {list(X.keys())}")
            for k, v in X.items():
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                if k == 'metadata_input':
                    if v.shape[1] != 3:
                        print(f"  ⚠️  WRONG SHAPE! Expected (32, 3), got {v.shape}")
                        sys.exit(1)
        else:
            print(f"✓ Batch X shape: {X.shape}")

        print(f"✓ Batch y shape: {y.shape}")

        # Check labels
        y_np = y.numpy()
        if len(y_np.shape) == 2:
            y_classes = np.argmax(y_np, axis=1)
        else:
            y_classes = y_np.astype(int)

        print(f"  Label distribution: {np.bincount(y_classes)}")

        # Check metadata values
        if isinstance(X, dict) and 'metadata_input' in X:
            meta = X['metadata_input'].numpy()
            print(f"\n✓ Metadata probability stats:")
            print(f"  Mean: {meta.mean(axis=0)}")
            print(f"  Std: {meta.std(axis=0)}")
            print(f"  Min: {meta.min(axis=0)}")
            print(f"  Max: {meta.max(axis=0)}")

            # Check if probabilities sum to 1
            row_sums = meta.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=0.01):
                print(f"  ⚠️  WARNING: Probabilities don't sum to 1!")
                print(f"  Row sums range: {row_sums.min():.4f} to {row_sums.max():.4f}")

    print(f"\n{'='*60}")
    print("TEST 6: PASSED ✓")
    print("="*60)

except Exception as e:
    print(f"\n⚠️  FAIL: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
