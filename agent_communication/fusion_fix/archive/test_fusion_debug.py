#!/usr/bin/env python3
"""Quick fusion diagnostics - run each test independently"""
import sys
sys.path.insert(0, '/workspace/DFUMultiClassification')

import numpy as np
import tensorflow as tf
from src.data.dataset_utils import prepare_cached_datasets
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths
import pandas as pd

# Get project paths
directory, result_dir, root = get_project_paths()
data_paths = {
    'bb_depth_csv': directory + 'dfuc2024_bb_depth.csv',
    'bb_thermal_csv': directory + 'dfuc2024_bb_thermal.csv',
    'meta_csv': directory + 'dfuc2024_meta.csv'
}

# Load data using same method as main.py
depth_bb_file = data_paths['bb_depth_csv']
thermal_bb_file = data_paths['bb_thermal_csv']
csv_file = data_paths['meta_csv']

print("\n" + "="*80)
print("TEST 1: Check RF predictions for thermal_map-only")
print("="*80)
data_thermal = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, ['thermal_map'])
data_thermal = data_thermal.sample(frac=0.3, random_state=42)  # 30% for speed
train_ds, valid_ds, _, _, _, _ = prepare_cached_datasets(
    data_thermal, ['thermal_map'], batch_size=32, run=0, image_size=32
)
for batch in train_ds.take(1):
    inputs, labels = batch
    print(f"thermal_map-only inputs: {inputs.keys()}")
    print(f"Labels shape: {labels.shape}, sample: {labels.numpy()[:5]}")
    if 'metadata_input' in inputs:
        print("ERROR: thermal_map-only has metadata_input (shouldn't exist)")
    else:
        print("✓ No metadata_input (correct)")

print("\n" + "="*80)
print("TEST 2: Check RF predictions for fusion (metadata+thermal_map)")
print("="*80)
data_fusion = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, ['metadata', 'thermal_map'])
data_fusion = data_fusion.sample(frac=0.3, random_state=42)  # 30% for speed
train_ds, valid_ds, _, _, _, _ = prepare_cached_datasets(
    data_fusion, ['metadata', 'thermal_map'], batch_size=32, run=0, image_size=32
)
for batch in train_ds.take(1):
    inputs, labels = batch
    print(f"Fusion inputs: {inputs.keys()}")
    print(f"Labels shape: {labels.shape}, sample: {labels.numpy()[:5]}")

    if 'metadata_input' in inputs:
        rf_preds = inputs['metadata_input'].numpy()
        print(f"\n✓ RF predictions exist")
        print(f"  Shape: {rf_preds.shape}")
        print(f"  First 3 samples:\n{rf_preds[:3]}")
        print(f"  Sum to 1.0? {[np.sum(p) for p in rf_preds[:5]]}")
        print(f"  All zeros? {np.all(rf_preds == 0)}")
        print(f"  Any NaN/Inf? {np.any(np.isnan(rf_preds)) or np.any(np.isinf(rf_preds))}")

        # Check alignment with labels
        print(f"\n  Checking label alignment:")
        for i in range(min(5, len(labels))):
            pred_class = np.argmax(rf_preds[i])
            true_class = labels.numpy()[i]
            print(f"    Sample {i}: RF predicts {pred_class}, true label {true_class}, probs {rf_preds[i]}")
    else:
        print("ERROR: Fusion missing metadata_input!")

print("\n" + "="*80)
print("TEST 3: Train thermal_map-only baseline")
print("="*80)
print("Edit production_config.py: INCLUDED_COMBINATIONS = [('thermal_map',)]")
print("Run: python src/main.py --mode search --cv_folds 1 --verbosity 2 --resume_mode fresh")

print("\n" + "="*80)
print("TEST 4: Train fusion and compare")
print("="*80)
print("Edit production_config.py: INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]")
print("Run: python src/main.py --mode search --cv_folds 1 --verbosity 2 --resume_mode fresh")

print("\n" + "="*80)
print("TEST 5: Manual fusion check")
print("="*80)
print("If RF and Image predictions look good individually, manually compute:")
print("  fusion_pred = 0.7 * rf_pred + 0.3 * image_pred")
print("Check if manual fusion gives reasonable Kappa")
