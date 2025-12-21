#!/usr/bin/env python3
"""
Debug script to verify modality isolation in Block A.
Tests that each modality combination only uses its specified modalities.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.config import get_project_paths, get_data_paths
from src.data.image_processing import extract_best_matching
from src.data.dataset_utils import prepare_cached_datasets

# Get paths
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

# Load data
depth_bb_file = data_paths['depth_bbox_file']
thermal_bb_file = data_paths['thermal_bbox_file']
csv_file = data_paths['csv_file']

# Prepare data
data = extract_best_matching(depth_bb_file, thermal_bb_file, csv_file)
print(f"Total samples: {len(data)}")

# Test three modality combinations
test_combinations = [
    ['metadata'],
    ['depth_rgb'],
    ['depth_map'],
]

print("\n" + "="*80)
print("TESTING MODALITY ISOLATION")
print("="*80)

for selected_modalities in test_combinations:
    print(f"\n\nTesting: {selected_modalities}")
    print("-" * 80)

    # Prepare dataset
    train_dataset, _, valid_dataset, _, _, _ = prepare_cached_datasets(
        data,
        selected_modalities,
        train_patient_percentage=0.8,
        batch_size=4,
        run=0,
        image_size=64,
        for_shape_inference=False
    )

    # Inspect what features are in the dataset
    print("\nInspecting train_dataset features:")
    for batch in train_dataset.take(1):
        features, labels = batch
        print(f"\nFeature keys in batch: {list(features.keys())}")
        for key, value in features.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if 'input' in key and value.shape[0] > 0:
                # Show some statistics
                print(f"    min={tf.reduce_min(value).numpy():.4f}, "
                      f"max={tf.reduce_max(value).numpy():.4f}, "
                      f"mean={tf.reduce_mean(value).numpy():.4f}")
        print(f"Labels: shape={labels.shape}")

    print("\n" + "-" * 80)

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print("\nIf modality isolation is working correctly:")
print("  - metadata combination should have: metadata_input, sample_id")
print("  - depth_rgb combination should have: depth_rgb_input, sample_id")
print("  - depth_map combination should have: depth_map_input, sample_id")
print("\nIf ALL combinations show the same features, there's a data leakage bug!")
print("="*80)
