#!/usr/bin/env python3
"""Verify that feature normalization is applied during CV test"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths

# Load data
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

data = prepare_dataset(
    depth_bb_file=data_paths['bb_depth_csv'],
    thermal_bb_file=data_paths['bb_thermal_csv'],
    csv_file=data_paths['csv_file'],
    selected_modalities=['metadata']
)

print(f"Data loaded: {len(data)} samples")

# Check numeric column ranges (before prepare_cached_datasets normalization)
numeric_cols = data.select_dtypes(include=[np.number]).columns
exclude = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
check_cols = [col for col in numeric_cols if col not in exclude][:5]  # Check first 5

print(f"\nMetadata columns (sample): {check_cols}")

print(f"\nSample numeric column ranges (BEFORE prepare_cached_datasets):")
for col in check_cols:
    print(f"  {col}: [{data[col].min():.2f}, {data[col].max():.2f}], mean={data[col].mean():.2f}")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)
print("If ranges are large (e.g., [0, 100]), normalization happens later in prepare_cached_datasets")
print("If ranges are ~[-3, 3] with mean~0, normalization already applied (unexpected)")
