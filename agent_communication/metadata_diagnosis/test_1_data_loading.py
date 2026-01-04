"""Test 1: Data Loading and Basic Statistics"""
import sys
sys.path.insert(0, '/home/user/DFUMultiClassification')

import pandas as pd
import numpy as np
from src.utils.config import get_data_paths, get_project_paths

print("="*60)
print("TEST 1: DATA LOADING")
print("="*60)

# Load data
_, _, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['metadata'])

print(f"\n✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")

# Check label distribution
label_col = 'Healing Phase Abs'
if label_col not in df.columns:
    print(f"⚠️  FAIL: '{label_col}' column not found!")
    sys.exit(1)

label_dist = df[label_col].value_counts().sort_index()
print(f"\n✓ Label distribution:")
for label, count in label_dist.items():
    pct = count/len(df)*100
    print(f"  Class {label}: {count} ({pct:.1f}%)")

# Check for required columns
required = ['Patient#', 'Appt#', 'DFU#', 'Age', 'Weight (Kg)', 'Height (cm)']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"⚠️  FAIL: Missing columns: {missing}")
    sys.exit(1)

print(f"\n✓ All required columns present")

# Check for NaN in critical columns
nan_counts = df[required + [label_col]].isna().sum()
if nan_counts.any():
    print(f"\n⚠️  WARNING: NaN values found:")
    print(nan_counts[nan_counts > 0])
else:
    print(f"\n✓ No NaN in critical columns")

# Check data types
print(f"\n✓ Key column types:")
print(f"  {label_col}: {df[label_col].dtype}")
print(f"  Age: {df['Age'].dtype}")
print(f"  Weight (Kg): {df['Weight (Kg)'].dtype}")

# Check label values
unique_labels = sorted(df[label_col].unique())
print(f"\n✓ Unique labels: {unique_labels}")
if not all(x in [0, 1, 2] for x in unique_labels):
    print(f"⚠️  FAIL: Labels not in [0, 1, 2]!")
    sys.exit(1)

print(f"\n{'='*60}")
print("TEST 1: PASSED ✓")
print("="*60)
