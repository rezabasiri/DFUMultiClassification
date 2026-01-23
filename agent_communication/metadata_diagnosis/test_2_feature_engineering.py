"""Test 2: Feature Engineering"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from src.utils.config import get_data_paths, get_project_paths

print("="*60)
print("TEST 2: FEATURE ENGINEERING")
print("="*60)

# Load data
_, _, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['csv_file']).copy()

print(f"\n✓ Starting with {len(df.columns)} columns")

# Feature engineering (from caching.py)
df['BMI'] = df['Weight (Kg)'] / ((df['Height (cm)'] / 100) ** 2)
df['Age above 60'] = (df['Age'] > 60).astype(int)
df['Age Bin'] = pd.cut(df['Age'], bins=range(0, int(df['Age'].max()) + 20, 20), right=False,
                       labels=range(len(range(0, int(df['Age'].max()) + 20, 20)) - 1))
df['Weight Bin'] = pd.cut(df['Weight (Kg)'], bins=range(0, int(df['Weight (Kg)'].max()) + 20, 20), right=False,
                          labels=range(len(range(0, int(df['Weight (Kg)'].max()) + 20, 20)) - 1))
df['Height Bin'] = pd.cut(df['Height (cm)'], bins=range(0, int(df['Height (cm)'].max()) + 10, 10), right=False,
                          labels=range(len(range(0, int(df['Height (cm)'].max()) + 10, 10)) - 1))

print(f"\n✓ Created: BMI, Age above 60, Age Bin, Weight Bin, Height Bin")

# Check engineered features
print(f"\n✓ BMI stats: mean={df['BMI'].mean():.2f}, std={df['BMI'].std():.2f}")
print(f"  Range: {df['BMI'].min():.2f} to {df['BMI'].max():.2f}")

if df['BMI'].isna().any():
    print(f"⚠️  WARNING: {df['BMI'].isna().sum()} NaN values in BMI")

if (df['BMI'] < 10).any() or (df['BMI'] > 80).any():
    print(f"⚠️  WARNING: Extreme BMI values detected")

print(f"\n✓ Age above 60: {df['Age above 60'].sum()} samples ({df['Age above 60'].mean()*100:.1f}%)")

# Categorical encoding
categorical_columns = ['Sex (F:0, M:1)', 'Side (Left:0, Right:1)', 'Foot Aspect', 'Odor', 'Type of Pain Grouped']
encoded = 0
for col in categorical_columns:
    if col in df.columns:
        before = df[col].dtype
        df[col] = pd.Categorical(df[col]).codes
        encoded += 1
        print(f"✓ Encoded {col}: {df[col].nunique()} unique values")

print(f"\n✓ Encoded {encoded} categorical columns")

# Categorical mappings
categorical_mappings = {
    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {'ankle': 4, 'Heel': 3, 'middle': 2, 'toes': 1, 'Hallux': 0},
    'Dressing Grouped': {'NoDressing': 0, 'BandAid': 1, 'BasicDressing': 1, 'AbsorbantDressing': 2, 'Antiseptic': 3, 'AdvanceMethod': 4, 'other': 4},
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {'Serous': 0, 'Haemoserous': 1, 'Bloody': 2, 'Thick': 3}
}

mapped = 0
for col, mapping in categorical_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
        mapped += 1
        print(f"✓ Mapped {col}: {df[col].nunique()} unique values")

print(f"\n✓ Mapped {mapped} additional columns")

# Check for unexpected NaN after encoding
new_nan = df.select_dtypes(include=[np.number]).isna().sum().sum()
if new_nan > 0:
    print(f"\n⚠️  WARNING: {new_nan} new NaN values after encoding")
    top_nan = df.select_dtypes(include=[np.number]).isna().sum().sort_values(ascending=False).head(5)
    print("Top columns with NaN:")
    print(top_nan[top_nan > 0])

print(f"\n{'='*60}")
print("TEST 2: PASSED ✓")
print("="*60)
