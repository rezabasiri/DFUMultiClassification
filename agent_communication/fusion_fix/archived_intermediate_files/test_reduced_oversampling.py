"""
Test: Reduced Oversampling Strategy

Goal: Reduce duplicate samples by accepting slight class imbalance
Current: Undersample P to 276, Oversample R to 276 → 158 duplicates (57% of R)
Proposed: Undersample P to 197, Oversample R to 197 → 79 duplicates (40% of R)

Expected: Less overfitting, better validation kappa
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from collections import Counter
import sys
sys.path.insert(0, '/workspace/DFUMultiClassification')

from src.data.dataset_utils import create_multimodal_dataset
from src.utils.production_config import INCLUDED_COMBINATIONS, DATA_PERCENTAGE

print("=" * 80)
print("Testing Reduced Oversampling Strategy @ 100% Data")
print("=" * 80)
print()

# Load full dataset
print("Loading 100% data...")
from src.data.caching import load_preprocessed_data
metadata_df, image_dict, label_encoder = load_preprocessed_data()

# Get original class distribution
print(f"Original data: {len(metadata_df)} samples")
original_dist = Counter(metadata_df['Healing Phase Abs'])
print(f"Class distribution: {original_dist}")
print()

# === CURRENT STRATEGY: combined (target MIDDLE class) ===
print("=" * 80)
print("CURRENT STRATEGY: 'combined' - Undersample P to MIDDLE (276)")
print("=" * 80)

# Find class counts
class_counts = {
    'I': original_dist['I'],  # 276
    'P': original_dist['P'],  # 496
    'R': original_dist['R']   # 118
}
sorted_counts = sorted(class_counts.values())
middle_count = sorted_counts[1]  # 276

print(f"Target count: {middle_count} (middle class)")
print()

# Perform sampling
df_I = metadata_df[metadata_df['Healing Phase Abs'] == 'I']
df_P = metadata_df[metadata_df['Healing Phase Abs'] == 'P']
df_R = metadata_df[metadata_df['Healing Phase Abs'] == 'R']

# Undersample P
df_P_sampled = df_P.sample(n=middle_count, random_state=42)
print(f"Undersampling P: {len(df_P)} → {len(df_P_sampled)}")

# Keep I
df_I_sampled = df_I
print(f"Keep I: {len(df_I)}")

# Oversample R (with duplicates)
oversample_factor = middle_count / len(df_R)
df_R_sampled = df_R.sample(n=middle_count, replace=True, random_state=42)
duplicates_R_current = middle_count - len(df_R)
print(f"Oversampling R: {len(df_R)} → {len(df_R_sampled)} ({oversample_factor:.2f}x)")
print(f"  Duplicates created: {duplicates_R_current} ({duplicates_R_current/middle_count*100:.1f}% of R class)")

# Combine
df_combined = pd.concat([df_I_sampled, df_P_sampled, df_R_sampled])
print(f"After 'combined' sampling: Counter({dict(Counter(df_combined['Healing Phase Abs']))})")
print(f"Final data: {len(df_combined)} samples")
total_duplicates_current = duplicates_R_current
print(f"Total duplicates: {total_duplicates_current} ({total_duplicates_current/len(df_combined)*100:.1f}% of dataset)")
print()

# === PROPOSED STRATEGY: reduced_combined (target between LOWER and MIDDLE) ===
print("=" * 80)
print("PROPOSED STRATEGY: 'reduced_combined' - Undersample P to ~197 (1.4x R)")
print("=" * 80)

# Target: 1.4x the R class (instead of 2.34x)
# This reduces oversampling while maintaining some balance
target_reduced = int(class_counts['R'] * 1.4)  # ~165
# Or alternatively, use sqrt of MIDDLE and R
# target_reduced = int(np.sqrt(middle_count * class_counts['R']))  # ~180

# For cleaner numbers, let's use 197 (exactly half of MIDDLE + R)
target_reduced = (middle_count + class_counts['R']) // 2  # (276 + 118) // 2 = 197

print(f"Target count: {target_reduced} (50% between R and MIDDLE)")
print()

# Perform sampling
# Undersample P to target
df_P_reduced = df_P.sample(n=target_reduced, random_state=42)
print(f"Undersampling P: {len(df_P)} → {len(df_P_reduced)}")

# Undersample I to target
df_I_reduced = df_I.sample(n=target_reduced, random_state=42)
print(f"Undersampling I: {len(df_I)} → {len(df_I_reduced)}")

# Oversample R to target
df_R_reduced = df_R.sample(n=target_reduced, replace=True, random_state=42)
duplicates_R_reduced = target_reduced - len(df_R)
print(f"Oversampling R: {len(df_R)} → {len(df_R_reduced)} ({target_reduced/len(df_R):.2f}x)")
print(f"  Duplicates created: {duplicates_R_reduced} ({duplicates_R_reduced/target_reduced*100:.1f}% of R class)")

# Combine
df_reduced = pd.concat([df_I_reduced, df_P_reduced, df_R_reduced])
print(f"After 'reduced_combined' sampling: Counter({dict(Counter(df_reduced['Healing Phase Abs']))})")
print(f"Final data: {len(df_reduced)} samples")
total_duplicates_reduced = duplicates_R_reduced
print(f"Total duplicates: {total_duplicates_reduced} ({total_duplicates_reduced/len(df_reduced)*100:.1f}% of dataset)")
print()

# === COMPARISON ===
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"{'Strategy':<20} {'Total Samples':<15} {'R Duplicates':<15} {'% Duplicates':<15}")
print("-" * 80)
print(f"{'combined':<20} {len(df_combined):<15} {duplicates_R_current:<15} {total_duplicates_current/len(df_combined)*100:>13.1f}%")
print(f"{'reduced_combined':<20} {len(df_reduced):<15} {duplicates_R_reduced:<15} {total_duplicates_reduced/len(df_reduced)*100:>13.1f}%")
print()
print(f"Reduction in duplicates: {duplicates_R_current - duplicates_R_reduced} samples ({(1 - duplicates_R_reduced/duplicates_R_current)*100:.1f}%)")
print()

# === QUICK RF TEST (Optional - just to show concept) ===
print("=" * 80)
print("Quick RF Overfitting Comparison (Training Kappa)")
print("=" * 80)

# Train RF on current strategy
X_current = df_combined.drop(['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs'], axis=1, errors='ignore')
y_current = df_combined['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})

rf_current = RandomForestClassifier(
    n_estimators=646,
    max_depth=14,
    min_samples_split=19,
    min_samples_leaf=2,
    max_features='log2',
    random_state=42,
    n_jobs=-1
)
rf_current.fit(X_current, y_current)
pred_current = rf_current.predict(X_current)
kappa_current = cohen_kappa_score(y_current, pred_current)
print(f"combined:         Training Kappa = {kappa_current:.4f}")

# Train RF on reduced strategy
X_reduced = df_reduced.drop(['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs'], axis=1, errors='ignore')
y_reduced = df_reduced['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})

rf_reduced = RandomForestClassifier(
    n_estimators=646,
    max_depth=14,
    min_samples_split=19,
    min_samples_leaf=2,
    max_features='log2',
    random_state=42,
    n_jobs=-1
)
rf_reduced.fit(X_reduced, y_reduced)
pred_reduced = rf_reduced.predict(X_reduced)
kappa_reduced = cohen_kappa_score(y_reduced, pred_reduced)
print(f"reduced_combined: Training Kappa = {kappa_reduced:.4f}")
print()

print("=" * 80)
print("Conclusions")
print("=" * 80)
print("✅ reduced_combined uses 50% fewer duplicates")
print("✅ Training kappa comparison shows relative overfitting")
print("✅ To test validation performance, need to implement in dataset_utils.py")
print()
print("Expected Results with reduced_combined:")
print("  - Less RF overfitting (training kappa closer to validation)")
print("  - Better generalization to validation set")
print("  - metadata-only: 0.18-0.19 (vs 0.16 with combined)")
print("  - fusion: 0.19-0.20 (vs 0.166 with combined)")
print("=" * 80)
