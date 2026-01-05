"""
Phase 7: Explicit Outlier Detection and Removal

Strategy:
- Use metadata-only (fast, targets RF quality bottleneck)
- Per-class Isolation Forest (handles class imbalance)
- Test multiple contamination rates (5%, 10%, 15%)
- Save cleaned datasets for training

Expected: If implicit outlier removal hypothesis is correct,
          cleaned 100% data should match 50% data performance (Kappa ~0.27)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import Counter
import os
import sys

# Add project root to path
sys.path.insert(0, '/workspace/DFUMultiClassification')

from src.data.caching import load_preprocessed_data
from src.utils.verbosity import vprint

print("=" * 80)
print("Phase 7: Explicit Outlier Detection using Isolation Forest")
print("=" * 80)
print()

# Load full dataset
print("Loading 100% preprocessed data...")
metadata_df, image_dict, label_encoder = load_preprocessed_data()
print(f"Loaded {len(metadata_df)} samples")
print()

# Get class distribution
class_dist = Counter(metadata_df['Healing Phase Abs'])
print("Original class distribution:")
for cls in ['I', 'P', 'R']:
    count = class_dist[cls]
    pct = count / len(metadata_df) * 100
    print(f"  {cls}: {count:3d} ({pct:5.1f}%)")
print()

# Prepare features for outlier detection
print("Preparing features for outlier detection...")
feature_cols = [col for col in metadata_df.columns
                if col not in ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']]
X = metadata_df[feature_cols].values
y = metadata_df['Healing Phase Abs'].values
patient_ids = metadata_df['Patient#'].values
print(f"Using {len(feature_cols)} features")
print()

# Test multiple contamination rates
contamination_rates = [0.05, 0.10, 0.15]

results_summary = []

for contamination in contamination_rates:
    print("=" * 80)
    print(f"Testing contamination rate: {contamination*100:.0f}%")
    print("=" * 80)
    print()

    # Per-class outlier detection
    outlier_mask = np.zeros(len(metadata_df), dtype=bool)
    outlier_counts = {}

    for cls in ['I', 'P', 'R']:
        print(f"Class {cls}:")

        # Get samples for this class
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        X_cls = X[cls_mask]

        if len(X_cls) < 10:
            print(f"  Skipping (too few samples: {len(X_cls)})")
            outlier_counts[cls] = 0
            continue

        # SAFETY: Reduce contamination for minority class R to preserve samples
        cls_contamination = contamination
        if cls == 'R' and len(X_cls) < 150:  # R class is minority
            # Cap at 10% max for R class to preserve samples
            cls_contamination = min(contamination, 0.10)
            if cls_contamination < contamination:
                print(f"  ⚠️  Reducing contamination for minority class: {contamination*100:.0f}% → {cls_contamination*100:.0f}%")

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=cls_contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )

        predictions = iso_forest.fit_predict(X_cls)

        # -1 = outlier, 1 = inlier
        outliers_cls = (predictions == -1)
        n_outliers = outliers_cls.sum()
        outlier_counts[cls] = n_outliers

        # SAFETY CHECK: Never remove more than 20% of minority class
        max_allowed = int(len(X_cls) * 0.20)
        if cls == 'R' and n_outliers > max_allowed:
            print(f"  ⚠️  Safety limit: Would remove {n_outliers}, limiting to {max_allowed} (20% max)")
            # Keep only the top outliers by score
            scores = iso_forest.decision_function(X_cls)
            outlier_indices = np.argsort(scores)[:max_allowed]  # Most outlier-like
            outliers_cls = np.zeros(len(X_cls), dtype=bool)
            outliers_cls[outlier_indices] = True
            n_outliers = max_allowed
            outlier_counts[cls] = n_outliers

        # Mark outliers in global mask
        outlier_mask[cls_indices[outliers_cls]] = True

        print(f"  Original: {len(X_cls)} samples")
        print(f"  Outliers detected: {n_outliers} ({n_outliers/len(X_cls)*100:.1f}%)")
        print(f"  Remaining: {len(X_cls) - n_outliers}")
        print()

    # Create cleaned dataset
    cleaned_df = metadata_df[~outlier_mask].copy()

    total_outliers = outlier_mask.sum()
    print(f"Total outliers detected: {total_outliers} ({total_outliers/len(metadata_df)*100:.1f}%)")
    print(f"Cleaned dataset size: {len(cleaned_df)} samples")
    print()

    # New class distribution
    cleaned_dist = Counter(cleaned_df['Healing Phase Abs'])
    print("Cleaned class distribution:")
    for cls in ['I', 'P', 'R']:
        orig_count = class_dist[cls]
        new_count = cleaned_dist[cls]
        removed = orig_count - new_count
        print(f"  {cls}: {new_count:3d} (removed {removed}, {removed/orig_count*100:.1f}%)")
    print()

    # Save cleaned dataset
    output_file = f'/workspace/DFUMultiClassification/data/cleaned/metadata_cleaned_{int(contamination*100):02d}pct.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    print()

    # Also save outlier list for analysis
    outlier_patients = metadata_df[outlier_mask][['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].copy()
    outlier_file = f'/workspace/DFUMultiClassification/data/cleaned/outliers_{int(contamination*100):02d}pct.csv'
    outlier_patients.to_csv(outlier_file, index=False)
    print(f"Outlier list saved to: {outlier_file}")
    print()

    results_summary.append({
        'contamination': contamination,
        'total_outliers': total_outliers,
        'outliers_I': outlier_counts.get('I', 0),
        'outliers_P': outlier_counts.get('P', 0),
        'outliers_R': outlier_counts.get('R', 0),
        'remaining': len(cleaned_df),
        'output_file': output_file
    })

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Cleaned datasets created:")
for result in results_summary:
    print(f"  {result['contamination']*100:.0f}% contamination: {result['remaining']} samples")
    print(f"    Outliers removed: I={result['outliers_I']}, P={result['outliers_P']}, R={result['outliers_R']}")
    print(f"    File: {result['output_file']}")
    print()

print("=" * 80)
print("Next Steps:")
print("=" * 80)
print("1. Train fusion model on each cleaned dataset")
print("2. Compare performance with 50% data (Kappa 0.279)")
print("3. If cleaned 100% ≈ 50% → Implicit outlier removal hypothesis CONFIRMED")
print("4. Use best contamination rate for production")
print()
print("Expected results:")
print("  5% contamination:  Moderate improvement (Kappa 0.19-0.21)")
print("  10% contamination: Good improvement (Kappa 0.22-0.24)")
print("  15% contamination: Best improvement (Kappa 0.25-0.27, matching 50%)")
print("=" * 80)
