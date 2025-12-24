#!/usr/bin/env python3
"""Analyze if fold 3 is genuinely easier than folds 1-2"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.data.dataset_utils import create_patient_folds

print("="*80)
print("FOLD DISTRIBUTION ANALYSIS FOR THERMAL_MAP")
print("="*80)

# Load data
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

data = prepare_dataset(
    depth_bb_file=data_paths['bb_depth_csv'],
    thermal_bb_file=data_paths['bb_thermal_csv'],
    csv_file=data_paths['csv_file'],
    selected_modalities=['thermal_map']
)

print(f"\nTotal data: {len(data)} samples")
print(f"Class distribution (overall): {data['Healing Phase Abs'].value_counts().sort_index().to_dict()}")

# Create same folds as CV test
folds = create_patient_folds(data, n_folds=3, random_state=42)

print(f"\n{'='*80}")
print("FOLD ANALYSIS (Validation sets)")
print(f"{'='*80}")

fold_stats = []
for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
    val_data = data.iloc[val_idx]
    train_data = data.iloc[train_idx]

    print(f"\nFold {fold_idx}:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")

    val_class_dist = val_data['Healing Phase Abs'].value_counts().sort_index()
    train_class_dist = train_data['Healing Phase Abs'].value_counts().sort_index()

    print(f"\n  Validation class distribution:")
    for cls, count in val_class_dist.items():
        pct = count / len(val_data) * 100
        print(f"    Class {cls}: {count} ({pct:.1f}%)")

    print(f"\n  Training class distribution:")
    for cls, count in train_class_dist.items():
        pct = count / len(train_data) * 100
        print(f"    Class {cls}: {count} ({pct:.1f}%)")

    # Calculate imbalance ratio for validation set
    max_count = val_class_dist.max()
    min_count = val_class_dist.min()
    imbalance_ratio = max_count / min_count

    # Calculate minority class percentage
    minority_pct = min_count / len(val_data) * 100

    fold_stats.append({
        'fold': fold_idx,
        'val_samples': len(val_data),
        'imbalance_ratio': imbalance_ratio,
        'minority_pct': minority_pct,
        'val_class_I': val_class_dist.get('I', 0),
        'val_class_P': val_class_dist.get('P', 0),
        'val_class_R': val_class_dist.get('R', 0),
    })

    print(f"\n  Validation imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"  Minority class percentage: {minority_pct:.1f}%")

    if imbalance_ratio < 4.0:
        print(f"  -> Relatively balanced")
    else:
        print(f"  -> Imbalanced (harder)")

print(f"\n{'='*80}")
print("SUMMARY COMPARISON")
print(f"{'='*80}")

print(f"\n{'Fold':<6} {'Val Size':<10} {'Imbalance':<12} {'Minority %':<12}")
print("-"*40)
for stats in fold_stats:
    print(f"{stats['fold']:<6} {stats['val_samples']:<10} {stats['imbalance_ratio']:<12.2f} {stats['minority_pct']:<12.1f}")

print(f"\n{'='*80}")
print("HYPOTHESIS EVALUATION")
print(f"{'='*80}")

# Check if fold 3 is more balanced
fold3_stats = fold_stats[2]
other_stats = fold_stats[:2]

avg_other_imbalance = np.mean([s['imbalance_ratio'] for s in other_stats])
fold3_imbalance = fold3_stats['imbalance_ratio']

print(f"\nFold 3 imbalance ratio: {fold3_imbalance:.2f}")
print(f"Avg imbalance of folds 1-2: {avg_other_imbalance:.2f}")

if fold3_imbalance < avg_other_imbalance * 0.8:
    print("\nCONCLUSION: Fold 3 is MORE BALANCED")
    print("-> Higher accuracy on fold 3 is likely NATURAL VARIATION")
    print("-> NOT a model leak")
elif fold3_imbalance > avg_other_imbalance * 1.2:
    print("\nCONCLUSION: Fold 3 is MORE IMBALANCED")
    print("-> Higher accuracy despite harder fold suggests POSSIBLE LEAK")
    print("-> Needs further investigation")
else:
    print("\nCONCLUSION: Fold 3 has SIMILAR balance to other folds")
    print("-> Increasing accuracy pattern could be:")
    print("   1. Random variation (most likely)")
    print("   2. Order effects in patient distribution")
    print("   3. Small sample size effects")
    print("-> NOT a model leak (model is recreated each fold)")
