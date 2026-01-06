"""
Phase 7b: Test Fusion on Cleaned Datasets

Tests the 3 cleaned datasets (5%, 10%, 15% outlier removal)
and compares with 50% data baseline (Kappa 0.279)
"""

import pandas as pd
import numpy as np
import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, '/workspace/DFUMultiClassification')

# Paths - use absolute paths
best_matching_file = '/workspace/DFUMultiClassification/results/best_matching.csv'
best_matching_backup = '/workspace/DFUMultiClassification/results/best_matching_original.csv'

# Cleaned dataset files
cleaned_files = {
    '05pct': '/workspace/DFUMultiClassification/data/cleaned/metadata_cleaned_05pct.csv',
    '10pct': '/workspace/DFUMultiClassification/data/cleaned/metadata_cleaned_10pct.csv',
    '15pct': '/workspace/DFUMultiClassification/data/cleaned/metadata_cleaned_15pct.csv'
}

def get_test_name():
    """Get which test to run from command line arg"""
    if len(sys.argv) < 2:
        print("Usage: python test_cleaned_data.py <test_name>")
        print("  test_name: 05pct, 10pct, 15pct, or restore")
        sys.exit(1)
    return sys.argv[1]

def backup_original():
    """Backup original best_matching.csv"""
    if not os.path.exists(best_matching_backup):
        print(f"Backing up original: {best_matching_file} -> {best_matching_backup}")
        shutil.copy(best_matching_file, best_matching_backup)
    else:
        print(f"Backup already exists: {best_matching_backup}")

def restore_original():
    """Restore original best_matching.csv"""
    if os.path.exists(best_matching_backup):
        print(f"Restoring original: {best_matching_backup} -> {best_matching_file}")
        shutil.copy(best_matching_backup, best_matching_file)
        print("Original restored!")
    else:
        print("No backup found!")

def filter_best_matching(cleaned_file):
    """Filter best_matching.csv to only include samples from cleaned dataset"""

    # Load cleaned metadata (contains Patient#, Appt#, DFU#)
    cleaned_df = pd.read_csv(cleaned_file)
    print(f"Cleaned dataset: {len(cleaned_df)} samples")

    # Load original best_matching
    if os.path.exists(best_matching_backup):
        original_df = pd.read_csv(best_matching_backup)
    else:
        original_df = pd.read_csv(best_matching_file)
    print(f"Original best_matching: {len(original_df)} samples")

    # Create key for matching
    cleaned_df['_key'] = cleaned_df['Patient#'].astype(str) + '_' + cleaned_df['Appt#'].astype(str) + '_' + cleaned_df['DFU#'].astype(str)
    original_df['_key'] = original_df['Patient#'].astype(str) + '_' + original_df['Appt#'].astype(str) + '_' + original_df['DFU#'].astype(str)

    # Filter original to only include cleaned samples
    filtered_df = original_df[original_df['_key'].isin(cleaned_df['_key'])].copy()
    filtered_df = filtered_df.drop('_key', axis=1)

    print(f"Filtered best_matching: {len(filtered_df)} samples")

    # Verify class distribution
    from collections import Counter
    dist = Counter(filtered_df['Healing Phase Abs'])
    print(f"Class distribution: I={dist['I']}, P={dist['P']}, R={dist['R']}")

    return filtered_df

def main():
    test_name = get_test_name()

    if test_name == 'restore':
        restore_original()
        return

    if test_name not in cleaned_files:
        print(f"Unknown test: {test_name}")
        print(f"Valid options: {list(cleaned_files.keys())} or 'restore'")
        sys.exit(1)

    print("=" * 80)
    print(f"Phase 7b: Testing {test_name} outlier removal")
    print("=" * 80)
    print()

    # Backup original first
    backup_original()

    # Filter best_matching
    cleaned_file = cleaned_files[test_name]
    filtered_df = filter_best_matching(cleaned_file)

    # Save filtered as best_matching.csv
    filtered_df.to_csv(best_matching_file, index=False)
    print(f"Saved filtered dataset to: {best_matching_file}")
    print()

    print("=" * 80)
    print("Ready to run training!")
    print("=" * 80)
    print()
    print("Now run:")
    print("  python src/main.py 2>&1 | tee agent_communication/fusion_fix/run_cleaned_{}.txt".format(test_name))
    print()
    print("After training, restore original with:")
    print("  python agent_communication/fusion_fix/test_cleaned_data.py restore")

if __name__ == '__main__':
    main()
