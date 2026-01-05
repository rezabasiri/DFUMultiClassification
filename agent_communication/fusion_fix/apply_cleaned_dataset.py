"""
Phase 7: Helper script to apply cleaned dataset

This script:
1. Backs up the original preprocessed cache
2. Replaces metadata with cleaned version
3. Allows normal training to run
4. Provides restore function

Usage:
  # Apply cleaned dataset
  python apply_cleaned_dataset.py --apply --contamination 10

  # Then run normal training:
  python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi

  # Restore original
  python apply_cleaned_dataset.py --restore
"""

import pandas as pd
import pickle
import shutil
import os
import argparse

CACHE_DIR = '/workspace/DFUMultiClassification/data/cache'
METADATA_CACHE = f'{CACHE_DIR}/preprocessed_metadata.pkl'
BACKUP_FILE = f'{CACHE_DIR}/preprocessed_metadata_BACKUP.pkl'

def apply_cleaned(contamination_pct):
    """Apply cleaned dataset by replacing cache"""

    cleaned_file = f'/workspace/DFUMultiClassification/data/cleaned/metadata_cleaned_{contamination_pct:02d}pct.csv'

    if not os.path.exists(cleaned_file):
        print(f"ERROR: Cleaned dataset not found: {cleaned_file}")
        print("Run detect_outliers.py first!")
        return False

    if not os.path.exists(METADATA_CACHE):
        print(f"ERROR: Original cache not found: {METADATA_CACHE}")
        return False

    print("=" * 80)
    print(f"Applying cleaned dataset ({contamination_pct}% contamination)")
    print("=" * 80)
    print()

    # Load original cache to get structure
    print(f"Loading original cache: {METADATA_CACHE}")
    with open(METADATA_CACHE, 'rb') as f:
        original_metadata = pickle.load(f)

    print(f"Original samples: {len(original_metadata)}")

    # Backup original (if not already backed up)
    if not os.path.exists(BACKUP_FILE):
        print(f"Creating backup: {BACKUP_FILE}")
        shutil.copy2(METADATA_CACHE, BACKUP_FILE)
    else:
        print(f"Backup already exists: {BACKUP_FILE}")

    # Load cleaned CSV
    print(f"Loading cleaned data: {cleaned_file}")
    cleaned_df = pd.read_csv(cleaned_file)
    print(f"Cleaned samples: {len(cleaned_df)}")
    print()

    # Save cleaned as new cache
    print(f"Replacing cache with cleaned data...")
    with open(METADATA_CACHE, 'wb') as f:
        pickle.dump(cleaned_df, f)

    print(f"✅ Cache replaced successfully!")
    print()
    print("Now you can run normal training:")
    print("  python src/main.py --mode search --cv_folds 3 --verbosity 2 \\")
    print("    --resume_mode fresh --device-mode multi \\")
    print(f"    2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_cleaned_{contamination_pct:02d}pct.txt")
    print()
    print("When done, restore original:")
    print("  python apply_cleaned_dataset.py --restore")
    print("=" * 80)

    return True


def restore_original():
    """Restore original cache from backup"""

    if not os.path.exists(BACKUP_FILE):
        print("ERROR: No backup found. Cannot restore.")
        return False

    print("=" * 80)
    print("Restoring original dataset")
    print("=" * 80)
    print()

    print(f"Restoring from backup: {BACKUP_FILE}")
    shutil.copy2(BACKUP_FILE, METADATA_CACHE)

    print("✅ Original cache restored!")
    print()
    print("You can now run training with original 100% data.")
    print("=" * 80)

    return True


def check_status():
    """Check current status"""

    print("=" * 80)
    print("Current Status")
    print("=" * 80)
    print()

    if not os.path.exists(METADATA_CACHE):
        print("❌ No cache file found!")
        return

    if not os.path.exists(BACKUP_FILE):
        print("✅ Using original dataset (no backup exists)")
        print("   Ready to apply cleaned dataset")
    else:
        print("⚠️  Backup exists - cleaned dataset may be active")
        print("   Run --restore to go back to original")

        # Check sizes
        with open(METADATA_CACHE, 'rb') as f:
            current_df = pickle.load(f)
        with open(BACKUP_FILE, 'rb') as f:
            backup_df = pickle.load(f)

        print()
        print(f"Current cache: {len(current_df)} samples")
        print(f"Backup:        {len(backup_df)} samples")

        if len(current_df) < len(backup_df):
            removed = len(backup_df) - len(current_df)
            pct = (removed / len(backup_df)) * 100
            print(f"Difference:    {removed} samples removed ({pct:.1f}%)")
        elif len(current_df) == len(backup_df):
            print("Status:        Same size (possibly restored)")

    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply/restore cleaned dataset')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--apply', action='store_true',
                       help='Apply cleaned dataset')
    group.add_argument('--restore', action='store_true',
                       help='Restore original dataset')
    group.add_argument('--status', action='store_true',
                       help='Check current status')

    parser.add_argument('--contamination', type=int,
                        choices=[5, 10, 15],
                        help='Contamination percentage (required with --apply)')

    args = parser.parse_args()

    if args.apply:
        if not args.contamination:
            parser.error("--contamination required with --apply")
        apply_cleaned(args.contamination)
    elif args.restore:
        restore_original()
    elif args.status:
        check_status()
