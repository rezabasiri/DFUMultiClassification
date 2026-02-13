#!/usr/bin/env python3
"""
Pre-flight check for multi-parameter search experiment.

Verifies that production_config.py is ready and all settings are correct.
Run this before starting the search to catch configuration issues early.

Usage:
    python agent_communication/contamination_search/preflight_check.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_config():
    """Check production_config.py settings"""
    print("="*80)
    print("PRE-FLIGHT CHECK FOR MULTI-PARAMETER SEARCH")
    print("="*80)
    print()

    # Import config
    try:
        from src.utils.production_config import (
            N_EPOCHS, IMAGE_SIZE, STAGE1_EPOCHS, OUTLIER_CONTAMINATION, OUTLIER_BATCH_SIZE,
            INCLUDED_COMBINATIONS, MODALITY_SEARCH_MODE, CV_N_SPLITS, OUTLIER_REMOVAL
        )
    except ImportError as e:
        print(f"❌ FAILED: Could not import production_config.py")
        print(f"   Error: {e}")
        return False

    all_good = True

    # Check 1: INCLUDED_COMBINATIONS must be set (will be tested across all param combinations)
    print("1. Checking INCLUDED_COMBINATIONS...")
    if MODALITY_SEARCH_MODE != 'custom':
        print(f"   ⚠️  WARNING: MODALITY_SEARCH_MODE = '{MODALITY_SEARCH_MODE}'")
        print(f"      Recommend setting to 'custom' for consistent testing")
        all_good = False
    elif not INCLUDED_COMBINATIONS:
        print(f"   ❌ FAILED: INCLUDED_COMBINATIONS is empty")
        print(f"      Must specify which modality combination to test")
        print(f"      Example: [('metadata', 'depth_rgb', 'depth_map', 'thermal_map')]")
        all_good = False
    else:
        print(f"   ✓ INCLUDED_COMBINATIONS: {INCLUDED_COMBINATIONS}")
        if len(INCLUDED_COMBINATIONS) > 1:
            print(f"   ⚠️  WARNING: Multiple combinations specified ({len(INCLUDED_COMBINATIONS)})")
            print(f"      Search will test parameters for EACH combination")
            print(f"      This will multiply search time by {len(INCLUDED_COMBINATIONS)}x")

    # Check 2: CV_N_SPLITS (should be 3 for full 3-fold CV)
    print()
    print("2. Checking CV_N_SPLITS...")
    if CV_N_SPLITS != 3:
        print(f"   ⚠️  WARNING: CV_N_SPLITS = {CV_N_SPLITS} (expected 3)")
        print(f"      Search expects 3-fold cross-validation")
    else:
        print(f"   ✓ CV_N_SPLITS = {CV_N_SPLITS}")

    # Check 3: OUTLIER_REMOVAL (should be enabled)
    print()
    print("3. Checking OUTLIER_REMOVAL...")
    if not OUTLIER_REMOVAL:
        print(f"   ⚠️  WARNING: OUTLIER_REMOVAL = {OUTLIER_REMOVAL}")
        print(f"      Outlier detection is disabled, contamination param will have no effect")
    else:
        print(f"   ✓ OUTLIER_REMOVAL = {OUTLIER_REMOVAL}")

    # Check 4: Current baseline values (will be overwritten during search)
    print()
    print("4. Current baseline values (will be modified during search):")
    print(f"   N_EPOCHS: {N_EPOCHS}")
    print(f"   IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"   STAGE1_EPOCHS: {STAGE1_EPOCHS}")
    print(f"   OUTLIER_CONTAMINATION: {OUTLIER_CONTAMINATION}")
    print(f"   OUTLIER_BATCH_SIZE: {OUTLIER_BATCH_SIZE}")

    # Check 5: Backup status
    print()
    print("5. Checking for existing backups...")
    backup_file = project_root / "src/utils/production_config.py.backup_multi"
    if backup_file.exists():
        print(f"   ⚠️  WARNING: Backup already exists: {backup_file}")
        print(f"      Previous search may have been interrupted")
        print(f"      Check if config needs to be restored manually")
    else:
        print(f"   ✓ No existing backup found (clean state)")

    # Check 6: Results directory
    print()
    print("6. Checking results directory...")
    csv_dir = project_root / "results/csv"
    if csv_dir.exists():
        print(f"   ✓ Results directory exists: {csv_dir}")
    else:
        print(f"   ⚠️  WARNING: Results directory not found: {csv_dir}")
        print(f"      Will be created during first run")

    # Summary
    print()
    print("="*80)
    if all_good:
        print("✓ PRE-FLIGHT CHECK PASSED")
        print()
        print("Ready to run:")
        print("  python agent_communication/contamination_search/search_multi_param.py --n-trials 20")
    else:
        print("⚠️  PRE-FLIGHT CHECK COMPLETED WITH WARNINGS")
        print()
        print("Review warnings above before starting search.")
        print("Some warnings are informational and may be acceptable.")
    print("="*80)

    return all_good


if __name__ == '__main__':
    check_config()
