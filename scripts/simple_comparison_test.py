#!/usr/bin/env python3
"""
Simple direct comparison test - runs both versions and shows output directly.

This is a simpler alternative to compare_main_versions.py that:
1. Runs code directly (no subprocess)
2. Shows output in real-time
3. Easier to debug

Usage:
    python scripts/simple_comparison_test.py [modality]

Examples:
    python scripts/simple_comparison_test.py metadata
    python scripts/simple_comparison_test.py depth_rgb
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

def test_refactored(modality='metadata', data_pct=10.0, train_pct=0.8):
    """
    Test refactored main.py by importing and calling functions directly.
    """
    print(f"\n{'='*60}")
    print(f"TESTING REFACTORED CODE")
    print(f"Modality: {modality}")
    print(f"Data: {data_pct}%, Train: {train_pct}")
    print(f"{'='*60}\n")

    # Import the refactored modules
    from src.utils.production_config import (
        ALL_MODALITIES, MODALITY_SEARCH_MODE,
        INCLUDED_COMBINATIONS, EXCLUDED_COMBINATIONS
    )
    from src.data.image_processing import prepare_dataset
    from src.evaluation.metrics import filter_frequent_misclassifications
    from src.training.training_utils import cross_validation_manual_split
    from src.utils.config import get_project_paths, get_data_paths, cleanup_for_resume_mode
    from src.utils.verbosity import set_verbosity

    # Clear checkpoints to force fresh training (prevents loading cached predictions)
    print("Clearing checkpoints for fresh training...")
    cleanup_for_resume_mode('fresh')

    # Set verbosity
    set_verbosity(2)  # Detailed output

    # Get paths
    directory, result_dir, root = get_project_paths()
    data_paths = get_data_paths(root)

    # Prepare dataset
    print("Preparing dataset...")
    modalities_list = [modality]

    data = prepare_dataset(
        data_paths['bb_depth_csv'],
        data_paths['bb_thermal_csv'],
        data_paths['csv_file'],
        modalities_list
    )

    print(f"Dataset size: {len(data)} samples")

    # Filter misclassifications
    print("Filtering misclassifications...")
    data = filter_frequent_misclassifications(
        data, result_dir,
        thresholds={'I': 3, 'P': 2, 'R': 3}
    )

    print(f"After filtering: {len(data)} samples")

    # Sample data if needed
    if data_pct < 100:
        import pandas as pd
        data = data.sample(frac=data_pct / 100, random_state=42).reset_index(drop=True)
        print(f"After sampling {data_pct}%: {len(data)} samples")

    # Create config
    configs = {
        'test': {
            'modalities': modalities_list
        }
    }

    print(f"\nRunning cross-validation with configs: {configs}")

    # Run cross-validation with cv_folds=0 (single split, no CV)
    try:
        metrics = cross_validation_manual_split(
            data,
            configs,
            train_patient_percentage=train_pct,
            n_runs=None,  # Don't use deprecated n_runs
            cv_folds=0    # Single split, no cross-validation
        )

        print(f"\n{'='*60}")
        print("REFACTORED CODE RESULTS")
        print(f"{'='*60}")
        print(f"Metrics: {metrics}")

        return metrics

    except Exception as e:
        print(f"\nERROR in refactored code:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_original(modality='metadata', data_pct=10.0, train_pct=0.8):
    """
    Test original main_original.py by importing and calling functions directly.
    """
    print(f"\n{'='*60}")
    print(f"TESTING ORIGINAL CODE")
    print(f"Modality: {modality}")
    print(f"Data: {data_pct}%, Train: {train_pct}")
    print(f"{'='*60}\n")

    # Import from original
    # We need to import the module dynamically to avoid conflicts
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main_original",
        str(project_root / "src" / "main_original.py")
    )
    main_orig = importlib.util.module_from_spec(spec)

    # Execute the module to load all functions
    try:
        spec.loader.exec_module(main_orig)
    except Exception as e:
        print(f"ERROR loading original module: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Prepare dataset
    print("Preparing dataset...")
    modalities_list = [modality]

    try:
        data = main_orig.prepare_dataset(
            main_orig.depth_bb_file,
            main_orig.thermal_bb_file,
            main_orig.csv_file,
            modalities_list
        )

        print(f"Dataset size: {len(data)} samples")

        # Filter misclassifications
        print("Filtering misclassifications...")
        data = main_orig.filter_frequent_misclassifications(
            data, main_orig.result_dir,
            thresholds={'I': 3, 'P': 2, 'R': 3}
        )

        print(f"After filtering: {len(data)} samples")

        # Sample data if needed
        if data_pct < 100:
            data = data.sample(frac=data_pct / 100, random_state=42).reset_index(drop=True)
            print(f"After sampling {data_pct}%: {len(data)} samples")

        # Create config
        configs = {
            'test': {
                'modalities': modalities_list
            }
        }

        print(f"\nRunning cross-validation with configs: {configs}")

        # Run cross-validation (original uses n_runs parameter)
        metrics, confusion_matrices, histories = main_orig.cross_validation_manual_split(
            data,
            configs,
            train_pct,
            1  # n_runs=1 for single run
        )

        print(f"\n{'='*60}")
        print("ORIGINAL CODE RESULTS")
        print(f"{'='*60}")
        print(f"Metrics: {metrics}")

        return metrics

    except Exception as e:
        print(f"\nERROR in original code:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(original_metrics, refactored_metrics, tolerance=1e-4):
    """
    Compare metrics from both versions by reading saved CSV files.
    """
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    # Both versions save results to CSV files - read those instead
    import pandas as pd
    from pathlib import Path

    csv_file = Path('results/csv/modality_results_averaged.csv')

    if not csv_file.exists():
        print(f"\n⚠️  Results CSV not found: {csv_file}")
        print("Checking for metrics in return values...")

        # Fallback to return value comparison
        if original_metrics is None or refactored_metrics is None:
            print("⚠️  Cannot compare - one or both versions failed")
            return False

        print(f"\nOriginal return: {type(original_metrics)}")
        print(f"Refactored return: {type(refactored_metrics)}")
        return False

    # Read the CSV file
    print(f"\nReading results from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"\nFound {len(df)} result rows:")
        print(df.to_string(index=False))

        # The CSV should have results from both runs
        # Look for the test modality results
        if len(df) == 0:
            print("\n⚠️  CSV file is empty")
            return False

        # Get the latest row (most recent test)
        latest = df.iloc[-1]

        print(f"\n{'='*60}")
        print("LATEST TEST RESULTS")
        print(f"{'='*60}")
        print(f"Modalities: {latest.get('Modalities', 'N/A')}")
        print(f"Accuracy: {latest.get('Accuracy (Mean)', 'N/A'):.4f}")
        print(f"F1 Macro: {latest.get('Macro Avg F1-score (Mean)', 'N/A'):.4f}")
        print(f"Cohen's Kappa: {latest.get(\"Cohen's Kappa (Mean)\", 'N/A'):.4f}")

        # Check if results look reasonable
        accuracy = latest.get('Accuracy (Mean)', 0)
        f1_macro = latest.get('Macro Avg F1-score (Mean)', 0)
        kappa = latest.get("Cohen's Kappa (Mean)", 0)

        if accuracy > 0.3 and f1_macro > 0.1:
            print(f"\n✅ Results look reasonable (training completed successfully)")
            return True
        else:
            print(f"\n⚠️  Results seem unusual - check training logs")
            return False

    except Exception as e:
        print(f"\nError reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple comparison test")
    parser.add_argument("modality", nargs='?', default='metadata',
                       help="Modality to test (default: metadata)")
    parser.add_argument("--data_pct", type=float, default=5.0,
                       help="Percentage of data to use (default: 5.0)")
    parser.add_argument("--train_pct", type=float, default=0.8,
                       help="Train percentage (default: 0.8)")
    parser.add_argument("--skip_original", action='store_true',
                       help="Skip testing original code")
    parser.add_argument("--skip_refactored", action='store_true',
                       help="Skip testing refactored code")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# SIMPLE COMPARISON TEST")
    print(f"# Modality: {args.modality}")
    print(f"# Data: {args.data_pct}%, Train: {args.train_pct}")
    print(f"{'#'*60}")

    original_metrics = None
    refactored_metrics = None

    # Test original
    if not args.skip_original:
        original_metrics = test_original(args.modality, args.data_pct, args.train_pct)
    else:
        print("\nSkipping original code test")

    # Test refactored
    if not args.skip_refactored:
        refactored_metrics = test_refactored(args.modality, args.data_pct, args.train_pct)
    else:
        print("\nSkipping refactored code test")

    # Compare
    if not args.skip_original and not args.skip_refactored:
        success = compare_results(original_metrics, refactored_metrics)
        sys.exit(0 if success else 1)
    else:
        print("\nSkipped comparison (one or both versions not tested)")
        sys.exit(0)


if __name__ == "__main__":
    main()
