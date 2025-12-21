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

        # Extract summary without printing full arrays
        if metrics and len(metrics) > 0 and isinstance(metrics, tuple) and len(metrics) > 0:
            all_metrics = metrics[0]
            if all_metrics and len(all_metrics) > 0:
                latest_metrics = all_metrics[-1]
                print(f"Run: {latest_metrics.get('run', 'N/A')}")
                print(f"Config: {latest_metrics.get('config', 'N/A')}")
                print(f"Modalities: {latest_metrics.get('modalities', 'N/A')}")
                print(f"Accuracy: {latest_metrics.get('accuracy', 0):.4f}")
                print(f"F1 Macro: {latest_metrics.get('f1_macro', 0):.4f}")
                print(f"F1 Weighted: {latest_metrics.get('f1_weighted', 0):.4f}")
                print(f"Cohen's Kappa: {latest_metrics.get('kappa', 0):.4f}")
                if 'f1_classes' in latest_metrics:
                    import numpy as np
                    f1_classes = latest_metrics['f1_classes']
                    print(f"F1 per class: I={f1_classes[0]:.4f}, P={f1_classes[1]:.4f}, R={f1_classes[2]:.4f}")
            else:
                print("No metrics returned")
        else:
            print(f"Metrics structure: {type(metrics)}")

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

        # Extract summary without printing full arrays
        if metrics and len(metrics) > 0:
            if isinstance(metrics, list):
                latest_metrics = metrics[-1] if metrics else None
                if latest_metrics:
                    print(f"Run: {latest_metrics.get('run', 'N/A')}")
                    print(f"Config: {latest_metrics.get('config', 'N/A')}")
                    print(f"Modalities: {latest_metrics.get('modalities', 'N/A')}")
                    print(f"Accuracy: {latest_metrics.get('accuracy', 0):.4f}")
                    print(f"F1 Macro: {latest_metrics.get('f1_macro', 0):.4f}")
                    print(f"F1 Weighted: {latest_metrics.get('f1_weighted', 0):.4f}")
                    print(f"Cohen's Kappa: {latest_metrics.get('kappa', 0):.4f}")
                    if 'f1_classes' in latest_metrics:
                        import numpy as np
                        f1_classes = latest_metrics['f1_classes']
                        print(f"F1 per class: I={f1_classes[0]:.4f}, P={f1_classes[1]:.4f}, R={f1_classes[2]:.4f}")
                else:
                    print("No metrics in list")
            else:
                print(f"Metrics structure: {type(metrics)}")
        else:
            print("No metrics returned")

        return metrics

    except Exception as e:
        print(f"\nERROR in original code:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(original_metrics, refactored_metrics, tolerance=1e-2):
    """
    Compare metrics from both versions directly from return values.
    """
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")

    # Extract metrics from return values
    orig_dict = None
    refac_dict = None

    # Original returns (metrics, confusion_matrices, histories)
    if original_metrics and isinstance(original_metrics, tuple) and len(original_metrics) > 0:
        metrics_list = original_metrics[0] if isinstance(original_metrics[0], list) else []
        if metrics_list and len(metrics_list) > 0:
            orig_dict = metrics_list[-1]

    # Refactored returns (all_metrics, all_confusion_matrices, all_histories)
    if refactored_metrics and isinstance(refactored_metrics, tuple) and len(refactored_metrics) > 0:
        metrics_list = refactored_metrics[0]
        if metrics_list and len(metrics_list) > 0:
            refac_dict = metrics_list[-1]

    if orig_dict is None or refac_dict is None:
        print("⚠️  Could not extract metrics from one or both versions")
        if orig_dict is None:
            print("  - Original metrics: Not available")
        if refac_dict is None:
            print("  - Refactored metrics: Not available")
        return False

    # Compare key metrics
    print(f"\n{'Metric':<20} {'Original':<15} {'Refactored':<15} {'Diff':<15} {'Status'}")
    print("-" * 75)

    metrics_to_compare = [
        ('accuracy', 'Accuracy'),
        ('f1_macro', 'F1 Macro'),
        ('f1_weighted', 'F1 Weighted'),
        ('kappa', "Cohen's Kappa")
    ]

    all_match = True
    for key, label in metrics_to_compare:
        orig_val = orig_dict.get(key, 0)
        refac_val = refac_dict.get(key, 0)
        diff = abs(orig_val - refac_val)
        rel_diff = (diff / abs(orig_val)) * 100 if orig_val != 0 else 0

        status = "✅" if diff <= tolerance else "❌"
        if diff > tolerance:
            all_match = False

        print(f"{label:<20} {orig_val:<15.4f} {refac_val:<15.4f} {rel_diff:<14.2f}% {status}")

    # Compare F1 per class
    if 'f1_classes' in orig_dict and 'f1_classes' in refac_dict:
        print(f"\n{'F1 Per Class':<20} {'Original':<15} {'Refactored':<15} {'Diff':<15} {'Status'}")
        print("-" * 75)
        class_labels = ['Inflammatory', 'Proliferative', 'Remodeling']
        orig_f1 = orig_dict['f1_classes']
        refac_f1 = refac_dict['f1_classes']

        for i, label in enumerate(class_labels):
            diff = abs(orig_f1[i] - refac_f1[i])
            rel_diff = (diff / abs(orig_f1[i])) * 100 if orig_f1[i] != 0 else 0
            status = "✅" if diff <= tolerance else "❌"
            if diff > tolerance:
                all_match = False
            print(f"{label:<20} {orig_f1[i]:<15.4f} {refac_f1[i]:<15.4f} {rel_diff:<14.2f}% {status}")

    print("\n" + "=" * 75)
    if all_match:
        print("✅ ALL METRICS MATCH (within tolerance)")
        print(f"   Tolerance: {tolerance*100:.1f}%")
        return True
    else:
        print("⚠️  SOME METRICS DIFFER (beyond tolerance)")
        print(f"   Tolerance: {tolerance*100:.1f}%")
        print("   This may be due to:")
        print("   - Different random initialization")
        print("   - Floating point precision")
        print("   - TensorFlow operation ordering")
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
