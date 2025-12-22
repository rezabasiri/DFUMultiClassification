#!/usr/bin/env python3
"""
Systematic comparison script to validate that the refactored main.py
produces identical results to the original main_original.py.

This ensures the refactoring didn't introduce any bugs.

Usage:
    python scripts/compare_main_versions.py [--modalities MODALITY1 MODALITY2 ...]

Examples:
    # Test all individual modalities
    python scripts/compare_main_versions.py

    # Test specific modalities
    python scripts/compare_main_versions.py --modalities metadata depth_rgb

    # Test combinations
    python scripts/compare_main_versions.py --modalities metadata+depth_rgb depth_rgb+depth_map
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_project_paths


class ComparisonRunner:
    """Runs both versions and compares results"""

    def __init__(self, output_dir="results/comparison"):
        self.project_root = project_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get project paths
        directory, result_dir, root = get_project_paths()
        self.result_dir = Path(result_dir)

        # Results storage
        self.results = {
            'original': {},
            'refactored': {},
            'comparison': {}
        }

    def prepare_original_config(self, modalities):
        """
        Modify main_original.py to run specific modality combinations.

        Args:
            modalities: List of modality strings (e.g., ['metadata'], ['depth_rgb'], etc.)

        Returns:
            Path to modified config file
        """
        # Read main_original.py
        original_file = self.project_root / "src" / "main_original.py"

        # We'll create a config dict string to inject
        config_entries = []
        for i, mod_list in enumerate(modalities):
            if isinstance(mod_list, str):
                mod_list = mod_list.split('+')

            config_entries.append(f"        'test_{i}': {{'modalities': {mod_list}}},")

        config_str = "\n".join(config_entries)

        # Note: We're NOT modifying the original file per user request
        # Instead, we'll use a different approach
        return config_str

    def run_original(self, modalities, data_pct=100, train_pct=0.8, n_runs=1):
        """
        Run main_original.py with specific modalities.

        Note: This requires temporarily modifying the configs dict in
        main_with_specialized_evaluation. We'll do this by creating a
        wrapper script that imports and calls the function.

        Args:
            modalities: List of modality combinations to test
            data_pct: Percentage of data to use
            train_pct: Train/validation split
            n_runs: Number of runs
        """
        print(f"\n{'='*60}")
        print(f"Running ORIGINAL main_original.py")
        print(f"Modalities: {modalities}")
        print(f"Data: {data_pct}%, Train: {train_pct}, Runs: {n_runs}")
        print(f"{'='*60}\n")

        # Create a wrapper script to run the original with custom configs
        wrapper_script = self.output_dir / "run_original_wrapper.py"

        # Build config dict
        config_entries = []
        for i, mod_list in enumerate(modalities):
            if isinstance(mod_list, str):
                mod_list = [m.strip() for m in mod_list.split('+')]
            config_entries.append(f"        'test_{i}': {{'modalities': {mod_list}}},")

        configs_str = "{\n" + "\n".join(config_entries) + "\n    }"

        wrapper_content = f"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from original main
os.chdir(str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import everything needed from main_original
# We'll import the module and modify its configs
import importlib.util
spec = importlib.util.spec_from_file_location("main_original", str(project_root / "src" / "main_original.py"))
main_orig = importlib.util.module_from_spec(spec)

# Execute the module to load all functions
spec.loader.exec_module(main_orig)

# Now call the function with our custom configs
# We need to replicate the logic from main_with_specialized_evaluation
# but with our custom modality list

configs = {configs_str}

print("Custom configs:", configs)
print("Starting original code execution...")

# Prepare dataset
data = main_orig.prepare_dataset(
    main_orig.depth_bb_file,
    main_orig.thermal_bb_file,
    main_orig.csv_file,
    list(set([mod for config in configs.values() for mod in config['modalities']]))
)

# Filter frequent misclassifications
data = main_orig.filter_frequent_misclassifications(
    data, main_orig.result_dir,
    thresholds={{'I': 3, 'P': 2, 'R': 3}}
)

if {data_pct} < 100:
    data = data.sample(frac={data_pct} / 100, random_state=42).reset_index(drop=True)

# Run cross-validation
print("\\nStarting cross-validation...")
metrics, confusion_matrices, histories = main_orig.cross_validation_manual_split(
    data, configs, {train_pct}, {n_runs}
)

print("\\nOriginal code execution complete!")
print("Metrics:", metrics)
"""

        wrapper_script.write_text(wrapper_content)

        # Run the wrapper
        cmd = [sys.executable, str(wrapper_script)]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Save output
            output_file = self.output_dir / "original_output.txt"
            output_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

            if result.returncode != 0:
                print(f"ERROR: Original code failed with return code {result.returncode}")
                print(f"See {output_file} for details")
                return None

            print("Original code completed successfully")

            # Extract metrics from results CSV
            return self.extract_metrics_from_results("original")

        except subprocess.TimeoutExpired:
            print("ERROR: Original code timed out after 1 hour")
            return None
        except Exception as e:
            print(f"ERROR running original code: {e}")
            return None

    def run_refactored(self, modalities, data_pct=100, train_pct=0.8, n_runs=1):
        """
        Run refactored main.py with specific modalities.

        Args:
            modalities: List of modality combinations to test
            data_pct: Percentage of data to use
            train_pct: Train/validation split
            n_runs: Number of runs (should be 1 for comparison with cv_folds=0)
        """
        print(f"\n{'='*60}")
        print(f"Running REFACTORED main.py")
        print(f"Modalities: {modalities}")
        print(f"Data: {data_pct}%, Train: {train_pct}, n_runs: {n_runs}, cv_folds: 0")
        print(f"{'='*60}\n")

        # We need to configure production_config.py to use our specific modalities
        # Read current config
        config_file = self.project_root / "src" / "utils" / "production_config.py"
        original_config = config_file.read_text()
        backup_config = self.output_dir / "production_config_backup.py"
        backup_config.write_text(original_config)

        try:
            # Modify config to use custom modalities
            mod_list = []
            for mod in modalities:
                if isinstance(mod, str):
                    mod_list.append([m.strip() for m in mod.split('+')])
                else:
                    mod_list.append(mod)

            # Find and replace INCLUDED_COMBINATIONS
            import re

            # Build the new combinations list
            new_combinations = "INCLUDED_COMBINATIONS = [\n"
            for mod in mod_list:
                new_combinations += f"    {mod},\n"
            new_combinations += "]\n"

            # Replace in config
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS = \[.*?\]',
                new_combinations.strip(),
                original_config,
                flags=re.DOTALL
            )

            # Also set to custom mode
            modified_config = re.sub(
                r"MODALITY_SEARCH_MODE = '[^']*'",
                "MODALITY_SEARCH_MODE = 'custom'",
                modified_config
            )

            # Write modified config
            config_file.write_text(modified_config)

            # Run main.py
            cmd = [
                sys.executable,
                str(self.project_root / "src" / "main.py"),
                "--mode", "search",
                "--data_percentage", str(data_pct),
                "--train_patient_percentage", str(train_pct),
                "--n_runs", str(n_runs),
                "--cv_folds", "0",  # No cross-validation, single split
                "--verbosity", "1",  # Normal verbosity
                "--resume_mode", "fresh"  # Start fresh
            ]

            print(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Save output
            output_file = self.output_dir / "refactored_output.txt"
            output_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

            if result.returncode != 0:
                print(f"ERROR: Refactored code failed with return code {result.returncode}")
                print(f"See {output_file} for details")
                return None

            print("Refactored code completed successfully")

            # Extract metrics from results CSV
            return self.extract_metrics_from_results("refactored")

        except subprocess.TimeoutExpired:
            print("ERROR: Refactored code timed out after 1 hour")
            return None
        except Exception as e:
            print(f"ERROR running refactored code: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Restore original config
            config_file.write_text(original_config)
            print(f"Restored original production_config.py")

    def extract_metrics_from_results(self, version):
        """
        Extract metrics from the results CSV files.

        Args:
            version: 'original' or 'refactored'

        Returns:
            Dict of metrics by modality combination
        """
        # Look for CSV files in results/csv/
        csv_dir = self.result_dir / "csv"

        if not csv_dir.exists():
            print(f"WARNING: CSV directory not found: {csv_dir}")
            return {}

        # Find the most recent results CSV
        csv_files = list(csv_dir.glob("modality_search_results_*.csv"))

        if not csv_files:
            print(f"WARNING: No results CSV found in {csv_dir}")
            return {}

        # Get most recent file
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Reading metrics from: {latest_csv}")

        # Read CSV
        df = pd.read_csv(latest_csv)

        # Extract metrics by modality combination
        metrics = {}

        for _, row in df.iterrows():
            mod_combo = row['Modality Combination']
            metrics[mod_combo] = {
                'accuracy': row.get('Mean Accuracy', np.nan),
                'accuracy_std': row.get('Std Accuracy', np.nan),
                'f1_macro': row.get('Mean F1 Macro', np.nan),
                'f1_macro_std': row.get('Std F1 Macro', np.nan),
                'kappa': row.get('Mean Kappa', np.nan),
                'kappa_std': row.get('Std Kappa', np.nan),
                'f1_class_0': row.get('Mean F1 Class 0 (I)', np.nan),
                'f1_class_1': row.get('Mean F1 Class 1 (P)', np.nan),
                'f1_class_2': row.get('Mean F1 Class 2 (R)', np.nan),
            }

        # Save to file
        output_file = self.output_dir / f"{version}_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"Saved {version} metrics to: {output_file}")

        return metrics

    def compare_metrics(self, original_metrics, refactored_metrics, tolerance=1e-6):
        """
        Compare metrics between original and refactored versions.

        Args:
            original_metrics: Dict of metrics from original code
            refactored_metrics: Dict of metrics from refactored code
            tolerance: Numerical tolerance for comparison

        Returns:
            Dict with comparison results
        """
        print(f"\n{'='*60}")
        print("COMPARING RESULTS")
        print(f"{'='*60}\n")

        comparison = {
            'identical': [],
            'different': [],
            'missing_in_refactored': [],
            'missing_in_original': [],
            'differences': {}
        }

        # Check all modalities in original
        for mod_combo in original_metrics:
            if mod_combo not in refactored_metrics:
                comparison['missing_in_refactored'].append(mod_combo)
                continue

            orig = original_metrics[mod_combo]
            refac = refactored_metrics[mod_combo]

            # Compare each metric
            all_identical = True
            diffs = {}

            for metric_name in orig:
                orig_val = orig[metric_name]
                refac_val = refac.get(metric_name, np.nan)

                # Handle NaN
                if pd.isna(orig_val) and pd.isna(refac_val):
                    continue

                if pd.isna(orig_val) or pd.isna(refac_val):
                    all_identical = False
                    diffs[metric_name] = {
                        'original': orig_val,
                        'refactored': refac_val,
                        'difference': 'NaN mismatch'
                    }
                    continue

                # Numerical comparison
                diff = abs(float(orig_val) - float(refac_val))

                if diff > tolerance:
                    all_identical = False
                    diffs[metric_name] = {
                        'original': float(orig_val),
                        'refactored': float(refac_val),
                        'difference': diff,
                        'relative_diff': (diff / abs(float(orig_val))) if orig_val != 0 else float('inf')
                    }

            if all_identical:
                comparison['identical'].append(mod_combo)
            else:
                comparison['different'].append(mod_combo)
                comparison['differences'][mod_combo] = diffs

        # Check for modalities only in refactored
        for mod_combo in refactored_metrics:
            if mod_combo not in original_metrics:
                comparison['missing_in_original'].append(mod_combo)

        # Print summary
        print(f"Identical results: {len(comparison['identical'])} modalities")
        for mod in comparison['identical']:
            print(f"  ✓ {mod}")

        if comparison['different']:
            print(f"\nDifferent results: {len(comparison['different'])} modalities")
            for mod in comparison['different']:
                print(f"  ✗ {mod}")
                diffs = comparison['differences'][mod]
                for metric, diff_info in diffs.items():
                    if isinstance(diff_info['difference'], str):
                        print(f"      {metric}: {diff_info['difference']}")
                    else:
                        print(f"      {metric}: diff={diff_info['difference']:.6e}, "
                              f"rel_diff={diff_info['relative_diff']:.2%}")

        if comparison['missing_in_refactored']:
            print(f"\nMissing in refactored: {comparison['missing_in_refactored']}")

        if comparison['missing_in_original']:
            print(f"\nMissing in original: {comparison['missing_in_original']}")

        # Save comparison
        output_file = self.output_dir / "comparison_results.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"\nFull comparison saved to: {output_file}")

        return comparison

    def run_comparison(self, modalities, data_pct=100, train_pct=0.8, n_runs=1):
        """
        Run full comparison for given modalities.

        Args:
            modalities: List of modality combinations to test
            data_pct: Percentage of data to use
            train_pct: Train/validation split
            n_runs: Number of runs
        """
        print(f"\n{'#'*60}")
        print(f"# COMPARISON TEST")
        print(f"# Modalities: {modalities}")
        print(f"# Config: data={data_pct}%, train={train_pct}, runs={n_runs}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*60}\n")

        # Run original
        print("\n[1/3] Running original code...")
        original_metrics = self.run_original(modalities, data_pct, train_pct, n_runs)

        if original_metrics is None:
            print("ERROR: Original code failed. Aborting comparison.")
            return None

        # Run refactored
        print("\n[2/3] Running refactored code...")
        refactored_metrics = self.run_refactored(modalities, data_pct, train_pct, n_runs)

        if refactored_metrics is None:
            print("ERROR: Refactored code failed. Aborting comparison.")
            return None

        # Compare
        print("\n[3/3] Comparing results...")
        comparison = self.compare_metrics(original_metrics, refactored_metrics)

        return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and refactored main.py implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--modalities",
        nargs='+',
        default=['metadata', 'depth_rgb', 'depth_map'],
        help="""Modality combinations to test.
        Use '+' to separate modalities in a combination.
        Examples: metadata depth_rgb 'metadata+depth_rgb'
        Default: metadata depth_rgb depth_map"""
    )

    parser.add_argument(
        "--data_percentage",
        type=float,
        default=10.0,
        help="Percentage of data to use (default: 10.0 for quick testing)"
    )

    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.8,
        help="Train/validation split (default: 0.8)"
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of runs (default: 1 for deterministic comparison)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results"
    )

    args = parser.parse_args()

    # Create runner
    runner = ComparisonRunner(output_dir=args.output_dir)

    # Run comparison
    comparison = runner.run_comparison(
        modalities=args.modalities,
        data_pct=args.data_percentage,
        train_pct=args.train_percentage,
        n_runs=args.n_runs
    )

    if comparison:
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Identical: {len(comparison['identical'])}")
        print(f"✗ Different: {len(comparison['different'])}")
        print(f"⚠ Missing in refactored: {len(comparison['missing_in_refactored'])}")
        print(f"⚠ Missing in original: {len(comparison['missing_in_original'])}")
        print(f"\nDetails saved to: {runner.output_dir}")

        # Return exit code
        if comparison['different'] or comparison['missing_in_refactored']:
            sys.exit(1)  # Failure
        else:
            sys.exit(0)  # Success
    else:
        print("\nERROR: Comparison failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
