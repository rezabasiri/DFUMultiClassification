"""
Automatic Dataset Polishing via Iterative Misclassification Filtering

This script automatically finds the optimal filtered dataset by:
1. Training metadata-only to identify misclassifications
2. Iteratively excluding problematic samples
3. Stopping when metadata performance meets minimum acceptable thresholds
4. Using the polished dataset for training all selected modalities

The key insight: If metadata can't learn certain samples even after filtering,
they're likely noise or annotation errors. Remove them before expensive
multi-modal training.

Threshold Scaling & Random Seeds:
Runs k-fold CV multiple times with DIFFERENT random seeds for independent fold assignments.
Each run: base_random_seed + run_number (e.g., seeds 43, 44, 45 for base=42, n_runs=3).
Misclassifications accumulate across runs → max count = n_runs.
Thresholds set as percentages of n_runs:
- P class (dominant, 60%): 50% of n_runs (aggressive filtering)
- I class (minority, 30%): 67% of n_runs (moderate filtering)
- R class (rarest, 10%): 100% of n_runs (conservative, only exclude if always wrong)

Example: n_runs=3 → P=1 (33%), I=2 (67%), R=3 (100%)
Example: n_runs=10 → P=5 (50%), I=6 (60%), R=10 (100%)

Each run generates different fold assignments → robust misclassification detection.

Usage:
    python scripts/auto_polish_dataset.py --modalities metadata depth_rgb depth_map
    python scripts/auto_polish_dataset.py --modalities metadata depth_rgb --min_f1_per_class 0.35
    python scripts/auto_polish_dataset.py --help
"""

import argparse
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_project_paths, cleanup_for_resume_mode


class DatasetPolisher:
    """Automatically polish dataset by iterative metadata-based filtering."""

    def __init__(self,
                 modalities,
                 min_f1_per_class=0.42,
                 min_macro_f1=0.48,
                 min_kappa=0.35,
                 max_iterations=5,
                 min_dataset_size=500,
                 cv_folds=1,  # Single split per run (fast polishing, use 5 for final training)
                 n_runs=3,  # Multiple runs with different seeds (3-5 recommended)
                 base_random_seed=42,  # Base seed (each run uses base_seed + run_number)
                 initial_thresholds=None,
                 threshold_reduction_factor=0.8):
        """
        Initialize the dataset polisher.

        Args:
            modalities: List of modalities to train (must include 'metadata')
            min_f1_per_class: Minimum F1 score required for each class (POC medical standard)
            min_macro_f1: Minimum macro F1 score (balanced performance)
            min_kappa: Minimum Cohen's Kappa ("fair" clinical agreement standard)
            max_iterations: Maximum polishing iterations
            min_dataset_size: Minimum dataset size (safety limit)
            cv_folds: Number of CV folds (default: 5)
            n_runs: Number of runs with different random seeds (default: 3)
            base_random_seed: Base seed, each run uses base_seed + run_idx (default: 42)
            initial_thresholds: Starting thresholds {'I': x, 'P': y, 'R': z}
            threshold_reduction_factor: How aggressively to lower thresholds (0.8 = 20% reduction)
        """
        if 'metadata' not in modalities:
            raise ValueError("Modalities must include 'metadata' for polishing")

        self.modalities = modalities
        self.min_f1_per_class = min_f1_per_class
        self.min_macro_f1 = min_macro_f1
        self.min_kappa = min_kappa
        self.max_iterations = max_iterations
        self.min_dataset_size = min_dataset_size
        self.cv_folds = cv_folds
        self.n_runs = n_runs
        self.base_random_seed = base_random_seed
        self.threshold_reduction_factor = threshold_reduction_factor

        # Thresholds proportional to n_runs (max misclass count = n_runs)
        # Each run uses different random seed → independent fold assignments
        # Strategy: Lower threshold for dominant P class, higher for rare R class
        if initial_thresholds is None:
            self.thresholds = {
                'P': max(1, int(0.50 * n_runs)),  # 50% - dominant class (60%), filter more
                'I': max(1, int(0.67 * n_runs)),  # 67% - minority class (30%), moderate
                'R': max(1, int(1.00 * n_runs))   # 100% - rarest class (10%), protect most
            }
        else:
            self.thresholds = initial_thresholds

        # Get project paths
        self.directory, self.result_dir, self.root = get_project_paths()

        # History tracking
        self.history = []
        self.polished_samples = None  # List of sample IDs to keep

    def run_metadata_training(self, iteration):
        """
        Run metadata-only training multiple times with different random seeds.

        Each run uses base_random_seed + run_idx for independent fold assignments.
        Misclassifications accumulate across runs → max count = n_runs.
        """
        print("\n" + "="*70)
        print(f"ITERATION {iteration}: METADATA-ONLY TRAINING ({self.n_runs} runs)")
        print("="*70)

        # Clean up from previous iteration (only on iteration 1, clean everything)
        if iteration == 1:
            cleanup_for_resume_mode('fresh')
        else:
            # For subsequent iterations, only clean predictions/models but keep patient splits
            cleanup_for_resume_mode('from_data')

        # Show thresholds with percentages
        thresh_str = "Thresholds (count/runs): "
        thresh_str += ", ".join([f"{k}={v}/{self.n_runs} ({100*v/self.n_runs:.0f}%)"
                                  for k, v in sorted(self.thresholds.items())])
        print(f"\n{thresh_str}")
        print(f"Running {self.cv_folds}-fold CV {self.n_runs} times with different random seeds")
        print(f"Seeds: {[self.base_random_seed + i for i in range(1, self.n_runs + 1)]}\n")

        # Temporarily override INCLUDED_COMBINATIONS to only train metadata
        config_path = project_root / 'src' / 'utils' / 'production_config.py'
        backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

        # Read current config
        with open(config_path, 'r') as f:
            original_config = f.read()

        # Create backup
        with open(backup_path, 'w') as f:
            f.write(original_config)

        try:
            # Modify INCLUDED_COMBINATIONS to only include metadata
            import re
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
                "INCLUDED_COMBINATIONS = [\n    ('metadata',),  # Temporary: polishing iteration\n]",
                original_config
            )

            with open(config_path, 'w') as f:
                f.write(modified_config)

            # Run training n_runs times with different random seeds
            # Each run must train fresh (delete predictions but keep patient splits)
            for run_idx in range(1, self.n_runs + 1):
                print(f"\n{'─'*70}")
                print(f"RUN {run_idx}/{self.n_runs} (seed={self.base_random_seed + run_idx})")
                print(f"{'─'*70}")

                # Clean up predictions/models/patient splits from previous run
                # We MUST delete patient splits to force regeneration with new random seed
                # This ensures each run has truly independent fold assignments
                if run_idx > 1:
                    import glob
                    import shutil
                    from src.utils.config import get_output_paths
                    output_paths = get_output_paths(self.result_dir)

                    # Delete predictions, models, AND patient splits (to force regeneration with new seed)
                    # Keep only misclassification CSV for accumulation across runs
                    patterns = [
                        os.path.join(output_paths['checkpoints'], '*predictions*.npy'),
                        os.path.join(output_paths['checkpoints'], '*pred*.npy'),
                        os.path.join(output_paths['checkpoints'], '*label*.npy'),
                        os.path.join(output_paths['checkpoints'], 'patient_split_*.npz'),  # DELETE to force new splits!
                        os.path.join(output_paths['models'], '*.h5'),
                    ]
                    for pattern in patterns:
                        for file_path in glob.glob(pattern):
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass

                # Set environment variable to control random seed for fold generation
                os.environ['CROSS_VAL_RANDOM_SEED'] = str(self.base_random_seed + run_idx)

                # ALWAYS use fresh mode - forces clean training with new random seed
                cmd = [
                    'python', 'src/main.py',
                    '--mode', 'search',
                    '--cv_folds', str(self.cv_folds),
                    '--verbosity', '1',  # Reduce noise
                    '--resume_mode', 'fresh',  # Force clean training each run
                ]

                # Add threshold arguments if not first iteration
                if iteration > 1:
                    cmd.extend(['--threshold_I', str(self.thresholds['I'])])
                    cmd.extend(['--threshold_P', str(self.thresholds['P'])])
                    cmd.extend(['--threshold_R', str(self.thresholds['R'])])

                if run_idx == 1:
                    print(f"⏳ First run may take 15-30 minutes (fresh mode)...")
                else:
                    print(f"⏳ Run {run_idx} - Training with seed {self.base_random_seed + run_idx} (fresh mode)...")

                # Run training with live output
                result = subprocess.run(cmd, cwd=project_root, env=os.environ.copy())

                if result.returncode != 0:
                    print(f"\n❌ Training failed on run {run_idx} with return code {result.returncode}")
                    return None

            # Clear environment variable
            if 'CROSS_VAL_RANDOM_SEED' in os.environ:
                del os.environ['CROSS_VAL_RANDOM_SEED']

            print(f"\n{'='*70}")
            print(f"✅ Completed {self.n_runs} runs with independent fold assignments")
            print(f"Misclassification counts accumulated (max count = {self.n_runs})")
            print(f"{'='*70}\n")

            # Extract performance metrics from CSV files (output not captured)
            return self.extract_metrics_from_files()

        finally:
            # Always restore original config
            with open(backup_path, 'r') as f:
                original_config = f.read()
            with open(config_path, 'w') as f:
                f.write(original_config)
            # Remove backup
            backup_path.unlink(missing_ok=True)

    def extract_metrics_from_files(self):
        """Extract performance metrics from CSV files."""
        metrics = {
            'f1_per_class': {'I': 0.0, 'P': 0.0, 'R': 0.0},
            'macro_f1': 0.0,
            'kappa': 0.0,
            'accuracy': 0.0
        }

        # Read from CSV files
        csv_file = os.path.join(self.result_dir, 'csv', 'modality_results_averaged.csv')
        if not os.path.exists(csv_file):
            print(f"⚠️  Metrics file not found: {csv_file}")
            return metrics

        try:
            # Check if file is empty
            if os.path.getsize(csv_file) == 0:
                print(f"⚠️  Metrics file is empty: {csv_file}")
                return metrics

            df = pd.read_csv(csv_file)
            if len(df) == 0:
                print(f"⚠️  No data in metrics file")
                return metrics

            # Get the metadata row (should be only one)
            metadata_rows = df[df['Modalities'].str.contains('metadata', case=False, na=False)]
            if len(metadata_rows) == 0:
                print(f"⚠️  No metadata results found in CSV")
                return metrics

            row = metadata_rows.iloc[-1]  # Latest metadata result
            metrics['macro_f1'] = float(row.get('Macro Avg F1-score (Mean)', 0.0))
            metrics['kappa'] = float(row.get("Cohen's Kappa (Mean)", 0.0))
            metrics['accuracy'] = float(row.get('Accuracy (Mean)', 0.0))

            # Try to get per-class F1 if available
            for i, cls in enumerate(['I', 'P', 'R']):
                col_name = f'Class {i} F1-score (Mean)'
                if col_name in row:
                    metrics['f1_per_class'][cls] = float(row[col_name])

            print(f"✓ Extracted metrics: Acc={metrics['accuracy']:.3f}, F1={metrics['macro_f1']:.3f}, Kappa={metrics['kappa']:.3f}")

        except Exception as e:
            print(f"⚠️  Could not read metrics from CSV: {e}")

        return metrics

    def extract_metrics(self, output):
        """Extract performance metrics from training output."""
        # Look for the final results section
        metrics = {
            'f1_per_class': {'I': 0.0, 'P': 0.0, 'R': 0.0},
            'macro_f1': 0.0,
            'kappa': 0.0,
            'accuracy': 0.0
        }

        # Parse output for metrics
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # Look for final averaged results
            if 'Final Averaged Results' in line or 'Gating Network Results' in line:
                # Look ahead for metrics
                for j in range(i, min(i+20, len(lines))):
                    l = lines[j].lower()

                    # Macro F1
                    if 'f1 macro' in l or 'f1_macro' in l:
                        try:
                            val = float(l.split(':')[-1].strip())
                            metrics['macro_f1'] = val
                        except:
                            pass

                    # Kappa
                    if 'kappa' in l:
                        try:
                            val = float(l.split(':')[-1].strip())
                            metrics['kappa'] = val
                        except:
                            pass

                    # Accuracy
                    if 'accuracy' in l and 'weighted' not in l:
                        try:
                            val = float(l.split(':')[-1].strip())
                            metrics['accuracy'] = val
                        except:
                            pass

                    # Per-class F1
                    if 'f1 per class' in l:
                        try:
                            # Format: "F1 per class: I=0.28, P=0.39, R=0.21"
                            parts = l.split(':')[-1].split(',')
                            for part in parts:
                                if '=' in part:
                                    cls, val = part.split('=')
                                    cls = cls.strip().upper()
                                    val = float(val.strip())
                                    if cls in ['I', 'P', 'R']:
                                        metrics['f1_per_class'][cls] = val
                        except:
                            pass

        # Fallback: try to read from CSV files
        if metrics['macro_f1'] == 0.0:
            csv_file = os.path.join(self.result_dir, 'csv', 'modality_results_averaged.csv')
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        row = df.iloc[-1]  # Last row
                        metrics['macro_f1'] = row.get('F1_Macro', 0.0)
                        metrics['kappa'] = row.get('Kappa', 0.0)
                        metrics['accuracy'] = row.get('Accuracy', 0.0)
                except Exception as e:
                    print(f"Warning: Could not read CSV: {e}")

        return metrics

    def check_performance(self, metrics):
        """Check if current performance meets minimum requirements."""
        checks = {
            'f1_per_class': {},
            'macro_f1': False,
            'kappa': False,
            'overall': False
        }

        # Check per-class F1
        for cls in ['I', 'P', 'R']:
            f1 = metrics['f1_per_class'].get(cls, 0.0)
            checks['f1_per_class'][cls] = f1 >= self.min_f1_per_class

        # Check macro F1
        checks['macro_f1'] = metrics['macro_f1'] >= self.min_macro_f1

        # Check kappa
        checks['kappa'] = metrics['kappa'] >= self.min_kappa

        # Overall: all per-class F1s must pass, AND (macro_f1 OR kappa)
        all_classes_pass = all(checks['f1_per_class'].values())
        aggregate_passes = checks['macro_f1'] or checks['kappa']
        checks['overall'] = all_classes_pass and aggregate_passes

        return checks

    def update_thresholds(self, metrics):
        """
        Update filtering thresholds based on current performance.

        Strategy: Lower thresholds for classes that are performing worst
        (lower threshold = exclude more samples)
        """
        new_thresholds = self.thresholds.copy()

        # Calculate how far each class is from target
        for cls in ['I', 'P', 'R']:
            f1 = metrics['f1_per_class'].get(cls, 0.0)

            if f1 < self.min_f1_per_class:
                # Performance is below target - reduce threshold to exclude more
                gap = self.min_f1_per_class - f1
                reduction = self.threshold_reduction_factor

                # More aggressive reduction for larger gaps
                if gap > 0.1:  # Very bad performance
                    reduction = 0.6  # 40% reduction
                elif gap > 0.05:  # Moderately bad
                    reduction = 0.7  # 30% reduction

                # Reduce threshold but never below 1
                new_thresholds[cls] = max(1, int(new_thresholds[cls] * reduction))

        # Ensure P (dominant class) has lower threshold than I and R
        if new_thresholds['P'] >= new_thresholds['I']:
            new_thresholds['P'] = max(1, new_thresholds['I'] - 1)
        if new_thresholds['P'] >= new_thresholds['R']:
            new_thresholds['P'] = max(1, new_thresholds['R'] - 1)

        return new_thresholds

    def analyze_misclassifications(self):
        """Analyze current misclassification patterns."""
        misclass_file = os.path.join(self.result_dir, 'frequent_misclassifications_total.csv')

        if not os.path.exists(misclass_file):
            print("⚠️  No misclassification file found")
            return None

        df = pd.read_csv(misclass_file)

        stats = {}
        for phase in ['I', 'P', 'R']:
            phase_df = df[df['True_Label'] == phase]
            if len(phase_df) > 0:
                stats[phase] = {
                    'count': len(phase_df),
                    'excluded': len(phase_df[phase_df['Misclass_Count'] >= self.thresholds[phase]]),
                    'mean_count': phase_df['Misclass_Count'].mean(),
                    'max_count': phase_df['Misclass_Count'].max()
                }

        return stats

    def save_polished_config(self, final_thresholds):
        """Save the polished dataset configuration."""
        # Copy misclassification file to saved version
        source = os.path.join(self.result_dir, 'frequent_misclassifications_total.csv')
        dest = os.path.join(self.result_dir, 'frequent_misclassifications_saved.csv')

        if os.path.exists(source):
            import shutil
            shutil.copy(source, dest)
            print(f"\n✅ Saved misclassification baseline: {dest}")

        # Save configuration
        config = {
            'timestamp': datetime.now().isoformat(),
            'final_thresholds': final_thresholds,
            'min_f1_per_class': self.min_f1_per_class,
            'min_macro_f1': self.min_macro_f1,
            'min_kappa': self.min_kappa,
            'iterations': len(self.history),
            'history': self.history
        }

        config_file = os.path.join(self.result_dir, 'polished_dataset_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Saved polishing configuration: {config_file}")

        return config_file

    def run_full_training(self, final_thresholds):
        """Run full training with all modalities on polished dataset."""
        print("\n" + "="*70)
        print("FINAL TRAINING: ALL MODALITIES ON POLISHED DATASET")
        print("="*70)

        # Clean up
        cleanup_for_resume_mode('fresh')

        # Prepare command
        cmd = [
            'python', 'src/main.py',
            '--mode', 'search',
            '--cv_folds', str(self.cv_folds),
            '--verbosity', '2',  # More verbose for final run
            '--threshold_I', str(final_thresholds['I']),
            '--threshold_P', str(final_thresholds['P']),
            '--threshold_R', str(final_thresholds['R'])
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        print(f"Using polished dataset with thresholds: {final_thresholds}")
        print("\n⏳ Training all modalities (this may take 30-60 minutes)...\n")

        # Run training with live output
        result = subprocess.run(cmd, cwd=project_root)

        return result.returncode == 0

    def polish(self):
        """Main polishing loop."""
        print("="*70)
        print("AUTOMATIC DATASET POLISHING")
        print("="*70)
        print(f"\nTarget Modalities: {', '.join(self.modalities)}")
        print(f"Performance Requirements:")
        print(f"  - Min F1 per class: {self.min_f1_per_class:.2f}")
        print(f"  - Min Macro F1: {self.min_macro_f1:.2f}")
        print(f"  - Min Cohen's Kappa: {self.min_kappa:.2f}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Min Dataset Size: {self.min_dataset_size}")

        for iteration in range(1, self.max_iterations + 1):
            # Run metadata training
            metrics = self.run_metadata_training(iteration)

            if metrics is None:
                print("\n❌ Training failed, stopping polishing")
                break

            # Analyze misclassifications
            misclass_stats = self.analyze_misclassifications()

            # Check performance
            checks = self.check_performance(metrics)

            # Record history
            iteration_data = {
                'iteration': iteration,
                'thresholds': self.thresholds.copy(),
                'metrics': metrics,
                'checks': checks,
                'misclass_stats': misclass_stats
            }
            self.history.append(iteration_data)

            # Print results
            print("\n" + "-"*70)
            print(f"ITERATION {iteration} RESULTS")
            print("-"*70)
            print(f"\nMetrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f} ({'✅' if checks['macro_f1'] else '❌'} target: {self.min_macro_f1:.2f})")
            print(f"  Kappa:    {metrics['kappa']:.4f} ({'✅' if checks['kappa'] else '❌'} target: {self.min_kappa:.2f})")
            print(f"\nPer-Class F1:")
            for cls in ['I', 'P', 'R']:
                f1 = metrics['f1_per_class'].get(cls, 0.0)
                status = '✅' if checks['f1_per_class'][cls] else '❌'
                print(f"  {cls}: {f1:.4f} ({status} target: {self.min_f1_per_class:.2f})")

            if misclass_stats:
                print(f"\nMisclassification Analysis (threshold-based exclusions):")
                for cls in ['I', 'P', 'R']:
                    if cls in misclass_stats:
                        stats = misclass_stats[cls]
                        pct = (stats['excluded'] / stats['count'] * 100) if stats['count'] > 0 else 0
                        print(f"  {cls}: {stats['excluded']}/{stats['count']} excluded ({pct:.1f}%)")

            # Check if we meet requirements
            if checks['overall']:
                print("\n" + "="*70)
                print("🎉 POLISHING COMPLETE - REQUIREMENTS MET!")
                print("="*70)

                # Save configuration
                config_file = self.save_polished_config(self.thresholds)

                # Prepare for full training
                print("\n📋 Next Steps:")
                print(f"1. Polished dataset configuration saved to: {config_file}")
                print(f"2. Edit src/main.py line ~1844 to use thresholds: {self.thresholds}")
                print(f"3. Or re-run this script with --apply to automatically train all modalities")

                return True, self.thresholds

            # Check if we should continue
            if iteration >= self.max_iterations:
                print("\n" + "="*70)
                print("⚠️  REACHED MAX ITERATIONS - STOPPING")
                print("="*70)
                print("\nBest iteration found:")
                self.print_best_iteration()
                break

            # Update thresholds for next iteration
            old_thresholds = self.thresholds.copy()
            self.thresholds = self.update_thresholds(metrics)

            print(f"\n📊 Updating thresholds for next iteration:")
            for cls in ['I', 'P', 'R']:
                print(f"  {cls}: {old_thresholds[cls]} → {self.thresholds[cls]} ({'⬇️' if self.thresholds[cls] < old_thresholds[cls] else '➡️'})")

            # Save updated baseline for next iteration
            source = os.path.join(self.result_dir, 'frequent_misclassifications_total.csv')
            dest = os.path.join(self.result_dir, 'frequent_misclassifications_saved.csv')
            if os.path.exists(source):
                import shutil
                shutil.copy(source, dest)

        # If we exit loop without success, return best iteration
        print("\n💡 Requirements not fully met. Using best iteration found.")
        return False, self.get_best_thresholds()

    def print_best_iteration(self):
        """Find and print the best iteration."""
        if not self.history:
            return

        # Score each iteration (higher is better)
        for item in self.history:
            metrics = item['metrics']
            score = (
                metrics['macro_f1'] * 0.4 +
                min(metrics['f1_per_class'].values()) * 0.4 +  # Worst class F1
                metrics['kappa'] * 0.2
            )
            item['score'] = score

        best = max(self.history, key=lambda x: x['score'])

        print(f"\nBest: Iteration {best['iteration']}")
        print(f"  Thresholds: {best['thresholds']}")
        print(f"  Macro F1: {best['metrics']['macro_f1']:.4f}")
        print(f"  Kappa: {best['metrics']['kappa']:.4f}")
        print(f"  Per-class F1: I={best['metrics']['f1_per_class']['I']:.4f}, " +
              f"P={best['metrics']['f1_per_class']['P']:.4f}, " +
              f"R={best['metrics']['f1_per_class']['R']:.4f}")

    def get_best_thresholds(self):
        """Get thresholds from best iteration."""
        if not self.history:
            return self.thresholds

        for item in self.history:
            metrics = item['metrics']
            score = (
                metrics['macro_f1'] * 0.4 +
                min(metrics['f1_per_class'].values()) * 0.4 +
                metrics['kappa'] * 0.2
            )
            item['score'] = score

        best = max(self.history, key=lambda x: x['score'])
        return best['thresholds']


def main():
    parser = argparse.ArgumentParser(
        description='Automatically polish dataset using metadata-based filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Polish dataset for metadata + images with POC medical standards (defaults)
  python scripts/auto_polish_dataset.py --modalities metadata depth_rgb depth_map
  # Uses: F1≥0.42 per class, Macro F1≥0.48, Kappa≥0.35 (fair clinical agreement)

  # Higher quality for production-ready model
  python scripts/auto_polish_dataset.py \\
      --modalities metadata depth_rgb \\
      --min_f1_per_class 0.50 \\
      --min_macro_f1 0.55 \\
      --min_kappa 0.45 \\
      --max_iterations 10

  # Quick test with relaxed requirements
  python scripts/auto_polish_dataset.py \\
      --modalities metadata depth_rgb \\
      --min_f1_per_class 0.35 \\
      --min_macro_f1 0.40 \\
      --min_kappa 0.25 \\
      --cv_folds 3 \\
      --max_iterations 3

How it works:
  1. Trains metadata-only to identify misclassifications
  2. Excludes frequently misclassified samples (problematic cases)
  3. Re-trains metadata to check if performance improves
  4. Repeats until performance meets minimum requirements
  5. Once polished, trains all modalities on clean dataset
        """
    )

    parser.add_argument('--modalities', nargs='+', required=True,
                        help='Modalities to train (must include metadata)')

    parser.add_argument('--min_f1_per_class', type=float, default=0.42,
                        help='Minimum F1 score for each class - POC medical standard (default: 0.42)')

    parser.add_argument('--min_macro_f1', type=float, default=0.48,
                        help='Minimum macro F1 score - balanced performance target (default: 0.48)')

    parser.add_argument('--min_kappa', type=float, default=0.35,
                        help='Minimum Cohen\'s Kappa - "fair" clinical agreement (default: 0.35)')

    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum polishing iterations (default: 5)')

    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')

    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs (default: 3)')

    parser.add_argument('--apply', action='store_true',
                        help='Automatically train all modalities after polishing')

    args = parser.parse_args()

    # Validate
    if 'metadata' not in args.modalities:
        print("❌ Error: --modalities must include 'metadata'")
        sys.exit(1)

    # Create polisher
    polisher = DatasetPolisher(
        modalities=args.modalities,
        min_f1_per_class=args.min_f1_per_class,
        min_macro_f1=args.min_macro_f1,
        min_kappa=args.min_kappa,
        max_iterations=args.max_iterations,
        cv_folds=args.cv_folds,
        n_runs=args.n_runs
    )

    # Run polishing
    success, final_thresholds = polisher.polish()

    # Optionally run full training
    if args.apply:
        print("\n" + "="*70)
        print("APPLYING POLISHED DATASET TO ALL MODALITIES")
        print("="*70)

        if len(args.modalities) > 1:
            polisher.run_full_training(final_thresholds)
        else:
            print("ℹ️  Only metadata specified, no additional training needed")
    else:
        print("\n💡 To train all modalities with polished dataset, re-run with --apply flag")
        print(f"   or manually edit src/main.py to use thresholds: {final_thresholds}")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
