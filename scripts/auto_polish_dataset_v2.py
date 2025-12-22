"""
Two-Phase Intelligent Dataset Polishing via Bayesian Threshold Optimization

This script uses a smart two-phase approach to find optimal misclassification
filtering thresholds:

PHASE 1: Misclassification Detection (Run Once)
- Runs training N times (e.g., 10) with different random seeds
- Accumulates misclassification counts across runs (max count = N)
- Creates comprehensive misclassification profile
- Time: ~30-60 minutes for N=10 runs

PHASE 2: Bayesian Threshold Optimization
- Uses Bayesian optimization to find optimal thresholds
- Search space: percentage-based (P: 30-70%, I: 50-90%, R: 80-100% of N)
- For each candidate threshold combination:
  * Filters dataset based on thresholds
  * Trains metadata with cv_folds=3, n_runs=1
  * Evaluates combined score: 0.4×macro_f1 + 0.4×min_f1 + 0.2×kappa
- Finds best thresholds in ~20 evaluations
- Time: ~1-2 hours for 20 evaluations with cv_folds=3

Key Advantages:
- Much fewer total runs than iterative approach (30 vs 50+)
- Systematically explores threshold space instead of arbitrary reduction
- Balances speed (Phase 1: n_runs=10) with precision (Phase 2: Bayesian)
- Safety constraint: rejects thresholds that filter >50% of data

Usage:
    # Run both phases automatically
    python scripts/auto_polish_dataset_v2.py --modalities metadata depth_rgb depth_map

    # Just Phase 1 (detection only)
    python scripts/auto_polish_dataset_v2.py --modalities metadata --phase1-only

    # Just Phase 2 (if Phase 1 already completed)
    python scripts/auto_polish_dataset_v2.py --modalities metadata --phase2-only

    # Custom optimization budget
    python scripts/auto_polish_dataset_v2.py --modalities metadata --n_evaluations 30
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


class BayesianDatasetPolisher:
    """Two-phase dataset polishing with Bayesian threshold optimization."""

    def __init__(self,
                 modalities,
                 min_f1_per_class=0.42,
                 min_macro_f1=0.48,
                 min_kappa=0.35,
                 phase1_n_runs=10,
                 phase1_cv_folds=1,
                 phase2_cv_folds=3,
                 phase2_n_evaluations=20,
                 base_random_seed=42,
                 min_dataset_fraction=0.5):
        """
        Initialize the Bayesian dataset polisher.

        Args:
            modalities: List of modalities to train (must include 'metadata')
            min_f1_per_class: Minimum F1 score for each class
            min_macro_f1: Minimum macro F1 score
            min_kappa: Minimum Cohen's Kappa
            phase1_n_runs: Number of runs in Phase 1 for misclass detection (default: 10)
            phase1_cv_folds: CV folds in Phase 1 (default: 1 for speed)
            phase2_cv_folds: CV folds in Phase 2 for evaluation (default: 3)
            phase2_n_evaluations: Number of Bayesian optimization iterations (default: 20)
            base_random_seed: Base random seed
            min_dataset_fraction: Minimum fraction of dataset to keep (default: 0.5)
        """
        if 'metadata' not in modalities:
            raise ValueError("Modalities must include 'metadata' for polishing")

        self.modalities = modalities
        self.min_f1_per_class = min_f1_per_class
        self.min_macro_f1 = min_macro_f1
        self.min_kappa = min_kappa
        self.phase1_n_runs = phase1_n_runs
        self.phase1_cv_folds = phase1_cv_folds
        self.phase2_cv_folds = phase2_cv_folds
        self.phase2_n_evaluations = phase2_n_evaluations
        self.base_random_seed = base_random_seed
        self.min_dataset_fraction = min_dataset_fraction

        # Get project paths
        self.directory, self.result_dir, self.root = get_project_paths()

        # Tracking
        self.original_dataset_size = None
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf

    def calculate_combined_score(self, metrics):
        """
        Calculate combined optimization score.

        Formula: 0.4×macro_f1 + 0.4×min_per_class_f1 + 0.2×kappa

        This balances:
        - Overall performance (macro F1)
        - Worst-class performance (min per-class F1)
        - Clinical agreement (Cohen's Kappa)
        """
        macro_f1 = metrics['macro_f1']
        min_f1 = min(metrics['f1_per_class'].values())
        kappa = metrics['kappa']

        score = 0.4 * macro_f1 + 0.4 * min_f1 + 0.2 * kappa
        return score

    def get_original_dataset_size(self):
        """Get original dataset size before any filtering."""
        # Read from data directory
        data_path = os.path.join(self.directory, 'balanced_combined_healing_phases.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return len(df)
        return None

    def get_filtered_dataset_size(self, thresholds):
        """
        Estimate how many samples would remain after filtering.

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}

        Returns:
            Number of samples that would remain
        """
        misclass_file = os.path.join(self.result_dir, 'frequent_misclassifications_total.csv')

        if not os.path.exists(misclass_file):
            print(f"⚠️  Misclassification file not found: {misclass_file}")
            return self.original_dataset_size

        df = pd.read_csv(misclass_file)

        # Count samples that would be excluded
        excluded_samples = set()
        for phase, threshold in thresholds.items():
            high_misclass = df[
                (df['True_Label'] == phase) &
                (df['Misclass_Count'] >= threshold)
            ]['Sample_ID'].tolist()
            excluded_samples.update(high_misclass)

        excluded_count = len(excluded_samples)
        remaining_count = self.original_dataset_size - excluded_count

        return remaining_count

    def phase1_detect_misclassifications(self):
        """
        Phase 1: Run training multiple times to accumulate misclassifications.

        Returns:
            bool: Success status
        """
        print("\n" + "="*70)
        print("PHASE 1: MISCLASSIFICATION DETECTION")
        print("="*70)
        print(f"\nRunning {self.phase1_n_runs} training runs with different random seeds")
        print(f"CV folds: {self.phase1_cv_folds} (fast detection)")
        print(f"Seeds: {[self.base_random_seed + i for i in range(1, self.phase1_n_runs + 1)]}")
        print(f"\nThis will create a comprehensive misclassification profile")
        print(f"Max misclassification count: {self.phase1_n_runs}\n")

        # Get original dataset size
        self.original_dataset_size = self.get_original_dataset_size()
        if self.original_dataset_size:
            print(f"Original dataset size: {self.original_dataset_size} samples")
            min_samples = int(self.original_dataset_size * self.min_dataset_fraction)
            print(f"Minimum dataset size after filtering: {min_samples} samples ({self.min_dataset_fraction*100:.0f}%)\n")

        # Clean up everything for fresh start
        cleanup_for_resume_mode('fresh')

        # Temporarily override INCLUDED_COMBINATIONS
        config_path = project_root / 'src' / 'utils' / 'production_config.py'
        backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

        with open(config_path, 'r') as f:
            original_config = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_config)

        try:
            import re
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
                "INCLUDED_COMBINATIONS = [\n    ('metadata',),  # Temporary: Phase 1 detection\n]",
                original_config
            )
            with open(config_path, 'w') as f:
                f.write(modified_config)

            # Run multiple times with different seeds
            for run_idx in range(1, self.phase1_n_runs + 1):
                print(f"\n{'─'*70}")
                print(f"RUN {run_idx}/{self.phase1_n_runs} (seed={self.base_random_seed + run_idx})")
                print(f"{'─'*70}")

                # Clean up predictions/models/patient splits from previous run
                # We MUST delete patient splits to force regeneration with new random seed
                if run_idx > 1:
                    import glob
                    from src.utils.config import get_output_paths
                    output_paths = get_output_paths(self.result_dir)

                    # Delete everything except misclassification CSV (which accumulates)
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

                # Set random seed via environment variable
                os.environ['CROSS_VAL_RANDOM_SEED'] = str(self.base_random_seed + run_idx)

                cmd = [
                    'python', 'src/main.py',
                    '--mode', 'search',
                    '--cv_folds', str(self.phase1_cv_folds),
                    '--verbosity', '1'
                ]

                if run_idx == 1:
                    print(f"⏳ Training run {run_idx}...")
                else:
                    print(f"⏳ Accumulating misclassifications (run {run_idx})...")

                result = subprocess.run(cmd, cwd=project_root, env=os.environ.copy())

                if result.returncode != 0:
                    print(f"\n❌ Training failed on run {run_idx}")
                    return False

            # Clear environment variable
            if 'CROSS_VAL_RANDOM_SEED' in os.environ:
                del os.environ['CROSS_VAL_RANDOM_SEED']

            print(f"\n{'='*70}")
            print(f"✅ PHASE 1 COMPLETE")
            print(f"{'='*70}")

            # Analyze misclassifications
            self.show_misclass_summary()

            return True

        finally:
            # Restore config
            with open(backup_path, 'r') as f:
                original_config = f.read()
            with open(config_path, 'w') as f:
                f.write(original_config)
            backup_path.unlink(missing_ok=True)

    def show_misclass_summary(self):
        """Show summary of accumulated misclassifications."""
        misclass_file = os.path.join(self.result_dir, 'frequent_misclassifications_total.csv')

        if not os.path.exists(misclass_file):
            print("\n⚠️  No misclassification file found")
            return

        df = pd.read_csv(misclass_file)

        print(f"\nMisclassification Summary:")
        print(f"  Total unique misclassified samples: {len(df)}")
        print(f"  Max count observed: {df['Misclass_Count'].max()}")

        print(f"\n  By class:")
        for phase in ['I', 'P', 'R']:
            phase_df = df[df['True_Label'] == phase]
            if len(phase_df) > 0:
                print(f"    {phase}: {len(phase_df)} samples, " +
                      f"counts {phase_df['Misclass_Count'].min()}-{phase_df['Misclass_Count'].max()}, " +
                      f"median={phase_df['Misclass_Count'].median():.1f}")

    def phase2_optimize_thresholds(self):
        """
        Phase 2: Use Bayesian optimization to find optimal thresholds.

        Returns:
            tuple: (success, best_thresholds)
        """
        print("\n" + "="*70)
        print("PHASE 2: BAYESIAN THRESHOLD OPTIMIZATION")
        print("="*70)

        # Check if scikit-optimize is available
        try:
            from skopt import gp_minimize
            from skopt.space import Integer
            from skopt.utils import use_named_args
        except ImportError:
            print("\n❌ scikit-optimize not found. Install with: pip install scikit-optimize")
            print("   Falling back to grid search...")
            return self.phase2_grid_search()

        # Define search space (percentage-based)
        # P: 30-70%, I: 50-90%, R: 80-100% of phase1_n_runs
        search_space = [
            Integer(int(0.30 * self.phase1_n_runs), int(0.70 * self.phase1_n_runs), name='threshold_P'),
            Integer(int(0.50 * self.phase1_n_runs), int(0.90 * self.phase1_n_runs), name='threshold_I'),
            Integer(int(0.80 * self.phase1_n_runs), int(1.00 * self.phase1_n_runs), name='threshold_R')
        ]

        print(f"\nSearch Space:")
        print(f"  P (dominant): {search_space[0].low}-{search_space[0].high} " +
              f"({search_space[0].low/self.phase1_n_runs*100:.0f}%-{search_space[0].high/self.phase1_n_runs*100:.0f}%)")
        print(f"  I (minority): {search_space[1].low}-{search_space[1].high} " +
              f"({search_space[1].low/self.phase1_n_runs*100:.0f}%-{search_space[1].high/self.phase1_n_runs*100:.0f}%)")
        print(f"  R (rarest):   {search_space[2].low}-{search_space[2].high} " +
              f"({search_space[2].low/self.phase1_n_runs*100:.0f}%-{search_space[2].high/self.phase1_n_runs*100:.0f}%)")

        print(f"\nOptimization Settings:")
        print(f"  Evaluations: {self.phase2_n_evaluations}")
        print(f"  CV folds per evaluation: {self.phase2_cv_folds}")
        print(f"  Score: 0.4×macro_f1 + 0.4×min_f1 + 0.2×kappa")
        print(f"  Constraint: Keep ≥{self.min_dataset_fraction*100:.0f}% of original dataset\n")

        # Objective function
        @use_named_args(search_space)
        def objective(**thresholds):
            """Objective function for Bayesian optimization (to minimize, so we negate)."""
            eval_num = len(self.optimization_history) + 1

            print(f"\n{'─'*70}")
            print(f"EVALUATION {eval_num}/{self.phase2_n_evaluations}")
            print(f"{'─'*70}")
            print(f"Trying thresholds: P={thresholds['threshold_P']}, " +
                  f"I={thresholds['threshold_I']}, R={thresholds['threshold_R']}")

            # Check dataset size constraint
            threshold_dict = {
                'P': thresholds['threshold_P'],
                'I': thresholds['threshold_I'],
                'R': thresholds['threshold_R']
            }

            filtered_size = self.get_filtered_dataset_size(threshold_dict)
            min_size = int(self.original_dataset_size * self.min_dataset_fraction)

            print(f"Dataset after filtering: {filtered_size}/{self.original_dataset_size} samples " +
                  f"({filtered_size/self.original_dataset_size*100:.1f}%)")

            if filtered_size < min_size:
                print(f"❌ Rejected: Below minimum size ({min_size})")
                penalty_score = -10.0  # Large penalty
                self.optimization_history.append({
                    'evaluation': eval_num,
                    'thresholds': threshold_dict,
                    'score': penalty_score,
                    'filtered_size': filtered_size,
                    'rejected': True
                })
                return -penalty_score  # Return positive for minimization

            # Train and evaluate
            metrics = self.train_with_thresholds(threshold_dict)

            if metrics is None:
                print(f"❌ Training failed")
                penalty_score = -10.0
                self.optimization_history.append({
                    'evaluation': eval_num,
                    'thresholds': threshold_dict,
                    'score': penalty_score,
                    'filtered_size': filtered_size,
                    'rejected': True,
                    'failed': True
                })
                return -penalty_score

            # Calculate score
            score = self.calculate_combined_score(metrics)

            print(f"Results:")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Min F1: {min(metrics['f1_per_class'].values()):.4f}")
            print(f"  Kappa: {metrics['kappa']:.4f}")
            print(f"  Combined Score: {score:.4f}")

            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_thresholds = threshold_dict
                print(f"  ✨ New best score!")

            self.optimization_history.append({
                'evaluation': eval_num,
                'thresholds': threshold_dict,
                'metrics': metrics,
                'score': score,
                'filtered_size': filtered_size,
                'rejected': False
            })

            # Return negative for minimization
            return -score

        # Run Bayesian optimization
        print(f"\n🔍 Starting Bayesian optimization...\n")

        result = gp_minimize(
            objective,
            search_space,
            n_calls=self.phase2_n_evaluations,
            random_state=self.base_random_seed,
            verbose=False
        )

        print(f"\n{'='*70}")
        print(f"✅ PHASE 2 COMPLETE")
        print(f"{'='*70}")
        print(f"\nBest thresholds found: {self.best_thresholds}")
        print(f"Best score: {self.best_score:.4f}")

        return True, self.best_thresholds

    def phase2_grid_search(self):
        """Fallback grid search if Bayesian optimization not available."""
        print("\n🔍 Using grid search (fallback)...")
        print("⚠️  This will be slower than Bayesian optimization")

        # Define coarse grid
        p_range = range(int(0.30 * self.phase1_n_runs), int(0.70 * self.phase1_n_runs) + 1, 2)
        i_range = range(int(0.50 * self.phase1_n_runs), int(0.90 * self.phase1_n_runs) + 1, 2)
        r_range = range(int(0.80 * self.phase1_n_runs), int(1.00 * self.phase1_n_runs) + 1)

        from itertools import product
        grid = list(product(p_range, i_range, r_range))

        # Limit to n_evaluations
        if len(grid) > self.phase2_n_evaluations:
            import random
            random.seed(self.base_random_seed)
            grid = random.sample(grid, self.phase2_n_evaluations)

        print(f"Testing {len(grid)} threshold combinations...\n")

        for eval_num, (p, i, r) in enumerate(grid, 1):
            threshold_dict = {'P': p, 'I': i, 'R': r}

            print(f"\nEvaluation {eval_num}/{len(grid)}")
            print(f"Thresholds: {threshold_dict}")

            # Check size constraint
            filtered_size = self.get_filtered_dataset_size(threshold_dict)
            min_size = int(self.original_dataset_size * self.min_dataset_fraction)

            if filtered_size < min_size:
                print(f"❌ Rejected: Dataset too small ({filtered_size} < {min_size})")
                continue

            # Train and evaluate
            metrics = self.train_with_thresholds(threshold_dict)
            if metrics is None:
                continue

            score = self.calculate_combined_score(metrics)
            print(f"Score: {score:.4f}")

            if score > self.best_score:
                self.best_score = score
                self.best_thresholds = threshold_dict
                print(f"✨ New best!")

            self.optimization_history.append({
                'evaluation': eval_num,
                'thresholds': threshold_dict,
                'metrics': metrics,
                'score': score,
                'filtered_size': filtered_size
            })

        return True, self.best_thresholds

    def train_with_thresholds(self, thresholds):
        """
        Train metadata with given thresholds and return metrics.

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}

        Returns:
            dict: Metrics or None if training failed
        """
        # Clean up from previous evaluation
        cleanup_for_resume_mode('from_data')

        # Temporarily override config
        config_path = project_root / 'src' / 'utils' / 'production_config.py'
        backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

        with open(config_path, 'r') as f:
            original_config = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_config)

        try:
            import re
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
                "INCLUDED_COMBINATIONS = [\n    ('metadata',),  # Temporary: Phase 2 evaluation\n]",
                original_config
            )
            with open(config_path, 'w') as f:
                f.write(modified_config)

            # Run training with thresholds
            cmd = [
                'python', 'src/main.py',
                '--mode', 'search',
                '--cv_folds', str(self.phase2_cv_folds),
                '--verbosity', '0',  # Minimal output during optimization
                '--threshold_I', str(thresholds['I']),
                '--threshold_P', str(thresholds['P']),
                '--threshold_R', str(thresholds['R'])
            ]

            print(f"⏳ Training with cv_folds={self.phase2_cv_folds}...")
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

            if result.returncode != 0:
                return None

            # Extract metrics from CSV
            return self.extract_metrics_from_files()

        finally:
            # Restore config
            with open(backup_path, 'r') as f:
                original_config = f.read()
            with open(config_path, 'w') as f:
                f.write(original_config)
            backup_path.unlink(missing_ok=True)

    def extract_metrics_from_files(self):
        """Extract performance metrics from CSV files."""
        metrics = {
            'f1_per_class': {'I': 0.0, 'P': 0.0, 'R': 0.0},
            'macro_f1': 0.0,
            'kappa': 0.0,
            'accuracy': 0.0
        }

        csv_file = os.path.join(self.result_dir, 'csv', 'modality_results_averaged.csv')
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            return None

        try:
            df = pd.read_csv(csv_file)
            metadata_rows = df[df['Modalities'].str.contains('metadata', case=False, na=False)]
            if len(metadata_rows) == 0:
                return None

            row = metadata_rows.iloc[-1]
            metrics['macro_f1'] = float(row.get('Macro Avg F1-score (Mean)', 0.0))
            metrics['kappa'] = float(row.get("Cohen's Kappa (Mean)", 0.0))
            metrics['accuracy'] = float(row.get('Accuracy (Mean)', 0.0))

            for i, cls in enumerate(['I', 'P', 'R']):
                col_name = f'Class {i} F1-score (Mean)'
                if col_name in row:
                    metrics['f1_per_class'][cls] = float(row[col_name])

            return metrics

        except Exception as e:
            print(f"⚠️  Could not extract metrics: {e}")
            return None

    def save_results(self):
        """Save optimization results to JSON."""
        results_file = os.path.join(self.result_dir, 'bayesian_optimization_results.json')

        results = {
            'timestamp': datetime.now().isoformat(),
            'phase1_n_runs': self.phase1_n_runs,
            'phase2_n_evaluations': self.phase2_n_evaluations,
            'best_thresholds': self.best_thresholds,
            'best_score': float(self.best_score),
            'optimization_history': self.optimization_history,
            'original_dataset_size': self.original_dataset_size
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Two-phase Bayesian dataset polishing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both phases
  python scripts/auto_polish_dataset_v2.py --modalities metadata depth_rgb depth_map

  # Just Phase 1 (detection)
  python scripts/auto_polish_dataset_v2.py --modalities metadata --phase1-only

  # Just Phase 2 (if Phase 1 already done)
  python scripts/auto_polish_dataset_v2.py --modalities metadata --phase2-only

  # More thorough optimization
  python scripts/auto_polish_dataset_v2.py --modalities metadata --n_evaluations 30
        """
    )

    parser.add_argument('--modalities', nargs='+', required=True,
                        help='Modalities to train (must include metadata)')

    parser.add_argument('--phase1-only', action='store_true',
                        help='Run only Phase 1 (misclassification detection)')

    parser.add_argument('--phase2-only', action='store_true',
                        help='Run only Phase 2 (threshold optimization)')

    parser.add_argument('--phase1-n-runs', type=int, default=10,
                        help='Number of runs in Phase 1 (default: 10)')

    parser.add_argument('--n-evaluations', type=int, default=20,
                        help='Number of Bayesian optimization evaluations (default: 20)')

    parser.add_argument('--min-dataset-fraction', type=float, default=0.5,
                        help='Minimum fraction of dataset to keep (default: 0.5)')

    args = parser.parse_args()

    if 'metadata' not in args.modalities:
        print("❌ Error: --modalities must include 'metadata'")
        sys.exit(1)

    polisher = BayesianDatasetPolisher(
        modalities=args.modalities,
        phase1_n_runs=args.phase1_n_runs,
        phase2_n_evaluations=args.n_evaluations,
        min_dataset_fraction=args.min_dataset_fraction
    )

    # Run phases
    if not args.phase2_only:
        print("\n🚀 Starting Phase 1: Misclassification Detection\n")
        success = polisher.phase1_detect_misclassifications()
        if not success:
            print("\n❌ Phase 1 failed")
            sys.exit(1)

    if not args.phase1_only:
        print("\n🚀 Starting Phase 2: Bayesian Threshold Optimization\n")
        success, best_thresholds = polisher.phase2_optimize_thresholds()
        if not success:
            print("\n❌ Phase 2 failed")
            sys.exit(1)

        polisher.save_results()

        print("\n" + "="*70)
        print("🎉 OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"\nOptimal thresholds: {best_thresholds}")
        print(f"Optimization score: {polisher.best_score:.4f}")
        print(f"\n💡 Use these thresholds for final training:")
        print(f"   python src/main.py --mode search --cv_folds 5 \\")
        print(f"       --threshold_I {best_thresholds['I']} \\")
        print(f"       --threshold_P {best_thresholds['P']} \\")
        print(f"       --threshold_R {best_thresholds['R']}")

    sys.exit(0)


if __name__ == '__main__':
    main()
