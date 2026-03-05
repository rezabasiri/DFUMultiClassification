#!/usr/bin/env python3
"""
Bayesian Search for Optimal OUTLIER_CONTAMINATION Value

Searches contamination values from 0.0 to 0.4 using Bayesian optimization.
Runs single-fold training to minimize computation time.

Usage:
    python agent_communication/contamination_search/search_contamination.py [--n-trials 15]
"""

import sys
import os
import subprocess
import json
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)


def extract_kappa_from_log(log_file):
    """Extract final validation kappa from training log."""
    if not Path(log_file).exists():
        return None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Look for the final kappa score in the log
    # Format: "Fold 0 - Final Validation Kappa: 0.XXXX"
    for line in reversed(lines):
        if 'Final Validation Kappa' in line or 'Kappa:' in line:
            try:
                # Extract number from line
                parts = line.split(':')[-1].strip()
                kappa = float(parts.split()[0])
                return kappa
            except:
                continue

    return None


def run_training(contamination, fold_num=1, data_percentage=100):
    """
    Run training with specific contamination value.

    Args:
        contamination: Outlier contamination value (0.0-0.4)
        fold_num: Which fold to run (1-indexed, must match CV_N_SPLITS in config)
        data_percentage: Percentage of data to use

    Returns:
        float: Validation kappa score (None if failed)
    """
    # Temporarily modify production_config.py
    config_file = project_root / "src/utils/production_config.py"
    config_backup = project_root / "src/utils/production_config.py.backup_search"

    # Backup original config
    if not config_backup.exists():
        import shutil
        shutil.copy(config_file, config_backup)

    # Read config
    with open(config_file, 'r') as f:
        lines = f.readlines()

    # Modify OUTLIER_CONTAMINATION and CV_N_SPLITS lines
    modified_lines = []
    for line in lines:
        if line.strip().startswith('OUTLIER_CONTAMINATION ='):
            modified_lines.append(f'OUTLIER_CONTAMINATION = {contamination}  # TEMPORARY: Bayesian search value\n')
        elif line.strip().startswith('CV_N_SPLITS ='):
            # Set to 1 to run single fold only (makes --fold 1 run just one fold)
            modified_lines.append(f'CV_N_SPLITS = 1  # TEMPORARY: Single fold for search\n')
        else:
            modified_lines.append(line)

    # Write modified config
    with open(config_file, 'w') as f:
        f.writelines(modified_lines)

    try:
        # Run training for single fold
        log_dir = project_root / "agent_communication/contamination_search/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"contamination_{contamination:.3f}.log"

        cmd = [
            "python", "src/main.py",
            "--data_percentage", str(data_percentage),
            "--device-mode", "multi",
            "--verbosity", "1",
            "--resume_mode", "fresh",
            "--fold", str(fold_num),  # Run specific fold (1-indexed)
        ]

        print(f"\n{'='*80}")
        print(f"Running: contamination={contamination:.3f}")
        print(f"{'='*80}")

        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            return None

        # Extract kappa from log
        kappa = extract_kappa_from_log(log_file)

        if kappa is None:
            print(f"Could not extract kappa from log: {log_file}")
            return None

        print(f"✓ Kappa: {kappa:.4f}")
        return kappa

    finally:
        # Restore original config
        if config_backup.exists():
            import shutil
            shutil.copy(config_backup, config_file)


def objective(trial):
    """Optuna objective function."""
    # Suggest contamination value
    contamination = trial.suggest_float('contamination', 0.0, 0.4)

    # Run training
    kappa = run_training(
        contamination=contamination,
        fold_num=1,  # Use fold 1 (1-indexed, with CV_N_SPLITS=1 this runs just one fold)
        data_percentage=100
    )

    if kappa is None:
        # Training failed, return worst possible score
        return -1.0

    return kappa


def main():
    parser = argparse.ArgumentParser(description='Bayesian search for optimal contamination')
    parser.add_argument('--n-trials', type=int, default=15,
                        help='Number of trials to run (default: 15)')
    parser.add_argument('--study-name', type=str, default='contamination_search',
                        help='Study name for Optuna (default: contamination_search)')
    args = parser.parse_args()

    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # Maximize kappa
        sampler=TPESampler(seed=42)
    )

    print("="*80)
    print("BAYESIAN SEARCH FOR OPTIMAL OUTLIER_CONTAMINATION")
    print("="*80)
    print(f"Search range: [0.0, 0.4]")
    print(f"Number of trials: {args.n_trials}")
    print(f"Single fold mode: CV_N_SPLITS temporarily set to 1")
    print("="*80)

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Save results
    results_dir = project_root / "agent_communication/contamination_search/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save study results
    results_file = results_dir / "search_results.json"
    results = {
        'best_contamination': study.best_params['contamination'],
        'best_kappa': study.best_value,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'contamination': trial.params['contamination'],
                'kappa': trial.value,
                'trial_number': trial.number
            }
            for trial in study.trials
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("SEARCH COMPLETE")
    print("="*80)
    print(f"Best contamination: {study.best_params['contamination']:.4f}")
    print(f"Best kappa: {study.best_value:.4f}")
    print(f"Results saved to: {results_file}")
    print("="*80)

    # Restore original config
    config_file = project_root / "src/utils/production_config.py"
    config_backup = project_root / "src/utils/production_config.py.backup_search"
    if config_backup.exists():
        import shutil
        shutil.copy(config_backup, config_file)
        config_backup.unlink()
        print(f"✓ Restored original production_config.py")


if __name__ == '__main__':
    main()
