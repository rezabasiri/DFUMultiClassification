#!/usr/bin/env python3
"""
Multi-Parameter Bayesian Search for Optimal Hyperparameters

Optimizes: N_EPOCHS, IMAGE_SIZE, STAGE1_EPOCHS, OUTLIER_CONTAMINATION, OUTLIER_BATCH_SIZE
Runs full 3-fold cross-validation for robust evaluation.

Usage:
    python agent_communication/contamination_search/search_multi_param.py [--n-trials 20]
"""

import sys
import os
import subprocess
import json
from pathlib import Path
import argparse
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)


def extract_kappa_from_csv():
    """
    Extract average validation kappa from results CSV file.

    Main.py runs 3 folds in subprocesses and saves aggregated results to:
    results/csv/modality_results_averaged.csv

    Format:
    Modalities,Accuracy (Mean),Accuracy (Std),...,Cohen's Kappa (Mean),Cohen's Kappa (Std)
    metadata+...,0.5773,0.0,...,0.3365,0.0

    Returns:
        float: Mean Cohen's Kappa across 3 folds (None if not found)
    """
    csv_file = project_root / 'results/csv/modality_results_averaged.csv'

    if not csv_file.exists():
        return None

    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None

            # Parse header to find kappa column index
            header = lines[0].strip().split(',')
            kappa_idx = None
            for i, col in enumerate(header):
                if "Cohen's Kappa (Mean)" in col:
                    kappa_idx = i
                    break

            if kappa_idx is None:
                return None

            # Get last result (latest run)
            last_line = lines[-1].strip().split(',')
            if len(last_line) > kappa_idx:
                kappa = float(last_line[kappa_idx])
                return kappa
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None

    return None


def run_training(n_epochs, image_size, stage1_epochs, contamination, batch_size, trial_number):
    """
    Run full 3-fold training with specific hyperparameters.

    Args:
        n_epochs: Total training epochs
        image_size: Image dimensions (32-256)
        stage1_epochs: Stage 1 fusion epochs
        contamination: Outlier contamination (0.05-0.40)
        batch_size: Outlier detection batch size
        trial_number: Current trial number (for saving results)

    Returns:
        float: Average validation kappa across 3 folds (None if failed)
    """
    # Temporarily modify production_config.py
    config_file = project_root / "src/utils/production_config.py"
    config_backup = project_root / "src/utils/production_config.py.backup_multi"

    # Backup original config (only once)
    if not config_backup.exists():
        import shutil
        shutil.copy(config_file, config_backup)

    # Read config
    with open(config_file, 'r') as f:
        lines = f.readlines()

    # Modify all target parameters
    modified_lines = []
    for line in lines:
        if line.strip().startswith('N_EPOCHS ='):
            modified_lines.append(f'N_EPOCHS = {n_epochs}  # TEMPORARY: Multi-param search trial {trial_number}\n')
        elif line.strip().startswith('IMAGE_SIZE ='):
            modified_lines.append(f'IMAGE_SIZE = {image_size}  # TEMPORARY: Multi-param search trial {trial_number}\n')
        elif line.strip().startswith('STAGE1_EPOCHS ='):
            modified_lines.append(f'STAGE1_EPOCHS = {stage1_epochs}  # TEMPORARY: Multi-param search trial {trial_number}\n')
        elif line.strip().startswith('OUTLIER_CONTAMINATION ='):
            modified_lines.append(f'OUTLIER_CONTAMINATION = {contamination}  # TEMPORARY: Multi-param search trial {trial_number}\n')
        elif line.strip().startswith('OUTLIER_BATCH_SIZE ='):
            modified_lines.append(f'OUTLIER_BATCH_SIZE = {batch_size}  # TEMPORARY: Multi-param search trial {trial_number}\n')
        else:
            modified_lines.append(line)

    # Write modified config
    with open(config_file, 'w') as f:
        f.writelines(modified_lines)

    try:
        # Run training for ALL 3 folds (no --fold argument = run all folds in subprocesses)
        log_dir = project_root / "agent_communication/contamination_search/logs_multi"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create descriptive log filename
        log_file = log_dir / f"ep{n_epochs}_img{image_size}_s1_{stage1_epochs}_cont{contamination:.2f}_bs{batch_size}.log"

        cmd = [
            "python", "src/main.py",
            "--data_percentage", "100",
            "--device-mode", "multi",
            "--verbosity", "1",
            "--resume_mode", "fresh",
            # NO --fold argument = runs all 3 folds in subprocesses with isolation
        ]

        print(f"\n{'='*80}")
        print(f"Running: epochs={n_epochs}, img_size={image_size}, stage1={stage1_epochs}, cont={contamination:.2f}, batch={batch_size}")
        print(f"Running all 3 folds in subprocesses (may take ~30-90 min depending on config)")
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

        # Extract average kappa from CSV (main.py saves aggregated results there)
        kappa = extract_kappa_from_csv()

        if kappa is None:
            print(f"Could not extract kappa from results/csv/modality_results_averaged.csv")
            print(f"Check log file: {log_file}")
            return None

        print(f"✓ Average Kappa (3 folds): {kappa:.4f}")

        # IMPORTANT: Save this trial's CSV result before next trial overwrites it
        # Copy to trial-specific file for record-keeping
        csv_source = project_root / 'results/csv/modality_results_averaged.csv'
        csv_backup = log_dir / f"trial_{trial_number}_results.csv"
        if csv_source.exists():
            import shutil
            shutil.copy(csv_source, csv_backup)
            print(f"  Saved trial results to: {csv_backup.name}")

        return kappa

    finally:
        # Restore original config
        if config_backup.exists():
            import shutil
            shutil.copy(config_backup, config_file)


def objective(trial):
    """Optuna objective function for multi-parameter optimization."""

    # Suggest hyperparameters
    # Based on contamination search: best was 0.29, but expanding range for joint optimization
    contamination = trial.suggest_float('contamination', 0.05, 0.40, step=0.01)  # Rounded to 2 decimals

    # Image size: powers of 2 or common sizes (32, 64, 96, 128, 160, 192, 224, 256)
    image_size = trial.suggest_categorical('image_size', [32, 64, 96, 128, 160, 192, 224, 256])

    # N_EPOCHS: reasonable range for full training
    n_epochs = trial.suggest_int('n_epochs', 50, 200, step=10)

    # STAGE1_EPOCHS: typically 5-20% of N_EPOCHS
    # Use relative percentage to ensure valid relationship
    stage1_ratio = trial.suggest_float('stage1_ratio', 0.05, 0.20)
    stage1_epochs = max(5, int(n_epochs * stage1_ratio))  # Minimum 5 epochs

    # OUTLIER_BATCH_SIZE: powers of 2 for GPU efficiency
    batch_size = trial.suggest_categorical('outlier_batch_size', [16, 32, 64, 128])

    # Run training with these parameters
    kappa = run_training(
        n_epochs=n_epochs,
        image_size=image_size,
        stage1_epochs=stage1_epochs,
        contamination=contamination,
        batch_size=batch_size,
        trial_number=trial.number
    )

    if kappa is None:
        # Training failed, return worst possible score
        return -1.0

    return kappa


def save_incremental_results(study, results_file):
    """Save current state of optimization after each trial."""
    # Reconstruct stage1_epochs for each trial for clearer reporting
    all_trials_data = []
    for trial in study.trials:
        trial_data = {
            'n_epochs': trial.params.get('n_epochs'),
            'image_size': trial.params.get('image_size'),
            'stage1_ratio': trial.params.get('stage1_ratio'),
            'stage1_epochs': max(5, int(trial.params.get('n_epochs', 0) * trial.params.get('stage1_ratio', 0.1))),
            'contamination': trial.params.get('contamination'),
            'outlier_batch_size': trial.params.get('outlier_batch_size'),
            'kappa': trial.value,
            'trial_number': trial.number,
            'state': trial.state.name  # COMPLETE, RUNNING, PRUNED, FAIL
        }
        all_trials_data.append(trial_data)

    # Best trial data (only if we have completed trials)
    completed_trials = [t for t in study.trials if t.value is not None]
    if completed_trials:
        best_trial = study.best_trial
        best_params = {
            'n_epochs': best_trial.params['n_epochs'],
            'image_size': best_trial.params['image_size'],
            'stage1_epochs': max(5, int(best_trial.params['n_epochs'] * best_trial.params['stage1_ratio'])),
            'contamination': best_trial.params['contamination'],
            'outlier_batch_size': best_trial.params['outlier_batch_size']
        }
        best_kappa = study.best_value
    else:
        best_params = None
        best_kappa = None

    results = {
        'best_params': best_params,
        'best_kappa': best_kappa,
        'n_trials_completed': len(completed_trials),
        'n_trials_total': len(study.trials),
        'all_trials': all_trials_data
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Multi-parameter Bayesian search')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of trials to run (default: 20)')
    parser.add_argument('--study-name', type=str, default='multi_param_search',
                        help='Study name for Optuna (default: multi_param_search)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh search (delete existing study database)')
    args = parser.parse_args()

    # Setup results directory and files
    results_dir = project_root / "agent_communication/contamination_search/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "multi_param_search_results.json"

    # Optuna database for persistence (SQLite)
    db_file = results_dir / "optuna_study.db"
    storage = f"sqlite:///{db_file}"

    # Handle fresh start
    # IMPORTANT: Only deletes multi_param_search files, NOT contamination search (search_results.json)
    if args.fresh:
        deleted_files = []
        if db_file.exists():
            db_file.unlink()
            deleted_files.append(db_file.name)
        if results_file.exists():
            results_file.unlink()
            deleted_files.append(results_file.name)

        # Also clean up any journal files
        journal_file = Path(str(db_file) + "-journal")
        if journal_file.exists():
            journal_file.unlink()
            deleted_files.append(journal_file.name)

        if deleted_files:
            print(f"🧹 Fresh start: Deleted multi-param search files:")
            for f in deleted_files:
                print(f"   ✓ {f}")
            print(f"   ℹ️  Contamination search results (search_results.json) preserved")
            print()

    # Create or load study (with persistent storage)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,  # Resume if study exists
        direction='maximize',  # Maximize average kappa
        sampler=TPESampler(seed=42)
    )

    # Check if resuming
    n_completed_before = len([t for t in study.trials if t.value is not None])
    if n_completed_before > 0:
        print(f"📂 Resuming existing study with {n_completed_before} completed trials")
        print(f"   Database: {db_file}")
        print(f"   Best kappa so far: {study.best_value:.4f}")
        print()

    print("="*80)
    print("MULTI-PARAMETER BAYESIAN SEARCH")
    print("="*80)
    print("Optimizing: N_EPOCHS, IMAGE_SIZE, STAGE1_EPOCHS, OUTLIER_CONTAMINATION, OUTLIER_BATCH_SIZE")
    print(f"Number of trials: {args.n_trials}")
    print("Full 3-fold cross-validation per trial")
    print("")
    print("Parameter ranges:")
    print("  - N_EPOCHS: 50-200 (step: 10)")
    print("  - IMAGE_SIZE: [32, 64, 96, 128, 160, 192, 224, 256]")
    print("  - STAGE1_EPOCHS: 5-20% of N_EPOCHS")
    print("  - OUTLIER_CONTAMINATION: 0.05-0.40 (step: 0.01, centered on 0.29 from previous search)")
    print("  - OUTLIER_BATCH_SIZE: [16, 32, 64, 128]")
    print("="*80)
    trials_remaining = args.n_trials - n_completed_before
    if trials_remaining <= 0:
        print(f"All {args.n_trials} trials already completed!")
        print("Use --fresh to start a new search, or increase --n-trials to continue.")
        print("="*80)
        return
    print(f"Trials remaining: {trials_remaining}/{args.n_trials}")
    print(f"Estimated time: ~{trials_remaining * 45 // 60} hours ({trials_remaining} trials × ~45 min/trial)")
    print("="*80)

    # Create callback for incremental saving
    def save_callback(study, trial):
        """Called after each trial completes."""
        save_incremental_results(study, results_file)
        print(f"💾 Progress saved to: {results_file.name}")
        print(f"   Completed trials: {len([t for t in study.trials if t.value is not None])}/{args.n_trials}")
        if study.best_value is not None:
            print(f"   Best kappa so far: {study.best_value:.4f}")

    # Run optimization with incremental saving
    study.optimize(objective, n_trials=trials_remaining, callbacks=[save_callback])

    # Final save
    save_incremental_results(study, results_file)

    print("\n" + "="*80)
    print("SEARCH COMPLETE")
    print("="*80)

    if study.best_trial is not None:
        best_trial = study.best_trial
        best_params = {
            'n_epochs': best_trial.params['n_epochs'],
            'image_size': best_trial.params['image_size'],
            'stage1_epochs': max(5, int(best_trial.params['n_epochs'] * best_trial.params['stage1_ratio'])),
            'contamination': best_trial.params['contamination'],
            'outlier_batch_size': best_trial.params['outlier_batch_size']
        }

        print("Best hyperparameters:")
        print(f"  N_EPOCHS: {best_params['n_epochs']}")
        print(f"  IMAGE_SIZE: {best_params['image_size']}")
        print(f"  STAGE1_EPOCHS: {best_params['stage1_epochs']}")
        print(f"  OUTLIER_CONTAMINATION: {best_params['contamination']:.2f}")
        print(f"  OUTLIER_BATCH_SIZE: {best_params['outlier_batch_size']}")
        print(f"\nBest average kappa (3 folds): {study.best_value:.4f}")
    else:
        print("No completed trials yet.")

    print(f"\nResults saved to: {results_file}")
    print(f"Study database: {db_file}")
    print("="*80)

    # Restore original config
    config_file = project_root / "src/utils/production_config.py"
    config_backup = project_root / "src/utils/production_config.py.backup_multi"
    if config_backup.exists():
        import shutil
        shutil.copy(config_backup, config_file)
        config_backup.unlink()
        print(f"✓ Restored original production_config.py")


if __name__ == '__main__':
    main()
