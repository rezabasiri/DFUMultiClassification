#!/usr/bin/env python3
"""
Check progress of multi-parameter search.

Shows current status, best parameters so far, and recent trials.

Usage:
    python agent_communication/contamination_search/check_progress.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_progress():
    results_file = project_root / "agent_communication/contamination_search/results/multi_param_search_results.json"
    db_file = project_root / "agent_communication/contamination_search/results/optuna_study.db"

    print("="*80)
    print("MULTI-PARAMETER SEARCH - PROGRESS CHECK")
    print("="*80)
    print()

    # Check if results exist
    if not results_file.exists():
        print("❌ No search results found.")
        print(f"   Expected file: {results_file}")
        print()
        print("Start a new search with:")
        print("  python agent_communication/contamination_search/search_multi_param.py --n-trials 20 --fresh")
        return

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Status
    n_completed = results.get('n_trials_completed', 0)
    n_total = results.get('n_trials_total', 0)
    best_kappa = results.get('best_kappa')

    print(f"Status: {n_completed}/{n_total} trials completed")
    if db_file.exists():
        print(f"Database: {db_file.name} (resume enabled)")
    print()

    # Best parameters so far
    if best_kappa is not None:
        print("Best Parameters So Far:")
        best_params = results['best_params']
        print(f"  N_EPOCHS: {best_params['n_epochs']}")
        print(f"  IMAGE_SIZE: {best_params['image_size']}")
        print(f"  STAGE1_EPOCHS: {best_params['stage1_epochs']}")
        print(f"  OUTLIER_CONTAMINATION: {best_params['contamination']:.2f}")
        print(f"  OUTLIER_BATCH_SIZE: {best_params['outlier_batch_size']}")
        print(f"  Kappa: {best_kappa:.4f}")
        print()
    else:
        print("No completed trials yet.")
        print()

    # Recent trials (last 5)
    all_trials = results.get('all_trials', [])
    if all_trials:
        completed_trials = [t for t in all_trials if t.get('kappa') is not None and t['kappa'] != -1.0]
        if completed_trials:
            print("Recent Trials (last 5):")
            recent = completed_trials[-5:]
            for trial in recent:
                print(f"  Trial {trial['trial_number']}: "
                      f"epochs={trial['n_epochs']}, "
                      f"size={trial['image_size']}, "
                      f"cont={trial['contamination']:.2f} → "
                      f"kappa={trial['kappa']:.4f}")
            print()

    # Next steps
    if n_completed < n_total:
        print(f"To continue search ({n_total - n_completed} trials remaining):")
        print(f"  python agent_communication/contamination_search/search_multi_param.py --n-trials {n_total}")
        print()
        print("To start fresh:")
        print("  python agent_communication/contamination_search/search_multi_param.py --n-trials 20 --fresh")
    else:
        print("✓ Search complete!")
        print()
        print("To extend search (add more trials):")
        print("  python agent_communication/contamination_search/search_multi_param.py --n-trials 30")

    print("="*80)


if __name__ == '__main__':
    check_progress()
