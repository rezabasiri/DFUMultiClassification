#!/usr/bin/env python3
"""
Analyze Bayesian Search Results

Generates plots and statistical summary of contamination search.

Usage:
    python agent_communication/contamination_search/analyze_results.py
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("WARNING: matplotlib not installed. Skipping plots.")
    plt = None


def load_results():
    """Load search results from JSON file."""
    results_file = project_root / "agent_communication/contamination_search/results/search_results.json"

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Run the search first: python agent_communication/contamination_search/search_contamination.py")
        sys.exit(1)

    with open(results_file, 'r') as f:
        return json.load(f)


def plot_optimization_history(results):
    """Plot optimization history (best kappa over trials)."""
    if plt is None:
        return

    trials = results['all_trials']
    trial_numbers = [t['trial_number'] for t in trials]
    kappas = [t['kappa'] for t in trials]

    # Compute running best
    best_kappas = []
    best_so_far = -float('inf')
    for kappa in kappas:
        if kappa > best_so_far:
            best_so_far = kappa
        best_kappas.append(best_so_far)

    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, kappas, 'o-', alpha=0.6, label='Trial Kappa')
    plt.plot(trial_numbers, best_kappas, 'r-', linewidth=2, label='Best Kappa')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Kappa')
    plt.title('Optimization History: Contamination Search')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_file = project_root / "agent_communication/contamination_search/results/optimization_history.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file.name}")
    plt.close()


def plot_contamination_vs_kappa(results):
    """Plot contamination values vs kappa scores."""
    if plt is None:
        return

    trials = results['all_trials']
    contaminations = [t['contamination'] for t in trials]
    kappas = [t['kappa'] for t in trials]

    # Highlight best point
    best_contamination = results['best_contamination']
    best_kappa = results['best_kappa']

    plt.figure(figsize=(10, 6))
    plt.scatter(contaminations, kappas, alpha=0.6, s=100, label='Trials')
    plt.scatter([best_contamination], [best_kappa], color='red', s=200,
                marker='*', label=f'Best ({best_contamination:.3f})', zorder=5)

    # Add baseline reference (current production value)
    plt.axvline(x=0.15, color='green', linestyle='--', alpha=0.5,
                label='Current (0.15)')

    plt.xlabel('Contamination')
    plt.ylabel('Validation Kappa')
    plt.title('Contamination vs Kappa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.02, 0.42)

    output_file = project_root / "agent_communication/contamination_search/results/contamination_vs_kappa.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file.name}")
    plt.close()


def print_summary(results):
    """Print statistical summary."""
    trials = results['all_trials']
    contaminations = [t['contamination'] for t in trials]
    kappas = [t['kappa'] for t in trials]

    print("\n" + "="*80)
    print("CONTAMINATION SEARCH SUMMARY")
    print("="*80)

    print(f"\n📊 Search Statistics:")
    print(f"  Total trials: {len(trials)}")
    print(f"  Contamination range: [{min(contaminations):.3f}, {max(contaminations):.3f}]")
    print(f"  Kappa range: [{min(kappas):.4f}, {max(kappas):.4f}]")
    print(f"  Kappa mean: {np.mean(kappas):.4f}")
    print(f"  Kappa std: {np.std(kappas):.4f}")

    print(f"\n🏆 Best Result:")
    print(f"  Best contamination: {results['best_contamination']:.4f}")
    print(f"  Best kappa: {results['best_kappa']:.4f}")

    # Find current baseline result if tested
    baseline_results = [t for t in trials if abs(t['contamination'] - 0.15) < 0.01]
    if baseline_results:
        baseline_kappa = baseline_results[0]['kappa']
        improvement = results['best_kappa'] - baseline_kappa
        print(f"\n📈 vs Current Baseline (0.15):")
        print(f"  Baseline kappa: {baseline_kappa:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement/baseline_kappa*100:+.1f}%)")

    # Top 5 trials
    sorted_trials = sorted(trials, key=lambda t: t['kappa'], reverse=True)[:5]
    print(f"\n🥇 Top 5 Trials:")
    for i, trial in enumerate(sorted_trials, 1):
        print(f"  {i}. contamination={trial['contamination']:.4f}, kappa={trial['kappa']:.4f}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    best_cont = results['best_contamination']
    current_cont = 0.15

    if abs(best_cont - current_cont) < 0.02:
        print(f"✓ Current value (0.15) is near-optimal")
        print(f"  No change recommended")
    else:
        diff_pct = abs(best_cont - current_cont) / current_cont * 100
        print(f"⚠️  Best value ({best_cont:.3f}) differs from current (0.15) by {diff_pct:.1f}%")
        print(f"  Recommendation: Update production_config.py:")
        print(f"    OUTLIER_CONTAMINATION = {best_cont:.3f}")

        if best_cont < 0.10:
            print(f"  Note: Low contamination preserves more samples (less aggressive outlier removal)")
        elif best_cont > 0.25:
            print(f"  Note: High contamination removes more samples (more aggressive outlier removal)")

    print("="*80)


def main():
    print("Loading results...")
    results = load_results()

    print(f"Analyzing {results['n_trials']} trials...")

    # Generate plots
    if plt is not None:
        plot_optimization_history(results)
        plot_contamination_vs_kappa(results)
    else:
        print("Skipping plots (matplotlib not available)")

    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
