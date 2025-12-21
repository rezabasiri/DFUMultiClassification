"""
Analyze misclassification patterns and suggest optimal per-class thresholds.

This script examines frequent_misclassifications_total.csv and recommends
class-specific thresholds that help reduce data imbalance while preserving
minority classes.

Usage:
    python scripts/suggest_thresholds.py
    python scripts/suggest_thresholds.py --csv results/frequent_misclassifications_total.csv
    python scripts/suggest_thresholds.py --strategy aggressive  # More filtering
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_class_distribution(csv_file):
    """Analyze misclassification patterns by class."""
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("\n💡 Run training first to generate misclassification data:")
        print("   python src/main.py --cv_folds 5 --n_runs 3")
        return None

    df = pd.read_csv(csv_file)

    if len(df) == 0:
        print("❌ Misclassification file is empty")
        return None

    print("="*70)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*70)

    stats = {}
    for phase in ['I', 'P', 'R']:
        phase_df = df[df['True_Label'] == phase]

        if len(phase_df) == 0:
            print(f"\n⚠️  No misclassifications found for class {phase}")
            continue

        stats[phase] = {
            'count': len(phase_df),
            'max': int(phase_df['Misclass_Count'].max()),
            'min': int(phase_df['Misclass_Count'].min()),
            'mean': float(phase_df['Misclass_Count'].mean()),
            'median': float(phase_df['Misclass_Count'].median()),
            'std': float(phase_df['Misclass_Count'].std()),
            'q25': float(phase_df['Misclass_Count'].quantile(0.25)),
            'q50': float(phase_df['Misclass_Count'].quantile(0.50)),
            'q75': float(phase_df['Misclass_Count'].quantile(0.75)),
            'q90': float(phase_df['Misclass_Count'].quantile(0.90)),
        }

        print(f"\n{phase} ({'Inflammatory' if phase == 'I' else 'Proliferative' if phase == 'P' else 'Remodeling'}):")
        print(f"  Total unique misclassified samples: {stats[phase]['count']}")
        print(f"  Misclass count range: {stats[phase]['min']} - {stats[phase]['max']}")
        print(f"  Mean: {stats[phase]['mean']:.2f}, Median: {stats[phase]['median']:.1f}, Std: {stats[phase]['std']:.2f}")
        print(f"  Quartiles: Q25={stats[phase]['q25']:.1f}, Q50={stats[phase]['q50']:.1f}, Q75={stats[phase]['q75']:.1f}, Q90={stats[phase]['q90']:.1f}")

    return stats


def suggest_thresholds(csv_file, strategy='balanced'):
    """
    Suggest class-specific thresholds based on misclassification patterns.

    Strategies:
    - 'conservative': High thresholds, minimal filtering (good for small datasets)
    - 'balanced': Moderate filtering, reduces imbalance (recommended)
    - 'aggressive': Low thresholds, maximum filtering (when you have lots of data)
    """
    stats = analyze_class_distribution(csv_file)

    if stats is None:
        return None

    print("\n" + "="*70)
    print(f"SUGGESTED THRESHOLDS (Strategy: {strategy.upper()})")
    print("="*70)

    # Strategy multipliers
    strategy_multipliers = {
        'conservative': {'I': 1.5, 'P': 1.2, 'R': 2.0},
        'balanced': {'I': 1.0, 'P': 0.7, 'R': 1.5},
        'aggressive': {'I': 0.7, 'P': 0.5, 'R': 1.2}
    }

    multipliers = strategy_multipliers.get(strategy, strategy_multipliers['balanced'])

    suggestions = {}
    total_excluded = 0

    for phase in ['I', 'P', 'R']:
        if phase not in stats:
            suggestions[phase] = 10  # Default if no data
            continue

        # Base threshold on 75th percentile (excludes top 25% of problematic cases)
        base_threshold = stats[phase]['q75']

        # Apply class-specific multiplier
        adjusted_threshold = base_threshold * multipliers[phase]

        # Ensure minimum threshold of 2 (need at least 2 misclassifications to exclude)
        suggested = max(2, int(adjusted_threshold))

        suggestions[phase] = suggested

        # Calculate how many samples will be excluded
        df = pd.read_csv(csv_file)
        phase_df = df[df['True_Label'] == phase]
        excluded = len(phase_df[phase_df['Misclass_Count'] >= suggested])
        total_excluded += excluded

        phase_name = {'I': 'Inflammatory', 'P': 'Proliferative', 'R': 'Remodeling'}[phase]
        print(f"\n{phase} ({phase_name}):")
        print(f"  Base threshold (75th percentile): {base_threshold:.1f}")
        print(f"  Multiplier for {strategy}: {multipliers[phase]:.2f}")
        print(f"  → Suggested threshold: {suggested}")
        print(f"  Will exclude: {excluded}/{stats[phase]['count']} samples ({excluded/stats[phase]['count']*100:.1f}%)")

        # Show distribution of excluded vs kept
        kept = stats[phase]['count'] - excluded
        print(f"  Keeping: {kept} samples")

    print("\n" + "="*70)
    print(f"SUMMARY")
    print("="*70)
    print(f"Total samples to exclude: {total_excluded}")
    print(f"\nClass-specific exclusion rates:")
    df = pd.read_csv(csv_file)
    for phase in ['I', 'P', 'R']:
        if phase not in stats:
            continue
        phase_df = df[df['True_Label'] == phase]
        excluded = len(phase_df[phase_df['Misclass_Count'] >= suggestions[phase]])
        print(f"  {phase}: {excluded/stats[phase]['count']*100:.1f}% excluded")

    # Show impact on class balance (approximate)
    print(f"\n💡 Impact on class balance:")
    print(f"  P (dominant class) gets highest exclusion → reduces imbalance ✅")
    print(f"  R (minority class) gets lowest exclusion → preserves representation ✅")

    print("\n" + "="*70)
    print("RECOMMENDED COMMANDS")
    print("="*70)

    # Save the baseline file
    saved_file = csv_file.replace('_total.csv', '_saved.csv')
    print(f"\n1. Save the misclassification baseline:")
    print(f"   cp {csv_file} {saved_file}")

    # Show how to use in main.py
    print(f"\n2. Edit src/main.py (line ~1844) to use these thresholds:")
    print(f"   data = filter_frequent_misclassifications(")
    print(f"       data,")
    print(f"       result_dir,")
    print(f"       thresholds={{'I': {suggestions['I']}, 'P': {suggestions['P']}, 'R': {suggestions['R']}}}")
    print(f"   )")

    # Or command-line if they add arguments
    print(f"\n3. Or if you add command-line arguments to main.py:")
    print(f"   python src/main.py \\")
    print(f"       --threshold_I {suggestions['I']} \\")
    print(f"       --threshold_P {suggestions['P']} \\")
    print(f"       --threshold_R {suggestions['R']} \\")
    print(f"       --resume_mode from_data")

    print("\n" + "="*70)

    return suggestions


def compare_strategies(csv_file):
    """Compare all three filtering strategies side-by-side."""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)

    df = pd.read_csv(csv_file)

    strategies = ['conservative', 'balanced', 'aggressive']
    all_suggestions = {}

    for strategy in strategies:
        stats = analyze_class_distribution(csv_file)
        if stats is None:
            return

        strategy_multipliers = {
            'conservative': {'I': 1.5, 'P': 1.2, 'R': 2.0},
            'balanced': {'I': 1.0, 'P': 0.7, 'R': 1.5},
            'aggressive': {'I': 0.7, 'P': 0.5, 'R': 1.2}
        }

        multipliers = strategy_multipliers[strategy]
        suggestions = {}

        for phase in ['I', 'P', 'R']:
            if phase not in stats:
                suggestions[phase] = 10
                continue

            base_threshold = stats[phase]['q75']
            adjusted_threshold = base_threshold * multipliers[phase]
            suggestions[phase] = max(2, int(adjusted_threshold))

        all_suggestions[strategy] = suggestions

    print(f"\n{'Strategy':<15} {'I Threshold':<12} {'P Threshold':<12} {'R Threshold':<12} {'Total Excluded'}")
    print("-" * 70)

    for strategy in strategies:
        suggestions = all_suggestions[strategy]
        total_excluded = 0

        for phase in ['I', 'P', 'R']:
            phase_df = df[df['True_Label'] == phase]
            excluded = len(phase_df[phase_df['Misclass_Count'] >= suggestions[phase]])
            total_excluded += excluded

        print(f"{strategy:<15} {suggestions['I']:<12} {suggestions['P']:<12} {suggestions['R']:<12} {total_excluded}")

    print("\n💡 Recommendation:")
    print("   - Start with 'balanced' for most cases")
    print("   - Use 'conservative' if you have limited data (<1000 samples)")
    print("   - Use 'aggressive' if you have lots of data (>5000 samples) and severe imbalance")


def main():
    parser = argparse.ArgumentParser(
        description='Suggest optimal misclassification filtering thresholds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/suggest_thresholds.py
  python scripts/suggest_thresholds.py --strategy aggressive
  python scripts/suggest_thresholds.py --compare
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        default='results/frequent_misclassifications_total.csv',
        help='Path to misclassification CSV file'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['conservative', 'balanced', 'aggressive'],
        default='balanced',
        help='Filtering strategy (default: balanced)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all strategies side-by-side'
    )

    args = parser.parse_args()

    # Get project root and construct absolute path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    csv_path = project_root / args.csv

    if args.compare:
        compare_strategies(str(csv_path))
    else:
        suggest_thresholds(str(csv_path), args.strategy)


if __name__ == '__main__':
    main()
