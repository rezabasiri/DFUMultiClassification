#!/usr/bin/env python3
"""
Compute statistical analyses for the DFU-MFNet paper.
Reads prediction files from results/checkpoints/ and outputs:
1. ROC/AUC curves and values per modality
2. One-way ANOVA across modality groups
3. Linear regression: number of modalities vs performance
4. Paired t-tests for key comparisons
5. McNemar's tests for sample-level significance

Outputs saved to paper/statistics/
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from sklearn.metrics import (
    roc_curve, auc, cohen_kappa_score, f1_score, accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJ_ROOT, "results", "checkpoints")
RESULTS_CSV = os.path.join(PROJ_ROOT, "results", "csv", "modality_combination_results.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5
CLASS_NAMES = ['I', 'P', 'R']
N_CLASSES = 3

ALL_MODALITIES = [
    'metadata', 'depth_rgb', 'depth_map', 'thermal_map',
    'metadata+depth_rgb', 'metadata+depth_map', 'metadata+thermal_map',
    'depth_rgb+depth_map', 'depth_rgb+thermal_map', 'depth_map+thermal_map',
    'metadata+depth_rgb+depth_map', 'metadata+depth_rgb+thermal_map',
    'metadata+depth_map+thermal_map', 'depth_rgb+depth_map+thermal_map',
    'metadata+depth_rgb+depth_map+thermal_map',
]


def load_predictions(modality, fold):
    """Load softmax predictions and true labels for a modality and fold."""
    pred_file = os.path.join(CHECKPOINT_DIR, f"pred_run{fold}_{modality}_valid.npy")
    label_file = os.path.join(CHECKPOINT_DIR, f"true_label_run{fold}_{modality}_valid.npy")
    if not os.path.exists(pred_file) or not os.path.exists(label_file):
        return None, None
    pred = np.load(pred_file)
    labels = np.load(label_file).astype(int)
    return pred, labels


def compute_roc_auc():
    """Compute ROC/AUC for all modalities (macro and per-class)."""
    print("=" * 80)
    print("1. ROC/AUC ANALYSIS")
    print("=" * 80)

    results = []
    roc_data = {}

    for mod in ALL_MODALITIES:
        all_preds = []
        all_labels = []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                all_preds.append(pred)
                all_labels.append(labels)

        if not all_preds:
            continue

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        labels_bin = label_binarize(labels, classes=[0, 1, 2])

        # Per-class AUC
        class_aucs = {}
        class_roc = {}
        for i, cls in enumerate(CLASS_NAMES):
            if labels_bin[:, i].sum() > 0 and labels_bin[:, i].sum() < len(labels_bin):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], preds[:, i])
                class_aucs[cls] = auc(fpr, tpr)
                class_roc[cls] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            else:
                class_aucs[cls] = 0.0

        # Macro AUC
        macro_auc = np.mean(list(class_aucs.values()))

        # Weighted AUC
        try:
            weighted_auc = roc_auc_score(labels_bin, preds, average='weighted', multi_class='ovr')
        except Exception:
            weighted_auc = macro_auc

        results.append({
            'Modality': mod,
            'AUC_macro': round(macro_auc, 4),
            'AUC_weighted': round(weighted_auc, 4),
            'AUC_I': round(class_aucs['I'], 4),
            'AUC_P': round(class_aucs['P'], 4),
            'AUC_R': round(class_aucs['R'], 4),
        })

        roc_data[mod] = class_roc
        print(f"  {mod:<45} macro={macro_auc:.4f} I={class_aucs['I']:.4f} P={class_aucs['P']:.4f} R={class_aucs['R']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "roc_auc_values.csv"), index=False)

    # Save ROC curve data for figure generation
    with open(os.path.join(OUTPUT_DIR, "roc_curve_data.json"), 'w') as f:
        # Only save key modalities to keep file manageable
        key_mods = ['metadata', 'depth_rgb', 'thermal_map', 'depth_map',
                    'metadata+depth_rgb+thermal_map', 'metadata+depth_rgb']
        filtered = {k: v for k, v in roc_data.items() if k in key_mods}
        json.dump(filtered, f)

    print(f"\n  Saved: roc_auc_values.csv, roc_curve_data.json")
    return df


def compute_anova():
    """One-way ANOVA comparing performance across modality group sizes."""
    print("\n" + "=" * 80)
    print("2. ONE-WAY ANOVA: Modality Count vs Performance")
    print("=" * 80)

    # Group modalities by number of modalities
    groups = defaultdict(list)  # n_modalities -> list of kappa values

    for mod in ALL_MODALITIES:
        n_mod = mod.count('+') + 1
        fold_kappas = []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                pred_class = np.argmax(pred, axis=1)
                kappa = cohen_kappa_score(labels, pred_class)
                fold_kappas.append(kappa)
        if fold_kappas:
            groups[n_mod].extend(fold_kappas)

    print(f"\n  Group sizes:")
    for n, vals in sorted(groups.items()):
        print(f"    {n} modalities: {len(vals)} fold-kappa values, mean={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # One-way ANOVA
    group_values = [groups[k] for k in sorted(groups.keys())]
    f_stat, p_value = stats.f_oneway(*group_values)
    print(f"\n  ANOVA F-statistic: {f_stat:.4f}")
    print(f"  ANOVA p-value: {p_value:.6f}")
    print(f"  Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")

    # Post-hoc: Tukey HSD
    all_vals = []
    all_groups = []
    for n, vals in sorted(groups.items()):
        all_vals.extend(vals)
        all_groups.extend([n] * len(vals))

    # Pairwise comparisons
    print(f"\n  Pairwise comparisons (Welch t-test with Bonferroni correction):")
    n_comparisons = len(groups) * (len(groups) - 1) // 2
    keys = sorted(groups.keys())
    pairwise = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            t, p = stats.ttest_ind(groups[keys[i]], groups[keys[j]], equal_var=False)
            p_corrected = min(p * n_comparisons, 1.0)  # Bonferroni
            sig = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else "ns"
            diff = np.mean(groups[keys[j]]) - np.mean(groups[keys[i]])
            pairwise.append({
                'Group A': f"{keys[i]}-modal",
                'Group B': f"{keys[j]}-modal",
                'Mean Diff': round(diff, 4),
                't-stat': round(t, 4),
                'p-value': round(p, 6),
                'p-corrected': round(p_corrected, 6),
                'Significance': sig,
            })
            print(f"    {keys[i]}-modal vs {keys[j]}-modal: diff={diff:+.4f}, p_corrected={p_corrected:.6f} {sig}")

    result = {
        'f_statistic': round(f_stat, 4),
        'p_value': round(p_value, 6),
        'significant': bool(p_value < 0.05),
        'group_means': {str(k): round(float(np.mean(v)), 4) for k, v in sorted(groups.items())},
        'group_stds': {str(k): round(float(np.std(v)), 4) for k, v in sorted(groups.items())},
        'pairwise': pairwise,
    }

    with open(os.path.join(OUTPUT_DIR, "anova_results.json"), 'w') as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(pairwise).to_csv(os.path.join(OUTPUT_DIR, "anova_pairwise.csv"), index=False)
    print(f"\n  Saved: anova_results.json, anova_pairwise.csv")
    return result


def compute_linear_regression():
    """Linear regression: number of modalities vs kappa."""
    print("\n" + "=" * 80)
    print("3. LINEAR REGRESSION: Modality Count vs Performance")
    print("=" * 80)

    x_vals = []
    y_vals = []

    for mod in ALL_MODALITIES:
        n_mod = mod.count('+') + 1
        fold_kappas = []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                pred_class = np.argmax(pred, axis=1)
                kappa = cohen_kappa_score(labels, pred_class)
                fold_kappas.append(kappa)
        if fold_kappas:
            mean_kappa = np.mean(fold_kappas)
            x_vals.append(n_mod)
            y_vals.append(mean_kappa)

    x = np.array(x_vals)
    y = np.array(y_vals)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(f"  Slope: {slope:.4f} (kappa increase per additional modality)")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    result = {
        'slope': round(slope, 4),
        'intercept': round(intercept, 4),
        'r_squared': round(r_value**2, 4),
        'r_value': round(r_value, 4),
        'p_value': round(p_value, 6),
        'std_err': round(std_err, 4),
        'significant': bool(p_value < 0.05),
        'data_points': [{'n_modalities': int(x), 'kappa': round(float(y), 4)} for x, y in zip(x_vals, y_vals)],
    }

    with open(os.path.join(OUTPUT_DIR, "linear_regression_results.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved: linear_regression_results.json")
    return result


def compute_paired_ttests():
    """Paired t-tests for key comparisons."""
    print("\n" + "=" * 80)
    print("4. PAIRED T-TESTS")
    print("=" * 80)

    comparisons = [
        ('metadata+depth_rgb+thermal_map', 'metadata', 'Best fusion vs metadata alone'),
        ('metadata+depth_rgb+thermal_map', 'depth_rgb', 'Best fusion vs best image modality'),
        ('metadata+depth_rgb+thermal_map', 'metadata+depth_rgb', 'Triple fusion vs double fusion'),
        ('metadata+depth_rgb', 'metadata', 'Double fusion vs metadata alone'),
        ('metadata+depth_rgb+depth_map+thermal_map', 'metadata+depth_rgb+thermal_map', 'All 4 modalities vs best triple'),
    ]

    results = []
    for mod_a, mod_b, description in comparisons:
        kappas_a = []
        kappas_b = []
        for fold in range(1, N_FOLDS + 1):
            pred_a, labels_a = load_predictions(mod_a, fold)
            pred_b, labels_b = load_predictions(mod_b, fold)
            if pred_a is not None and pred_b is not None:
                kappas_a.append(cohen_kappa_score(labels_a, np.argmax(pred_a, axis=1)))
                kappas_b.append(cohen_kappa_score(labels_b, np.argmax(pred_b, axis=1)))

        if len(kappas_a) >= 2:
            t_stat, p_value = stats.ttest_rel(kappas_a, kappas_b)
            diff = np.mean(kappas_a) - np.mean(kappas_b)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            results.append({
                'Comparison': description,
                'Modality A': mod_a,
                'Modality B': mod_b,
                'Mean A': round(np.mean(kappas_a), 4),
                'Mean B': round(np.mean(kappas_b), 4),
                'Diff': round(diff, 4),
                't-stat': round(t_stat, 4),
                'p-value': round(p_value, 6),
                'Significance': sig,
            })
            print(f"  {description}:")
            print(f"    {mod_a}: {np.mean(kappas_a):.4f} vs {mod_b}: {np.mean(kappas_b):.4f}")
            print(f"    diff={diff:+.4f}, t={t_stat:.4f}, p={p_value:.6f} {sig}")
            print()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "paired_ttests.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "paired_ttests.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: paired_ttests.csv, paired_ttests.json")
    return results


def compute_mcnemar():
    """McNemar's test for sample-level comparisons."""
    print("\n" + "=" * 80)
    print("5. McNEMAR'S TESTS")
    print("=" * 80)

    comparisons = [
        ('metadata+depth_rgb+thermal_map', 'metadata'),
        ('metadata+depth_rgb+thermal_map', 'depth_rgb'),
        ('metadata+depth_rgb+thermal_map', 'thermal_map'),
        ('metadata+depth_rgb', 'metadata'),
    ]

    results = []
    for mod_a, mod_b in comparisons:
        total_b = total_c = 0
        for fold in range(1, N_FOLDS + 1):
            pred_a, labels_a = load_predictions(mod_a, fold)
            pred_b, labels_b = load_predictions(mod_b, fold)
            if pred_a is not None and pred_b is not None:
                correct_a = np.argmax(pred_a, axis=1) == labels_a
                correct_b = np.argmax(pred_b, axis=1) == labels_b
                total_b += np.sum(~correct_a & correct_b)  # A wrong, B correct
                total_c += np.sum(correct_a & ~correct_b)  # A correct, B wrong

        if total_b + total_c > 0:
            chi2 = (abs(total_b - total_c) - 1) ** 2 / (total_b + total_c)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2 = 0.0
            p_value = 1.0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        results.append({
            'Modality A': mod_a,
            'Modality B': mod_b,
            'A_correct_B_wrong': int(total_c),
            'A_wrong_B_correct': int(total_b),
            'chi2': round(chi2, 4),
            'p-value': round(p_value, 6),
            'Significance': sig,
        })
        print(f"  {mod_a} vs {mod_b}: b={total_b}, c={total_c}, chi2={chi2:.4f}, p={p_value:.6f} {sig}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "mcnemar_tests.csv"), index=False)
    print(f"\n  Saved: mcnemar_tests.csv")
    return results


def main():
    print("=" * 80)
    print("STATISTICAL ANALYSES FOR DFU-MFNet PAPER")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    roc_df = compute_roc_auc()
    anova = compute_anova()
    linreg = compute_linear_regression()
    ttests = compute_paired_ttests()
    mcnemar = compute_mcnemar()

    # Summary
    summary = {
        'roc_auc': roc_df.to_dict('records') if roc_df is not None else [],
        'anova': anova,
        'linear_regression': linreg,
        'paired_ttests': ttests,
        'mcnemar': mcnemar,
    }
    with open(os.path.join(OUTPUT_DIR, "all_statistics_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ALL STATISTICS COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
