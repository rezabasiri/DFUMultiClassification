#!/usr/bin/env python3
"""
Comprehensive Multimodal Analysis for DFU Classification Paper.

Generates statistical analyses, figures, and tables examining:
1. Error correlation between modalities (complementarity vs redundancy)
2. Confidence calibration per modality
3. Per-class difficulty analysis
4. Fusion gain decomposition
5. Sample-level agreement/disagreement patterns
6. Confusion matrix comparisons
7. Prediction diversity and ensemble potential

All outputs saved to agent_communication/multimodal_analysis/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, classification_report,
    f1_score, accuracy_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['I (Inflammatory)', 'P (Proliferative)', 'R (Remodeling)']
CLASS_SHORT = ['I', 'P', 'R']
N_FOLDS = 5

STANDALONE_MODALITIES = ['metadata', 'depth_rgb', 'thermal_map', 'depth_map']
FUSION_MODALITIES = [
    'metadata+depth_rgb', 'metadata+thermal_map', 'metadata+depth_rgb+thermal_map',
    'depth_rgb+thermal_map', 'metadata+depth_rgb+depth_map+thermal_map'
]
ALL_MODALITIES = STANDALONE_MODALITIES + FUSION_MODALITIES

# Plot style
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white'
})


# ============================================================
# Data Loading
# ============================================================
def load_predictions():
    """Load all prediction files across folds and modalities."""
    data = {}
    for fold in range(1, N_FOLDS + 1):
        data[fold] = {}
        for mod in ALL_MODALITIES:
            prefix = f"run{fold}_{mod}_valid"
            pred_path = CHECKPOINT_DIR / f"pred_{prefix}.npy"
            label_path = CHECKPOINT_DIR / f"true_label_{prefix}.npy"
            ids_path = CHECKPOINT_DIR / f"sample_ids_{prefix}.npy"
            if pred_path.exists():
                data[fold][mod] = {
                    'pred_probs': np.load(pred_path),
                    'true_labels': np.load(label_path).astype(int),
                    'sample_ids': np.load(ids_path),
                    'pred_labels': np.load(pred_path).argmax(axis=1)
                }
    return data


def get_correctness(data, fold, mod):
    """Return boolean array: True where prediction is correct."""
    d = data[fold][mod]
    return d['pred_labels'] == d['true_labels']


# ============================================================
# 1. Error Correlation Analysis
# ============================================================
def error_correlation_analysis(data, log):
    """Compute pairwise error correlation metrics between standalone modalities."""
    log.write("\n" + "="*80 + "\n")
    log.write("1. ERROR CORRELATION ANALYSIS (Standalone Modalities)\n")
    log.write("="*80 + "\n\n")

    mods = [m for m in STANDALONE_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]
    pairs = list(combinations(mods, 2))

    # Aggregate across folds
    results = []
    for m1, m2 in pairs:
        q_stats, disagree_rates, error_overlaps, kappas_err = [], [], [], []
        cond_a_given_b, cond_b_given_a = [], []

        for fold in range(1, N_FOLDS + 1):
            if m1 not in data[fold] or m2 not in data[fold]:
                continue
            c1 = get_correctness(data, fold, m1)
            c2 = get_correctness(data, fold, m2)
            n = len(c1)

            # Contingency: N11=both correct, N00=both wrong, N10=A correct B wrong, N01=A wrong B correct
            n11 = np.sum(c1 & c2)
            n00 = np.sum(~c1 & ~c2)
            n10 = np.sum(c1 & ~c2)
            n01 = np.sum(~c1 & c2)

            # Q-statistic
            num = n11 * n00 - n01 * n10
            den = n11 * n00 + n01 * n10
            q = num / den if den != 0 else 1.0
            q_stats.append(q)

            # Disagreement measure
            disagree = (n01 + n10) / n
            disagree_rates.append(disagree)

            # Error overlap
            at_least_one_wrong = n00 + n01 + n10
            overlap = n00 / at_least_one_wrong if at_least_one_wrong > 0 else 0
            error_overlaps.append(overlap)

            # Kappa between error vectors
            k = cohen_kappa_score(c1.astype(int), c2.astype(int))
            kappas_err.append(k)

            # Conditional error rates
            b_correct_mask = c2
            if b_correct_mask.sum() > 0:
                cond_a_given_b.append((~c1[b_correct_mask]).mean())
            a_correct_mask = c1
            if a_correct_mask.sum() > 0:
                cond_b_given_a.append((~c2[a_correct_mask]).mean())

        row = {
            'Pair': f"{m1} vs {m2}",
            'Q-statistic': np.mean(q_stats),
            'Q_std': np.std(q_stats),
            'Disagreement': np.mean(disagree_rates),
            'Error_Overlap': np.mean(error_overlaps),
            'Error_Kappa': np.mean(kappas_err),
            'P(A_wrong|B_correct)': np.mean(cond_a_given_b) if cond_a_given_b else 0,
            'P(B_wrong|A_correct)': np.mean(cond_b_given_a) if cond_b_given_a else 0,
        }
        results.append(row)

        log.write(f"  {m1} vs {m2}:\n")
        log.write(f"    Q-statistic:     {row['Q-statistic']:.4f} +/-{row['Q_std']:.4f}\n")
        log.write(f"    Disagreement:    {row['Disagreement']:.4f}\n")
        log.write(f"    Error overlap:   {row['Error_Overlap']:.4f}\n")
        log.write(f"    Error kappa:     {row['Error_Kappa']:.4f}\n")
        log.write(f"    P({m1} wrong | {m2} correct): {row['P(A_wrong|B_correct)']:.4f}\n")
        log.write(f"    P({m2} wrong | {m1} correct): {row['P(B_wrong|A_correct)']:.4f}\n\n")

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "error_correlation_pairwise.csv", index=False)

    # Interpretation
    log.write("  Interpretation:\n")
    for _, r in df.iterrows():
        q = r['Q-statistic']
        if q < 0.3:
            interp = "highly complementary (fusion should help)"
        elif q < 0.6:
            interp = "moderately correlated (fusion may help)"
        else:
            interp = "highly correlated (fusion adds little)"
        log.write(f"    {r['Pair']}: Q={q:.3f} -> {interp}\n")

    return df


def plot_error_correlation_heatmap(data):
    """Plot Q-statistic heatmap for standalone modalities."""
    mods = [m for m in STANDALONE_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]
    n = len(mods)
    q_matrix = np.eye(n)

    for i, m1 in enumerate(mods):
        for j, m2 in enumerate(mods):
            if i >= j:
                continue
            q_vals = []
            for fold in range(1, N_FOLDS + 1):
                c1 = get_correctness(data, fold, m1)
                c2 = get_correctness(data, fold, m2)
                n11 = np.sum(c1 & c2)
                n00 = np.sum(~c1 & ~c2)
                n10 = np.sum(c1 & ~c2)
                n01 = np.sum(~c1 & c2)
                num = n11 * n00 - n01 * n10
                den = n11 * n00 + n01 * n10
                q_vals.append(num / den if den != 0 else 1.0)
            q_matrix[i, j] = q_matrix[j, i] = np.mean(q_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(q_matrix, dtype=bool), k=1)
    sns.heatmap(q_matrix, annot=True, fmt='.3f', xticklabels=mods, yticklabels=mods,
                cmap='RdYlGn_r', vmin=-0.2, vmax=1.0, ax=ax, mask=mask,
                cbar_kws={'label': 'Q-statistic (lower = more complementary)'})
    ax.set_title('Pairwise Error Correlation (Q-statistic)\nAveraged Across 5 Folds')
    fig.savefig(FIGURES_DIR / "error_correlation_heatmap.png")
    plt.close(fig)


# ============================================================
# 2. Confidence Calibration Analysis
# ============================================================
def confidence_calibration_analysis(data, log):
    """Analyze prediction confidence and calibration per modality."""
    log.write("\n" + "="*80 + "\n")
    log.write("2. CONFIDENCE AND CALIBRATION ANALYSIS\n")
    log.write("="*80 + "\n\n")

    mods_to_analyze = [m for m in ALL_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]
    results = []

    for mod in mods_to_analyze:
        all_probs, all_labels, all_correct = [], [], []
        for fold in range(1, N_FOLDS + 1):
            d = data[fold][mod]
            all_probs.append(d['pred_probs'])
            all_labels.append(d['true_labels'])
            all_correct.append(get_correctness(data, fold, mod))

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        correct = np.concatenate(all_correct)

        max_conf = probs.max(axis=1)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # Confidence when correct vs wrong
        conf_correct = max_conf[correct].mean() if correct.sum() > 0 else 0
        conf_wrong = max_conf[~correct].mean() if (~correct).sum() > 0 else 0

        # ECE (Expected Calibration Error)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for b in range(n_bins):
            mask = (max_conf >= bin_edges[b]) & (max_conf < bin_edges[b+1])
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = max_conf[mask].mean()
            ece += mask.sum() / len(max_conf) * abs(bin_acc - bin_conf)

        row = {
            'Modality': mod,
            'Mean_Confidence': max_conf.mean(),
            'Conf_When_Correct': conf_correct,
            'Conf_When_Wrong': conf_wrong,
            'Conf_Separation': conf_correct - conf_wrong,
            'Mean_Entropy': entropy.mean(),
            'ECE': ece,
            'Accuracy': correct.mean(),
        }
        results.append(row)
        log.write(f"  {mod}:\n")
        log.write(f"    Mean confidence:    {row['Mean_Confidence']:.4f}\n")
        log.write(f"    Conf when correct:  {row['Conf_When_Correct']:.4f}\n")
        log.write(f"    Conf when wrong:    {row['Conf_When_Wrong']:.4f}\n")
        log.write(f"    Conf separation:    {row['Conf_Separation']:.4f} (higher = better calibrated)\n")
        log.write(f"    Mean entropy:       {row['Mean_Entropy']:.4f}\n")
        log.write(f"    ECE:                {row['ECE']:.4f}\n")
        log.write(f"    Accuracy:           {row['Accuracy']:.4f}\n\n")

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "confidence_calibration.csv", index=False)
    return df


def plot_confidence_distributions(data):
    """Plot confidence distributions for standalone modalities (correct vs wrong)."""
    mods = [m for m in STANDALONE_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]

    fig, axes = plt.subplots(1, len(mods), figsize=(4*len(mods), 4), sharey=True)
    if len(mods) == 1:
        axes = [axes]

    for ax, mod in zip(axes, mods):
        confs_correct, confs_wrong = [], []
        for fold in range(1, N_FOLDS+1):
            d = data[fold][mod]
            max_conf = d['pred_probs'].max(axis=1)
            correct = get_correctness(data, fold, mod)
            confs_correct.extend(max_conf[correct])
            confs_wrong.extend(max_conf[~correct])

        ax.hist(confs_correct, bins=30, alpha=0.6, label='Correct', color='#2ecc71', density=True)
        ax.hist(confs_wrong, bins=30, alpha=0.6, label='Wrong', color='#e74c3c', density=True)
        ax.set_title(mod)
        ax.set_xlabel('Max Confidence')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Density')
    fig.suptitle('Prediction Confidence: Correct vs Incorrect Samples', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_distributions.png")
    plt.close(fig)


def plot_reliability_diagrams(data):
    """Plot reliability (calibration) diagrams per standalone modality."""
    mods = [m for m in STANDALONE_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]

    fig, axes = plt.subplots(1, len(mods), figsize=(4*len(mods), 4))
    if len(mods) == 1:
        axes = [axes]

    for ax, mod in zip(axes, mods):
        all_probs, all_labels = [], []
        for fold in range(1, N_FOLDS+1):
            d = data[fold][mod]
            all_probs.append(d['pred_probs'])
            all_labels.append(d['true_labels'])
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)

        # One-vs-all calibration per class
        for c in range(3):
            y_true_bin = (labels == c).astype(int)
            y_prob = probs[:, c]
            try:
                fraction_pos, mean_pred = calibration_curve(y_true_bin, y_prob, n_bins=8, strategy='uniform')
                ax.plot(mean_pred, fraction_pos, 'o-', label=CLASS_SHORT[c], markersize=4)
            except ValueError:
                pass

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
        ax.set_title(mod)
        ax.set_xlabel('Mean Predicted Probability')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    axes[0].set_ylabel('Fraction of Positives')
    fig.suptitle('Reliability Diagrams (Calibration)', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reliability_diagrams.png")
    plt.close(fig)


# ============================================================
# 3. Per-Class Difficulty Analysis
# ============================================================
def per_class_analysis(data, log):
    """Analyze which classes are hardest per modality."""
    log.write("\n" + "="*80 + "\n")
    log.write("3. PER-CLASS DIFFICULTY ANALYSIS\n")
    log.write("="*80 + "\n\n")

    mods = [m for m in ALL_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]
    rows = []

    for mod in mods:
        all_preds, all_labels = [], []
        for fold in range(1, N_FOLDS+1):
            d = data[fold][mod]
            all_preds.append(d['pred_labels'])
            all_labels.append(d['true_labels'])
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        f1s = f1_score(labels, preds, average=None, labels=[0,1,2])
        acc = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)

        # Per-class accuracy
        class_accs = []
        for c in range(3):
            mask = labels == c
            class_accs.append((preds[mask] == labels[mask]).mean() if mask.sum() > 0 else 0)

        rows.append({
            'Modality': mod,
            'Kappa': kappa,
            'Accuracy': acc,
            'F1_I': f1s[0], 'F1_P': f1s[1], 'F1_R': f1s[2],
            'Acc_I': class_accs[0], 'Acc_P': class_accs[1], 'Acc_R': class_accs[2],
            'Hardest_Class': CLASS_SHORT[np.argmin(f1s)],
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "per_class_difficulty.csv", index=False)

    log.write(f"  {'Modality':<40} {'Kappa':>7} {'Acc':>7} {'F1-I':>7} {'F1-P':>7} {'F1-R':>7} {'Hardest':>8}\n")
    log.write("  " + "-"*85 + "\n")
    for _, r in df.iterrows():
        log.write(f"  {r['Modality']:<40} {r['Kappa']:>7.4f} {r['Accuracy']:>7.4f} "
                  f"{r['F1_I']:>7.3f} {r['F1_P']:>7.3f} {r['F1_R']:>7.3f} {r['Hardest_Class']:>8}\n")

    return df


def plot_per_class_f1_heatmap(class_df):
    """Heatmap of F1 per class per modality."""
    mods = class_df['Modality'].tolist()
    f1_data = class_df[['F1_I', 'F1_P', 'F1_R']].values

    fig, ax = plt.subplots(figsize=(6, max(4, len(mods)*0.45)))
    sns.heatmap(f1_data, annot=True, fmt='.3f', xticklabels=CLASS_SHORT,
                yticklabels=mods, cmap='YlOrRd', vmin=0.1, vmax=0.85, ax=ax,
                cbar_kws={'label': 'F1 Score'})
    ax.set_title('Per-Class F1 Score by Modality')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_f1_heatmap.png")
    plt.close(fig)


# ============================================================
# 4. Confusion Matrix Comparison
# ============================================================
def plot_confusion_matrices(data):
    """Side-by-side normalized confusion matrices for standalone + best fusion."""
    mods_to_plot = ['metadata', 'depth_rgb', 'thermal_map', 'metadata+depth_rgb+thermal_map']
    mods_to_plot = [m for m in mods_to_plot if all(m in data[f] for f in range(1, N_FOLDS+1))]

    fig, axes = plt.subplots(1, len(mods_to_plot), figsize=(4.5*len(mods_to_plot), 4))
    if len(mods_to_plot) == 1:
        axes = [axes]

    for ax, mod in zip(axes, mods_to_plot):
        all_preds, all_labels = [], []
        for fold in range(1, N_FOLDS+1):
            d = data[fold][mod]
            all_preds.append(d['pred_labels'])
            all_labels.append(d['true_labels'])
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        cm = confusion_matrix(labels, preds, labels=[0,1,2], normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=CLASS_SHORT,
                    yticklabels=CLASS_SHORT, cmap='Blues', vmin=0, vmax=1, ax=ax)
        kappa = cohen_kappa_score(labels, preds)
        short_name = mod.replace('metadata', 'meta').replace('thermal_map', 'therm').replace('depth_rgb', 'depth')
        ax.set_title(f'{short_name}\n(κ={kappa:.3f})')
        ax.set_xlabel('Predicted')

    axes[0].set_ylabel('True')
    fig.suptitle('Normalized Confusion Matrices', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices_comparison.png")
    plt.close(fig)


# ============================================================
# 5. Fusion Gain Decomposition
# ============================================================
def fusion_gain_analysis(data, log):
    """Analyze where fusion helps and where it hurts relative to best standalone."""
    log.write("\n" + "="*80 + "\n")
    log.write("5. FUSION GAIN DECOMPOSITION\n")
    log.write("="*80 + "\n\n")

    fusion_mod = 'metadata+depth_rgb+thermal_map'
    standalone_mods = ['metadata', 'depth_rgb', 'thermal_map']

    # Check availability
    if not all(all(m in data[f] for f in range(1, N_FOLDS+1)) for m in [fusion_mod] + standalone_mods):
        log.write("  Skipped: not all modalities available\n")
        return None

    total_rescued = 0
    total_degraded = 0
    total_both_correct = 0
    total_both_wrong = 0
    total_n = 0

    rescued_by_class = Counter()
    degraded_by_class = Counter()

    for fold in range(1, N_FOLDS + 1):
        fusion_correct = get_correctness(data, fold, fusion_mod)
        # "Best standalone" = correct if ANY standalone gets it right
        any_standalone_correct = np.zeros(len(fusion_correct), dtype=bool)
        for mod in standalone_mods:
            any_standalone_correct |= get_correctness(data, fold, mod)

        # Also check: best single standalone (metadata, usually best)
        meta_correct = get_correctness(data, fold, 'metadata')
        labels = data[fold][fusion_mod]['true_labels']

        # Categories
        rescued = fusion_correct & ~meta_correct  # fusion got right, metadata wrong
        degraded = ~fusion_correct & meta_correct  # fusion got wrong, metadata right
        both_correct = fusion_correct & meta_correct
        both_wrong = ~fusion_correct & ~meta_correct

        total_rescued += rescued.sum()
        total_degraded += degraded.sum()
        total_both_correct += both_correct.sum()
        total_both_wrong += both_wrong.sum()
        total_n += len(fusion_correct)

        for c in range(3):
            rescued_by_class[c] += (rescued & (labels == c)).sum()
            degraded_by_class[c] += (degraded & (labels == c)).sum()

    log.write(f"  Fusion: {fusion_mod}\n")
    log.write(f"  Reference: metadata (best standalone)\n\n")
    log.write(f"  Total samples (across folds): {total_n}\n")
    log.write(f"  Both correct:    {total_both_correct} ({total_both_correct/total_n*100:.1f}%)\n")
    log.write(f"  Fusion rescued:  {total_rescued} ({total_rescued/total_n*100:.1f}%) — fusion right, metadata wrong\n")
    log.write(f"  Fusion degraded: {total_degraded} ({total_degraded/total_n*100:.1f}%) — fusion wrong, metadata right\n")
    log.write(f"  Both wrong:      {total_both_wrong} ({total_both_wrong/total_n*100:.1f}%)\n")
    log.write(f"  Net gain:        {total_rescued - total_degraded:+d} samples\n\n")

    log.write(f"  Per-class breakdown:\n")
    log.write(f"    {'Class':<6} {'Rescued':>10} {'Degraded':>10} {'Net':>8}\n")
    for c in range(3):
        r = rescued_by_class[c]
        d = degraded_by_class[c]
        log.write(f"    {CLASS_SHORT[c]:<6} {r:>10} {d:>10} {r-d:>+8}\n")

    return {
        'rescued': total_rescued, 'degraded': total_degraded,
        'both_correct': total_both_correct, 'both_wrong': total_both_wrong,
        'rescued_by_class': dict(rescued_by_class),
        'degraded_by_class': dict(degraded_by_class),
    }


def plot_fusion_gain_waterfall(gain_data):
    """Waterfall chart showing fusion gain/loss per class."""
    if gain_data is None:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    categories = ['Both\nCorrect', 'Fusion\nRescued', 'Fusion\nDegraded', 'Both\nWrong']
    values = [gain_data['both_correct'], gain_data['rescued'],
              gain_data['degraded'], gain_data['both_wrong']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(v), ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Number of Samples (across all folds)')
    ax.set_title('Fusion (meta+depth+thermal) vs Metadata Alone\nSample-Level Outcome')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fusion_gain_waterfall.png")
    plt.close(fig)


# ============================================================
# 6. Prediction Diversity (Error Dendrogram)
# ============================================================
def plot_error_dendrogram(data):
    """Hierarchical clustering of modalities by error pattern similarity."""
    mods = [m for m in ALL_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]
    n = len(mods)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            disagrees = []
            for fold in range(1, N_FOLDS+1):
                c1 = get_correctness(data, fold, mods[i])
                c2 = get_correctness(data, fold, mods[j])
                disagrees.append(np.mean(c1 != c2))
            d = np.mean(disagrees)
            dist_matrix[i, j] = dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')

    fig, ax = plt.subplots(figsize=(10, 5))
    short_names = [m.replace('metadata', 'meta').replace('thermal_map', 'therm')
                   .replace('depth_rgb', 'depth').replace('depth_map', 'dmap') for m in mods]
    dendrogram(Z, labels=short_names, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_ylabel('Disagreement Rate (higher = more diverse)')
    ax.set_title('Modality Clustering by Prediction Error Patterns')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_dendrogram.png")
    plt.close(fig)


# ============================================================
# 7. Modality Performance Radar Chart
# ============================================================
def plot_radar_chart(class_df):
    """Radar chart comparing modalities across metrics."""
    standalone = class_df[class_df['Modality'].isin(STANDALONE_MODALITIES + ['metadata+depth_rgb+thermal_map'])]
    if standalone.empty:
        return

    metrics = ['Kappa', 'F1_I', 'F1_P', 'F1_R', 'Accuracy']
    metric_labels = ['Kappa', 'F1 (I)', 'F1 (P)', 'F1 (R)', 'Accuracy']

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22']

    for idx, (_, row) in enumerate(standalone.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        short = row['Modality'].replace('metadata', 'meta').replace('thermal_map', 'therm').replace('depth_rgb', 'depth')
        ax.plot(angles, values, 'o-', linewidth=2, label=short, color=colors[idx % len(colors)], markersize=4)
        ax.fill(angles, values, alpha=0.05, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 0.85)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title('Modality Performance Comparison', y=1.08)
    fig.savefig(FIGURES_DIR / "radar_chart.png")
    plt.close(fig)


# ============================================================
# 8. Sample Hardness Analysis
# ============================================================
def sample_hardness_analysis(data, log):
    """Identify universally hard samples and class-conditional difficulty."""
    log.write("\n" + "="*80 + "\n")
    log.write("8. SAMPLE HARDNESS ANALYSIS\n")
    log.write("="*80 + "\n\n")

    mods = [m for m in STANDALONE_MODALITIES if all(m in data[f] for f in range(1, N_FOLDS+1))]

    # Track per-sample correctness across modalities
    sample_results = defaultdict(lambda: {'correct_count': 0, 'total_count': 0, 'label': None})

    for fold in range(1, N_FOLDS+1):
        ids = data[fold][mods[0]]['sample_ids']
        labels = data[fold][mods[0]]['true_labels']
        for i in range(len(ids)):
            sid = tuple(ids[i])
            sample_results[sid]['label'] = labels[i]
            for mod in mods:
                correct = get_correctness(data, fold, mod)
                sample_results[sid]['correct_count'] += int(correct[i])
                sample_results[sid]['total_count'] += 1

    # Categorize
    n_mods = len(mods)
    always_correct = sum(1 for s in sample_results.values() if s['correct_count'] == s['total_count'])
    always_wrong = sum(1 for s in sample_results.values() if s['correct_count'] == 0)
    total = len(sample_results)

    log.write(f"  Standalone modalities analyzed: {mods}\n")
    log.write(f"  Total unique samples: {total}\n")
    log.write(f"  Always correct (all modalities): {always_correct} ({always_correct/total*100:.1f}%)\n")
    log.write(f"  Always wrong (no modality correct): {always_wrong} ({always_wrong/total*100:.1f}%)\n\n")

    # By class
    log.write(f"  Per-class hardness:\n")
    log.write(f"    {'Class':<6} {'Total':>8} {'Always OK':>10} {'Always Wrong':>12} {'Hard %':>8}\n")
    for c in range(3):
        class_samples = {k: v for k, v in sample_results.items() if v['label'] == c}
        n_class = len(class_samples)
        n_ok = sum(1 for v in class_samples.values() if v['correct_count'] == v['total_count'])
        n_bad = sum(1 for v in class_samples.values() if v['correct_count'] == 0)
        log.write(f"    {CLASS_SHORT[c]:<6} {n_class:>8} {n_ok:>10} {n_bad:>12} {n_bad/n_class*100 if n_class>0 else 0:>7.1f}%\n")

    # Hardness distribution histogram
    correct_fracs = [s['correct_count'] / s['total_count'] for s in sample_results.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct_fracs, bins=n_mods+1, edgecolor='white', color='#3498db', alpha=0.8)
    ax.set_xlabel(f'Fraction of {n_mods} Modalities Correct')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Hardness Distribution\n(How many standalone modalities classify each sample correctly)')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Always wrong')
    ax.axvline(1, color='green', linestyle='--', alpha=0.5, label='Always correct')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_hardness_distribution.png")
    plt.close(fig)


# ============================================================
# 9. Fusion vs Standalone Scatter
# ============================================================
def plot_fusion_vs_standalone_scatter(data):
    """Scatter: per-sample max standalone confidence vs fusion confidence, colored by outcome."""
    fusion_mod = 'metadata+depth_rgb+thermal_map'
    standalone_mods = ['metadata', 'depth_rgb', 'thermal_map']

    if not all(all(m in data[f] for f in range(1, N_FOLDS+1)) for m in [fusion_mod] + standalone_mods):
        return

    fusion_confs, standalone_max_confs, outcomes = [], [], []

    for fold in range(1, N_FOLDS+1):
        fd = data[fold][fusion_mod]
        fusion_conf = fd['pred_probs'].max(axis=1)
        fusion_correct = get_correctness(data, fold, fusion_mod)

        # Max confidence from any standalone (on the predicted class of each standalone)
        standalone_confs = np.stack([data[fold][m]['pred_probs'].max(axis=1) for m in standalone_mods])
        best_standalone_conf = standalone_confs.max(axis=0)

        meta_correct = get_correctness(data, fold, 'metadata')

        fusion_confs.extend(fusion_conf)
        standalone_max_confs.extend(best_standalone_conf)
        for i in range(len(fusion_conf)):
            if fusion_correct[i] and meta_correct[i]:
                outcomes.append('Both correct')
            elif fusion_correct[i] and not meta_correct[i]:
                outcomes.append('Fusion rescued')
            elif not fusion_correct[i] and meta_correct[i]:
                outcomes.append('Fusion degraded')
            else:
                outcomes.append('Both wrong')

    fig, ax = plt.subplots(figsize=(7, 6))
    colors_map = {'Both correct': '#2ecc71', 'Fusion rescued': '#3498db',
                  'Fusion degraded': '#e74c3c', 'Both wrong': '#95a5a6'}
    for cat in ['Both wrong', 'Fusion degraded', 'Fusion rescued', 'Both correct']:
        mask = [o == cat for o in outcomes]
        ax.scatter(np.array(standalone_max_confs)[mask], np.array(fusion_confs)[mask],
                   s=8, alpha=0.4, label=f'{cat} ({sum(mask)})', color=colors_map[cat])

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Best Standalone Max Confidence')
    ax.set_ylabel('Fusion Max Confidence')
    ax.set_title('Fusion vs Standalone Confidence\n(meta+depth+thermal vs best standalone)')
    ax.legend(fontsize=8, markerscale=3)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fusion_vs_standalone_scatter.png")
    plt.close(fig)


# ============================================================
# 10. Statistical Significance Tests
# ============================================================
def statistical_tests(data, log):
    """Paired t-tests and McNemar's tests between key modality pairs."""
    log.write("\n" + "="*80 + "\n")
    log.write("10. STATISTICAL SIGNIFICANCE TESTS\n")
    log.write("="*80 + "\n\n")

    pairs_to_test = [
        ('metadata', 'metadata+depth_rgb+thermal_map'),
        ('metadata', 'depth_rgb'),
        ('metadata', 'thermal_map'),
        ('depth_rgb', 'thermal_map'),
        ('metadata+depth_rgb', 'metadata+depth_rgb+thermal_map'),
        ('metadata+thermal_map', 'metadata+depth_rgb+thermal_map'),
    ]

    results = []
    for m1, m2 in pairs_to_test:
        if not all(all(m in data[f] for f in range(1, N_FOLDS+1)) for m in [m1, m2]):
            continue

        # Per-fold kappas for paired t-test
        k1s, k2s = [], []
        # For McNemar's: aggregate correctness
        all_c1, all_c2 = [], []

        for fold in range(1, N_FOLDS+1):
            d1, d2 = data[fold][m1], data[fold][m2]
            k1 = cohen_kappa_score(d1['true_labels'], d1['pred_labels'])
            k2 = cohen_kappa_score(d2['true_labels'], d2['pred_labels'])
            k1s.append(k1)
            k2s.append(k2)

            c1 = get_correctness(data, fold, m1)
            c2 = get_correctness(data, fold, m2)
            all_c1.extend(c1)
            all_c2.extend(c2)

        # Paired t-test on kappas
        t_stat, t_p = stats.ttest_rel(k1s, k2s)

        # McNemar's test
        c1 = np.array(all_c1)
        c2 = np.array(all_c2)
        b = np.sum(c1 & ~c2)  # m1 correct, m2 wrong
        c_val = np.sum(~c1 & c2)  # m1 wrong, m2 correct
        if b + c_val > 0:
            mcnemar_chi2 = (abs(b - c_val) - 1)**2 / (b + c_val)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, df=1)
        else:
            mcnemar_chi2 = 0
            mcnemar_p = 1.0

        row = {
            'Pair': f"{m1} vs {m2}",
            'Mean_K1': np.mean(k1s), 'Mean_K2': np.mean(k2s),
            'Diff': np.mean(k2s) - np.mean(k1s),
            't_stat': t_stat, 't_p': t_p,
            'McNemar_chi2': mcnemar_chi2, 'McNemar_p': mcnemar_p,
            'Significant_t': t_p < 0.05,
            'Significant_McNemar': mcnemar_p < 0.05,
        }
        results.append(row)

        sig_t = "YES" if t_p < 0.05 else "no"
        sig_m = "YES" if mcnemar_p < 0.05 else "no"
        log.write(f"  {m1} vs {m2}:\n")
        log.write(f"    Mean kappa: {np.mean(k1s):.4f} vs {np.mean(k2s):.4f} (diff={np.mean(k2s)-np.mean(k1s):+.4f})\n")
        log.write(f"    Paired t-test: t={t_stat:.3f}, p={t_p:.4f} -> {sig_t}\n")
        log.write(f"    McNemar's test: chi2={mcnemar_chi2:.3f}, p={mcnemar_p:.4f} -> {sig_m}\n\n")

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)
    return df


# ============================================================
# 11. Modality Agreement Venn-style Analysis
# ============================================================
def modality_agreement_analysis(data, log):
    """Compute agreement regions for 3 standalone modalities."""
    log.write("\n" + "="*80 + "\n")
    log.write("11. MODALITY AGREEMENT ANALYSIS (meta, depth_rgb, thermal_map)\n")
    log.write("="*80 + "\n\n")

    mods = ['metadata', 'depth_rgb', 'thermal_map']
    if not all(all(m in data[f] for f in range(1, N_FOLDS+1)) for m in mods):
        log.write("  Skipped: not all modalities available\n")
        return

    regions = Counter()
    total = 0

    for fold in range(1, N_FOLDS+1):
        cm = get_correctness(data, fold, 'metadata')
        cd = get_correctness(data, fold, 'depth_rgb')
        ct = get_correctness(data, fold, 'thermal_map')
        n = len(cm)
        total += n

        regions['All correct'] += np.sum(cm & cd & ct)
        regions['Only meta'] += np.sum(cm & ~cd & ~ct)
        regions['Only depth'] += np.sum(~cm & cd & ~ct)
        regions['Only thermal'] += np.sum(~cm & ~cd & ct)
        regions['Meta+Depth'] += np.sum(cm & cd & ~ct)
        regions['Meta+Thermal'] += np.sum(cm & ~cd & ct)
        regions['Depth+Thermal'] += np.sum(~cm & cd & ct)
        regions['None correct'] += np.sum(~cm & ~cd & ~ct)

    log.write(f"  Total samples (across folds): {total}\n\n")
    log.write(f"  {'Region':<20} {'Count':>8} {'Pct':>8}\n")
    log.write("  " + "-"*40 + "\n")
    for region in ['All correct', 'Only meta', 'Only depth', 'Only thermal',
                   'Meta+Depth', 'Meta+Thermal', 'Depth+Thermal', 'None correct']:
        c = regions[region]
        log.write(f"  {region:<20} {c:>8} {c/total*100:>7.1f}%\n")

    # Complementarity score
    unique_contributions = regions['Only meta'] + regions['Only depth'] + regions['Only thermal']
    any_correct = total - regions['None correct']
    complementarity = unique_contributions / any_correct if any_correct > 0 else 0
    log.write(f"\n  Unique contributions: {unique_contributions} ({unique_contributions/total*100:.1f}%)\n")
    log.write(f"  Complementarity score: {complementarity:.4f} (fraction of correct samples from unique modality)\n")

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(regions.keys())
    values = [regions[l] for l in labels]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']
    bars = ax.bar(labels, values, color=colors, edgecolor='white')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{v}\n({v/total*100:.1f}%)', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Sample Count (across all folds)')
    ax.set_title('Modality Agreement Regions\n(metadata, depth_rgb, thermal_map)')
    plt.xticks(rotation=30, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "modality_agreement_regions.png")
    plt.close(fig)


# ============================================================
# 12. Performance vs Number of Modalities
# ============================================================
def plot_modality_count_vs_performance(class_df):
    """Plot how kappa changes with number of modalities."""
    df = class_df.copy()
    df['N_Modalities'] = df['Modality'].apply(lambda x: len(x.split('+')))

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in df.iterrows():
        has_meta = 'metadata' in row['Modality']
        color = '#e74c3c' if has_meta else '#3498db'
        marker = 's' if has_meta else 'o'
        ax.scatter(row['N_Modalities'] + np.random.uniform(-0.1, 0.1), row['Kappa'],
                   s=60, color=color, marker=marker, alpha=0.7, zorder=3)

    # Mean per count
    for n in sorted(df['N_Modalities'].unique()):
        subset = df[df['N_Modalities'] == n]
        ax.hlines(subset['Kappa'].mean(), n-0.3, n+0.3, colors='black', linewidth=2, zorder=4)

    ax.set_xlabel('Number of Modalities')
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title('Performance vs Number of Modalities')
    ax.set_xticks([1, 2, 3, 4])
    legend_elements = [Patch(facecolor='#e74c3c', label='Includes metadata'),
                       Patch(facecolor='#3498db', label='Images only')]
    ax.legend(handles=legend_elements)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "modality_count_vs_performance.png")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    print("Loading predictions...")
    data = load_predictions()
    available_folds = list(data.keys())
    available_mods = set()
    for fold in available_folds:
        available_mods.update(data[fold].keys())
    print(f"  Folds: {available_folds}")
    print(f"  Modalities: {sorted(available_mods)}")

    log_path = OUTPUT_DIR / "analysis_log.txt"
    with open(log_path, 'w') as log:
        log.write("MULTIMODAL ANALYSIS FOR DFU CLASSIFICATION\n")
        log.write("=" * 80 + "\n")
        log.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        log.write(f"Folds: {N_FOLDS}\n")
        log.write(f"Available modalities: {sorted(available_mods)}\n\n")

        # 1. Error correlation
        print("1. Error correlation analysis...")
        err_df = error_correlation_analysis(data, log)
        plot_error_correlation_heatmap(data)

        # 2. Confidence calibration
        print("2. Confidence calibration...")
        conf_df = confidence_calibration_analysis(data, log)
        plot_confidence_distributions(data)
        plot_reliability_diagrams(data)

        # 3. Per-class difficulty
        print("3. Per-class difficulty...")
        class_df = per_class_analysis(data, log)
        plot_per_class_f1_heatmap(class_df)

        # 4. Confusion matrices
        print("4. Confusion matrices...")
        plot_confusion_matrices(data)

        # 5. Fusion gain decomposition
        print("5. Fusion gain decomposition...")
        gain_data = fusion_gain_analysis(data, log)
        plot_fusion_gain_waterfall(gain_data)

        # 6. Error dendrogram
        print("6. Error dendrogram...")
        plot_error_dendrogram(data)

        # 7. Radar chart
        print("7. Radar chart...")
        plot_radar_chart(class_df)

        # 8. Sample hardness
        print("8. Sample hardness...")
        sample_hardness_analysis(data, log)

        # 9. Fusion vs standalone scatter
        print("9. Fusion vs standalone scatter...")
        plot_fusion_vs_standalone_scatter(data)

        # 10. Statistical tests
        print("10. Statistical significance tests...")
        stat_df = statistical_tests(data, log)

        # 11. Agreement analysis
        print("11. Modality agreement analysis...")
        modality_agreement_analysis(data, log)

        # 12. Modality count vs performance
        print("12. Modality count vs performance...")
        plot_modality_count_vs_performance(class_df)

        log.write("\n" + "="*80 + "\n")
        log.write("ANALYSIS COMPLETE\n")
        log.write(f"Figures saved to: {FIGURES_DIR}\n")
        log.write(f"Tables saved to: {TABLES_DIR}\n")

    print(f"\nDone! Log: {log_path}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables: {TABLES_DIR}")


if __name__ == '__main__':
    main()
