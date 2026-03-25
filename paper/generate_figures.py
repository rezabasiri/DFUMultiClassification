#!/usr/bin/env python3
"""
Generate all figures needed for the DFU-MFNet paper.

Figures generated:
- Fig 4: Per-class F1 scores (phase_f1_scores.png)
- Fig 5: Modality agreement patterns (modality_agreement.png)
- Fig 6: Performance progression by modality count (performance_progression.png)
- Fig 7: Generative augmentation dose-response (dose_response.png)
- Fig 8: Cross-run comparison (cross_run_comparison.png)
- Fig S1: ROC curves (roc_curves.png)
- Fig S2: Confusion matrices grid (confusion_matrices.png)

Existing figures NOT regenerated (check results/visualizations/ and agent_communication/):
- Fig 1: Framework overview (manual diagram)
- Fig 2: Generated samples (results/visualizations/gen*.png)
- Fig 3: Architecture diagram (manual diagram)

Outputs saved to paper/figures/
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix, cohen_kappa_score, f1_score, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJ_ROOT, "results", "checkpoints")
STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistics")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5
CLASS_NAMES = ['I', 'P', 'R']
CLASS_FULL = ['Inflammatory', 'Proliferative', 'Remodeling']

ALL_MODALITIES = [
    'metadata', 'depth_rgb', 'depth_map', 'thermal_map',
    'metadata+depth_rgb', 'metadata+depth_map', 'metadata+thermal_map',
    'depth_rgb+depth_map', 'depth_rgb+thermal_map', 'depth_map+thermal_map',
    'metadata+depth_rgb+depth_map', 'metadata+depth_rgb+thermal_map',
    'metadata+depth_map+thermal_map', 'depth_rgb+depth_map+thermal_map',
    'metadata+depth_rgb+depth_map+thermal_map',
]

# Short names for display
def short_name(mod):
    return mod.replace('metadata', 'M').replace('depth_rgb', 'D').replace('thermal_map', 'T').replace('depth_map', 'DM')

# Colors
COLORS = {
    'I': '#2196F3', 'P': '#FF9800', 'R': '#4CAF50',
    'kappa': '#F44336', 'accuracy': '#9C27B0', 'f1': '#795548',
    'metadata': '#2196F3', 'depth_rgb': '#FF9800', 'thermal_map': '#4CAF50', 'depth_map': '#9C27B0',
    'metadata+depth_rgb+thermal_map': '#F44336', 'metadata+depth_rgb': '#795548',
}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_predictions(modality, fold):
    pred_file = os.path.join(CHECKPOINT_DIR, f"pred_run{fold}_{modality}_valid.npy")
    label_file = os.path.join(CHECKPOINT_DIR, f"true_label_run{fold}_{modality}_valid.npy")
    if not os.path.exists(pred_file) or not os.path.exists(label_file):
        return None, None
    return np.load(pred_file), np.load(label_file).astype(int)


def get_all_metrics():
    """Compute metrics for all modalities."""
    results = {}
    for mod in ALL_MODALITIES:
        fold_kappas, fold_accs, fold_f1s = [], [], []
        fold_f1_class = {0: [], 1: [], 2: []}
        all_preds, all_labels = [], []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                pc = np.argmax(pred, axis=1)
                fold_kappas.append(cohen_kappa_score(labels, pc))
                fold_accs.append(accuracy_score(labels, pc))
                fold_f1s.append(f1_score(labels, pc, average='macro', zero_division=0))
                f1c = f1_score(labels, pc, labels=[0, 1, 2], average=None, zero_division=0)
                for c in range(3):
                    fold_f1_class[c].append(f1c[c])
                all_preds.extend(pc)
                all_labels.extend(labels)
        if fold_kappas:
            results[mod] = {
                'kappa': np.mean(fold_kappas), 'kappa_std': np.std(fold_kappas),
                'acc': np.mean(fold_accs), 'acc_std': np.std(fold_accs),
                'f1': np.mean(fold_f1s),
                'f1_I': np.mean(fold_f1_class[0]), 'f1_P': np.mean(fold_f1_class[1]), 'f1_R': np.mean(fold_f1_class[2]),
                'fold_kappas': fold_kappas,
                'n_mod': mod.count('+') + 1,
            }
    return results


# === FIGURE 4: Per-class F1 Scores ===

def fig4_phase_f1_scores(metrics):
    """Grouped bar chart of per-class F1 for key modalities."""
    key_mods = [
        'metadata', 'depth_rgb', 'thermal_map', 'depth_map',
        'metadata+depth_rgb', 'metadata+thermal_map',
        'metadata+depth_rgb+thermal_map', 'metadata+depth_rgb+depth_map+thermal_map',
    ]
    mods = [m for m in key_mods if m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mods))
    width = 0.22

    bars_i = ax.bar(x - width, [metrics[m]['f1_I'] for m in mods], width, label='F1 Inflammatory', color=COLORS['I'], alpha=0.85)
    bars_p = ax.bar(x, [metrics[m]['f1_P'] for m in mods], width, label='F1 Proliferative', color=COLORS['P'], alpha=0.85)
    bars_r = ax.bar(x + width, [metrics[m]['f1_R'] for m in mods], width, label='F1 Remodeling', color=COLORS['R'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in mods], rotation=30, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores Across Modality Combinations')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.95)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars_i, bars_p, bars_r]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase_f1_scores.png"))
    plt.close(fig)
    print("  Saved: phase_f1_scores.png")


# === FIGURE 5: Modality Agreement Patterns ===

def fig5_modality_agreement(metrics):
    """Modality agreement patterns between standalone modalities."""
    mods = ['metadata', 'depth_rgb', 'thermal_map']
    available = [m for m in mods if m in metrics]

    patterns = defaultdict(int)
    total = 0

    for fold in range(1, N_FOLDS + 1):
        preds = {}
        labels = None
        for m in available:
            p, l = load_predictions(m, fold)
            if p is not None:
                preds[m] = np.argmax(p, axis=1) == l
                labels = l
        if len(preds) == len(available):
            for i in range(len(labels)):
                pattern = tuple(int(preds[m][i]) for m in available)
                patterns[pattern] += 1
                total += 1

    label_map = {
        (1,1,1): 'All correct', (1,1,0): 'M+D only', (1,0,1): 'M+T only',
        (0,1,1): 'D+T only', (1,0,0): 'M only', (0,1,0): 'D only',
        (0,0,1): 'T only', (0,0,0): 'All wrong',
    }
    color_map = {
        (1,1,1): '#4CAF50', (1,1,0): '#8BC34A', (1,0,1): '#CDDC39',
        (0,1,1): '#FFC107', (1,0,0): '#FF9800', (0,1,0): '#FF5722',
        (0,0,1): '#E91E63', (0,0,0): '#9E9E9E',
    }

    order = [(1,1,1), (1,1,0), (1,0,1), (0,1,1), (1,0,0), (0,1,0), (0,0,1), (0,0,0)]
    names = [label_map[p] for p in order]
    counts = [patterns.get(p, 0) for p in order]
    colors = [color_map[p] for p in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names[::-1], counts[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, counts[::-1]):
        if count > 0:
            pct = count / total * 100
            ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{count} ({pct:.1f}%)', va='center', fontsize=9)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Modality Agreement Patterns\n(M=Metadata, D=RGB, T=Thermal)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "modality_agreement.png"))
    plt.close(fig)
    print("  Saved: modality_agreement.png")


# === FIGURE 6: Performance Progression by Modality Count ===

def fig6_performance_progression(metrics):
    """Scatter + regression: modality count vs kappa."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_all, y_all = [], []
    for mod, m in metrics.items():
        n = m['n_mod']
        x_all.append(n)
        y_all.append(m['kappa'])
        color = '#F44336' if 'metadata' in mod else '#2196F3'
        marker = 'o' if 'metadata' in mod else 's'
        ax.scatter(n + np.random.uniform(-0.1, 0.1), m['kappa'], c=color, s=60,
                   marker=marker, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Regression line
    from scipy import stats as sp_stats
    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_arr, y_arr)
    x_line = np.linspace(0.8, 4.2, 50)
    ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=2,
            label=f'Linear fit: R$^2$={r_val**2:.3f}, p={p_val:.4f}')

    # Box plots per group
    groups = defaultdict(list)
    for mod, m in metrics.items():
        groups[m['n_mod']].append(m['kappa'])
    positions = sorted(groups.keys())
    bp = ax.boxplot([groups[p] for p in positions], positions=positions, widths=0.3,
                     patch_artist=True, showfliers=False, zorder=0)
    for patch in bp['boxes']:
        patch.set_facecolor('#E0E0E0')
        patch.set_alpha(0.4)

    ax.scatter([], [], c='#F44336', marker='o', label='Includes metadata')
    ax.scatter([], [], c='#2196F3', marker='s', label='Image only')
    ax.set_xlabel('Number of Modalities')
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title('Classification Performance vs Number of Modalities')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "performance_progression.png"))
    plt.close(fig)
    print("  Saved: performance_progression.png")


# === FIGURE 7: Dose-Response Curve ===

def fig7_dose_response():
    """Generative augmentation dose-response curve."""
    # Data from all runs (Run 2 = 0%, Run 3a = 6%, Run 3b = 15%, Run 3c = 25%)
    probs = [0, 6, 15, 25]

    # M+D+T target combo
    combo_kappa = [0.573, 0.586, 0.584, 0.579]
    combo_f1r = [0.519, 0.541, 0.551, 0.545]
    combo_acc = [0.717, 0.724, 0.726, 0.716]

    # Gating ensemble
    gate_kappa = [0.537, 0.498, 0.528, 0.520]
    gate_f1r = [0.583, 0.561, 0.590, 0.559]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: target combo
    ax = axes[0]
    ax.plot(probs, combo_kappa, '-o', color='#F44336', linewidth=2, markersize=8, label='Kappa')
    ax.plot(probs, combo_f1r, '-s', color='#4CAF50', linewidth=2, markersize=8, label='F1-R')
    ax.plot(probs, combo_acc, '-^', color='#2196F3', linewidth=2, markersize=8, label='Accuracy')
    ax.set_xlabel('Augmentation Probability (%)')
    ax.set_ylabel('Score')
    ax.set_title('(a) Target Combination\n(Metadata + RGB + Thermal)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(probs)
    ax.set_ylim(0.45, 0.80)

    # Annotate optimal
    ax.annotate('Optimal', xy=(15, 0.551), xytext=(20, 0.57),
                arrowprops=dict(arrowstyle='->', color='#4CAF50'), fontsize=9, color='#4CAF50')

    # Right: gating ensemble
    ax = axes[1]
    ax.plot(probs, gate_kappa, '-o', color='#F44336', linewidth=2, markersize=8, label='Kappa')
    ax.plot(probs, gate_f1r, '-s', color='#4CAF50', linewidth=2, markersize=8, label='F1-R')
    ax.set_xlabel('Augmentation Probability (%)')
    ax.set_ylabel('Score')
    ax.set_title('(b) Gating Network Ensemble\n(Simple Average, Metadata Combinations)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(probs)
    ax.set_ylim(0.45, 0.65)

    ax.annotate('Best F1-R', xy=(15, 0.590), xytext=(20, 0.61),
                arrowprops=dict(arrowstyle='->', color='#4CAF50'), fontsize=9, color='#4CAF50')

    fig.suptitle('Generative Augmentation Dose-Response', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dose_response.png"))
    plt.close(fig)
    print("  Saved: dose_response.png")


# === FIGURE 8: Cross-Run Comparison ===

def fig8_cross_run_comparison():
    """Cross-run comparison bar chart."""
    runs = ['Run 1\n(Baseline)', 'Run 2\n(+Gating)', 'Run 3a\n(+6% GenAug)', 'Run 3b\n(+15% GenAug)']

    # Best combo kappa
    combo_kappa = [0.613, 0.584, 0.586, 0.584]
    # M+D+T kappa
    mdt_kappa = [0.613, 0.573, 0.586, 0.584]
    # Gating kappa
    gate_kappa = [0, 0.537, 0.498, 0.528]
    # Gating F1-R
    gate_f1r = [0, 0.583, 0.561, 0.590]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Kappa comparison
    ax = axes[0]
    x = np.arange(len(runs))
    width = 0.25
    ax.bar(x - width, mdt_kappa, width, label='M+D+T (target)', color='#F44336', alpha=0.85)
    ax.bar(x, [k if k > 0 else 0 for k in gate_kappa], width, label='Gating Ensemble', color='#2196F3', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("(a) Cohen's Kappa Across Experimental Conditions")
    ax.legend()
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (mk, gk) in enumerate(zip(mdt_kappa, gate_kappa)):
        ax.text(i - width, mk + 0.01, f'{mk:.3f}', ha='center', fontsize=8)
        if gk > 0:
            ax.text(i, gk + 0.01, f'{gk:.3f}', ha='center', fontsize=8)

    # Right: F1 per class for M+D+T across runs
    ax = axes[1]
    f1_i = [0.689, 0.633, 0.644, 0.636]
    f1_p = [0.760, 0.780, 0.786, 0.780]
    f1_r = [0.544, 0.519, 0.541, 0.551]

    ax.bar(x - width, f1_i, width, label='F1-I', color=COLORS['I'], alpha=0.85)
    ax.bar(x, f1_p, width, label='F1-P', color=COLORS['P'], alpha=0.85)
    ax.bar(x + width, f1_r, width, label='F1-R', color=COLORS['R'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_title('(b) Per-Class F1 (M+D+T) Across Conditions')
    ax.legend()
    ax.set_ylim(0, 0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cross_run_comparison.png"))
    plt.close(fig)
    print("  Saved: cross_run_comparison.png")


# === SUPPLEMENTARY: ROC Curves ===

def figS1_roc_curves():
    """ROC curves for key modalities."""
    key_mods = ['metadata', 'depth_rgb', 'thermal_map', 'depth_map',
                'metadata+depth_rgb+thermal_map', 'metadata+depth_rgb']
    mod_colors = {
        'metadata': '#2196F3', 'depth_rgb': '#FF9800', 'thermal_map': '#4CAF50',
        'depth_map': '#9C27B0', 'metadata+depth_rgb+thermal_map': '#F44336',
        'metadata+depth_rgb': '#795548',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[cls_idx]
        for mod in key_mods:
            all_preds, all_labels = [], []
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

            fpr, tpr, _ = roc_curve(labels_bin[:, cls_idx], preds[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            label = f'{short_name(mod)} (AUC={roc_auc:.3f})'
            ax.plot(fpr, tpr, color=mod_colors.get(mod, '#999'), linewidth=2, label=label)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC: {CLASS_FULL[cls_idx]} ({cls_name})')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle('Receiver Operating Characteristic Curves by Class', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plt.close(fig)
    print("  Saved: roc_curves.png")


# === SUPPLEMENTARY: Confusion Matrices Grid ===

def figS2_confusion_matrices():
    """Confusion matrices for key modalities in a grid."""
    key_mods = ['metadata', 'depth_rgb', 'thermal_map',
                'metadata+depth_rgb+thermal_map', 'metadata+depth_rgb+depth_map+thermal_map']
    mods = [m for m in key_mods if any(
        os.path.exists(os.path.join(CHECKPOINT_DIR, f"pred_run{f}_{m}_valid.npy"))
        for f in range(1, N_FOLDS + 1)
    )]

    n = len(mods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, mod in enumerate(mods):
        ax = axes[idx]
        all_preds, all_labels = [], []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                all_preds.extend(np.argmax(pred, axis=1))
                all_labels.extend(labels)

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm_pct[i,j]:.0f}%\n({cm[i,j]})', ha='center', va='center',
                        fontsize=9, color='white' if cm_pct[i,j] > 50 else 'black')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted')
        if idx == 0:
            ax.set_ylabel('True')
        ax.set_title(short_name(mod), fontsize=11)

    fig.suptitle('Confusion Matrices (Row-Normalized %)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"))
    plt.close(fig)
    print("  Saved: confusion_matrices.png")


def main():
    print("=" * 80)
    print("GENERATING FIGURES FOR DFU-MFNet PAPER")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    metrics = get_all_metrics()
    print(f"Loaded metrics for {len(metrics)} modalities")

    print("\nMain figures:")
    fig4_phase_f1_scores(metrics)
    fig5_modality_agreement(metrics)
    fig6_performance_progression(metrics)
    fig7_dose_response()
    fig8_cross_run_comparison()

    print("\nSupplementary figures:")
    figS1_roc_curves()
    figS2_confusion_matrices()

    print(f"\n{'=' * 80}")
    print("ALL FIGURES GENERATED")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
