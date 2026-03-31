#!/usr/bin/env python3
"""
Generate updated figures for the FUSE4DFU paper (v2).

Figures generated:
- Fig 5: Per-class F1 scores, redesigned (phase_f1_scores.png)
- Fig 6: Performance progression by modality count (performance_progression.png)
- Fig 7: ROC curves per class (roc_curves_main.png) [replaces modality_agreement]
- Fig 8: Ensemble analysis + calibration (ensemble_calibration.png) [replaces dose_response]

Outputs saved to paper/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from collections import defaultdict
from sklearn.metrics import (
    cohen_kappa_score, f1_score, accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJ_ROOT, "results", "checkpoints")
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

METADATA_COMBOS = [m for m in ALL_MODALITIES if 'metadata' in m]
OPTIMIZED_ENSEMBLE = [
    'metadata', 'metadata+depth_map', 'metadata+thermal_map',
    'metadata+depth_rgb+depth_map',
]

# Clear display names
DISPLAY_NAMES = {
    'metadata': 'Metadata',
    'depth_rgb': 'RGB',
    'depth_map': 'Depth',
    'thermal_map': 'Thermal',
    'metadata+depth_rgb': 'Meta+RGB',
    'metadata+depth_map': 'Meta+Depth',
    'metadata+thermal_map': 'Meta+Thermal',
    'depth_rgb+depth_map': 'RGB+Depth',
    'depth_rgb+thermal_map': 'RGB+Thermal',
    'depth_map+thermal_map': 'Depth+Thermal',
    'metadata+depth_rgb+depth_map': 'Meta+RGB+Depth',
    'metadata+depth_rgb+thermal_map': 'Meta+RGB+Thermal',
    'metadata+depth_map+thermal_map': 'Meta+Depth+Thermal',
    'depth_rgb+depth_map+thermal_map': 'RGB+Depth+Thermal',
    'metadata+depth_rgb+depth_map+thermal_map': 'All Four',
}

# Colorblind-friendly palette (Wong 2011)
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN = '#009E73'
CB_VERMILLION = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_SKYBLUE = '#56B4E9'
CB_YELLOW = '#F0E442'

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'legend.title_fontsize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_predictions(modality, fold):
    pred_file = os.path.join(CHECKPOINT_DIR, f"pred_run{fold}_{modality}_valid.npy")
    label_file = os.path.join(CHECKPOINT_DIR, f"true_label_run{fold}_{modality}_valid.npy")
    if not os.path.exists(pred_file) or not os.path.exists(label_file):
        return None, None
    return np.load(pred_file), np.load(label_file).astype(int)


def get_all_metrics():
    results = {}
    for mod in ALL_MODALITIES:
        fold_kappas, fold_accs = [], []
        fold_f1_class = {0: [], 1: [], 2: []}
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                pc = np.argmax(pred, axis=1)
                fold_kappas.append(cohen_kappa_score(labels, pc))
                fold_accs.append(accuracy_score(labels, pc))
                f1c = f1_score(labels, pc, labels=[0, 1, 2], average=None, zero_division=0)
                for c in range(3):
                    fold_f1_class[c].append(f1c[c])
        if fold_kappas:
            results[mod] = {
                'kappa': np.mean(fold_kappas),
                'kappa_std': np.std(fold_kappas),
                'acc': np.mean(fold_accs),
                'f1_I': np.mean(fold_f1_class[0]),
                'f1_P': np.mean(fold_f1_class[1]),
                'f1_R': np.mean(fold_f1_class[2]),
                'fold_kappas': fold_kappas,
                'n_mod': mod.count('+') + 1,
            }
    return results


def compute_ensemble_predictions(combo_list, fold):
    """Average softmax predictions from multiple combos for a given fold."""
    preds_list = []
    labels = None
    for mod in combo_list:
        pred, lab = load_predictions(mod, fold)
        if pred is not None:
            preds_list.append(pred)
            labels = lab
    if not preds_list:
        return None, None
    return np.mean(preds_list, axis=0), labels


def compute_calibration(preds, labels, n_bins=10):
    """Compute ECE with standard fixed bins [0,1], return only populated bins for plotting."""
    confidences = np.max(preds, axis=1)
    predictions = np.argmax(preds, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if i == 0:
            mask = mask | (confidences == bin_edges[i])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    total = sum(bin_counts)
    ece = sum(
        abs(a - c) * n / total
        for a, c, n in zip(bin_accs, bin_confs, bin_counts) if n > 0
    )
    return (np.array(bin_accs), np.array(bin_confs), np.array(bin_counts),
            ece, confidences)


# ======================================================================
# FIGURE 5: Per-Class F1 Scores (Redesigned)
# ======================================================================

def fig5_phase_f1_scores(metrics):
    key_mods = [
        'metadata', 'depth_rgb', 'thermal_map', 'depth_map',
        'metadata+depth_rgb', 'metadata+thermal_map',
        'metadata+depth_rgb+thermal_map', 'metadata+depth_rgb+depth_map+thermal_map',
    ]
    mods = [m for m in key_mods if m in metrics]
    labels = [DISPLAY_NAMES[m] for m in mods]

    # Compute optimized ensemble F1 per class
    ens_f1_class = {0: [], 1: [], 2: []}
    for fold in range(1, N_FOLDS + 1):
        avg_pred, lab = compute_ensemble_predictions(OPTIMIZED_ENSEMBLE, fold)
        if avg_pred is not None:
            pc = np.argmax(avg_pred, axis=1)
            f1c = f1_score(lab, pc, labels=[0, 1, 2], average=None, zero_division=0)
            for c in range(3):
                ens_f1_class[c].append(f1c[c])
    ens_f1 = {c: np.mean(v) for c, v in ens_f1_class.items() if v}

    # Append ensemble as final group
    labels.append('Optimized\nEnsemble')

    fig, ax = plt.subplots(figsize=(15, 5.5))
    x = np.arange(len(labels))
    width = 0.24

    colors_3 = [CB_BLUE, CB_ORANGE, CB_GREEN]

    vals_i = [metrics[m]['f1_I'] for m in mods] + [ens_f1.get(0, 0)]
    vals_p = [metrics[m]['f1_P'] for m in mods] + [ens_f1.get(1, 0)]
    vals_r = [metrics[m]['f1_R'] for m in mods] + [ens_f1.get(2, 0)]

    bars_i = ax.bar(x - width, vals_i, width, label='I',
                    color=colors_3[0], edgecolor='white', linewidth=0.5)
    bars_p = ax.bar(x, vals_p, width, label='P',
                    color=colors_3[1], edgecolor='white', linewidth=0.5)
    bars_r = ax.bar(x + width, vals_r, width, label='R',
                    color=colors_3[2], edgecolor='white', linewidth=0.5)

    # Hatch the ensemble bars to distinguish them
    for bars in [bars_i, bars_p, bars_r]:
        bars[-1].set_hatch('//')
        bars[-1].set_edgecolor('#555555')

    for bars in [bars_i, bars_p, bars_r]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=12,
                        fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.0)
    ax.legend(framealpha=0.95, edgecolor='#cccccc', loc='upper left',
              handlelength=1.2)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # Separators between single, fusion, ensemble
    ax.axvline(x=3.5, color='#444444', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.axvline(x=7.5, color='#444444', linestyle='--', alpha=0.6, linewidth=1.2)
    ax.text(1.5, 0.96, 'Single Modality', ha='center', fontsize=14,
            color='#333333', style='italic', fontweight='medium')
    ax.text(5.5, 0.96, 'Multimodal Fusion', ha='center', fontsize=14,
            color='#333333', style='italic', fontweight='medium')
    ax.text(8.0, 0.96, 'Ensemble', ha='center', fontsize=14,
            color='#333333', style='italic', fontweight='medium')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase_f1_scores.png"))
    plt.close(fig)
    print("  Saved: phase_f1_scores.png")


# ======================================================================
# FIGURE 6: Performance Progression by Modality Count
# ======================================================================

def fig6_performance_progression(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    np.random.seed(42)
    x_all, y_all = [], []
    for mod, m in metrics.items():
        n = m['n_mod']
        x_all.append(n)
        y_all.append(m['kappa'])
        color = CB_VERMILLION if 'metadata' in mod else CB_BLUE
        marker = 'o' if 'metadata' in mod else 's'
        jitter = np.random.uniform(-0.08, 0.08)
        ax.scatter(n + jitter, m['kappa'], c=color, s=50, marker=marker,
                   alpha=0.75, edgecolors='black', linewidth=0.4, zorder=3)

    # Regression line
    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_arr, y_arr)
    x_line = np.linspace(0.8, 4.2, 50)
    ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1.5,
            label=f'Linear fit (R$^2$={r_val**2:.2f}, p={p_val:.3f})')

    # Box plots
    groups = defaultdict(list)
    for mod, m in metrics.items():
        groups[m['n_mod']].append(m['kappa'])
    positions = sorted(groups.keys())
    bp = ax.boxplot([groups[p] for p in positions], positions=positions, widths=0.3,
                    patch_artist=True, showfliers=False, zorder=1)
    for patch in bp['boxes']:
        patch.set_facecolor('#E8E8E8')
        patch.set_alpha(0.5)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('#888888')
            line.set_linewidth(0.8)

    ax.scatter([], [], c=CB_VERMILLION, marker='o', label='Includes metadata',
               edgecolors='black', linewidth=0.4)
    ax.scatter([], [], c=CB_BLUE, marker='s', label='Image only',
               edgecolors='black', linewidth=0.4)

    ax.set_xlabel('Number of Modalities')
    ax.set_ylabel("Cohen's Kappa")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.5)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='#cccccc')
    ax.grid(alpha=0.2, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "performance_progression.png"))
    plt.close(fig)
    print("  Saved: performance_progression.png")


# ======================================================================
# FIGURE 7: ROC Curves Per Class (NEW)
# ======================================================================

def fig7_roc_curves():
    # Single modalities + optimized ensemble only (clean, not crowded)
    key_mods = ['metadata', 'depth_rgb', 'thermal_map']
    mod_colors = {
        'metadata': CB_BLUE,
        'depth_rgb': CB_VERMILLION,
        'thermal_map': CB_GREEN,
    }
    mod_styles = {
        'metadata': '-',
        'depth_rgb': '-',
        'thermal_map': '-',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for cls_idx, (cls_short, cls_full) in enumerate(zip(CLASS_NAMES, CLASS_FULL)):
        ax = axes[cls_idx]

        # Plot single modalities
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
            display = DISPLAY_NAMES[mod]
            ax.plot(fpr, tpr, color=mod_colors[mod], linewidth=2,
                    linestyle=mod_styles[mod],
                    label=f'{display} ({roc_auc:.2f})')

        # Plot optimized ensemble
        all_ens_preds, all_ens_labels = [], []
        for fold in range(1, N_FOLDS + 1):
            avg_pred, lab = compute_ensemble_predictions(OPTIMIZED_ENSEMBLE, fold)
            if avg_pred is not None:
                all_ens_preds.append(avg_pred)
                all_ens_labels.append(lab)
        if all_ens_preds:
            ens_preds = np.concatenate(all_ens_preds)
            ens_labels = np.concatenate(all_ens_labels)
            ens_labels_bin = label_binarize(ens_labels, classes=[0, 1, 2])
            fpr, tpr, _ = roc_curve(ens_labels_bin[:, cls_idx],
                                    ens_preds[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CB_ORANGE, linewidth=2.5,
                    linestyle='-',
                    label=f'Opt. Ensemble ({roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('False Positive Rate')
        if cls_idx == 0:
            ax.set_ylabel('True Positive Rate')
        ax.set_title(f'({chr(97 + cls_idx)}) {cls_full}')
        ax.legend(loc='lower right', framealpha=0.95,
                  edgecolor='#cccccc', title='AUC',
                  bbox_to_anchor=(1.0, 0.0))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(alpha=0.15, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves_main.png"))
    plt.close(fig)
    print("  Saved: roc_curves_main.png")


# ======================================================================
# FIGURE 8: Ensemble Analysis + Calibration (NEW)
# ======================================================================

def fig8_ensemble_calibration():
    # --- Compute kappa for each metadata combo ---
    combo_kappas = {}
    for mod in METADATA_COMBOS:
        fold_kappas = []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                fold_kappas.append(
                    cohen_kappa_score(labels, np.argmax(pred, axis=1)))
        if fold_kappas:
            combo_kappas[mod] = np.mean(fold_kappas)

    # Full ensemble (8 combos)
    full_kappas = []
    for fold in range(1, N_FOLDS + 1):
        avg_pred, labels = compute_ensemble_predictions(METADATA_COMBOS, fold)
        if avg_pred is not None:
            full_kappas.append(
                cohen_kappa_score(labels, np.argmax(avg_pred, axis=1)))
    full_kappa = np.mean(full_kappas) if full_kappas else 0

    # Optimized ensemble (4 combos)
    opt_kappas = []
    for fold in range(1, N_FOLDS + 1):
        avg_pred, labels = compute_ensemble_predictions(OPTIMIZED_ENSEMBLE, fold)
        if avg_pred is not None:
            opt_kappas.append(
                cohen_kappa_score(labels, np.argmax(avg_pred, axis=1)))
    opt_kappa = np.mean(opt_kappas) if opt_kappas else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- Subplot (a): Ensemble Composition ---
    sorted_combos = sorted(combo_kappas.items(), key=lambda x: x[1])
    bar_names = [DISPLAY_NAMES[m] for m, _ in sorted_combos]
    bar_kappas = [k for _, k in sorted_combos]

    bar_names.extend(['Full Ensemble (8)', 'Opt. Ensemble (4)'])
    bar_kappas.extend([full_kappa, opt_kappa])

    opt_display = {DISPLAY_NAMES[m] for m in OPTIMIZED_ENSEMBLE}
    colors = []
    for name in bar_names:
        if 'Ensemble' in name or 'Optimized' in name:
            colors.append(CB_ORANGE)
        elif name in opt_display:
            colors.append(CB_BLUE)
        else:
            colors.append(CB_SKYBLUE)

    # Shift bars up to create gap at bottom for legend
    gap = 3.5
    y_pos = np.arange(len(bar_names)) + gap
    bars = ax1.barh(y_pos, bar_kappas, color=colors,
                    edgecolor='white', linewidth=0.5, height=0.7)

    for bar, k in zip(bars, bar_kappas):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{k:.2f}', va='center', fontsize=14)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(bar_names)
    ax1.set_xlabel("Cohen's Kappa")
    ax1.set_title('(a) Ensemble Composition Analysis')
    ax1.set_xlim(0, max(bar_kappas) + 0.08)
    ax1.set_ylim(-0.5, y_pos[-1] + 0.8)
    ax1.grid(axis='x', alpha=0.2, linestyle='--')

    sep_y = len(combo_kappas) + gap - 0.5
    ax1.axhline(y=sep_y, color='gray', linestyle=':', alpha=0.5)

    legend_elements = [
        Patch(facecolor=CB_BLUE, label='In optimized ensemble'),
        Patch(facecolor=CB_SKYBLUE, label='Other metadata combos'),
        Patch(facecolor=CB_ORANGE, label='Ensemble result'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right',
               framealpha=0.95, edgecolor='#cccccc')

    # --- Subplot (b): Calibration (reliability diagram + confidence histogram) ---
    cal_configs = [
        ('metadata', 'Metadata', CB_BLUE, 'o', '-'),
        ('metadata+depth_rgb+thermal_map', 'Meta+RGB+Thermal', CB_ORANGE, 's', '-'),
    ]

    cal_data = {}  # display -> (bin_accs, bin_confs, bin_counts, ece, confidences, color, marker, ls)
    for mod, display, color, marker, ls in cal_configs:
        all_preds, all_labels = [], []
        for fold in range(1, N_FOLDS + 1):
            pred, labels = load_predictions(mod, fold)
            if pred is not None:
                all_preds.append(pred)
                all_labels.append(labels)
        if all_preds:
            preds = np.concatenate(all_preds)
            labs = np.concatenate(all_labels)
            bin_accs, bin_confs, bin_counts, ece, confs = \
                compute_calibration(preds, labs, n_bins=10)
            cal_data[display] = (bin_accs, bin_confs, bin_counts, ece,
                                 confs, color, marker, ls)

    # Optimized ensemble
    all_ens_preds, all_ens_labels = [], []
    for fold in range(1, N_FOLDS + 1):
        avg_pred, labels = compute_ensemble_predictions(OPTIMIZED_ENSEMBLE, fold)
        if avg_pred is not None:
            all_ens_preds.append(avg_pred)
            all_ens_labels.append(labels)
    if all_ens_preds:
        ens_preds = np.concatenate(all_ens_preds)
        ens_labs = np.concatenate(all_ens_labels)
        bin_accs, bin_confs, bin_counts, ece, confs = \
            compute_calibration(ens_preds, ens_labs, n_bins=10)
        cal_data['Opt. Ensemble'] = (bin_accs, bin_confs, bin_counts, ece,
                                     confs, CB_GREEN, 'D', '--')

    # Plot reliability curves (only populated bins)
    for display, (bin_accs, bin_confs, bin_counts, ece,
                  confs, color, marker, ls) in cal_data.items():
        valid = np.array(bin_counts) > 0
        ax2.plot(bin_confs[valid], bin_accs[valid], marker=marker,
                 linestyle=ls, color=color, linewidth=2, markersize=6,
                 label=f'{display} (ECE={ece:.2f})')

    # Determine axis range from data
    all_valid_confs = []
    for key, val in cal_data.items():
        bc, cnt = val[1], val[2]
        all_valid_confs.extend(bc[np.array(cnt) > 0].tolist())
    x_min = min(all_valid_confs)
    x_max = max(all_valid_confs)
    pad = 0.03

    # Perfect calibration diagonal
    diag_x = np.linspace(x_min - pad, min(x_max + pad, 1.0), 50)
    ax2.plot(diag_x, diag_x, 'k--', alpha=0.4, linewidth=1,
             label='Perfect calibration')

    ax2.set_xlabel('Mean Predicted Confidence')
    ax2.set_ylabel('Fraction Correct')
    ax2.set_title('(b) Probability Calibration')
    ax2.legend(loc='lower right', framealpha=0.95,
               edgecolor='#cccccc', bbox_to_anchor=(1.0, 0.0))

    # Zoom to data range
    ax2.set_xlim(x_min - pad, x_max + pad)
    y_lo = min(ba[np.array(cnt) > 0].min()
               for _, (ba, _, cnt, *_rest) in cal_data.items())
    ax2.set_ylim(max(0, y_lo - 0.05), 1.0)
    ax2.grid(alpha=0.15, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ensemble_calibration.png"))
    plt.close(fig)
    print("  Saved: ensemble_calibration.png")


def main():
    print("=" * 80)
    print("GENERATING UPDATED FIGURES FOR FUSE4DFU PAPER (v2)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    metrics = get_all_metrics()
    print(f"Loaded metrics for {len(metrics)} modalities\n")

    print("Generating figures:")
    fig5_phase_f1_scores(metrics)
    fig6_performance_progression(metrics)
    fig7_roc_curves()
    fig8_ensemble_calibration()

    print(f"\n{'=' * 80}")
    print("ALL FIGURES GENERATED")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
