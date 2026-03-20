#!/usr/bin/env python3
"""
Gating Network Optimization Audit.

Searches for the best gating network configuration to combine predictions
from 15 modality-specific models. Uses pre-computed combo_pred files from
a previous main.py run, avoiding re-training the modality models.

Explores:
  - Ensemble strategies: attention gating, weighted average, stacking (LR/RF/GB), simple average
  - Model subsets: top-3, top-5, top-7, top-10, all-15
  - Architecture params: heads, key_dim, dropout, L2, hidden dims
  - Training params: LR, epochs, batch size, patience
  - Loss variants: cross-entropy, focal, entropy-regularized
  - Temperature scaling

Baseline: best single combination (kappa ~0.584, acc ~0.746)
"""

import os
import sys
import json
import time
import hashlib
import warnings
import datetime
import csv
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from collections import Counter

import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
CHECKPOINT_DIR = os.path.join(PROJ_ROOT, "results", "checkpoints")
AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(AUDIT_DIR, "gating_search_results.csv")
LOG_DIR = os.path.join(AUDIT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

N_FOLDS = 5
N_CLASSES = 3
CLASS_NAMES = ['I', 'P', 'R']

# All 15 modality combos (underscore-safe names as stored in files)
ALL_COMBOS = [
    'metadata', 'depth_rgb', 'depth_map', 'thermal_map',
    'metadata_depth_rgb', 'metadata_depth_map', 'metadata_thermal_map',
    'depth_rgb_depth_map', 'depth_rgb_thermal_map', 'depth_map_thermal_map',
    'metadata_depth_rgb_depth_map', 'metadata_depth_rgb_thermal_map',
    'metadata_depth_map_thermal_map', 'depth_rgb_depth_map_thermal_map',
    'metadata_depth_rgb_depth_map_thermal_map',
]

# Setup logging
log_filename = datetime.datetime.now().strftime("gating_search_%Y%m%d_%H%M%S.log")
LOG_FILE = os.path.join(LOG_DIR, log_filename)

def log(msg):
    line = f"{msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# ============================================================
# DATA LOADING
# ============================================================

def load_combo_predictions(combo_name, fold, dtype='valid'):
    """Load predictions and labels for a combo/fold."""
    pred_path = os.path.join(CHECKPOINT_DIR, f"combo_pred_{combo_name}_run{fold}_{dtype}.npy")
    label_path = os.path.join(CHECKPOINT_DIR, f"combo_label_{combo_name}_run{fold}_{dtype}.npy")
    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        return None, None
    return np.load(pred_path).astype(np.float32), np.load(label_path).astype(np.int64)


def load_all_data(combo_names):
    """Load train/valid predictions for all combos and folds. Returns dict keyed by fold."""
    data = {}
    for fold in range(1, N_FOLDS + 1):
        fold_data = {'train_preds': [], 'valid_preds': [], 'train_labels': None, 'valid_labels': None, 'combo_names': []}
        for combo in combo_names:
            train_pred, train_labels = load_combo_predictions(combo, fold, 'train')
            valid_pred, valid_labels = load_combo_predictions(combo, fold, 'valid')
            if train_pred is not None and valid_pred is not None:
                fold_data['train_preds'].append(train_pred)
                fold_data['valid_preds'].append(valid_pred)
                fold_data['combo_names'].append(combo)
                if fold_data['train_labels'] is None:
                    fold_data['train_labels'] = train_labels
                    fold_data['valid_labels'] = valid_labels
        data[fold] = fold_data
    return data


def rank_combos_by_kappa(data):
    """Rank combos by average validation kappa across folds."""
    from sklearn.metrics import cohen_kappa_score
    combo_kappas = {}
    for fold in range(1, N_FOLDS + 1):
        for i, combo in enumerate(data[fold]['combo_names']):
            pred_class = np.argmax(data[fold]['valid_preds'][i], axis=1)
            kappa = cohen_kappa_score(data[fold]['valid_labels'], pred_class)
            if combo not in combo_kappas:
                combo_kappas[combo] = []
            combo_kappas[combo].append(kappa)

    ranked = [(combo, np.mean(kappas)) for combo, kappas in combo_kappas.items()]
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ============================================================
# ENSEMBLE STRATEGIES
# ============================================================

def simple_average(train_preds_list, valid_preds_list, train_labels, valid_labels, **kwargs):
    """Simple average of all model predictions."""
    avg_valid = np.mean(np.stack(valid_preds_list, axis=0), axis=0)
    return avg_valid


def weighted_average_optimal(train_preds_list, valid_preds_list, train_labels, valid_labels, **kwargs):
    """Learn optimal weights via scipy minimize on training set."""
    from scipy.optimize import minimize
    from sklearn.metrics import cohen_kappa_score

    n_models = len(train_preds_list)
    train_stack = np.stack(train_preds_list, axis=0)  # (n_models, n_samples, 3)
    valid_stack = np.stack(valid_preds_list, axis=0)

    def neg_kappa(weights):
        w = np.abs(weights)
        w = w / w.sum()
        combined = np.tensordot(w, train_stack, axes=([0], [0]))
        pred_class = np.argmax(combined, axis=1)
        return -cohen_kappa_score(train_labels, pred_class)

    best_result = None
    best_val = float('inf')
    # Multiple restarts
    for _ in range(5):
        x0 = np.random.dirichlet(np.ones(n_models))
        result = minimize(neg_kappa, x0, method='Nelder-Mead', options={'maxiter': 2000})
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    w = np.abs(best_result.x)
    w = w / w.sum()
    combined_valid = np.tensordot(w, valid_stack, axes=([0], [0]))
    return combined_valid


def stacking_logistic(train_preds_list, valid_preds_list, train_labels, valid_labels, C=1.0, **kwargs):
    """Logistic regression stacking."""
    from sklearn.linear_model import LogisticRegression

    X_train = np.concatenate(train_preds_list, axis=1)
    X_valid = np.concatenate(valid_preds_list, axis=1)

    model = LogisticRegression(C=C, max_iter=1000, multi_class='multinomial', solver='lbfgs',
                               class_weight='balanced', random_state=42)
    model.fit(X_train, train_labels)
    probs = model.predict_proba(X_valid)
    return probs


def stacking_gradient_boosting(train_preds_list, valid_preds_list, train_labels, valid_labels,
                                n_estimators=100, max_depth=3, learning_rate=0.1, **kwargs):
    """Gradient boosting stacking."""
    from sklearn.ensemble import GradientBoostingClassifier

    X_train = np.concatenate(train_preds_list, axis=1)
    X_valid = np.concatenate(valid_preds_list, axis=1)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        random_state=42, subsample=0.8
    )
    model.fit(X_train, train_labels)
    probs = model.predict_proba(X_valid)
    return probs


def stacking_mlp(train_preds_list, valid_preds_list, train_labels, valid_labels,
                  hidden_dims=[64, 32], dropout=0.3, lr=1e-3, epochs=100, batch_size=64, **kwargs):
    """MLP stacking with Keras."""
    import tensorflow as tf
    from tensorflow import keras

    X_train = np.concatenate(train_preds_list, axis=1).astype(np.float32)
    X_valid = np.concatenate(valid_preds_list, axis=1).astype(np.float32)
    y_train = keras.utils.to_categorical(train_labels, N_CLASSES)

    # Build MLP
    inputs = keras.Input(shape=(X_train.shape[1],))
    x = inputs
    for dim in hidden_dims:
        x = keras.layers.Dense(dim, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(N_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.arange(N_CLASSES), y=train_labels)
    class_weight = {i: cw[i] for i in range(N_CLASSES)}

    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.15, class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
        ],
        verbose=0
    )

    probs = model.predict(X_valid, verbose=0)
    keras.backend.clear_session()
    return probs


def attention_gating(train_preds_list, valid_preds_list, train_labels, valid_labels,
                     num_heads=4, key_dim=16, dropout=0.1, l2_reg=1e-4, lr=1e-3,
                     epochs=100, batch_size=64, use_class_attention=True, **kwargs):
    """Attention-based gating network (simplified from main.py)."""
    import tensorflow as tf
    from tensorflow import keras

    n_models = len(train_preds_list)
    X_train = np.stack(train_preds_list, axis=1).astype(np.float32)  # (N, n_models, 3)
    X_valid = np.stack(valid_preds_list, axis=1).astype(np.float32)
    y_train = keras.utils.to_categorical(train_labels, N_CLASSES)

    # Build attention model
    inputs = keras.Input(shape=(n_models, N_CLASSES))

    # Model-level attention
    attn_out = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(inputs, inputs)
    attn_out = keras.layers.LayerNormalization()(attn_out + inputs)

    if use_class_attention:
        # Class-level attention (transpose: models attend to classes)
        transposed = keras.layers.Permute((2, 1))(attn_out)  # (N, 3, n_models)
        class_attn = keras.layers.MultiHeadAttention(
            num_heads=min(4, n_models), key_dim=max(8, n_models), dropout=dropout
        )(transposed, transposed)
        class_attn = keras.layers.LayerNormalization()(class_attn + transposed)
        pooled = keras.layers.GlobalAveragePooling1D()(class_attn)  # (N, n_models)
        x = keras.layers.Dense(N_CLASSES, kernel_regularizer=keras.regularizers.l2(l2_reg))(pooled)
    else:
        # Simple pooling
        pooled = keras.layers.GlobalAveragePooling1D()(attn_out)  # (N, 3)
        x = keras.layers.Dense(N_CLASSES, kernel_regularizer=keras.regularizers.l2(l2_reg))(pooled)

    # Residual from mean prediction
    mean_pred = keras.layers.Lambda(lambda inp: tf.reduce_mean(inp, axis=1))(inputs)
    x = x + mean_pred

    outputs = keras.layers.Softmax()(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.arange(N_CLASSES), y=train_labels)
    class_weight = {i: cw[i] for i in range(N_CLASSES)}

    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.15, class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
        ],
        verbose=0
    )

    probs = model.predict(X_valid, verbose=0)
    keras.backend.clear_session()
    return probs


def rank_weighted_average(train_preds_list, valid_preds_list, train_labels, valid_labels, decay=0.5, **kwargs):
    """Weight models by their rank (best model gets highest weight)."""
    from sklearn.metrics import cohen_kappa_score

    # Compute per-model kappa on training set
    kappas = []
    for pred in train_preds_list:
        pred_class = np.argmax(pred, axis=1)
        kappas.append(cohen_kappa_score(train_labels, pred_class))

    # Exponential decay weights by rank
    ranked_indices = np.argsort(kappas)[::-1]
    weights = np.zeros(len(kappas))
    for rank, idx in enumerate(ranked_indices):
        weights[idx] = decay ** rank
    weights = weights / weights.sum()

    valid_stack = np.stack(valid_preds_list, axis=0)
    combined = np.tensordot(weights, valid_stack, axes=([0], [0]))
    return combined


def temperature_scaled_average(train_preds_list, valid_preds_list, train_labels, valid_labels, **kwargs):
    """Learn per-model temperature scaling then average."""
    from scipy.optimize import minimize

    n_models = len(train_preds_list)
    train_stack = np.stack(train_preds_list, axis=0)
    valid_stack = np.stack(valid_preds_list, axis=0)

    def apply_temperature(preds, temps):
        """Apply temperature scaling to logits."""
        # Convert probs to logits, scale, convert back
        logits = np.log(np.clip(preds, 1e-10, 1.0))
        scaled = logits / np.abs(temps).reshape(-1, 1, 1)
        exp_scaled = np.exp(scaled - np.max(scaled, axis=2, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=2, keepdims=True)

    def neg_log_likelihood(temps):
        scaled = apply_temperature(train_stack, temps)
        avg = np.mean(scaled, axis=0)
        # NLL
        y_onehot = np.eye(N_CLASSES)[train_labels]
        nll = -np.mean(np.sum(y_onehot * np.log(np.clip(avg, 1e-10, 1.0)), axis=1))
        return nll

    x0 = np.ones(n_models)
    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead', options={'maxiter': 1000})

    scaled_valid = apply_temperature(valid_stack, result.x)
    combined = np.mean(scaled_valid, axis=0)
    return combined


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(probs, labels):
    """Compute all metrics."""
    from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

    pred_class = np.argmax(probs, axis=1)
    return {
        'kappa': cohen_kappa_score(labels, pred_class),
        'accuracy': accuracy_score(labels, pred_class),
        'f1_macro': f1_score(labels, pred_class, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, pred_class, average='weighted', zero_division=0),
        'f1_I': f1_score(labels, pred_class, labels=[0], average=None, zero_division=0)[0],
        'f1_P': f1_score(labels, pred_class, labels=[1], average=None, zero_division=0)[0],
        'f1_R': f1_score(labels, pred_class, labels=[2], average=None, zero_division=0)[0],
    }


def run_5fold_evaluation(strategy_fn, data, combo_names, **kwargs):
    """Run strategy across 5 folds and return averaged metrics."""
    fold_metrics = []
    for fold in range(1, N_FOLDS + 1):
        fd = data[fold]
        # Filter to requested combos
        indices = [i for i, c in enumerate(fd['combo_names']) if c in combo_names]
        if len(indices) < 2 and strategy_fn != simple_average:
            # Need at least 2 for most strategies, 1 for simple average
            continue

        train_preds = [fd['train_preds'][i] for i in indices]
        valid_preds = [fd['valid_preds'][i] for i in indices]

        try:
            probs = strategy_fn(
                train_preds, valid_preds,
                fd['train_labels'], fd['valid_labels'],
                **kwargs
            )
            metrics = evaluate_predictions(probs, fd['valid_labels'])
            fold_metrics.append(metrics)
        except Exception as e:
            log(f"    Fold {fold} failed: {e}")
            fold_metrics.append({'kappa': 0, 'accuracy': 0, 'f1_macro': 0, 'f1_weighted': 0,
                                'f1_I': 0, 'f1_P': 0, 'f1_R': 0})

    if not fold_metrics:
        return None

    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std = {f'{k}_std': np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    avg.update(std)
    avg['fold_kappas'] = [m['kappa'] for m in fold_metrics]
    return avg


# ============================================================
# SEARCH CONFIGURATIONS
# ============================================================

def generate_configs(ranked_combos):
    """Generate all configurations to test."""
    configs = []

    # Model subsets
    top_3 = [c[0] for c in ranked_combos[:3]]
    top_5 = [c[0] for c in ranked_combos[:5]]
    top_7 = [c[0] for c in ranked_combos[:7]]
    top_10 = [c[0] for c in ranked_combos[:10]]
    all_15 = [c[0] for c in ranked_combos]

    # Only combos with metadata
    meta_combos = [c[0] for c in ranked_combos if 'metadata' in c[0]]
    # Only standalone modalities
    standalone = [c[0] for c in ranked_combos if '_' not in c[0]]

    subsets = {
        'top3': top_3, 'top5': top_5, 'top7': top_7, 'top10': top_10,
        'all15': all_15, 'meta_only': meta_combos, 'standalone': standalone,
    }

    # === SIMPLE STRATEGIES ===
    for subset_name, subset in subsets.items():
        configs.append({
            'name': f'simple_avg_{subset_name}',
            'strategy': 'simple_average',
            'combos': subset,
            'params': {},
        })

        configs.append({
            'name': f'rank_weighted_{subset_name}',
            'strategy': 'rank_weighted',
            'combos': subset,
            'params': {'decay': 0.5},
        })

    # Rank weighted with different decays
    for decay in [0.3, 0.5, 0.7, 0.9]:
        for subset_name in ['top5', 'top7', 'all15']:
            configs.append({
                'name': f'rank_weighted_{subset_name}_d{decay}',
                'strategy': 'rank_weighted',
                'combos': subsets[subset_name],
                'params': {'decay': decay},
            })

    # === OPTIMAL WEIGHTED AVERAGE ===
    for subset_name in ['top3', 'top5', 'top7', 'top10']:
        configs.append({
            'name': f'opt_weighted_{subset_name}',
            'strategy': 'optimal_weighted',
            'combos': subsets[subset_name],
            'params': {},
        })

    # === TEMPERATURE SCALED ===
    for subset_name in ['top5', 'top7', 'all15']:
        configs.append({
            'name': f'temp_scaled_{subset_name}',
            'strategy': 'temperature_scaled',
            'combos': subsets[subset_name],
            'params': {},
        })

    # === STACKING: LOGISTIC REGRESSION ===
    for C in [0.01, 0.1, 1.0, 10.0]:
        for subset_name in ['top5', 'top7', 'all15']:
            configs.append({
                'name': f'stack_lr_C{C}_{subset_name}',
                'strategy': 'stacking_lr',
                'combos': subsets[subset_name],
                'params': {'C': C},
            })

    # === STACKING: GRADIENT BOOSTING ===
    for n_est, depth, lr in [(50, 3, 0.1), (100, 3, 0.1), (100, 4, 0.05), (200, 3, 0.05)]:
        for subset_name in ['top5', 'top7', 'all15']:
            configs.append({
                'name': f'stack_gb_n{n_est}_d{depth}_lr{lr}_{subset_name}',
                'strategy': 'stacking_gb',
                'combos': subsets[subset_name],
                'params': {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr},
            })

    # === STACKING: MLP ===
    for hidden, dropout, lr in [([32], 0.3, 1e-3), ([64, 32], 0.3, 1e-3), ([128, 64], 0.4, 5e-4),
                                 ([32, 16], 0.2, 1e-3), ([64], 0.3, 5e-4)]:
        for subset_name in ['top5', 'top7', 'all15']:
            h_str = '_'.join(map(str, hidden))
            configs.append({
                'name': f'stack_mlp_h{h_str}_d{dropout}_{subset_name}',
                'strategy': 'stacking_mlp',
                'combos': subsets[subset_name],
                'params': {'hidden_dims': hidden, 'dropout': dropout, 'lr': lr, 'epochs': 150, 'batch_size': 64},
            })

    # === ATTENTION GATING ===
    for heads, key_dim, dropout in [(2, 8, 0.1), (4, 16, 0.1), (4, 16, 0.2), (8, 32, 0.1),
                                     (2, 8, 0.2), (4, 32, 0.15)]:
        for use_class in [True, False]:
            for subset_name in ['top5', 'top7', 'all15']:
                ca_str = 'ca' if use_class else 'noca'
                configs.append({
                    'name': f'attn_h{heads}_k{key_dim}_d{dropout}_{ca_str}_{subset_name}',
                    'strategy': 'attention',
                    'combos': subsets[subset_name],
                    'params': {
                        'num_heads': heads, 'key_dim': key_dim, 'dropout': dropout,
                        'l2_reg': 1e-4, 'lr': 1e-3, 'epochs': 150, 'batch_size': 64,
                        'use_class_attention': use_class,
                    },
                })

    # === ATTENTION with different LRs ===
    for lr in [5e-4, 1e-3, 3e-3]:
        configs.append({
            'name': f'attn_h4_k16_lr{lr}_ca_top7',
            'strategy': 'attention',
            'combos': subsets['top7'],
            'params': {
                'num_heads': 4, 'key_dim': 16, 'dropout': 0.1,
                'l2_reg': 1e-4, 'lr': lr, 'epochs': 200, 'batch_size': 64,
                'use_class_attention': True,
            },
        })

    return configs


# ============================================================
# MAIN SEARCH
# ============================================================

STRATEGY_MAP = {
    'simple_average': simple_average,
    'rank_weighted': rank_weighted_average,
    'optimal_weighted': weighted_average_optimal,
    'temperature_scaled': temperature_scaled_average,
    'stacking_lr': stacking_logistic,
    'stacking_gb': stacking_gradient_boosting,
    'stacking_mlp': stacking_mlp,
    'attention': attention_gating,
}


def load_completed():
    """Load already-completed config names from CSV."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        return {row['name'] for row in reader}


def save_result(result):
    """Append a result to CSV."""
    file_exists = os.path.exists(RESULTS_CSV)
    fieldnames = [
        'name', 'strategy', 'n_combos', 'combo_subset',
        'kappa', 'kappa_std', 'accuracy', 'accuracy_std',
        'f1_macro', 'f1_macro_std', 'f1_weighted',
        'f1_I', 'f1_P', 'f1_R',
        'fold_kappas', 'params', 'elapsed_seconds',
    ]
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main():
    log("=" * 80)
    log("GATING NETWORK OPTIMIZATION AUDIT")
    log(f"  Logging to: {LOG_FILE}")
    log(f"  Results: {RESULTS_CSV}")
    log("=" * 80)

    # Load all prediction data
    log("\nLoading all combo predictions...")
    data = load_all_data(ALL_COMBOS)
    for fold in range(1, N_FOLDS + 1):
        n = len(data[fold]['combo_names'])
        log(f"  Fold {fold}: {n} combos, train={data[fold]['train_preds'][0].shape[0]}, valid={data[fold]['valid_preds'][0].shape[0]}")

    # Rank combos
    log("\nRanking combos by validation kappa:")
    ranked = rank_combos_by_kappa(data)
    for i, (combo, kappa) in enumerate(ranked):
        log(f"  {i+1:>2}. {combo:<45} kappa={kappa:.4f}")

    # Compute single-combo baseline
    log("\nBaseline (best single combo, no ensemble):")
    best_combo = ranked[0]
    log(f"  {best_combo[0]}: kappa={best_combo[1]:.4f}")

    # Generate configs
    configs = generate_configs(ranked)
    completed = load_completed()
    remaining = [c for c in configs if c['name'] not in completed]

    log(f"\nTotal configs: {len(configs)}")
    log(f"Already completed: {len(completed)}")
    log(f"Remaining: {len(remaining)}")

    # Run search
    log(f"\n{'='*80}")
    log("STARTING SEARCH")
    log(f"{'='*80}")

    for i, cfg in enumerate(remaining):
        t0 = time.time()
        strategy_fn = STRATEGY_MAP[cfg['strategy']]
        combo_names = cfg['combos']

        log(f"\n[{i+1}/{len(remaining)}] {cfg['name']}")
        log(f"  Strategy: {cfg['strategy']}, Combos: {len(combo_names)}, Params: {cfg['params']}")

        try:
            metrics = run_5fold_evaluation(strategy_fn, data, combo_names, **cfg['params'])
        except Exception as e:
            log(f"  FAILED: {e}")
            traceback.print_exc()
            metrics = None

        elapsed = time.time() - t0

        if metrics is None:
            log(f"  FAILED (no metrics)")
            continue

        # Determine subset name
        subset_parts = cfg['name'].split('_')
        subset_name = subset_parts[-1] if subset_parts[-1] in ['top3','top5','top7','top10','all15','meta_only','standalone'] else 'custom'

        result = {
            'name': cfg['name'],
            'strategy': cfg['strategy'],
            'n_combos': len(combo_names),
            'combo_subset': subset_name,
            'kappa': f"{metrics['kappa']:.6f}",
            'kappa_std': f"{metrics['kappa_std']:.6f}",
            'accuracy': f"{metrics['accuracy']:.6f}",
            'accuracy_std': f"{metrics['accuracy_std']:.6f}",
            'f1_macro': f"{metrics['f1_macro']:.6f}",
            'f1_macro_std': f"{metrics['f1_macro_std']:.6f}",
            'f1_weighted': f"{metrics['f1_weighted']:.6f}",
            'f1_I': f"{metrics['f1_I']:.6f}",
            'f1_P': f"{metrics['f1_P']:.6f}",
            'f1_R': f"{metrics['f1_R']:.6f}",
            'fold_kappas': str(metrics['fold_kappas']),
            'params': json.dumps(cfg['params']),
            'elapsed_seconds': f"{elapsed:.1f}",
        }
        save_result(result)

        delta = metrics['kappa'] - best_combo[1]
        marker = " *** NEW BEST ***" if delta > 0 else ""
        log(f"  kappa={metrics['kappa']:.4f}±{metrics['kappa_std']:.4f} acc={metrics['accuracy']:.4f} "
            f"F1={metrics['f1_macro']:.4f} [I={metrics['f1_I']:.3f} P={metrics['f1_P']:.3f} R={metrics['f1_R']:.3f}] "
            f"delta={delta:+.4f} ({elapsed:.1f}s){marker}")

    # Final summary
    log(f"\n{'='*80}")
    log("SEARCH COMPLETE — SUMMARY")
    log(f"{'='*80}")

    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: -float(r['kappa']))

        log(f"\nBaseline (best single combo): kappa={best_combo[1]:.4f}")
        log(f"\nTOP 20 CONFIGURATIONS:")
        log(f"{'Rank':>4} {'Name':<50} {'Kappa':>8} {'±Std':>8} {'Acc':>8} {'F1':>8} {'Strategy':<20} {'#Combos':>7}")
        log("-" * 125)
        for j, r in enumerate(rows[:20]):
            delta = float(r['kappa']) - best_combo[1]
            beat = "✓" if delta > 0 else " "
            log(f"{j+1:>4} {r['name']:<50} {float(r['kappa']):>8.4f} {float(r['kappa_std']):>8.4f} "
                f"{float(r['accuracy']):>8.4f} {float(r['f1_macro']):>8.4f} {r['strategy']:<20} {r['n_combos']:>7} {beat}")

        # Best by strategy
        log(f"\nBEST PER STRATEGY:")
        strategies = {}
        for r in rows:
            s = r['strategy']
            if s not in strategies or float(r['kappa']) > float(strategies[s]['kappa']):
                strategies[s] = r
        for s, r in sorted(strategies.items(), key=lambda x: -float(x[1]['kappa'])):
            log(f"  {s:<25} {r['name']:<45} kappa={float(r['kappa']):.4f}")

        # Save best config
        best = rows[0]
        best_config_path = os.path.join(AUDIT_DIR, "gating_best_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best, f, indent=2)
        log(f"\nBest config saved to: {best_config_path}")

    log(f"\nDone. Results in: {RESULTS_CSV}")


if __name__ == '__main__':
    main()
