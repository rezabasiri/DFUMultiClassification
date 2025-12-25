#!/usr/bin/env python3
"""
Comprehensive Class Imbalance Strategy Comparison - V5 (FULL DATA)

Using 100% of data for more reliable results.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.data.dataset_utils import create_patient_folds
from src.evaluation.metrics import filter_frequent_misclassifications

# Force CPU for consistency
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)


def focal_loss(gamma=2.0, alpha=None):
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_weight = tf.reduce_sum(alpha_tensor * y_true, axis=-1, keepdims=True)
            weight = weight * alpha_weight
        fl = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    return focal_loss_fn


def compute_alpha_weights(y_train, scheme='balanced'):
    classes = np.unique(y_train)
    if scheme == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
    elif scheme == 'sqrt':
        counts = np.bincount(y_train)
        max_count = max(counts)
        weights = np.sqrt(max_count / counts)
    else:
        weights = np.ones(len(classes))
    return dict(zip(classes, weights))


def build_model(input_dim, loss='categorical_crossentropy'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    return model


def apply_sampling(X_train, y_train, strategy, random_state=42):
    if strategy == 'none':
        return X_train, y_train

    min_samples = min(Counter(y_train).values())
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    if strategy == 'oversample':
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == 'smote':
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=random_state, n_neighbors=k_neighbors)
    elif strategy == 'borderline_smote':
        sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors))
    elif strategy == 'smote_enn':
        sampler = SMOTEENN(random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sampler.fit_resample(X_train, y_train)


def evaluate_strategy(X, y, data, folds, strategy_name, use_alpha, sampling_strategy,
                      alpha_scheme='balanced', use_focal=False):
    fold_results = []

    for fold_idx, (train_patients, val_patients) in enumerate(folds):
        train_mask = data['Patient#'].isin(train_patients)
        val_mask = data['Patient#'].isin(val_patients)

        X_train = X[train_mask.values]
        y_train = y[train_mask.values]
        X_val = X[val_mask.values]
        y_val = y[val_mask.values]

        if use_alpha or use_focal:
            class_weights_dict = compute_alpha_weights(y_train, scheme=alpha_scheme)
            class_weights = class_weights_dict if use_alpha else None
            focal_alpha = [class_weights_dict[i] for i in range(3)] if use_focal else None
        else:
            class_weights = None
            focal_alpha = None

        try:
            X_train_sampled, y_train_sampled = apply_sampling(
                X_train, y_train, sampling_strategy, random_state=42+fold_idx
            )
        except Exception as e:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sampled)
        X_val_scaled = scaler.transform(X_val)

        y_train_onehot = tf.keras.utils.to_categorical(y_train_sampled, num_classes=3)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)

        tf.keras.backend.clear_session()
        if use_focal:
            loss = focal_loss(gamma=2.0, alpha=focal_alpha)
        else:
            loss = 'categorical_crossentropy'

        model = build_model(X_train_scaled.shape[1], loss=loss)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        model.fit(
            X_train_scaled, y_train_onehot,
            validation_data=(X_val_scaled, y_val_onehot),
            epochs=50,
            batch_size=32,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)

        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_per_class = f1_score(y_val, y_pred, average=None)
        min_f1 = min(f1_per_class)
        kappa = cohen_kappa_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)

        fold_results.append({
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'min_f1': min_f1,
            'kappa': kappa,
            'balanced_acc': balanced_acc
        })

    if not fold_results:
        return None

    return {
        'strategy': strategy_name,
        'avg_f1_macro': np.mean([r['f1_macro'] for r in fold_results]),
        'std_f1_macro': np.std([r['f1_macro'] for r in fold_results]),
        'avg_f1_weighted': np.mean([r['f1_weighted'] for r in fold_results]),
        'avg_min_f1': np.mean([r['min_f1'] for r in fold_results]),
        'avg_kappa': np.mean([r['kappa'] for r in fold_results]),
        'avg_balanced_acc': np.mean([r['balanced_acc'] for r in fold_results]),
        'avg_f1_per_class': np.mean([r['f1_per_class'] for r in fold_results], axis=0),
        'n_folds': len(fold_results)
    }


def main():
    print("="*100)
    print("COMPREHENSIVE CLASS IMBALANCE STRATEGY COMPARISON - V5 (FULL DATA)")
    print("="*100)
    print("\nUsing 100% of data for reliable results")
    print("Metrics: F1 Macro, F1 Weighted, Min F1, Cohen's Kappa, Balanced Accuracy")
    print("3-fold patient-level CV\n")

    # Load FULL data
    print("Loading FULL dataset...")
    directory, result_dir, root = get_project_paths()
    data_paths = get_data_paths(root)

    data = prepare_dataset(
        depth_bb_file=data_paths['bb_depth_csv'],
        thermal_bb_file=data_paths['bb_thermal_csv'],
        csv_file=data_paths['csv_file'],
        selected_modalities=['metadata']
    )

    data = filter_frequent_misclassifications(data, result_dir)
    # NO SAMPLING - use full data

    print(f"Dataset: {len(data)} samples (FULL)")
    print(f"Unique patients: {data['Patient#'].nunique()}")

    exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    X = data[feature_cols].fillna(0).values
    label_map = {'I': 0, 'P': 1, 'R': 2}
    y = np.array([label_map[label] for label in data['Healing Phase Abs'].values])

    print(f"Features: {len(feature_cols)}")
    class_dist = Counter(y)
    print(f"Class distribution: I={class_dist[0]}, P={class_dist[1]}, R={class_dist[2]}")
    print(f"Imbalance ratio: {max(class_dist.values())/min(class_dist.values()):.1f}:1\n")

    folds = create_patient_folds(data, n_folds=3, random_state=42)

    strategies = [
        ("1. Baseline", False, 'none', 'balanced', False),
        ("2. RandomOverSampler", False, 'oversample', 'balanced', False),
        ("3. RandomUnderSampler", False, 'undersample', 'balanced', False),
        ("4. SMOTE", False, 'smote', 'balanced', False),
        ("5. ADASYN", False, 'adasyn', 'balanced', False),
        ("6. Borderline-SMOTE", False, 'borderline_smote', 'balanced', False),
        ("7. SMOTE+Tomek", False, 'smote_tomek', 'balanced', False),
        ("8. SMOTE+ENN", False, 'smote_enn', 'balanced', False),
        ("9. Alpha (balanced)", True, 'none', 'balanced', False),
        ("10. Alpha (sqrt)", True, 'none', 'sqrt', False),
        ("11. Focal Loss", False, 'none', 'balanced', True),
        ("12. Oversample+Alpha", True, 'oversample', 'balanced', False),
        ("13. SMOTE+Alpha", True, 'smote', 'balanced', False),
        ("14. Oversample+Focal", False, 'oversample', 'balanced', True),
        ("15. SMOTE+Focal", False, 'smote', 'balanced', True),
        ("16. Oversample+Alpha(sqrt)", True, 'oversample', 'sqrt', False),
        ("17. SMOTE+Alpha(sqrt)", True, 'smote', 'sqrt', False),
        ("18. SMOTE+Tomek+Alpha", True, 'smote_tomek', 'balanced', False),
    ]

    results = []
    total_start = time.time()

    print("="*100)
    print("RUNNING EXPERIMENTS")
    print("="*100)

    for strategy_name, use_alpha, sampling, alpha_scheme, use_focal in strategies:
        start = time.time()
        print(f"\nTesting: {strategy_name}...", end=" ", flush=True)
        result = evaluate_strategy(
            X, y, data, folds, strategy_name, use_alpha, sampling,
            alpha_scheme, use_focal
        )
        elapsed = time.time() - start
        if result:
            results.append(result)
            print(f"({elapsed:.1f}s) F1m={result['avg_f1_macro']:.3f} Kappa={result['avg_kappa']:.3f}")
        else:
            print(f"FAILED ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n\nTotal time: {total_elapsed/60:.1f} minutes")

    # Get baseline for improvement calculation
    baseline = results[0]

    # Calculate improvements
    for r in results[1:]:
        r['imp_f1_macro'] = ((r['avg_f1_macro'] - baseline['avg_f1_macro']) / baseline['avg_f1_macro']) * 100
        r['imp_f1_weighted'] = ((r['avg_f1_weighted'] - baseline['avg_f1_weighted']) / baseline['avg_f1_weighted']) * 100
        r['imp_min_f1'] = ((r['avg_min_f1'] - baseline['avg_min_f1']) / baseline['avg_min_f1']) * 100
        r['imp_kappa'] = ((r['avg_kappa'] - baseline['avg_kappa']) / baseline['avg_kappa']) * 100
        r['imp_balanced_acc'] = ((r['avg_balanced_acc'] - baseline['avg_balanced_acc']) / baseline['avg_balanced_acc']) * 100
        r['avg_improvement'] = (r['imp_f1_macro'] + r['imp_f1_weighted'] + r['imp_min_f1'] +
                                r['imp_kappa'] + r['imp_balanced_acc']) / 5

    baseline['avg_improvement'] = 0

    # Sort by average improvement
    results_sorted = sorted(results[1:], key=lambda x: x['avg_improvement'], reverse=True)

    print("\n" + "="*120)
    print("RESULTS - PERCENTAGE IMPROVEMENT FROM BASELINE (sorted by avg improvement)")
    print("="*120)
    print(f"\nBaseline: F1m={baseline['avg_f1_macro']:.4f}, F1w={baseline['avg_f1_weighted']:.4f}, "
          f"MinF1={baseline['avg_min_f1']:.4f}, κ={baseline['avg_kappa']:.4f}, BA={baseline['avg_balanced_acc']:.4f}")

    print(f"\n{'Strategy':<25} {'F1 Macro':<12} {'F1 Wght':<12} {'Min F1':<12} {'Kappa':<12} {'Bal Acc':<12} {'AVG IMP':<10}")
    print("-"*120)

    for r in results_sorted:
        print(f"{r['strategy']:<25} {r['imp_f1_macro']:>+8.1f}%   {r['imp_f1_weighted']:>+8.1f}%   "
              f"{r['imp_min_f1']:>+8.1f}%   {r['imp_kappa']:>+8.1f}%   {r['imp_balanced_acc']:>+8.1f}%   "
              f"{r['avg_improvement']:>+7.1f}%")

    print("\n" + "="*120)
    print("TOP 3 STRATEGIES BY OVERALL IMPROVEMENT")
    print("="*120)

    for rank, r in enumerate(results_sorted[:3], 1):
        print(f"\n🏆 #{rank}: {r['strategy']}")
        print(f"   Average improvement: {r['avg_improvement']:+.1f}%")
        print(f"   ├─ F1 Macro:      {r['imp_f1_macro']:+.1f}% (absolute: {r['avg_f1_macro']:.4f})")
        print(f"   ├─ F1 Weighted:   {r['imp_f1_weighted']:+.1f}% (absolute: {r['avg_f1_weighted']:.4f})")
        print(f"   ├─ Min F1:        {r['imp_min_f1']:+.1f}% (absolute: {r['avg_min_f1']:.4f})")
        print(f"   ├─ Kappa:         {r['imp_kappa']:+.1f}% (absolute: {r['avg_kappa']:.4f})")
        print(f"   └─ Balanced Acc:  {r['imp_balanced_acc']:+.1f}% (absolute: {r['avg_balanced_acc']:.4f})")
        print(f"   F1 per class: I={r['avg_f1_per_class'][0]:.3f}, P={r['avg_f1_per_class'][1]:.3f}, R={r['avg_f1_per_class'][2]:.3f}")

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'imbalance_strategy_results_v5_full.txt')

    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE CLASS IMBALANCE STRATEGY COMPARISON - V5 (FULL DATA)\n")
        f.write("="*120 + "\n\n")
        f.write(f"Dataset: {len(data)} samples (FULL)\n")
        f.write(f"Class distribution: I={class_dist[0]}, P={class_dist[1]}, R={class_dist[2]}\n")
        f.write(f"Imbalance ratio: {max(class_dist.values())/min(class_dist.values()):.1f}:1\n")
        f.write(f"Cross-validation: 3-fold patient-level CV\n")
        f.write(f"Runtime: {total_elapsed/60:.1f} minutes\n\n")

        f.write(f"Baseline: F1m={baseline['avg_f1_macro']:.4f}, F1w={baseline['avg_f1_weighted']:.4f}, "
                f"MinF1={baseline['avg_min_f1']:.4f}, κ={baseline['avg_kappa']:.4f}, BA={baseline['avg_balanced_acc']:.4f}\n\n")

        f.write("RESULTS (sorted by avg improvement):\n")
        f.write("-"*120 + "\n")
        for r in results_sorted:
            f.write(f"{r['strategy']}: avg_imp={r['avg_improvement']:+.1f}%, "
                   f"F1m={r['avg_f1_macro']:.4f}, MinF1={r['avg_min_f1']:.4f}, κ={r['avg_kappa']:.4f}\n")

        f.write("\n\nTOP 3:\n")
        for i, r in enumerate(results_sorted[:3], 1):
            f.write(f"\n{i}. {r['strategy']} - {r['avg_improvement']:+.1f}% avg improvement\n")

    print(f"\n📁 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
