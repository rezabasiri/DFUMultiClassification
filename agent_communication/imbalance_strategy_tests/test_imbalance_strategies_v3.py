#!/usr/bin/env python3
"""
Fast Class Imbalance Strategy Comparison - V3

Focused on strategies that actually work quickly:
- Skipping slow samplers: ClusterCentroids, SVM-SMOTE, NearMiss (too slow)
- Keeping fast, effective samplers: RandomOverSampler, SMOTE, ADASYN, Borderline-SMOTE
- Testing class weights and focal loss

All tested on identical data with 3-fold patient-level CV.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
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
    """Focal loss for class imbalance."""
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
    """Compute class weights from training data BEFORE any sampling."""
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
    """Build simple model for metadata classification."""
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
    """Apply sampling strategy. Returns resampled X, y."""
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
    """Evaluate a single strategy across all folds."""
    fold_results = []

    for fold_idx, (train_patients, val_patients) in enumerate(folds):
        train_mask = data['Patient#'].isin(train_patients)
        val_mask = data['Patient#'].isin(val_patients)

        X_train = X[train_mask.values]
        y_train = y[train_mask.values]
        X_val = X[val_mask.values]
        y_val = y[val_mask.values]

        # Compute alpha BEFORE sampling
        if use_alpha or use_focal:
            class_weights_dict = compute_alpha_weights(y_train, scheme=alpha_scheme)
            class_weights = class_weights_dict if use_alpha else None
            focal_alpha = [class_weights_dict[i] for i in range(3)] if use_focal else None
        else:
            class_weights = None
            focal_alpha = None

        # Apply sampling
        try:
            X_train_sampled, y_train_sampled = apply_sampling(
                X_train, y_train, sampling_strategy, random_state=42+fold_idx
            )
        except Exception as e:
            print(f"    Fold {fold_idx+1}: Sampling failed - {e}")
            continue

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sampled)
        X_val_scaled = scaler.transform(X_val)

        # One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train_sampled, num_classes=3)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)

        # Build model
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
        f1_per_class = f1_score(y_val, y_pred, average=None)
        min_f1 = min(f1_per_class)

        fold_results.append({
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class,
            'min_f1': min_f1
        })

    if not fold_results:
        return None

    return {
        'strategy': strategy_name,
        'avg_f1_macro': np.mean([r['f1_macro'] for r in fold_results]),
        'std_f1_macro': np.std([r['f1_macro'] for r in fold_results]),
        'avg_min_f1': np.mean([r['min_f1'] for r in fold_results]),
        'avg_f1_per_class': np.mean([r['f1_per_class'] for r in fold_results], axis=0),
        'n_folds': len(fold_results)
    }


def main():
    print("="*80)
    print("FAST CLASS IMBALANCE STRATEGY COMPARISON - V3")
    print("="*80)
    print("\nTesting practical strategies (skipping slow ones):")
    print("  Fast samplers: Oversample, Undersample, SMOTE, ADASYN, Borderline-SMOTE")
    print("  Hybrids: SMOTE+Tomek, SMOTE+ENN")
    print("  Weighting: None, Balanced, Sqrt")
    print("  Loss: Cross-Entropy, Focal Loss")
    print("\n  Using patient-level 3-fold CV on identical data splits\n")

    # Load data
    print("Loading data...")
    directory, result_dir, root = get_project_paths()
    data_paths = get_data_paths(root)

    data = prepare_dataset(
        depth_bb_file=data_paths['bb_depth_csv'],
        thermal_bb_file=data_paths['bb_thermal_csv'],
        csv_file=data_paths['csv_file'],
        selected_modalities=['metadata']
    )

    data = filter_frequent_misclassifications(data, result_dir)
    data = data.sample(frac=0.3, random_state=42).reset_index(drop=True)

    print(f"Dataset: {len(data)} samples")
    print(f"Unique patients: {data['Patient#'].nunique()}")

    exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    X = data[feature_cols].fillna(0).values
    label_map = {'I': 0, 'P': 1, 'R': 2}
    y = np.array([label_map[label] for label in data['Healing Phase Abs'].values])

    print(f"Features: {len(feature_cols)}")
    print(f"Class distribution: {dict(Counter(y))}")

    folds = create_patient_folds(data, n_folds=3, random_state=42)

    # Define strategies - (name, use_alpha, sampling, alpha_scheme, use_focal)
    strategies = [
        # Baseline
        ("1. Baseline (none)", False, 'none', 'balanced', False),

        # Sampling only
        ("2. RandomOverSampler", False, 'oversample', 'balanced', False),
        ("3. RandomUnderSampler", False, 'undersample', 'balanced', False),
        ("4. SMOTE", False, 'smote', 'balanced', False),
        ("5. ADASYN", False, 'adasyn', 'balanced', False),
        ("6. Borderline-SMOTE", False, 'borderline_smote', 'balanced', False),
        ("7. SMOTE+Tomek", False, 'smote_tomek', 'balanced', False),
        ("8. SMOTE+ENN", False, 'smote_enn', 'balanced', False),

        # Weights only
        ("9. Alpha (balanced)", True, 'none', 'balanced', False),
        ("10. Alpha (sqrt)", True, 'none', 'sqrt', False),

        # Focal loss only
        ("11. Focal Loss", False, 'none', 'balanced', True),

        # Best samplers + Alpha
        ("12. Oversample + Alpha", True, 'oversample', 'balanced', False),
        ("13. SMOTE + Alpha", True, 'smote', 'balanced', False),
        ("14. SMOTE+ENN + Alpha", True, 'smote_enn', 'balanced', False),

        # Best samplers + Focal
        ("15. Oversample + Focal", False, 'oversample', 'balanced', True),
        ("16. SMOTE + Focal", False, 'smote', 'balanced', True),

        # Sqrt weight combos
        ("17. Oversample + Alpha(sqrt)", True, 'oversample', 'sqrt', False),
        ("18. SMOTE + Alpha(sqrt)", True, 'smote', 'sqrt', False),
    ]

    results = []
    total_start = time.time()

    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)

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
            print(f"({elapsed:.1f}s)")
            print(f"  F1 Macro: {result['avg_f1_macro']:.4f} ± {result['std_f1_macro']:.4f}")
            print(f"  Min F1:   {result['avg_min_f1']:.4f}")
            print(f"  F1 [I, P, R]: [{result['avg_f1_per_class'][0]:.3f}, "
                  f"{result['avg_f1_per_class'][1]:.3f}, {result['avg_f1_per_class'][2]:.3f}]")
        else:
            print(f"FAILED ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n\nTotal time: {total_elapsed/60:.1f} minutes")

    # Sort by F1 macro
    results.sort(key=lambda x: x['avg_f1_macro'], reverse=True)

    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by F1 Macro)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Strategy':<30} {'F1 Macro':<15} {'Min F1':<10} {'F1 [I, P, R]'}")
    print("-"*90)

    for rank, r in enumerate(results, 1):
        f1_str = f"[{r['avg_f1_per_class'][0]:.3f}, {r['avg_f1_per_class'][1]:.3f}, {r['avg_f1_per_class'][2]:.3f}]"
        print(f"{rank:<5} {r['strategy']:<30} {r['avg_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}  "
              f"{r['avg_min_f1']:.4f}     {f1_str}")

    # Best by criteria
    best_f1 = results[0]
    best_min_f1 = max(results, key=lambda x: x['avg_min_f1'])
    best_stability = min(results, key=lambda x: x['std_f1_macro'])

    print("\n" + "="*80)
    print("BEST STRATEGIES")
    print("="*80)

    print(f"\n🏆 Best F1 Macro: {best_f1['strategy']}")
    print(f"   F1: {best_f1['avg_f1_macro']:.4f}, Min F1: {best_f1['avg_min_f1']:.4f}")

    print(f"\n🎯 Best Min F1 (minority class): {best_min_f1['strategy']}")
    print(f"   F1: {best_min_f1['avg_f1_macro']:.4f}, Min F1: {best_min_f1['avg_min_f1']:.4f}")

    print(f"\n📊 Most Stable: {best_stability['strategy']}")
    print(f"   F1: {best_stability['avg_f1_macro']:.4f} ± {best_stability['std_f1_macro']:.4f}")

    # Key comparisons
    print("\n" + "="*80)
    print("KEY COMPARISONS")
    print("="*80)

    oversample = next((r for r in results if "RandomOverSampler" in r['strategy']), None)
    alpha = next((r for r in results if r['strategy'] == "9. Alpha (balanced)"), None)
    oversample_alpha = next((r for r in results if r['strategy'] == "12. Oversample + Alpha"), None)
    focal = next((r for r in results if r['strategy'] == "11. Focal Loss"), None)

    print("\nSampling vs Weighting vs Combined:")
    if oversample:
        print(f"  Oversample only:     F1={oversample['avg_f1_macro']:.4f}, Min={oversample['avg_min_f1']:.4f}")
    if alpha:
        print(f"  Alpha only:          F1={alpha['avg_f1_macro']:.4f}, Min={alpha['avg_min_f1']:.4f}")
    if oversample_alpha:
        print(f"  Oversample + Alpha:  F1={oversample_alpha['avg_f1_macro']:.4f}, Min={oversample_alpha['avg_min_f1']:.4f}")
    if focal:
        print(f"  Focal Loss only:     F1={focal['avg_f1_macro']:.4f}, Min={focal['avg_min_f1']:.4f}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\n✅ Use: {best_f1['strategy']}")
    if oversample and alpha and oversample_alpha:
        if oversample['avg_f1_macro'] > oversample_alpha['avg_f1_macro']:
            print("\n⚠️  NOTE: Combining sampling + alpha HURTS performance!")
            print("   Stick with sampling alone for best results.")

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'imbalance_strategy_results_v3.txt')

    with open(output_file, 'w') as f:
        f.write("FAST CLASS IMBALANCE STRATEGY COMPARISON - V3\n")
        f.write("="*90 + "\n\n")
        f.write(f"Dataset: {len(data)} samples (30% of full dataset)\n")
        f.write(f"Class distribution: {dict(Counter(y))}\n")
        f.write(f"Cross-validation: 3-fold patient-level CV\n")
        f.write(f"Total runtime: {total_elapsed/60:.1f} minutes\n\n")

        f.write("RESULTS (sorted by F1 Macro):\n")
        f.write("-"*90 + "\n")
        for rank, r in enumerate(results, 1):
            f.write(f"{rank:2d}. {r['strategy']}\n")
            f.write(f"    F1 Macro: {r['avg_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}\n")
            f.write(f"    Min F1: {r['avg_min_f1']:.4f}\n")
            f.write(f"    F1 per class: I={r['avg_f1_per_class'][0]:.3f}, "
                   f"P={r['avg_f1_per_class'][1]:.3f}, R={r['avg_f1_per_class'][2]:.3f}\n\n")

        f.write("\n" + "="*90 + "\n")
        f.write(f"BEST OVERALL: {best_f1['strategy']}\n")
        f.write(f"BEST MINORITY CLASS: {best_min_f1['strategy']}\n")

    print(f"\n📁 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
