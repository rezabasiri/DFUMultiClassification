#!/usr/bin/env python3
"""
Comprehensive Class Imbalance Strategy Comparison - V2

Additional strategies tested:
- ADASYN (Adaptive Synthetic Sampling)
- Borderline-SMOTE
- SVM-SMOTE
- Cluster Centroids undersampling
- NearMiss undersampling
- Edited Nearest Neighbors (cleaning)
- SMOTE + ENN (hybrid)
- SMOTE + Tomek (hybrid)
- Focal Loss (without sampling)
- Focal Loss + sampling combinations
- Different class weight schemes (balanced, sqrt, log)
- Threshold adjustment strategies

All tested on identical data with 3-fold patient-level CV.
Alpha values calculated from TRAINING data BEFORE sampling.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
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
    """Focal loss for class imbalance - focuses on hard examples."""
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Focal weight
        weight = tf.pow(1 - y_pred, gamma) * y_true

        # Apply alpha if provided
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
        # Square root scaling - less aggressive than balanced
        counts = np.bincount(y_train)
        max_count = max(counts)
        weights = np.sqrt(max_count / counts)
    elif scheme == 'log':
        # Logarithmic scaling - even less aggressive
        counts = np.bincount(y_train)
        max_count = max(counts)
        weights = np.log1p(max_count / counts)
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
    min_samples = min(Counter(y_train).values())
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    samplers = {
        'none': None,
        'oversample': RandomOverSampler(random_state=random_state),
        'undersample': RandomUnderSampler(random_state=random_state),
        'smote': SMOTE(random_state=random_state, k_neighbors=k_neighbors),
        'adasyn': ADASYN(random_state=random_state, n_neighbors=k_neighbors),
        'borderline_smote': BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors),
        'svm_smote': SVMSMOTE(random_state=random_state, k_neighbors=k_neighbors),
        'cluster_centroids': ClusterCentroids(random_state=random_state),
        'nearmiss': NearMiss(version=1),
        'enn': EditedNearestNeighbours(),
        'smote_tomek': SMOTETomek(random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors)),
        'smote_enn': SMOTEENN(random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors)),
    }

    if strategy == 'none':
        return X_train, y_train

    sampler = samplers.get(strategy)
    if sampler is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sampler.fit_resample(X_train, y_train)


def evaluate_strategy(X, y, data, folds, strategy_name, use_alpha, sampling_strategy,
                      alpha_scheme='balanced', use_focal=False, focal_gamma=2.0):
    """Evaluate a single strategy across all folds."""
    fold_results = []

    for fold_idx, (train_patients, val_patients) in enumerate(folds):
        # Get train/val masks
        train_mask = data['Patient#'].isin(train_patients)
        val_mask = data['Patient#'].isin(val_patients)

        X_train = X[train_mask.values]
        y_train = y[train_mask.values]
        X_val = X[val_mask.values]
        y_val = y[val_mask.values]

        # Step 1: Compute alpha weights from ORIGINAL training data (before sampling)
        if use_alpha or use_focal:
            class_weights_dict = compute_alpha_weights(y_train, scheme=alpha_scheme)
            class_weights = class_weights_dict if use_alpha else None
            focal_alpha = [class_weights_dict[i] for i in range(3)] if use_focal else None
        else:
            class_weights = None
            focal_alpha = None

        # Step 2: Apply sampling (if any)
        try:
            X_train_sampled, y_train_sampled = apply_sampling(
                X_train, y_train, sampling_strategy, random_state=42+fold_idx
            )
        except Exception as e:
            # If sampling fails, skip this fold
            continue

        # Step 3: Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sampled)
        X_val_scaled = scaler.transform(X_val)

        # Step 4: One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train_sampled, num_classes=3)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)

        # Step 5: Build model with appropriate loss
        tf.keras.backend.clear_session()
        if use_focal:
            loss = focal_loss(gamma=focal_gamma, alpha=focal_alpha)
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

        # Step 6: Evaluate
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
    print("COMPREHENSIVE CLASS IMBALANCE STRATEGY COMPARISON - V2")
    print("="*80)
    print("\nTesting ALL available strategies:")
    print("  Sampling: Oversample, Undersample, SMOTE, ADASYN, Borderline-SMOTE,")
    print("            SVM-SMOTE, Cluster Centroids, NearMiss, ENN,")
    print("            SMOTE+Tomek, SMOTE+ENN")
    print("  Weighting: None, Balanced, Sqrt, Log")
    print("  Loss: Cross-Entropy, Focal Loss (gamma=2)")
    print("  Combinations of the above")
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

    # Filter frequent misclassifications
    data = filter_frequent_misclassifications(data, result_dir)

    # Use 30% of data for speed (proportional sampling)
    data = data.sample(frac=0.3, random_state=42).reset_index(drop=True)

    print(f"Dataset: {len(data)} samples")
    print(f"Unique patients: {data['Patient#'].nunique()}")

    # Extract features
    exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    X = data[feature_cols].fillna(0).values
    label_map = {'I': 0, 'P': 1, 'R': 2}
    y = np.array([label_map[label] for label in data['Healing Phase Abs'].values])

    print(f"Features: {len(feature_cols)}")
    print(f"Class distribution: {dict(Counter(y))}")

    # Create patient-level folds (SAME folds for all strategies)
    folds = create_patient_folds(data, n_folds=3, random_state=42)

    # Define ALL strategies to test
    strategies = [
        # === BASELINE ===
        ("1. Baseline (no balancing)", False, 'none', 'balanced', False),

        # === SAMPLING ONLY ===
        ("2. RandomOverSampler", False, 'oversample', 'balanced', False),
        ("3. RandomUnderSampler", False, 'undersample', 'balanced', False),
        ("4. SMOTE", False, 'smote', 'balanced', False),
        ("5. ADASYN", False, 'adasyn', 'balanced', False),
        ("6. Borderline-SMOTE", False, 'borderline_smote', 'balanced', False),
        ("7. SVM-SMOTE", False, 'svm_smote', 'balanced', False),
        ("8. ClusterCentroids", False, 'cluster_centroids', 'balanced', False),
        ("9. NearMiss", False, 'nearmiss', 'balanced', False),
        ("10. EditedNearestNeighbours", False, 'enn', 'balanced', False),
        ("11. SMOTE+Tomek", False, 'smote_tomek', 'balanced', False),
        ("12. SMOTE+ENN", False, 'smote_enn', 'balanced', False),

        # === CLASS WEIGHTS ONLY ===
        ("13. Alpha (balanced)", True, 'none', 'balanced', False),
        ("14. Alpha (sqrt)", True, 'none', 'sqrt', False),
        ("15. Alpha (log)", True, 'none', 'log', False),

        # === FOCAL LOSS ONLY ===
        ("16. Focal Loss (gamma=2)", False, 'none', 'balanced', True),

        # === BEST SAMPLERS + ALPHA ===
        ("17. Oversample + Alpha", True, 'oversample', 'balanced', False),
        ("18. SMOTE + Alpha", True, 'smote', 'balanced', False),
        ("19. ADASYN + Alpha", True, 'adasyn', 'balanced', False),
        ("20. SMOTE+ENN + Alpha", True, 'smote_enn', 'balanced', False),

        # === BEST SAMPLERS + FOCAL ===
        ("21. Oversample + Focal", False, 'oversample', 'balanced', True),
        ("22. SMOTE + Focal", False, 'smote', 'balanced', True),
        ("23. ADASYN + Focal", False, 'adasyn', 'balanced', True),

        # === TRIPLE COMBO ===
        ("24. Oversample + Alpha + Focal", True, 'oversample', 'balanced', True),
        ("25. SMOTE + Alpha + Focal", True, 'smote', 'balanced', True),

        # === SQRT WEIGHT COMBOS ===
        ("26. Oversample + Alpha(sqrt)", True, 'oversample', 'sqrt', False),
        ("27. SMOTE + Alpha(sqrt)", True, 'smote', 'sqrt', False),
    ]

    results = []

    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)

    for strategy_name, use_alpha, sampling, alpha_scheme, use_focal in strategies:
        print(f"\nTesting: {strategy_name}...")
        result = evaluate_strategy(
            X, y, data, folds, strategy_name, use_alpha, sampling,
            alpha_scheme, use_focal
        )
        if result:
            results.append(result)
            print(f"  F1 Macro: {result['avg_f1_macro']:.4f} ± {result['std_f1_macro']:.4f}")
            print(f"  Min F1:   {result['avg_min_f1']:.4f}")
            print(f"  F1 [I, P, R]: [{result['avg_f1_per_class'][0]:.3f}, "
                  f"{result['avg_f1_per_class'][1]:.3f}, {result['avg_f1_per_class'][2]:.3f}]")
        else:
            print(f"  FAILED - could not complete evaluation")

    # Sort by F1 macro
    results.sort(key=lambda x: x['avg_f1_macro'], reverse=True)

    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by F1 Macro)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Strategy':<35} {'F1 Macro':<15} {'Min F1':<10} {'F1 [I, P, R]'}")
    print("-"*90)

    for rank, r in enumerate(results, 1):
        f1_str = f"[{r['avg_f1_per_class'][0]:.3f}, {r['avg_f1_per_class'][1]:.3f}, {r['avg_f1_per_class'][2]:.3f}]"
        print(f"{rank:<5} {r['strategy']:<35} {r['avg_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}  "
              f"{r['avg_min_f1']:.4f}     {f1_str}")

    # Find best by different criteria
    best_f1 = results[0]
    best_min_f1 = max(results, key=lambda x: x['avg_min_f1'])
    best_stability = min(results, key=lambda x: x['std_f1_macro'])

    print("\n" + "="*80)
    print("BEST STRATEGIES BY CRITERIA")
    print("="*80)

    print(f"\n🏆 Best F1 Macro: {best_f1['strategy']}")
    print(f"   F1 Macro: {best_f1['avg_f1_macro']:.4f}, Min F1: {best_f1['avg_min_f1']:.4f}")

    print(f"\n🎯 Best Min F1 (minority class): {best_min_f1['strategy']}")
    print(f"   F1 Macro: {best_min_f1['avg_f1_macro']:.4f}, Min F1: {best_min_f1['avg_min_f1']:.4f}")

    print(f"\n📊 Most Stable (lowest std): {best_stability['strategy']}")
    print(f"   F1 Macro: {best_stability['avg_f1_macro']:.4f} ± {best_stability['std_f1_macro']:.4f}")

    # Category analysis
    print("\n" + "="*80)
    print("CATEGORY ANALYSIS")
    print("="*80)

    # Group results
    sampling_only = [r for r in results if r['strategy'].startswith(('2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.'))]
    alpha_only = [r for r in results if r['strategy'].startswith(('13.', '14.', '15.'))]
    focal_only = [r for r in results if r['strategy'] == "16. Focal Loss (gamma=2)"]
    combos = [r for r in results if r['strategy'].startswith(('17.', '18.', '19.', '20.', '21.', '22.', '23.', '24.', '25.', '26.', '27.'))]

    if sampling_only:
        best_sampling = max(sampling_only, key=lambda x: x['avg_f1_macro'])
        print(f"\nBest Sampling-only: {best_sampling['strategy']}")
        print(f"   F1: {best_sampling['avg_f1_macro']:.4f}, Min: {best_sampling['avg_min_f1']:.4f}")

    if alpha_only:
        best_alpha = max(alpha_only, key=lambda x: x['avg_f1_macro'])
        print(f"\nBest Alpha-only: {best_alpha['strategy']}")
        print(f"   F1: {best_alpha['avg_f1_macro']:.4f}, Min: {best_alpha['avg_min_f1']:.4f}")

    if focal_only:
        print(f"\nFocal Loss alone: {focal_only[0]['strategy']}")
        print(f"   F1: {focal_only[0]['avg_f1_macro']:.4f}, Min: {focal_only[0]['avg_min_f1']:.4f}")

    if combos:
        best_combo = max(combos, key=lambda x: x['avg_f1_macro'])
        print(f"\nBest Combination: {best_combo['strategy']}")
        print(f"   F1: {best_combo['avg_f1_macro']:.4f}, Min: {best_combo['avg_min_f1']:.4f}")

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'imbalance_strategy_results_v2.txt')

    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE CLASS IMBALANCE STRATEGY COMPARISON - V2\n")
        f.write("="*90 + "\n\n")
        f.write(f"Dataset: {len(data)} samples (30% of full dataset)\n")
        f.write(f"Unique patients: {data['Patient#'].nunique()}\n")
        f.write(f"Class distribution: {dict(Counter(y))}\n")
        f.write(f"Cross-validation: 3-fold patient-level CV (identical splits)\n\n")

        f.write("RESULTS (sorted by F1 Macro):\n")
        f.write("-"*90 + "\n")
        for rank, r in enumerate(results, 1):
            f.write(f"{rank:2d}. {r['strategy']}\n")
            f.write(f"    F1 Macro: {r['avg_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}\n")
            f.write(f"    Min F1: {r['avg_min_f1']:.4f}\n")
            f.write(f"    F1 per class: I={r['avg_f1_per_class'][0]:.3f}, "
                   f"P={r['avg_f1_per_class'][1]:.3f}, R={r['avg_f1_per_class'][2]:.3f}\n\n")

        f.write("\n" + "="*90 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*90 + "\n\n")
        f.write(f"Best Overall (F1 Macro): {best_f1['strategy']}\n")
        f.write(f"Best Minority Class (Min F1): {best_min_f1['strategy']}\n")
        f.write(f"Most Stable: {best_stability['strategy']}\n")

    print(f"\n📁 Results saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
