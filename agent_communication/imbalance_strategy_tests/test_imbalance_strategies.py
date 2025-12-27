#!/usr/bin/env python3
"""
Test different class imbalance strategies to find optimal approach.

Strategies tested:
1. No balancing (baseline)
2. Class weights (alpha) only
3. Oversampling only
4. Undersampling only
5. SMOTE only
6. Combo sampling only
7. Class weights + Oversampling
8. Class weights + Undersampling
9. Class weights + SMOTE
10. Class weights + Combo

Key requirement: Alpha values calculated from TRAINING data BEFORE sampling.
Uses patient-level CV to prevent data leakage.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
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


def compute_alpha_weights(y_train):
    """Compute class weights (alpha) from training data BEFORE any sampling."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def build_model(input_dim):
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
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def apply_sampling(X_train, y_train, strategy, random_state=42):
    """Apply sampling strategy. Returns resampled X, y."""
    if strategy == 'none':
        return X_train, y_train
    elif strategy == 'oversample':
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == 'smote':
        # SMOTE needs at least k_neighbors samples per class
        min_samples = min(Counter(y_train).values())
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    elif strategy == 'combo':
        # SMOTETomek: SMOTE + Tomek links cleaning
        min_samples = min(Counter(y_train).values())
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        sampler = SMOTETomek(random_state=random_state, smote=SMOTE(k_neighbors=k_neighbors))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sampler.fit_resample(X_train, y_train)


def evaluate_strategy(X, y, data, folds, strategy_name, use_alpha, sampling_strategy):
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
        class_weights = compute_alpha_weights(y_train) if use_alpha else None

        # Step 2: Apply sampling (if any)
        try:
            X_train_sampled, y_train_sampled = apply_sampling(
                X_train, y_train, sampling_strategy, random_state=42+fold_idx
            )
        except Exception as e:
            # If sampling fails (e.g., not enough samples), skip this fold
            print(f"    Fold {fold_idx+1}: Sampling failed - {e}")
            continue

        # Step 3: Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sampled)
        X_val_scaled = scaler.transform(X_val)

        # Step 4: One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train_sampled, num_classes=3)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)

        # Step 5: Build and train model
        tf.keras.backend.clear_session()
        model = build_model(X_train_scaled.shape[1])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        model.fit(
            X_train_scaled, y_train_onehot,
            validation_data=(X_val_scaled, y_val_onehot),
            epochs=50,
            batch_size=32,
            class_weight=class_weights,  # Use alpha weights if enabled
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
    print("CLASS IMBALANCE STRATEGY COMPARISON")
    print("="*80)
    print("\nTesting which combination of strategies works best:")
    print("  - Sampling: None, Oversample, Undersample, SMOTE, Combo (SMOTETomek)")
    print("  - Alpha weights: Calculated from training data BEFORE sampling")
    print("  - Using patient-level CV to prevent data leakage")
    print("  - Using 30% of data for speed\n")

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

    # Filter frequent misclassifications (optional - use filtered data)
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

    # Create patient-level folds
    folds = create_patient_folds(data, n_folds=3, random_state=42)

    # Define strategies to test
    strategies = [
        # (name, use_alpha, sampling_strategy)
        ("Baseline (no balancing)", False, 'none'),
        ("Alpha only", True, 'none'),
        ("Oversample only", False, 'oversample'),
        ("Undersample only", False, 'undersample'),
        ("SMOTE only", False, 'smote'),
        ("Combo (SMOTETomek) only", False, 'combo'),
        ("Alpha + Oversample", True, 'oversample'),
        ("Alpha + Undersample", True, 'undersample'),
        ("Alpha + SMOTE", True, 'smote'),
        ("Alpha + Combo", True, 'combo'),
    ]

    results = []

    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)

    for strategy_name, use_alpha, sampling in strategies:
        print(f"\nTesting: {strategy_name}...")
        result = evaluate_strategy(X, y, data, folds, strategy_name, use_alpha, sampling)
        if result:
            results.append(result)
            print(f"  F1 Macro: {result['avg_f1_macro']:.4f} ± {result['std_f1_macro']:.4f}")
            print(f"  Min F1:   {result['avg_min_f1']:.4f}")
            print(f"  F1 per class [I, P, R]: [{result['avg_f1_per_class'][0]:.3f}, "
                  f"{result['avg_f1_per_class'][1]:.3f}, {result['avg_f1_per_class'][2]:.3f}]")
        else:
            print(f"  FAILED - could not complete evaluation")

    # Sort by F1 macro
    results.sort(key=lambda x: x['avg_f1_macro'], reverse=True)

    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by F1 Macro)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Strategy':<30} {'F1 Macro':<15} {'Min F1':<10} {'F1 [I, P, R]'}")
    print("-"*80)

    for rank, r in enumerate(results, 1):
        f1_str = f"[{r['avg_f1_per_class'][0]:.3f}, {r['avg_f1_per_class'][1]:.3f}, {r['avg_f1_per_class'][2]:.3f}]"
        print(f"{rank:<5} {r['strategy']:<30} {r['avg_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}  "
              f"{r['avg_min_f1']:.4f}     {f1_str}")

    # Find best strategy
    best = results[0]
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\n🏆 Best Strategy: {best['strategy']}")
    print(f"   F1 Macro: {best['avg_f1_macro']:.4f}")
    print(f"   Min F1:   {best['avg_min_f1']:.4f}")

    # Compare alpha-only vs sampling-only vs combined
    alpha_only = next((r for r in results if r['strategy'] == "Alpha only"), None)
    oversample_only = next((r for r in results if r['strategy'] == "Oversample only"), None)
    alpha_oversample = next((r for r in results if r['strategy'] == "Alpha + Oversample"), None)

    print("\n📊 Key Comparisons:")
    if alpha_only:
        print(f"   Alpha only:        F1={alpha_only['avg_f1_macro']:.4f}, Min={alpha_only['avg_min_f1']:.4f}")
    if oversample_only:
        print(f"   Oversample only:   F1={oversample_only['avg_f1_macro']:.4f}, Min={oversample_only['avg_min_f1']:.4f}")
    if alpha_oversample:
        print(f"   Alpha + Oversample: F1={alpha_oversample['avg_f1_macro']:.4f}, Min={alpha_oversample['avg_min_f1']:.4f}")

    # Save results to file
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'imbalance_strategy_results.txt')

    with open(output_file, 'w') as f:
        f.write("CLASS IMBALANCE STRATEGY COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {len(data)} samples (30% of full dataset)\n")
        f.write(f"Unique patients: {data['Patient#'].nunique()}\n")
        f.write(f"Class distribution: {dict(Counter(y))}\n")
        f.write(f"Cross-validation: 3-fold patient-level CV\n\n")

        f.write("RESULTS (sorted by F1 Macro):\n")
        f.write("-"*80 + "\n")
        for rank, r in enumerate(results, 1):
            f.write(f"{rank}. {r['strategy']}\n")
            f.write(f"   F1 Macro: {r['avg_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}\n")
            f.write(f"   Min F1: {r['avg_min_f1']:.4f}\n")
            f.write(f"   F1 per class: I={r['avg_f1_per_class'][0]:.3f}, "
                   f"P={r['avg_f1_per_class'][1]:.3f}, R={r['avg_f1_per_class'][2]:.3f}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST STRATEGY: {best['strategy']}\n")
        f.write("="*80 + "\n")

    print(f"\n📁 Results saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
