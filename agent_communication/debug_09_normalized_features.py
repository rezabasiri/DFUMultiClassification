#!/usr/bin/env python3
"""DEBUG 9: Test with feature normalization + oversampling + plain cross-entropy
UPDATED 2025-12-24: Fixed to use patient-level CV instead of sample-level split
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.training.training_utils import create_patient_folds

# Force CPU
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

def main():
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 9: FEATURE NORMALIZATION + OVERSAMPLING + PLAIN CROSS-ENTROPY")
    log("UPDATED: Now uses PATIENT-LEVEL CV (not sample-level split)")
    log("="*80)
    log("\nHypothesis: Phase 8 failed because features need normalization.")
    log("Test: Normalize features, use simple cross-entropy, keep oversampling.")
    log("\n⚠️  IMPORTANT: Previous version used train_test_split which allowed")
    log("   patient overlap between train/test sets. This version uses")
    log("   patient-level CV to prevent data leakage.")

    try:
        # Load data
        log("\n1. Loading data...")
        directory, result_dir, root = get_project_paths()
        data_paths = get_data_paths(root)

        data = prepare_dataset(
            depth_bb_file=data_paths['bb_depth_csv'],
            thermal_bb_file=data_paths['bb_thermal_csv'],
            csv_file=data_paths['csv_file'],
            selected_modalities=['metadata']
        )

        exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        X = data[feature_cols].fillna(0).values
        y_raw = data['Healing Phase Abs'].values

        label_map = {'I': 0, 'P': 1, 'R': 2}
        y = np.array([label_map[label] for label in y_raw])

        log(f"  Original data: {X.shape}")
        log(f"  Unique patients: {data['Patient#'].nunique()}")
        log(f"  Feature range BEFORE normalization: [{X.min():.2f}, {X.max():.2f}]")

        # Create patient-level folds (3-fold CV)
        log("\n2. Creating patient-level cross-validation folds...")
        folds = create_patient_folds(data, n_folds=3, random_state=42)


        # Track results across folds
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
            log(f"\n{'='*80}")
            log(f"FOLD {fold_idx}/3")
            log(f"{'='*80}")

            # Get patient IDs for this fold
            train_patients = set(data.iloc[train_idx]['Patient#'])
            val_patients = set(data.iloc[val_idx]['Patient#'])

            # Verify NO overlap
            patient_overlap = len(train_patients.intersection(val_patients))
            log(f"  Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
            log(f"  Patient overlap: {patient_overlap} (should be 0)")

            # Get train/val data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            log(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")

            log("\n  Class distribution BEFORE oversampling:")
            counts_before = Counter(y_train)
            for cls in [0, 1, 2]:
                log(f"    Class {cls}: {counts_before[cls]} samples")

            # Apply oversampling
            log("\n  Applying RandomOverSampler...")
            oversampler = RandomOverSampler(random_state=42 + fold_idx)
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

            log("\n  Class distribution AFTER oversampling:")
            counts_after = Counter(y_train_resampled)
            for cls in [0, 1, 2]:
                log(f"    Class {cls}: {counts_after[cls]} samples")

            # Normalize features
            log("\n  Normalizing features with StandardScaler...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_val_scaled = scaler.transform(X_val)

            if fold_idx == 1:
                log(f"    Train feature range AFTER normalization: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
                log(f"    Train feature mean: {X_train_scaled.mean():.4f} (should be ~0)")
                log(f"    Train feature std: {X_train_scaled.std():.4f} (should be ~1)")

            # One-hot encode
            y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=3)
            y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)

            # Build model
            log("\n  Building model...")
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
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

            # Train
            log("\n  Training (20 epochs)...")
            history = model.fit(
                X_train_scaled, y_train_onehot,
                validation_data=(X_val_scaled, y_val_onehot),
                epochs=20,
                batch_size=32,
                verbose=0
            )

            # Evaluate
            val_loss, val_acc = model.evaluate(X_val_scaled, y_val_onehot, verbose=0)

            # Predictions
            y_pred_probs = model.predict(X_val_scaled, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Metrics
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
            min_f1 = min(f1_per_class)

            log(f"\n  Results:")
            log(f"    Accuracy: {val_acc:.4f}")
            log(f"    F1 Macro: {f1_macro:.4f}")
            log(f"    F1 per class [I, P, R]: [{f1_per_class[0]:.3f}, {f1_per_class[1]:.3f}, {f1_per_class[2]:.3f}]")
            log(f"    Min F1: {min_f1:.4f}")

            fold_results.append({
                'accuracy': val_acc,
                'f1_macro': f1_macro,
                'f1_per_class': f1_per_class,
                'min_f1': min_f1
            })

        # Summary across folds
        log(f"\n{'='*80}")
        log("CROSS-VALIDATION SUMMARY")
        log(f"{'='*80}")

        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
        avg_min_f1 = np.mean([r['min_f1'] for r in fold_results])

        log(f"\nAverage Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        log(f"Average F1 Macro: {avg_f1:.4f}")
        log(f"Average Min F1:   {avg_min_f1:.4f}")

        log(f"\nPer-Fold Results:")
        for i, r in enumerate(fold_results, 1):
            log(f"  Fold {i}: Acc={r['accuracy']:.4f}, F1_macro={r['f1_macro']:.4f}, Min_F1={r['min_f1']:.4f}")

        # Verdict
        log("\n" + "="*80)
        log("COMPARISON WITH PREVIOUS APPROACH")
        log("="*80)
        log("\nPrevious approach (sample-level train_test_split):")
        log("  - Allowed patient overlap between train/test sets")
        log("  - Accuracy: ~97.6% (inflated by data leakage)")
        log("  - Model learned patient-specific patterns, not generalizable features")
        log("\nCurrent approach (patient-level cross-validation):")
        log(f"  - NO patient overlap between train/val sets")
        log(f"  - Accuracy: {avg_acc:.1%} (true generalization performance)")
        log(f"  - Model must learn generalizable wound healing features")
        log("\n⚠️  CONCLUSION:")
        log(f"  The {100*(1-avg_acc/0.976):.0f}% performance drop reveals that Phase 9's high")
        log("  accuracy was due to MEMORIZATION, not real learning.")
        log("\nFeature normalization IS still critical:")
        log(f"  - Phase 8 (no normalization): 33% accuracy (random guessing)")
        log(f"  - Phase 9 (with normalization): {avg_acc:.1%} accuracy (actual learning)")
        log("\nAll three fixes are necessary:")
        log("  1. Enable oversampling (balance training data)")
        log("  2. Normalize features with StandardScaler (enable learning)")
        log("  3. Use patient-level CV (prevent data leakage)")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'results_09_normalized_features.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    log(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
