#!/usr/bin/env python3
"""DEBUG 6: Test training with focal loss + inverse frequency alpha"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.models.losses import get_focal_ordinal_loss

# Force CPU for debugging
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

def main():
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 6: TRAINING WITH FOCAL LOSS + INVERSE FREQUENCY ALPHA")
    log("="*80)

    try:
        # Load data using main.py's method
        log("\n1. Loading data via prepare_dataset()...")
        directory, result_dir, root = get_project_paths()
        data_paths = get_data_paths(root)

        depth_bb_file = data_paths['bb_depth_csv']
        thermal_bb_file = data_paths['bb_thermal_csv']
        csv_file = data_paths['csv_file']

        data = prepare_dataset(
            depth_bb_file=depth_bb_file,
            thermal_bb_file=thermal_bb_file,
            csv_file=csv_file,
            selected_modalities=['metadata']
        )

        exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        X = data[feature_cols].fillna(0).values
        y_raw = data['Healing Phase Abs'].values

        # Convert string labels to integers
        label_map = {'I': 0, 'P': 1, 'R': 2}
        if y_raw.dtype == object:
            y = np.array([label_map[label] for label in y_raw])
            log(f"  Labels converted: {set(y_raw)} -> {set(y)}")
        else:
            y = y_raw

        log(f"  Features: {X.shape}")
        log(f"  Labels: {y.shape}")

        # Calculate inverse frequency alpha values (same as main.py)
        log("\n2. Calculating inverse frequency alpha values...")
        counts = Counter(y)
        total_samples = len(y)
        class_frequencies = {cls: count/total_samples for cls, count in counts.items()}

        log(f"  Class distribution:")
        for cls in [0, 1, 2]:
            log(f"    Class {cls}: {counts[cls]} ({class_frequencies[cls]*100:.1f}%)")

        # Inverse frequency for each class (no capping)
        alpha_values = [1.0/class_frequencies[i] for i in [0, 1, 2]]

        # Normalize to sum=3.0
        alpha_sum = sum(alpha_values)
        alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]

        log(f"\n  Alpha values (inverse frequency, normalized to sum=3):")
        log(f"    [I, P, R] = [{alpha_values[0]:.3f}, {alpha_values[1]:.3f}, {alpha_values[2]:.3f}]")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        log(f"\n3. Data split:")
        log(f"  Train: {len(X_train)}, Class dist: {np.bincount(y_train)}")
        log(f"  Test: {len(X_test)}, Class dist: {np.bincount(y_test)}")

        # One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

        # Build model
        log("\n4. Building model with focal loss...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Use focal loss with inverse frequency alpha (like main.py should)
        focal_loss = get_focal_ordinal_loss(
            num_classes=3,
            ordinal_weight=0.05,  # Same as main.py default
            gamma=2.0,  # Same as main.py default
            alpha=alpha_values  # CRITICAL: Use inverse frequency weights
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss,  # Use focal loss instead of plain cross-entropy
            metrics=['accuracy']
        )

        log(f"  Model has {model.count_params()} parameters")
        log(f"  Loss: Focal + Ordinal (gamma=2.0, ordinal_weight=0.05)")
        log(f"  Alpha: {[round(a, 3) for a in alpha_values]}")

        # Train
        log("\n5. Training (20 epochs with focal loss)...")
        log("-" * 80)

        history = model.fit(
            X_train, y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=20,
            batch_size=32,
            verbose=0
        )

        # Print training progress
        for epoch in range(len(history.history['loss'])):
            log(f"  Epoch {epoch+1:2d}: loss={history.history['loss'][epoch]:.4f}, "
                f"acc={history.history['accuracy'][epoch]:.4f} | "
                f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                f"val_acc={history.history['val_accuracy'][epoch]:.4f}")

        # Evaluate
        log("\n6. Evaluation:")
        log("-" * 80)
        test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
        log(f"  Test Loss: {test_loss:.4f}")
        log(f"  Test Accuracy: {test_acc:.4f}")

        # Predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Detailed metrics
        log("\n7. Detailed Metrics:")
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        log(f"  F1 Macro: {f1_macro:.4f}")
        log(f"  F1 per class: {f1_per_class}")

        # Prediction distribution
        log("\n8. Prediction distribution:")
        pred_counts = np.bincount(y_pred, minlength=3)
        for cls in range(3):
            pct = pred_counts[cls] / len(y_pred) * 100
            log(f"  Class {cls}: {pred_counts[cls]} predictions ({pct:.1f}%)")

        # Issue detection
        log("\n9. Issue Detection:")
        min_f1 = min(f1_per_class)
        unpredicted_classes = [i for i, count in enumerate(pred_counts) if count == 0]

        if min_f1 == 0.0:
            log(f"❌ At least one class has F1=0: {f1_per_class}")
        else:
            log(f"✅ All classes have F1>0: {f1_per_class}")

        if unpredicted_classes:
            log(f"⚠️  Classes never predicted: {unpredicted_classes}")
        else:
            log(f"✅ All classes were predicted at least once")

        # Comparison with Phase 2
        log("\n10. Comparison with Phase 2 (plain cross-entropy):")
        log("  Phase 2 (no focal loss): F1=[0.0, 0.753, 0.0], Min F1=0.0")
        log(f"  Phase 6 (focal loss):     F1={[f'{x:.3f}' for x in f1_per_class]}, Min F1={min_f1:.3f}")

        if min_f1 > 0.0:
            log("\n✅ FOCAL LOSS WORKS! Minority classes are being learned.")
        else:
            log("\n❌ FOCAL LOSS FAILED! Still predicting only majority class.")

        # Verdict
        log("\n" + "="*80)
        if min_f1 > 0.0:
            log("✅ PASS: Focal loss + alpha successfully handles class imbalance")
            log("="*80)
            log("\nThis means the bug in main.py is NOT the focal loss itself.")
            log("Possible causes:")
            log("  1. Config overrides alpha to [1,1,1]")
            log("  2. Learning rate or other hyperparameters")
            log("  3. Distributed training issues")
            log("  4. Data pipeline creates imbalanced batches")
        else:
            log("❌ FAIL: Even with focal loss + alpha, model only predicts majority class")
            log("="*80)
            log("\nThis suggests:")
            log("  1. Alpha values may not be sufficient for this level of imbalance")
            log("  2. May need data oversampling in addition to loss weighting")
            log("  3. Model architecture or learning rate issues")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'results_06_focal_loss_training.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    log(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
