#!/usr/bin/env python3
"""DEBUG 7: Test if enabling oversampling fixes the Min F1=0 issue"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.models.losses import get_focal_ordinal_loss

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
    log("DEBUG 7: TEST OVERSAMPLING FIX")
    log("="*80)

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

        # Calculate alpha values BEFORE oversampling (from original distribution)
        log("\n2. Calculating alpha values from ORIGINAL distribution...")
        counts_orig = Counter(y)
        total_orig = len(y)

        log("  Original class distribution:")
        for cls in [0, 1, 2]:
            log(f"    Class {cls}: {counts_orig[cls]} ({counts_orig[cls]/total_orig*100:.1f}%)")

        class_freq = {cls: count/total_orig for cls, count in counts_orig.items()}
        alpha_values = [1.0/class_freq[i] for i in [0, 1, 2]]
        alpha_sum = sum(alpha_values)
        alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]

        log(f"  Alpha values: [{alpha_values[0]:.3f}, {alpha_values[1]:.3f}, {alpha_values[2]:.3f}]")

        # Split BEFORE oversampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        log("\n3. Class distribution BEFORE oversampling:")
        counts_before = Counter(y_train)
        for cls in [0, 1, 2]:
            log(f"  Class {cls}: {counts_before[cls]} samples")

        # Apply oversampling to TRAINING data only
        log("\n4. Applying RandomOverSampler to training data...")
        oversampler = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

        log("\n5. Class distribution AFTER oversampling:")
        counts_after = Counter(y_train_resampled)
        for cls in [0, 1, 2]:
            log(f"  Class {cls}: {counts_after[cls]} samples ({counts_after[cls]/counts_before[cls]:.1f}x increase)")

        # One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

        # Build model
        log("\n6. Building model with focal loss + alpha...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        focal_loss = get_focal_ordinal_loss(
            num_classes=3,
            ordinal_weight=0.05,
            gamma=2.0,
            alpha=alpha_values
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_loss,
            metrics=['accuracy']
        )

        log(f"  Model parameters: {model.count_params()}")
        log(f"  Training samples: {len(X_train_resampled)} (after oversampling)")

        # Train (quick test - 10 epochs)
        log("\n7. Training (10 epochs - quick test)...")
        log("-" * 80)

        history = model.fit(
            X_train_resampled, y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=10,
            batch_size=32,
            verbose=0
        )

        for epoch in range(len(history.history['loss'])):
            log(f"  Epoch {epoch+1:2d}: loss={history.history['loss'][epoch]:.4f}, "
                f"acc={history.history['accuracy'][epoch]:.4f} | "
                f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                f"val_acc={history.history['val_accuracy'][epoch]:.4f}")

        # Evaluate
        log("\n8. Evaluation:")
        log("-" * 80)
        test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
        log(f"  Test Loss: {test_loss:.4f}")
        log(f"  Test Accuracy: {test_acc:.4f}")

        # Predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        min_f1 = min(f1_per_class)

        log(f"  F1 Macro: {f1_macro:.4f}")
        log(f"  F1 per class: [{f1_per_class[0]:.3f}, {f1_per_class[1]:.3f}, {f1_per_class[2]:.3f}]")
        log(f"  Min F1: {min_f1:.4f}")

        # Prediction distribution
        log("\n9. Prediction distribution:")
        pred_counts = np.bincount(y_pred, minlength=3)
        for cls in range(3):
            pct = pred_counts[cls] / len(y_pred) * 100
            log(f"  Class {cls}: {pred_counts[cls]} predictions ({pct:.1f}%)")

        # Comparison
        log("\n10. COMPARISON:")
        log("  Phase 2 (plain cross-entropy, no oversampling): Min F1=0.000")
        log("  Phase 6 (focal loss, no oversampling):          Min F1=0.000")
        log(f"  Phase 7 (focal loss + oversampling):            Min F1={min_f1:.3f}")

        # Verdict
        log("\n" + "="*80)
        if min_f1 > 0.0 and all(pred_counts > 0):
            log("✅ SUCCESS! Oversampling + focal loss solves the Min F1=0 issue")
            log("="*80)
            log("\nAll 3 classes are being predicted!")
            log("The fix is confirmed: apply_sampling=True in dataset_utils.py:714")
        else:
            log("❌ FAIL: Still having issues even with oversampling")
            log("="*80)
            if min_f1 == 0.0:
                unpredicted = [i for i, c in enumerate(pred_counts) if c == 0]
                log(f"  Classes not predicted: {unpredicted}")
            log("  May need to adjust hyperparameters or use stronger augmentation")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'results_07_oversampling_fix.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    log(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
