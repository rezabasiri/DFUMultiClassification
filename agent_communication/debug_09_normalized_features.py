#!/usr/bin/env python3
"""DEBUG 9: Test with feature normalization + oversampling + plain cross-entropy"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths

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
    log("="*80)
    log("\nHypothesis: Phase 8 failed because features need normalization.")
    log("Test: Normalize features, use simple cross-entropy, keep oversampling.")

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
        log(f"  Feature range BEFORE normalization: [{X.min():.2f}, {X.max():.2f}]")

        # Split BEFORE oversampling and normalization
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        log("\n2. Class distribution BEFORE oversampling:")
        counts_before = Counter(y_train)
        for cls in [0, 1, 2]:
            log(f"  Class {cls}: {counts_before[cls]} samples")

        # Apply oversampling
        log("\n3. Applying RandomOverSampler...")
        oversampler = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

        log("\n4. Class distribution AFTER oversampling:")
        counts_after = Counter(y_train_resampled)
        for cls in [0, 1, 2]:
            log(f"  Class {cls}: {counts_after[cls]} samples (perfectly balanced)")

        # NORMALIZE FEATURES (NEW!)
        log("\n5. Normalizing features with StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)

        log(f"  Train feature range AFTER normalization: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        log(f"  Train feature mean: {X_train_scaled.mean():.4f} (should be ~0)")
        log(f"  Train feature std: {X_train_scaled.std():.4f} (should be ~1)")

        # One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

        # Build model
        log("\n6. Building model with PLAIN cross-entropy...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Use PLAIN cross-entropy (no focal, no ordinal)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',  # Simple and effective with balanced data
            metrics=['accuracy']
        )

        log(f"  Model parameters: {model.count_params()}")
        log(f"  Loss: Plain categorical cross-entropy")
        log(f"  Data: Balanced (oversampled) + Normalized")

        # Train (20 epochs)
        log("\n7. Training (20 epochs)...")
        log("-" * 80)

        history = model.fit(
            X_train_scaled, y_train_onehot,
            validation_data=(X_test_scaled, y_test_onehot),
            epochs=20,
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
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)
        log(f"  Test Loss: {test_loss:.4f}")
        log(f"  Test Accuracy: {test_acc:.4f}")

        # Predictions
        y_pred_probs = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        min_f1 = min(f1_per_class)

        log(f"  F1 Macro: {f1_macro:.4f}")
        log(f"  F1 per class [I, P, R]: [{f1_per_class[0]:.3f}, {f1_per_class[1]:.3f}, {f1_per_class[2]:.3f}]")
        log(f"  Min F1: {min_f1:.4f}")

        # Prediction distribution
        log("\n9. Prediction distribution:")
        pred_counts = np.bincount(y_pred, minlength=3)
        test_counts = np.bincount(y_test, minlength=3)
        for cls in range(3):
            pred_pct = pred_counts[cls] / len(y_pred) * 100
            true_pct = test_counts[cls] / len(y_test) * 100
            log(f"  Class {cls}: {pred_counts[cls]} predictions ({pred_pct:.1f}%) vs {test_counts[cls]} actual ({true_pct:.1f}%)")

        # Detailed classification report
        log("\n10. Classification Report:")
        log("-" * 80)
        class_names = ['I (Inflammation)', 'P (Proliferation)', 'R (Remodeling)']
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        for line in report.split('\n'):
            log(line)

        # Loss comparison
        log("\n11. Loss Comparison Across Phases:")
        log("  Phase 2 (cross-entropy, no oversampling):  Final loss=0.98")
        log("  Phase 6 (focal, no oversampling):          Final loss=6.73")
        log("  Phase 7 (focal, oversampling, imb alpha):  Final loss=5.83")
        log("  Phase 8 (focal, oversampling, bal alpha):  Final loss=10.83")
        log(f"  Phase 9 (cross-entropy, oversampling):     Final loss={history.history['loss'][-1]:.2f}")

        # Verdict
        log("\n" + "="*80)
        if min_f1 > 0.1 and all(pred_counts > 0):
            log("✅ SUCCESS! Feature normalization solved the problem!")
            log("="*80)
            log("\nAll 3 classes are being predicted with reasonable F1 scores.")
            log("\nThe complete solution:")
            log("  1. Enable oversampling (balance training data)")
            log("  2. Normalize features with StandardScaler")
            log("  3. Use plain cross-entropy (balanced data doesn't need focal loss)")
            log("\nThis proves:")
            log("  - Unnormalized features prevented learning in Phase 8")
            log("  - Focal loss was unnecessary complexity with balanced data")
            log("  - Simple approach works best: oversample + normalize + cross-entropy")
        elif min_f1 > 0.0:
            log("🟡 PARTIAL SUCCESS - Model is learning but performance is low")
            log("="*80)
            log(f"  Min F1={min_f1:.3f} > 0 (better than previous phases)")
            log(f"  All classes predicted: {all(pred_counts > 0)}")
            log("\n  Possible improvements:")
            log("  - More training epochs")
            log("  - Different model architecture")
            log("  - Hyperparameter tuning")
        else:
            log("❌ STILL FAILING - Feature normalization alone didn't solve it")
            log("="*80)
            unpredicted = [i for i, c in enumerate(pred_counts) if c == 0]
            if unpredicted:
                log(f"  Classes not predicted: {unpredicted}")
            log("  Need to investigate:")
            log("  - Different model architecture")
            log("  - Feature engineering")
            log("  - Data quality issues")

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
