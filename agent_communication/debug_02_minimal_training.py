#!/usr/bin/env python3
"""DEBUG 2: Test if basic training works using main.py's data pipeline"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths

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
    log("DEBUG 2: MINIMAL TRAINING TEST (CPU only, using main.py data)")
    log("="*80)

    try:
        # Load data using main.py's method
        log("\n1. Loading data via prepare_dataset()...")
        directory, result_dir, root = get_project_paths()

        depth_bb_file = os.path.join(root, 'raw', 'bb_depth_annotation.csv')
        thermal_bb_file = os.path.join(root, 'raw', 'bb_thermal_annotation.csv')
        csv_file = os.path.join(root, 'raw', 'DataMaster_Processed_V12_WithMissing.csv')

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
        y = data['Healing Phase Abs'].values

        log(f"  Features: {X.shape}")
        log(f"  Labels: {y.shape}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        log(f"\n2. Data split:")
        log(f"  Train: {len(X_train)}, Class dist: {np.bincount(y_train)}")
        log(f"  Test: {len(X_test)}, Class dist: {np.bincount(y_test)}")

        # One-hot encode
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

        # Build model
        log("\n3. Building model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
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

        log(f"  Model has {model.count_params()} parameters")

        # Train
        log("\n4. Training (10 epochs)...")
        log("-" * 80)

        history = model.fit(
            X_train, y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=10,
            batch_size=32,
            verbose=0
        )

        # Show epoch history
        for epoch in range(len(history.history['loss'])):
            train_loss = history.history['loss'][epoch]
            val_loss = history.history['val_loss'][epoch]
            train_acc = history.history['accuracy'][epoch]
            val_acc = history.history['val_accuracy'][epoch]
            log(f"  Epoch {epoch+1:2d}: loss={train_loss:.4f}, acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Evaluate
        log("\n5. Evaluation:")
        log("-" * 80)

        test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
        log(f"  Test Loss: {test_loss:.4f}")
        log(f"  Test Accuracy: {test_acc:.4f}")

        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        log(f"\n6. Detailed Metrics:")
        log(f"  F1 Macro: {f1_macro:.4f}")
        log(f"  F1 per class: {f1_per_class}")

        # Prediction distribution
        pred_counts = np.bincount(y_pred, minlength=3)
        log(f"\n7. Prediction distribution:")
        for i, count in enumerate(pred_counts):
            pct = count/len(y_pred)*100
            log(f"  Class {i}: {count} predictions ({pct:.1f}%)")

        # Check for issues
        log("\n8. Issue Detection:")
        issues = []

        if test_acc < 0.25:
            issues.append(f"❌ Accuracy too low: {test_acc:.4f} (worse than random!)")

        if f1_macro < 0.20:
            issues.append(f"❌ F1 Macro too low: {f1_macro:.4f}")

        if min(f1_per_class) == 0.0:
            issues.append(f"❌ At least one class has F1=0: {f1_per_class}")

        if 0 in pred_counts:
            zero_classes = np.where(pred_counts == 0)[0]
            issues.append(f"⚠️  Classes never predicted: {zero_classes}")

        if issues:
            for issue in issues:
                log(issue)
            success = False
        else:
            log("  ✓ All checks passed!")
            success = True

        log("\n" + "="*80)
        if success:
            log("✅ PASS: Minimal training works correctly")
        else:
            log("❌ FAIL: Training has issues (but TensorFlow works)")
        log("="*80)

        return success, output

    except Exception as e:
        log(f"\n❌ EXCEPTION: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False, output

if __name__ == "__main__":
    success, output = main()

    # Save output
    output_file = 'agent_communication/results_02_minimal_training.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    print(f"\n✓ Results saved to: {output_file}")
    sys.exit(0 if success else 1)
