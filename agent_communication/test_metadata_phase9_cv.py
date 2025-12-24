#!/usr/bin/env python3
"""Test Phase 9-style metadata model with PROPER cross-validation to verify generalizability"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.data.dataset_utils import create_patient_folds
from src.evaluation.metrics import filter_frequent_misclassifications

print("="*80)
print("GENERALIZABILITY TEST: Phase 9-style Model with Patient-Level CV")
print("="*80)
print("\nThis test uses PATIENT-LEVEL cross-validation to ensure:")
print("  1. No patient appears in both train and test sets")
print("  2. Model must generalize to unseen patients")
print("  3. Results are not due to memorization")

# Load data
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

data = prepare_dataset(
    depth_bb_file=data_paths['bb_depth_csv'],
    thermal_bb_file=data_paths['bb_thermal_csv'],
    csv_file=data_paths['csv_file'],
    selected_modalities=['metadata']
)

data = filter_frequent_misclassifications(data, result_dir)
print(f"\nData after filtering: {len(data)} samples")
print(f"Unique patients: {data['Patient#'].nunique()}")

# Extract features
exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
numeric_cols = data.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")

# Convert labels
label_map = {'I': 0, 'P': 1, 'R': 2}
y_all = np.array([label_map[label] for label in data['Healing Phase Abs'].values])

# Create PATIENT-LEVEL folds (same as CV test)
folds = create_patient_folds(data, n_folds=3, random_state=42)

print(f"\n{'='*80}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*80}")

all_fold_results = []
all_y_true = []
all_y_pred = []

for fold_idx, (train_patients, val_patients) in enumerate(folds, 1):
    print(f"\n--- Fold {fold_idx} ---")

    # Get train/val data using PATIENT NUMBERS (not indices!)
    train_mask = data['Patient#'].isin(train_patients)
    val_mask = data['Patient#'].isin(val_patients)

    X_train = data.loc[train_mask, feature_cols].fillna(0).values
    X_val = data.loc[val_mask, feature_cols].fillna(0).values
    y_train = y_all[train_mask.values]
    y_val = y_all[val_mask.values]

    # Verify no patient overlap
    train_patient_set = set(train_patients)
    val_patient_set = set(val_patients)
    overlap = train_patient_set & val_patient_set
    print(f"  Train patients: {len(train_patient_set)}, Val patients: {len(val_patient_set)}")
    print(f"  Patient overlap: {len(overlap)} (should be 0)")

    if overlap:
        print(f"  WARNING: Patient leak detected! Overlapping patients: {overlap}")

    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"  Train class distribution: {np.bincount(y_train)}")
    print(f"  Val class distribution: {np.bincount(y_val)}")

    # Apply oversampling to training data only
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"  After oversampling: {np.bincount(y_train_resampled)}")

    # Normalize (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)

    # One-hot encode
    y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=3)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)

    # Build fresh model (Phase 9 style)
    tf.keras.backend.clear_session()
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

    # Train with early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train_onehot,
        validation_data=(X_val_scaled, y_val_onehot),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val_scaled, y_val_onehot, verbose=0)
    y_pred = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)

    f1_per_class = f1_score(y_val, y_pred, average=None)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    min_f1 = min(f1_per_class)

    print(f"\n  Results:")
    print(f"    Accuracy: {val_acc:.4f}")
    print(f"    F1 Macro: {f1_macro:.4f}")
    print(f"    F1 per class [I, P, R]: [{f1_per_class[0]:.4f}, {f1_per_class[1]:.4f}, {f1_per_class[2]:.4f}]")
    print(f"    Min F1: {min_f1:.4f}")
    print(f"    Epochs trained: {len(history.history['loss'])}")

    all_fold_results.append({
        'fold': fold_idx,
        'accuracy': val_acc,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'min_f1': min_f1
    })

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

# Summary
print(f"\n{'='*80}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*80}")

avg_acc = np.mean([r['accuracy'] for r in all_fold_results])
std_acc = np.std([r['accuracy'] for r in all_fold_results])
avg_f1 = np.mean([r['f1_macro'] for r in all_fold_results])
avg_min_f1 = np.mean([r['min_f1'] for r in all_fold_results])

print(f"\nAverage Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
print(f"Average F1 Macro: {avg_f1:.4f}")
print(f"Average Min F1:   {avg_min_f1:.4f}")

print(f"\nPer-Fold Results:")
for r in all_fold_results:
    print(f"  Fold {r['fold']}: Acc={r['accuracy']:.4f}, F1_macro={r['f1_macro']:.4f}, Min_F1={r['min_f1']:.4f}")

# Overall confusion matrix
print(f"\nOverall Confusion Matrix (all folds combined):")
cm = confusion_matrix(all_y_true, all_y_pred)
print(f"           Predicted")
print(f"           I    P    R")
print(f"Actual I  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
print(f"       P  {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
print(f"       R  {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")

# Overall classification report
print(f"\nOverall Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['I', 'P', 'R']))

print(f"\n{'='*80}")
print("GENERALIZABILITY ASSESSMENT")
print(f"{'='*80}")

print(f"\nKey Metrics:")
print(f"  1. Patient-level CV ensures no data leakage")
print(f"  2. Average accuracy across folds: {avg_acc:.1%}")
print(f"  3. Standard deviation: {std_acc:.4f} (consistency check)")
print(f"  4. Min F1 across folds: {avg_min_f1:.4f} (minority class learning)")

if avg_acc > 0.85 and avg_min_f1 > 0.5:
    print(f"\n✅ CONCLUSION: Model GENERALIZES well")
    print(f"   - High accuracy maintained across patient-level folds")
    print(f"   - All classes being predicted (Min F1 > 0.5)")
    print(f"   - NOT memorizing - truly learning patterns")
elif avg_acc > 0.7 and avg_min_f1 > 0.3:
    print(f"\n⚠️  CONCLUSION: Model shows MODERATE generalization")
    print(f"   - Decent accuracy but room for improvement")
    print(f"   - Some class imbalance issues remain")
elif avg_acc > 0.5:
    print(f"\n⚠️  CONCLUSION: Model shows WEAK generalization")
    print(f"   - Better than random but struggling")
    print(f"   - May be partially memorizing easy patterns")
else:
    print(f"\n❌ CONCLUSION: Model does NOT generalize")
    print(f"   - Performance at or below random chance")
    print(f"   - Previous results likely due to memorization or data leak")

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"Phase 9 (train/test split):    97.59% accuracy")
print(f"This test (patient-level CV):  {avg_acc*100:.2f}% accuracy")
print(f"CV test (minimal architecture): 50.14% accuracy")

if avg_acc > 0.9:
    print(f"\n→ Phase 9 results are VALIDATED - model genuinely learns from metadata")
elif avg_acc > 0.7:
    print(f"\n→ Phase 9 results PARTIALLY validated - some generalization but not as strong")
else:
    print(f"\n→ Phase 9 results NOT validated - train/test split may have allowed memorization")
