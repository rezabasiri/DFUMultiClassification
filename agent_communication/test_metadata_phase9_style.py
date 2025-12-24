#!/usr/bin/env python3
"""Test metadata with Phase 9-style simple model (not full multimodal architecture)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler

from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.evaluation.metrics import filter_frequent_misclassifications

print("="*80)
print("TEST: Metadata with Phase 9-style Simple Model")
print("="*80)

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
print(f"Data after filtering: {len(data)} samples")

# Extract features (only numeric metadata)
exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
numeric_cols = data.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

X = data[feature_cols].fillna(0).values  # Fill NaN with 0 like Phase 9
y_raw = data['Healing Phase Abs'].values

# Convert labels
label_map = {'I': 0, 'P': 1, 'R': 2}
y = np.array([label_map[label] for label in y_raw])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
print(f"Classes (train): {np.unique(y_train, return_counts=True)}")

# Apply oversampling (like Phase 9)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
print(f"After oversampling: {np.unique(y_train_resampled, return_counts=True)}")

# Normalize (like Phase 9)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# One-hot encode
y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=3)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Build SIMPLE model (like Phase 9)
print("\n" + "="*80)
print("Building Phase 9-style Simple Sequential Model:")
print("  Dense(128) -> Dropout(0.3) -> Dense(64) -> Dropout(0.3) -> Dense(3)")
print("  Loss: categorical_crossentropy")
print("  LR: 0.001")
print("="*80)

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

print(f"Model parameters: {model.count_params()}")

print(f"\nTraining simple Sequential model (20 epochs)...")
history = model.fit(
    X_train_scaled, y_train_onehot,
    validation_data=(X_test_scaled, y_test_onehot),
    epochs=20,
    batch_size=32,
    verbose=2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)
y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
f1_per_class = f1_score(y_test, y_pred, average=None)
min_f1 = min(f1_per_class)

print(f"\n{'='*80}")
print("RESULTS:")
print(f"{'='*80}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"F1 per class [I, P, R]: {f1_per_class}")
print(f"Min F1: {min_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['I', 'P', 'R']))

print(f"\n{'='*80}")
print("COMPARISON:")
print(f"{'='*80}")
print(f"Phase 9 (debug script):  97.59% accuracy, Min F1=0.964")
print(f"CV test (multimodal):    50.14% accuracy, Min F1=0.126")
print(f"This test (Phase 9 style): {test_acc*100:.2f}% accuracy, Min F1={min_f1:.4f}")

if test_acc > 0.9:
    print("\n" + "="*80)
    print("CONCLUSION: SIMPLE MODEL WORKS WELL!")
    print("="*80)
    print("The CV test underperforms because it uses a complex multimodal")
    print("architecture (minimal metadata branch) instead of a dedicated")
    print("metadata-optimized model like Phase 9's simple Sequential model.")
elif test_acc > 0.7:
    print("\n" + "="*80)
    print("CONCLUSION: Model works but performance varies")
    print("="*80)
    print("Simple model outperforms CV test but doesn't reach Phase 9 levels.")
    print("This could be due to data filtering differences or random variation.")
else:
    print("\n" + "="*80)
    print("CONCLUSION: Investigation needed")
    print("="*80)
    print("Simple model does not match Phase 9 performance.")
