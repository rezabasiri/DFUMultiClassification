#!/usr/bin/env python3
"""
Test Workflow Script for DFU Multi-Classification
Tests the complete pipeline with demo data and limited computational resources.

Usage: python test_workflow.py
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.config import get_project_paths, get_data_paths, CLASS_LABELS
from src.utils.debug import clear_gpu_memory
from src.data.image_processing import (
    create_best_matching_dataset,
    prepare_dataset
)
from src.data.dataset_utils import prepare_cached_datasets
from src.models.builders import create_multimodal_model
from src.models.losses import get_focal_ordinal_loss
from src.evaluation.metrics import track_misclassifications

print("=" * 80)
print("DFU MULTI-CLASSIFICATION WORKFLOW TEST")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# STEP 1: Environment Configuration
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: ENVIRONMENT CONFIGURATION")
print("=" * 80)

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print(f"✓ Random seed set to: {RANDOM_SEED}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU detected: {len(gpus)} device(s)")
    for i, gpu in enumerate(gpus):
        print(f"  - GPU {i}: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
else:
    print("! No GPU detected - using CPU only")
    print("  (This is fine for testing with demo data)")

# Get paths
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

print(f"\n📁 Path Configuration:")
print(f"  - Project directory: {directory}")
print(f"  - Results directory: {result_dir}")
print(f"  - Data root: {root}")

# Create necessary directories
ck_path = os.path.join(result_dir, "test_checkpoints")
os.makedirs(ck_path, exist_ok=True)
print(f"✓ Checkpoint directory: {ck_path}")

# =============================================================================
# STEP 2: Data Discovery
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA DISCOVERY")
print("=" * 80)

print("\n🔍 Scanning for image files...")
for modality, path in [
    ("Depth RGB", data_paths['image_folder']),
    ("Depth Map", data_paths['depth_folder']),
    ("Thermal RGB", data_paths['thermal_rgb_folder']),
    ("Thermal Map", data_paths['thermal_folder'])
]:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        print(f"  ✓ {modality:15s}: {len(files):3d} files in {path}")
    else:
        print(f"  ✗ {modality:15s}: Directory not found - {path}")

print("\n🔍 Checking CSV files...")
for csv_name, csv_path in [
    ("Master CSV", data_paths['csv_file']),
    ("Depth BB", data_paths['bb_depth_csv']),
    ("Thermal BB", data_paths['bb_thermal_csv'])
]:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  ✓ {csv_name:15s}: {len(df):4d} rows - {csv_path}")
        if csv_name == "Master CSV":
            if 'Healing_Phase_cat' in df.columns:
                phase_counts = df['Healing_Phase_cat'].value_counts()
                print(f"     Phase distribution: {dict(phase_counts)}")
    else:
        print(f"  ✗ {csv_name:15s}: File not found - {csv_path}")

# =============================================================================
# STEP 3: Dataset Preparation
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATASET PREPARATION")
print("=" * 80)

print("\n📊 Creating best matching dataset...")
best_matching_csv = os.path.join(result_dir, 'best_matching.csv')

try:
    if os.path.exists(best_matching_csv):
        print(f"  ℹ Using existing file: {best_matching_csv}")
        best_matching_df = pd.read_csv(best_matching_csv)
    else:
        print(f"  ⏳ Creating new matching dataset...")
        best_matching_df = create_best_matching_dataset(
            data_paths['bb_depth_csv'],
            data_paths['bb_thermal_csv'],
            data_paths['csv_file'],
            data_paths['depth_folder'],
            data_paths['thermal_folder'],
            best_matching_csv
        )

    print(f"✓ Best matching dataset: {len(best_matching_df)} samples")

    # Check for 'Healing Phase Abs' column - this is the standard column name
    if 'Healing Phase Abs' not in best_matching_df.columns or best_matching_df['Healing Phase Abs'].isna().all():
        print(f"  ⚠ Warning: 'Healing Phase Abs' column is missing or empty")
        print(f"  Creating synthetic labels for testing purposes...")
        # Create synthetic labels for testing (distribute across I, P, R)
        import random
        random.seed(42)
        synthetic_labels = random.choices(['I', 'P', 'R'], k=len(best_matching_df))
        best_matching_df['Healing Phase Abs'] = synthetic_labels
        print(f"  ✓ Created synthetic labels: {len(synthetic_labels)} samples")

    # Show phase distribution
    phase_dist = best_matching_df['Healing Phase Abs'].value_counts()
    print(f"  Phase distribution (column: 'Healing Phase Abs'):")
    for phase, count in phase_dist.items():
        print(f"    - {phase}: {count} samples ({count/len(best_matching_df)*100:.1f}%)")

    print(f"\n  Sample patient IDs: {best_matching_df['Patient#'].unique()[:5].tolist()}")

except Exception as e:
    print(f"✗ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 4: Configure Test Parameters
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: TEST CONFIGURATION")
print("=" * 80)

# Test with minimal computational requirements
TEST_CONFIG = {
    'selected_modalities': ['metadata', 'depth_rgb'],  # Start with just 2 modalities
    'batch_size': 4,  # Small batch size for limited memory
    'n_epochs': 5,  # Few epochs for quick testing
    'image_size': 64,  # Small image size
    'train_patient_percentage': 0.67,  # 67% train (5 patients), 33% val (3 patients) for better phase distribution
    'use_augmentation': False,  # Disable augmentation for faster testing
    'max_split_diff': 0.3,  # Allow 30% class distribution difference for small test dataset (9 patients)
}

print(f"\n⚙️ Test Configuration:")
for key, value in TEST_CONFIG.items():
    print(f"  - {key:25s}: {value}")

# =============================================================================
# STEP 5: Data Splitting
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: DATA SPLITTING")
print("=" * 80)

print("\n✂️ Splitting data into train/validation sets...")
try:
    # Get unique patients
    unique_patients = best_matching_df['Patient#'].unique()
    print(f"  Total unique patients: {len(unique_patients)}")

    # Split patients
    n_train = int(len(unique_patients) * TEST_CONFIG['train_patient_percentage'])
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:]

    print(f"  Train patients: {len(train_patients)} - IDs: {train_patients.tolist()}")
    print(f"  Val patients: {len(val_patients)} - IDs: {val_patients.tolist()}")

    # Split data
    train_data = best_matching_df[best_matching_df['Patient#'].isin(train_patients)].copy()
    val_data = best_matching_df[best_matching_df['Patient#'].isin(val_patients)].copy()

    print(f"\n✓ Train samples: {len(train_data)}")
    print(f"✓ Val samples: {len(val_data)}")

    # Show phase distribution using 'Healing Phase Abs'
    if 'Healing Phase Abs' in train_data.columns:
        print(f"\n  Train phase distribution:")
        train_dist = train_data['Healing Phase Abs'].value_counts()
        for phase, count in train_dist.items():
            print(f"    - {phase}: {count} samples")

        print(f"\n  Val phase distribution:")
        val_dist = val_data['Healing Phase Abs'].value_counts()
        for phase, count in val_dist.items():
            print(f"    - {phase}: {count} samples")
    else:
        print(f"  ⚠ Warning: 'Healing Phase Abs' column not found")

except Exception as e:
    print(f"✗ Error splitting data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 6: Dataset Creation
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: DATASET CREATION")
print("=" * 80)

print("\n🔨 Creating TensorFlow datasets...")
try:
    # Note: We're passing the FULL dataset, the function will split it internally by patient
    # This ensures proper patient-level splitting
    datasets_output = prepare_cached_datasets(
        best_matching_df,  # Pass full dataset, not pre-split train_data
        TEST_CONFIG['selected_modalities'],
        train_patient_percentage=TEST_CONFIG['train_patient_percentage'],
        batch_size=TEST_CONFIG['batch_size'],
        cache_dir=None,  # Don't cache for test
        gen_manager=None,  # No generative augmentation for test
        aug_config=None,   # No augmentation config
        run=0,
        max_split_diff=TEST_CONFIG['max_split_diff']  # Relaxed threshold for small test dataset
    )

    # Unpack the returned values
    train_dataset, pre_aug_dataset, val_dataset, steps_per_epoch, validation_steps, alpha_values = datasets_output

    # Compute class weights from 'Healing Phase Abs' column
    if 'Healing Phase Abs' in best_matching_df.columns:
        from sklearn.utils.class_weight import compute_class_weight
        # Convert to numeric (I->0, P->1, R->2)
        phase_map = {'I': 0, 'P': 1, 'R': 2}
        phase_values = best_matching_df['Healing Phase Abs'].map(phase_map)

        unique_classes = np.sort(phase_values.unique())
        class_weights_array = compute_class_weight('balanced', classes=unique_classes,
                                                    y=phase_values)
        class_weights = dict(zip(unique_classes, class_weights_array))
    else:
        class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        print(f"  ⚠ Warning: 'Healing Phase Abs' column not found, using balanced weights")

    print(f"✓ Train dataset created")
    print(f"✓ Validation dataset created")
    print(f"✓ Class weights computed: {class_weights}")

    # Test dataset iteration
    print(f"\n🧪 Testing dataset iteration...")
    for batch_idx, (inputs, labels) in enumerate(train_dataset.take(1)):
        print(f"  Batch {batch_idx}:")
        print(f"    Input types: {type(inputs)}")

        if isinstance(inputs, dict):
            for key, value in inputs.items():
                print(f"    - {key:15s}: shape {value.shape}, dtype {value.dtype}")

        print(f"    Labels shape: {labels.shape}, dtype {labels.dtype}")
        print(f"    Sample label values: {labels[:3].numpy()}")

    print(f"✓ Dataset iteration successful")

except Exception as e:
    print(f"✗ Error creating datasets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 7: Model Building
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL BUILDING")
print("=" * 80)

print("\n🏗️ Building multimodal model...")
try:
    # Prepare input shapes
    input_shapes = {}

    if 'metadata' in TEST_CONFIG['selected_modalities']:
        # Count metadata features
        metadata_cols = [col for col in train_data.columns
                        if col not in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                                     'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                                     'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax',
                                     'Healing_Phase_cat', 'Patient#', 'Appt#', 'DFU#']]
        input_shapes['metadata'] = (len(metadata_cols),)
        print(f"  Metadata features: {len(metadata_cols)}")

    if 'depth_rgb' in TEST_CONFIG['selected_modalities']:
        input_shapes['depth_rgb'] = (TEST_CONFIG['image_size'], TEST_CONFIG['image_size'], 3)
        print(f"  Depth RGB shape: {input_shapes['depth_rgb']}")

    if 'depth_map' in TEST_CONFIG['selected_modalities']:
        input_shapes['depth_map'] = (TEST_CONFIG['image_size'], TEST_CONFIG['image_size'], 3)
        print(f"  Depth Map shape: {input_shapes['depth_map']}")

    if 'thermal_rgb' in TEST_CONFIG['selected_modalities']:
        input_shapes['thermal_rgb'] = (TEST_CONFIG['image_size'], TEST_CONFIG['image_size'], 3)
        print(f"  Thermal RGB shape: {input_shapes['thermal_rgb']}")

    if 'thermal_map' in TEST_CONFIG['selected_modalities']:
        input_shapes['thermal_map'] = (TEST_CONFIG['image_size'], TEST_CONFIG['image_size'], 3)
        print(f"  Thermal Map shape: {input_shapes['thermal_map']}")

    print(f"\n  Building model with {len(input_shapes)} modalities...")
    model = create_multimodal_model(
        input_shapes,
        TEST_CONFIG['selected_modalities'],
        class_weights
    )

    print(f"✓ Model built successfully")
    print(f"\n  Model summary:")
    print(f"    Total parameters: {model.count_params():,}")
    print(f"    Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"    Input layers: {len(model.inputs)}")
    print(f"    Output shape: {model.output_shape}")

except Exception as e:
    print(f"✗ Error building model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 8: Model Compilation
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: MODEL COMPILATION")
print("=" * 80)

print("\n⚙️ Compiling model...")
try:
    # Use simple loss for testing
    loss_fn = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.5, gamma=2.0, alpha=0.25)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )

    print(f"✓ Model compiled")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: Focal Ordinal Loss")
    print(f"  Metrics: accuracy")

except Exception as e:
    print(f"✗ Error compiling model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 9: Training
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: MODEL TRAINING")
print("=" * 80)

print(f"\n🚂 Training model for {TEST_CONFIG['n_epochs']} epochs...")
print(f"  (This may take a few minutes depending on your hardware)")

try:
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=TEST_CONFIG['n_epochs'],
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✓ Training completed")
    print(f"\n  Training History:")
    for epoch in range(len(history.history['loss'])):
        print(f"    Epoch {epoch+1}/{TEST_CONFIG['n_epochs']}:")
        print(f"      - loss: {history.history['loss'][epoch]:.4f}")
        print(f"      - accuracy: {history.history['accuracy'][epoch]:.4f}")
        if 'val_loss' in history.history:
            print(f"      - val_loss: {history.history['val_loss'][epoch]:.4f}")
            print(f"      - val_accuracy: {history.history['val_accuracy'][epoch]:.4f}")

except Exception as e:
    print(f"✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 10: Evaluation
# =============================================================================
print("\n" + "=" * 80)
print("STEP 10: MODEL EVALUATION")
print("=" * 80)

print("\n📊 Evaluating model on validation set...")
try:
    # Evaluate
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)

    print(f"✓ Validation Results:")
    print(f"  - Loss: {val_loss:.4f}")
    print(f"  - Accuracy: {val_accuracy:.4f}")

    # Get predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    true_labels = []

    for inputs, labels in val_dataset:
        preds = model.predict(inputs, verbose=0)
        predictions.extend(np.argmax(preds, axis=1))
        true_labels.extend(np.argmax(labels.numpy(), axis=1))

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f"✓ Generated {len(predictions)} predictions")

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n  Classification Report:")
    report = classification_report(true_labels, predictions,
                                  target_names=CLASS_LABELS,
                                  zero_division=0)
    print(report)

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(f"  {'':10s} ", end="")
    for label in CLASS_LABELS:
        print(f"{label:>10s}", end="")
    print()
    for i, label in enumerate(CLASS_LABELS):
        print(f"  {label:10s} ", end="")
        for j in range(len(CLASS_LABELS)):
            print(f"{cm[i,j]:10d}", end="")
        print()

except Exception as e:
    print(f"✗ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 11: Save Test Results
# =============================================================================
print("\n" + "=" * 80)
print("STEP 11: SAVING TEST RESULTS")
print("=" * 80)

try:
    test_results_path = os.path.join(result_dir, 'test_workflow_results.txt')

    with open(test_results_path, 'w') as f:
        f.write("DFU Multi-Classification Workflow Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Configuration:\n")
        for key, value in TEST_CONFIG.items():
            f.write(f"  {key}: {value}\n")

        f.write(f"\nData:\n")
        f.write(f"  Total samples: {len(best_matching_df)}\n")
        f.write(f"  Train samples: {len(train_data)}\n")
        f.write(f"  Val samples: {len(val_data)}\n")

        f.write(f"\nModel:\n")
        f.write(f"  Total parameters: {model.count_params():,}\n")

        f.write(f"\nResults:\n")
        f.write(f"  Validation Loss: {val_loss:.4f}\n")
        f.write(f"  Validation Accuracy: {val_accuracy:.4f}\n")

        f.write(f"\nClassification Report:\n")
        f.write(report)

    print(f"✓ Results saved to: {test_results_path}")

except Exception as e:
    print(f"⚠ Warning: Could not save results: {e}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST WORKFLOW COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"\n✅ Summary:")
print(f"  - Data loaded: {len(best_matching_df)} samples")
print(f"  - Model trained: {TEST_CONFIG['n_epochs']} epochs")
print(f"  - Val accuracy: {val_accuracy:.2%}")
print(f"  - Results saved: {test_results_path}")

print(f"\n📝 Next Steps:")
print(f"  1. Review the results in: {test_results_path}")
print(f"  2. If successful, you can run the full training with more data")
print(f"  3. Adjust parameters in src/main.py for production training")
print(f"  4. Use more modalities: ['metadata', 'depth_rgb', 'thermal_rgb', 'depth_map', 'thermal_map']")

print(f"\n✨ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
