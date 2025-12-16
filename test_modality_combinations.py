"""
Test script to verify the refactored code can handle all modality combinations (1-5).
This validates the dynamic modality selection system works correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import get_project_paths, get_data_paths, CLASS_LABELS, IMAGE_SIZE, RANDOM_SEED
from src.data.dataset_utils import prepare_cached_datasets
from src.models.builders import create_multimodal_model
from src.models.losses import get_focal_ordinal_loss

print("=" * 80)
print("MODALITY COMBINATION TEST")
print("=" * 80)
print("\nTesting all modality combinations (1-5) to verify dynamic system works correctly")
print("This test validates the refactored code maintains the original dynamic behavior\n")

# Configuration for testing
BASE_CONFIG = {
    'batch_size': 4,
    'n_epochs': 3,  # 3 epochs per combination
    'image_size': 64,
    'train_patient_percentage': 0.67,
    'max_split_diff': 0.3,  # Relaxed for small demo dataset
}

# Define all possible modality combinations to test
MODALITY_COMBINATIONS = [
    # 1 modality
    ['metadata'],
    ['depth_rgb'],

    # 2 modalities (already tested, but include for completeness)
    ['metadata', 'depth_rgb'],

    # 3 modalities
    ['metadata', 'depth_rgb', 'depth_map'],
    ['metadata', 'depth_rgb', 'thermal_rgb'],

    # 4 modalities
    ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb'],

    # 5 modalities (all)
    ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map'],
]

def test_modality_combination(modalities, config):
    """Test a specific modality combination"""
    print("\n" + "=" * 80)
    print(f"Testing: {len(modalities)} modality(ies) - {', '.join(modalities)}")
    print("=" * 80)

    try:
        # Step 1: Load data
        print("\n[1/5] Loading data...")
        data_paths = get_data_paths()
        best_matching_csv = data_paths['best_matching_csv']

        if not os.path.exists(best_matching_csv):
            print(f"❌ Error: {best_matching_csv} not found!")
            return False

        best_matching = pd.read_csv(best_matching_csv)
        print(f"  ✓ Loaded {len(best_matching)} samples")

        # Step 2: Prepare datasets
        print(f"\n[2/5] Preparing datasets with {len(modalities)} modality(ies)...")
        train_dataset, val_dataset, steps_per_epoch, validation_steps, class_weights, train_data, val_data = \
            prepare_cached_datasets(
                best_matching,
                selected_modalities=modalities,
                train_patient_percentage=config['train_patient_percentage'],
                batch_size=config['batch_size'],
                cache_dir=None,
                gen_manager=None,
                aug_config=None,
                run=0,
                max_split_diff=config['max_split_diff'],
                image_size=config['image_size']
            )

        print(f"  ✓ Train steps: {steps_per_epoch}, Val steps: {validation_steps}")
        print(f"  ✓ Class weights: {class_weights}")

        # Step 3: Determine input shapes
        print("\n[3/5] Building model architecture...")
        input_shapes = {}

        for modality in modalities:
            if modality == 'metadata':
                # Metadata uses 3 RF probability features
                input_shapes['metadata'] = (3,)
                print(f"  ✓ {modality}: shape = (3,)")
            elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                input_shapes[modality] = (config['image_size'], config['image_size'], 3)
                print(f"  ✓ {modality}: shape = ({config['image_size']}, {config['image_size']}, 3)")

        # Step 4: Create model
        model = create_multimodal_model(
            input_shapes=input_shapes,
            selected_modalities=modalities,
            class_weights=class_weights,
            strategy=None  # Single GPU/CPU for testing
        )

        print(f"\n  Model Summary:")
        print(f"  - Input modalities: {len(modalities)}")
        print(f"  - Total parameters: {model.count_params():,}")
        print(f"  - Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

        # Step 5: Compile and train (minimal)
        print("\n[4/5] Compiling model...")

        alpha = [class_weights[i] for i in range(len(class_weights))]
        focal_loss_fn = get_focal_ordinal_loss(alpha=alpha, gamma=2.0)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=focal_loss_fn,
            metrics=['accuracy']
        )
        print("  ✓ Model compiled")

        # Step 6: Training test
        print(f"\n[5/5] Running {config['n_epochs']} epoch(s) to verify training works...")

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config['n_epochs'],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1  # Show training progress
        )

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        print(f"  ✓ Training completed")
        print(f"  ✓ Final train accuracy: {final_train_acc:.2%}")
        print(f"  ✓ Final val accuracy: {final_val_acc:.2%}")

        # Clean up
        del model
        del train_dataset
        del val_dataset
        tf.keras.backend.clear_session()

        print(f"\n✅ SUCCESS: {len(modalities)} modality combination works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {len(modalities)} modality combination")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all modality combination tests"""

    print(f"\nBase configuration:")
    for key, value in BASE_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\n\nWill test {len(MODALITY_COMBINATIONS)} different combinations:")
    for i, combo in enumerate(MODALITY_COMBINATIONS, 1):
        print(f"  {i}. {len(combo)} modality(ies): {', '.join(combo)}")

    print("\n" + "=" * 80)
    print("Ready to start testing...")
    print("=" * 80)

    results = {}

    for combo in MODALITY_COMBINATIONS:
        success = test_modality_combination(combo, BASE_CONFIG)
        combo_key = f"{len(combo)}_modalities_{'-'.join(combo[:2])}"  # Shortened key
        results[combo_key] = success

    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL TESTS")
    print("=" * 80)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    print("\nDetailed results:")
    for i, (combo, result) in enumerate(zip(MODALITY_COMBINATIONS, results.values()), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i}. {len(combo)} modalities ({', '.join(combo)}): {status}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The refactored code successfully handles all modality combinations.")
        print("The dynamic modality system from the original code has been preserved.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the errors above.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
