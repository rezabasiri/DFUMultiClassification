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
print("MODALITY COMBINATION TEST - ALL 31 COMBINATIONS")
print("=" * 80)
print("\nTesting ALL possible modality combinations (1-5) to verify dynamic system works correctly")
print("Total: 5 (1-mod) + 10 (2-mod) + 10 (3-mod) + 5 (4-mod) + 1 (5-mod) = 31 combinations")
print("This test validates the refactored code maintains the original dynamic behavior\n")

# Configuration for testing
BASE_CONFIG = {
    'batch_size': 4,
    'n_epochs': 3,  # 3 epochs per combination
    'image_size': 64,
    'train_patient_percentage': 0.67,
    'max_split_diff': 0.3,  # Relaxed for small demo dataset
}

# Define ALL possible modality combinations (31 total)
from itertools import combinations

ALL_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']

# Generate all combinations: 1, 2, 3, 4, and 5 modalities
MODALITY_COMBINATIONS = []

# 1 modality: 5 combinations
for r in range(1, len(ALL_MODALITIES) + 1):
    for combo in combinations(ALL_MODALITIES, r):
        MODALITY_COMBINATIONS.append(list(combo))

# Total: C(5,1) + C(5,2) + C(5,3) + C(5,4) + C(5,5) = 5 + 10 + 10 + 5 + 1 = 31 combinations

def test_modality_combination(modalities, config):
    """Test a specific modality combination"""
    print("\n" + "=" * 80)
    print(f"Testing: {len(modalities)} modality(ies) - {', '.join(modalities)}")
    print("=" * 80)

    # Clear TensorFlow session state BEFORE starting
    tf.keras.backend.clear_session()

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

        # Prepare datasets - uses default cache_dir (results/tf_records)
        print(f"\n[2/5] Preparing datasets with {len(modalities)} modality(ies)...")
        train_dataset, pre_aug_dataset, val_dataset, steps_per_epoch, validation_steps, class_weights = \
            prepare_cached_datasets(
                best_matching,
                selected_modalities=modalities,
                train_patient_percentage=config['train_patient_percentage'],
                batch_size=config['batch_size'],
                cache_dir=None,  # Use default centralized location: results/tf_records
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
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"  ✓ Training completed")
        print(f"  ✓ Final train accuracy: {final_train_acc:.2%}")
        print(f"  ✓ Final val accuracy: {final_val_acc:.2%}")

        # Prepare metrics to return
        metrics = {
            'train_acc': final_train_acc,
            'val_acc': final_val_acc,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'modalities': modalities,
            'success': True
        }

        # Clean up
        del model
        del train_dataset
        del pre_aug_dataset
        del val_dataset
        tf.keras.backend.clear_session()

        print(f"\n✅ SUCCESS: {len(modalities)} modality combination works correctly!")
        return metrics

    except Exception as e:
        print(f"\n❌ FAILED: {len(modalities)} modality combination")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'modalities': modalities,
            'success': False,
            'error': str(e)
        }

def main():
    """Run all modality combination tests"""
    from datetime import datetime

    # Create results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"modality_test_results_{timestamp}.txt"

    print(f"\nBase configuration:")
    for key, value in BASE_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\n\nWill test {len(MODALITY_COMBINATIONS)} different combinations:")
    for i, combo in enumerate(MODALITY_COMBINATIONS, 1):
        print(f"  {i}. {len(combo)} modality(ies): {', '.join(combo)}")

    print(f"\nResults will be saved to: {results_file}")

    print("\n" + "=" * 80)
    print("Ready to start testing...")
    print("=" * 80)

    results = []

    # Write header to file
    with open(results_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MODALITY COMBINATION TEST RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {BASE_CONFIG}\n")
        f.write("=" * 100 + "\n\n")

    for i, combo in enumerate(MODALITY_COMBINATIONS, 1):
        metrics = test_modality_combination(combo, BASE_CONFIG)
        results.append(metrics)

        # Write result to file immediately
        with open(results_file, 'a') as f:
            f.write(f"\n[{i}/{len(MODALITY_COMBINATIONS)}] {', '.join(combo)}\n")
            if metrics['success']:
                f.write(f"  ✅ SUCCESS\n")
                f.write(f"  Train Accuracy: {metrics['train_acc']:.4f}\n")
                f.write(f"  Val Accuracy:   {metrics['val_acc']:.4f}\n")
                f.write(f"  Train Loss:     {metrics['train_loss']:.4f}\n")
                f.write(f"  Val Loss:       {metrics['val_loss']:.4f}\n")
            else:
                f.write(f"  ❌ FAILED: {metrics.get('error', 'Unknown error')}\n")
            f.write("-" * 100 + "\n")

    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL TESTS")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total - passed

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    # Group results by number of modalities
    print("\nResults by modality count:")
    for num_mods in range(1, 6):
        combos_at_level = [r for r in results if len(r['modalities']) == num_mods]
        passed_at_level = sum(1 for r in combos_at_level if r['success'])
        total_at_level = len(combos_at_level)
        print(f"  {num_mods} modality(ies): {passed_at_level}/{total_at_level} passed")

    # Performance Analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    successful_results = [r for r in results if r['success']]

    if successful_results:
        # Overall statistics
        avg_val_acc = sum(r['val_acc'] for r in successful_results) / len(successful_results)
        avg_val_loss = sum(r['val_loss'] for r in successful_results) / len(successful_results)

        print(f"\nOverall Performance ({len(successful_results)} successful tests):")
        print(f"  Average Val Accuracy: {avg_val_acc:.4f}")
        print(f"  Average Val Loss:     {avg_val_loss:.4f}")

        # Best performers
        print("\n📊 Top 5 Performers (by Validation Accuracy):")
        sorted_by_acc = sorted(successful_results, key=lambda x: x['val_acc'], reverse=True)
        for i, r in enumerate(sorted_by_acc[:5], 1):
            combo_str = ', '.join(r['modalities'])
            print(f"  {i}. [{len(r['modalities'])} mod] {combo_str:50s} Val Acc: {r['val_acc']:.4f}, Val Loss: {r['val_loss']:.4f}")

        # By modality count
        print("\n📈 Performance by Modality Count:")
        for num_mods in range(1, 6):
            combos_at_level = [r for r in successful_results if len(r['modalities']) == num_mods]
            if combos_at_level:
                avg_acc = sum(r['val_acc'] for r in combos_at_level) / len(combos_at_level)
                avg_loss = sum(r['val_loss'] for r in combos_at_level) / len(combos_at_level)
                print(f"  {num_mods} modality(ies): Avg Val Acc = {avg_acc:.4f}, Avg Val Loss = {avg_loss:.4f}")

        # Write analysis to file
        with open(results_file, 'a') as f:
            f.write("\n\n" + "=" * 100 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Overall Performance ({len(successful_results)} successful tests):\n")
            f.write(f"  Average Val Accuracy: {avg_val_acc:.4f}\n")
            f.write(f"  Average Val Loss:     {avg_val_loss:.4f}\n\n")

            f.write("Top 5 Performers (by Validation Accuracy):\n")
            for i, r in enumerate(sorted_by_acc[:5], 1):
                combo_str = ', '.join(r['modalities'])
                f.write(f"  {i}. [{len(r['modalities'])} mod] {combo_str:50s} Val Acc: {r['val_acc']:.4f}, Val Loss: {r['val_loss']:.4f}\n")

            f.write("\nPerformance by Modality Count:\n")
            for num_mods in range(1, 6):
                combos_at_level = [r for r in successful_results if len(r['modalities']) == num_mods]
                if combos_at_level:
                    avg_acc = sum(r['val_acc'] for r in combos_at_level) / len(combos_at_level)
                    avg_loss = sum(r['val_loss'] for r in combos_at_level) / len(combos_at_level)
                    f.write(f"  {num_mods} modality(ies): Avg Val Acc = {avg_acc:.4f}, Avg Val Loss = {avg_loss:.4f}\n")

    if failed == 0:
        print("\n🎉 ALL 31 TESTS PASSED! The refactored code successfully handles all modality combinations.")
        print("The dynamic modality system from the original code has been fully preserved.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the errors above.")

    print(f"\n📄 Detailed results saved to: {results_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
