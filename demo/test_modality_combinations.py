"""
Test script to verify the refactored code can handle all modality combinations (1-5).
This validates the dynamic modality selection system works correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

# Add project root to path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config import get_project_paths, get_data_paths, CLASS_LABELS, IMAGE_SIZE, RANDOM_SEED
from src.utils.demo_config import get_demo_config, get_modality_config
from src.data.dataset_utils import prepare_cached_datasets
from src.models.builders import create_multimodal_model
from src.models.losses import get_focal_ordinal_loss

# Load configuration
DEMO_CONFIG = get_demo_config()
MODALITY_CONFIG = get_modality_config()

print("=" * 80)
print("MODALITY COMBINATION TEST - ALL 31 COMBINATIONS")
print("=" * 80)
if DEMO_CONFIG['verbose_testing']:
    print("\nTesting ALL possible modality combinations (1-5) to verify dynamic system works correctly")
    print("Total: 5 (1-mod) + 10 (2-mod) + 10 (3-mod) + 5 (4-mod) + 1 (5-mod) = 31 combinations")
    print("This test validates the refactored code maintains the original dynamic behavior\n")

# Define ALL possible modality combinations (31 total)
from itertools import combinations

ALL_MODALITIES = MODALITY_CONFIG['all_modalities']

# Generate all combinations: 1, 2, 3, 4, and 5 modalities
MODALITY_COMBINATIONS = []
for r in range(1, len(ALL_MODALITIES) + 1):
    for combo in combinations(ALL_MODALITIES, r):
        MODALITY_COMBINATIONS.append(list(combo))

# Total: C(5,1) + C(5,2) + C(5,3) + C(5,4) + C(5,5) = 5 + 10 + 10 + 5 + 1 = 31 combinations

def run_single_fold(modalities, config, fold_num=0, total_folds=1):
    """Run a single training fold/run and return metrics"""
    from sklearn.metrics import f1_score

    # Load data
    data_paths = get_data_paths()
    # Use demo-specific best matching CSV instead of production one
    result_dir = get_project_paths()[1]  # Get result directory
    best_matching_csv = os.path.join(result_dir, 'demo_best_matching.csv')

    if not os.path.exists(best_matching_csv):
        raise FileNotFoundError(f"Demo best matching CSV not found: {best_matching_csv}\n"
                                f"Please run demo/test_workflow.py first to generate it.")

    best_matching = pd.read_csv(best_matching_csv)

    # Create demo-specific cache directory under results
    demo_cache_dir = os.path.join(result_dir, 'demo_tf_records')
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Prepare datasets
    train_dataset, pre_aug_dataset, val_dataset, steps_per_epoch, validation_steps, class_weights = \
        prepare_cached_datasets(
            best_matching,
            selected_modalities=modalities,
            train_patient_percentage=config['train_patient_percentage'],
            batch_size=config['batch_size'],
            cache_dir=demo_cache_dir,  # Use demo-specific cache directory
            gen_manager=None,
            aug_config=None,
            run=fold_num,  # Use fold number as run seed for different splits
            max_split_diff=config['max_split_diff'],
            image_size=config['image_size']
        )

    # Determine input shapes
    input_shapes = {}
    for modality in modalities:
        if modality == 'metadata':
            input_shapes['metadata'] = (3,)
        elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
            input_shapes[modality] = (config['image_size'], config['image_size'], 3)

    # Create and compile model
    model = create_multimodal_model(
        input_shapes=input_shapes,
        selected_modalities=modalities,
        class_weights=class_weights,
        strategy=None
    )

    alpha = [class_weights[i] for i in range(len(class_weights))]
    focal_loss_fn = get_focal_ordinal_loss(alpha=alpha, gamma=config['focal_loss_gamma'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=focal_loss_fn,
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['n_epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=config['verbose_training']
    )

    # Get final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    # Calculate F1 score on validation set
    y_true = []
    y_pred = []
    for batch in val_dataset:
        inputs, labels = batch
        predictions = model.predict(inputs, verbose=config['verbose_predictions'])
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    val_f1_macro = f1_score(y_true, y_pred, average='macro')
    val_f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # Clean up
    del model, train_dataset, pre_aug_dataset, val_dataset
    tf.keras.backend.clear_session()

    return {
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
    }

def test_modality_combination(modalities, config):
    """Test a specific modality combination with optional cross-validation"""
    if config['verbose_testing']:
        print("\n" + "=" * 80)
        print(f"Testing: {len(modalities)} modality(ies) - {', '.join(modalities)}")
        if config['use_cv']:
            print(f"Mode: {config['n_folds']}-Fold Cross-Validation")
        else:
            print(f"Mode: Single Run")
        print("=" * 80)

    # Clear TensorFlow session state BEFORE starting
    tf.keras.backend.clear_session()

    try:
        if config['use_cv']:
            # Cross-validation mode
            n_folds = config['n_folds']
            if config['verbose_testing']:
                print(f"\nRunning {n_folds}-fold cross-validation...")

            fold_results = []
            for fold in range(n_folds):
                if config['verbose_testing']:
                    print(f"\n  Fold {fold + 1}/{n_folds}...")
                fold_metrics = run_single_fold(modalities, config, fold_num=fold, total_folds=n_folds)
                fold_results.append(fold_metrics)
                if config['verbose_testing']:
                    print(f"    ✓ Val Acc: {fold_metrics['val_acc']:.{config['metrics_precision']}f}, Val F1 (macro): {fold_metrics['val_f1_macro']:.{config['metrics_precision']}f}")

            # Aggregate results across folds
            metrics = {
                'train_acc': np.mean([r['train_acc'] for r in fold_results]),
                'val_acc': np.mean([r['val_acc'] for r in fold_results]),
                'train_loss': np.mean([r['train_loss'] for r in fold_results]),
                'val_loss': np.mean([r['val_loss'] for r in fold_results]),
                'val_f1_macro': np.mean([r['val_f1_macro'] for r in fold_results]),
                'val_f1_weighted': np.mean([r['val_f1_weighted'] for r in fold_results]),
                'val_acc_std': np.std([r['val_acc'] for r in fold_results]),
                'val_f1_macro_std': np.std([r['val_f1_macro'] for r in fold_results]),
                'modalities': modalities,
                'success': True,
                'n_folds': n_folds
            }

            if config['verbose_testing']:
                prec = config['metrics_precision']
                print(f"\n  Cross-validation Results (mean ± std):")
                print(f"    Val Accuracy:     {metrics['val_acc']:.{prec}f} ± {metrics['val_acc_std']:.{prec}f}")
                print(f"    Val F1 (macro):   {metrics['val_f1_macro']:.{prec}f} ± {metrics['val_f1_macro_std']:.{prec}f}")
                print(f"    Val F1 (weighted): {metrics['val_f1_weighted']:.{prec}f}")

        else:
            # Single run mode
            if config['verbose_testing']:
                print(f"\nRunning single training run with {config['n_epochs']} epochs...")
            fold_metrics = run_single_fold(modalities, config, fold_num=0, total_folds=1)

            metrics = {
                **fold_metrics,
                'modalities': modalities,
                'success': True
            }

            if config['verbose_testing']:
                prec = config['metrics_precision']
                print(f"\n  ✓ Training completed")
                print(f"  ✓ Train Accuracy: {metrics['train_acc']:.{prec}f}")
                print(f"  ✓ Val Accuracy:   {metrics['val_acc']:.{prec}f}")
                print(f"  ✓ Val F1 (macro): {metrics['val_f1_macro']:.{prec}f}")
                print(f"  ✓ Val F1 (weighted): {metrics['val_f1_weighted']:.{prec}f}")

        if config['verbose_testing']:
            print(f"\n✅ SUCCESS: {len(modalities)} modality combination works correctly!")
        return metrics

    except Exception as e:
        if config['verbose_testing']:
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
    from src.utils.demo_config import RESULTS_DEMO_DIR, RESULTS_FILE_PREFIX, RESULTS_FILE_EXTENSION

    # Create results file in results/demo directory
    results_demo_dir = os.path.join(project_root, RESULTS_DEMO_DIR)
    os.makedirs(results_demo_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_demo_dir, f"{RESULTS_FILE_PREFIX}_{timestamp}{RESULTS_FILE_EXTENSION}")

    if DEMO_CONFIG['verbose_testing']:
        print(f"\nConfiguration:")
        for key, value in DEMO_CONFIG.items():
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
        f.write(f"Configuration: {DEMO_CONFIG}\n")
        f.write("=" * 100 + "\n\n")

    for i, combo in enumerate(MODALITY_COMBINATIONS, 1):
        metrics = test_modality_combination(combo, DEMO_CONFIG)
        results.append(metrics)

        # Write result to file immediately
        prec = DEMO_CONFIG['metrics_precision']
        with open(results_file, 'a') as f:
            f.write(f"\n[{i}/{len(MODALITY_COMBINATIONS)}] {', '.join(combo)}\n")
            if metrics['success']:
                f.write(f"  ✅ SUCCESS\n")
                if 'n_folds' in metrics:
                    f.write(f"  Mode: {metrics['n_folds']}-Fold Cross-Validation\n")
                    f.write(f"  Val Accuracy:     {metrics['val_acc']:.{prec}f} ± {metrics.get('val_acc_std', 0):.{prec}f}\n")
                    f.write(f"  Val F1 (macro):   {metrics['val_f1_macro']:.{prec}f} ± {metrics.get('val_f1_macro_std', 0):.{prec}f}\n")
                    f.write(f"  Val F1 (weighted): {metrics['val_f1_weighted']:.{prec}f}\n")
                    f.write(f"  Train Accuracy:   {metrics['train_acc']:.{prec}f}\n")
                    f.write(f"  Train Loss:       {metrics['train_loss']:.{prec}f}\n")
                    f.write(f"  Val Loss:         {metrics['val_loss']:.{prec}f}\n")
                else:
                    f.write(f"  Train Accuracy:   {metrics['train_acc']:.{prec}f}\n")
                    f.write(f"  Val Accuracy:     {metrics['val_acc']:.{prec}f}\n")
                    f.write(f"  Val F1 (macro):   {metrics['val_f1_macro']:.{prec}f}\n")
                    f.write(f"  Val F1 (weighted): {metrics['val_f1_weighted']:.{prec}f}\n")
                    f.write(f"  Train Loss:       {metrics['train_loss']:.{prec}f}\n")
                    f.write(f"  Val Loss:         {metrics['val_loss']:.{prec}f}\n")
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
        prec = DEMO_CONFIG['metrics_precision']
        top_n = DEMO_CONFIG['top_n_performers']

        # Overall statistics
        avg_val_acc = sum(r['val_acc'] for r in successful_results) / len(successful_results)
        avg_val_loss = sum(r['val_loss'] for r in successful_results) / len(successful_results)
        avg_val_f1_macro = sum(r['val_f1_macro'] for r in successful_results) / len(successful_results)
        avg_val_f1_weighted = sum(r['val_f1_weighted'] for r in successful_results) / len(successful_results)

        print(f"\nOverall Performance ({len(successful_results)} successful tests):")
        print(f"  Average Val Accuracy:     {avg_val_acc:.{prec}f}")
        print(f"  Average Val F1 (macro):   {avg_val_f1_macro:.{prec}f}")
        print(f"  Average Val F1 (weighted): {avg_val_f1_weighted:.{prec}f}")
        print(f"  Average Val Loss:         {avg_val_loss:.{prec}f}")

        # Best performers by F1 score
        print(f"\n📊 Top {top_n} Performers (by Validation F1-macro):")
        sorted_by_f1 = sorted(successful_results, key=lambda x: x['val_f1_macro'], reverse=True)
        for i, r in enumerate(sorted_by_f1[:top_n], 1):
            combo_str = ', '.join(r['modalities'])
            std_str = f" ± {r.get('val_f1_macro_std', 0):.{prec}f}" if 'n_folds' in r else ""
            print(f"  {i}. [{len(r['modalities'])} mod] {combo_str:50s} F1: {r['val_f1_macro']:.{prec}f}{std_str}, Acc: {r['val_acc']:.{prec}f}")

        # By modality count
        print("\n📈 Performance by Modality Count:")
        for num_mods in range(1, 6):
            combos_at_level = [r for r in successful_results if len(r['modalities']) == num_mods]
            if combos_at_level:
                avg_acc = sum(r['val_acc'] for r in combos_at_level) / len(combos_at_level)
                avg_f1 = sum(r['val_f1_macro'] for r in combos_at_level) / len(combos_at_level)
                avg_loss = sum(r['val_loss'] for r in combos_at_level) / len(combos_at_level)
                print(f"  {num_mods} modality(ies): Avg Val Acc = {avg_acc:.{prec}f}, Avg F1 = {avg_f1:.{prec}f}, Avg Loss = {avg_loss:.{prec}f}")

        # Write analysis to file
        with open(results_file, 'a') as f:
            f.write("\n\n" + "=" * 100 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Overall Performance ({len(successful_results)} successful tests):\n")
            f.write(f"  Average Val Accuracy:     {avg_val_acc:.{prec}f}\n")
            f.write(f"  Average Val F1 (macro):   {avg_val_f1_macro:.{prec}f}\n")
            f.write(f"  Average Val F1 (weighted): {avg_val_f1_weighted:.{prec}f}\n")
            f.write(f"  Average Val Loss:         {avg_val_loss:.{prec}f}\n\n")

            f.write(f"Top {top_n} Performers (by Validation F1-macro):\n")
            for i, r in enumerate(sorted_by_f1[:top_n], 1):
                combo_str = ', '.join(r['modalities'])
                std_str = f" ± {r.get('val_f1_macro_std', 0):.{prec}f}" if 'n_folds' in r else ""
                f.write(f"  {i}. [{len(r['modalities'])} mod] {combo_str:50s} F1: {r['val_f1_macro']:.{prec}f}{std_str}, Acc: {r['val_acc']:.{prec}f}\n")

            f.write("\nPerformance by Modality Count:\n")
            for num_mods in range(1, 6):
                combos_at_level = [r for r in successful_results if len(r['modalities']) == num_mods]
                if combos_at_level:
                    avg_acc = sum(r['val_acc'] for r in combos_at_level) / len(combos_at_level)
                    avg_f1 = sum(r['val_f1_macro'] for r in combos_at_level) / len(combos_at_level)
                    avg_loss = sum(r['val_loss'] for r in combos_at_level) / len(combos_at_level)
                    f.write(f"  {num_mods} modality(ies): Avg Val Acc = {avg_acc:.{prec}f}, Avg F1 = {avg_f1:.{prec}f}, Avg Loss = {avg_loss:.{prec}f}\n")

    if failed == 0:
        print("\n🎉 ALL 31 TESTS PASSED! The refactored code successfully handles all modality combinations.")
        print("The dynamic modality system from the original code has been fully preserved.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the errors above.")

    print(f"\n📄 Detailed results saved to: {results_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
