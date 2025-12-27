"""
Configuration module for path management across different platforms.
Centralizes directory paths used throughout the project.
"""

import os

def get_project_paths():
    """
    Determine the appropriate directory paths based on the environment.

    Returns:
        tuple: (directory, result_dir, root) paths
    """
    # Default to current repository structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = project_root
    result_dir = os.path.join(project_root, 'results')
    # print(f"Using local repository structure: {directory}")

    root = os.path.join(directory, "data")

    return directory, result_dir, root


def get_output_paths(result_dir=None):
    """
    Get organized output paths for saving results, models, and other artifacts.
    Creates directories if they don't exist.

    Args:
        result_dir: Base results directory (optional, will use get_project_paths if not provided)

    Returns:
        dict: Dictionary containing paths to organized output directories
    """
    if result_dir is None:
        _, result_dir, _ = get_project_paths()

    output_paths = {
        'models': os.path.join(result_dir, 'models'),           # .h5 model weight files
        'checkpoints': os.path.join(result_dir, 'checkpoints'), # .npy prediction/label files
        'csv': os.path.join(result_dir, 'csv'),                 # Result CSV files
        'misclassifications': os.path.join(result_dir, 'misclassifications'),  # Misclassification tracking
        'visualizations': os.path.join(result_dir, 'visualizations'),  # Plots and visualizations
        'logs': os.path.join(result_dir, 'logs'),               # Training logs
        'tf_records': os.path.join(result_dir, 'tf_records'),   # TensorFlow cache files
    }

    # Create directories if they don't exist
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    return output_paths


def get_data_paths(root=None):
    """
    Get paths to image folders.

    Args:
        root: Root data directory (optional, will use get_project_paths if not provided)

    Returns:
        dict: Dictionary containing paths to different image modalities
    """
    # Always get result_dir from get_project_paths
    _, result_dir, default_root = get_project_paths()

    # Use provided root or default
    if root is None:
        root = default_root

    return {
        'image_folder': os.path.join(root, "raw/Depth_RGB"),
        'depth_folder': os.path.join(root, "raw/Depth_Map_IMG"),
        'thermal_folder': os.path.join(root, "raw/Thermal_Map_IMG"),
        'thermal_rgb_folder': os.path.join(root, "raw/Thermal_RGB"),
        'csv_file': os.path.join(root, "raw/DataMaster_Processed_V12_WithMissing.csv"),
        'best_matching_csv': os.path.join(result_dir, "best_matching.csv"),
        'bb_depth_csv': os.path.join(root, "raw/bounding_box_depth.csv"),
        'bb_thermal_csv': os.path.join(root, "raw/bounding_box_thermal.csv")
    }


# Image processing parameters
IMAGE_SIZE = 128
IMAGE_SIZE_MAIN = 64  # Used in main training script

# Random seeds for reproducibility
RANDOM_SEED = 42

# Class labels
CLASS_LABELS = ['I', 'P', 'R']


def cleanup_for_resume_mode(resume_mode='auto', result_dir=None):
    """
    Clean up files based on resume mode to control checkpoint behavior.

    Resume modes:
    - 'fresh': Delete everything (models, predictions, cache, CSV results) - start from scratch
    - 'auto': Keep all checkpoints, resume from latest available state (default)
    - 'from_data': Keep processed data, delete models/predictions/splits - retrain from scratch
    - 'from_models': Keep model weights, delete predictions - regenerate predictions
    - 'from_predictions': Keep Block A predictions, delete Block B gating models - retrain only ensemble

    Args:
        resume_mode: One of ['fresh', 'auto', 'from_data', 'from_models', 'from_predictions']
        result_dir: Base results directory (optional)

    Returns:
        dict: Statistics about deleted files
    """
    import glob
    import shutil

    if result_dir is None:
        _, result_dir, _ = get_project_paths()

    output_paths = get_output_paths(result_dir)

    deleted_stats = {
        'models': 0,
        'predictions': 0,
        'csv_results': 0,
        'patient_splits': 0,
        'tf_cache': 0,
        'progress_files': 0
    }

    def delete_files(pattern_list, stat_key):
        """Helper to delete files matching patterns."""
        count = 0
        for pattern in pattern_list:
            try:
                files = glob.glob(pattern, recursive=True)
                for file_path in files:
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            count += 1
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            count += 1
                    except Exception as e:
                        print(f"Warning: Could not delete {file_path}: {e}")
            except Exception as e:
                print(f"Warning: Error processing pattern {pattern}: {e}")
        deleted_stats[stat_key] = count

    if resume_mode == 'fresh':
        print("\n🧹 FRESH START MODE: Deleting all checkpoints...")
        print("="*80)

        # Delete model weights (both HDF5 and TF checkpoint formats)
        delete_files([
            os.path.join(output_paths['models'], '*.h5'),
            os.path.join(output_paths['models'], '*.ckpt*'),  # TF checkpoint format
            os.path.join(output_paths['models'], 'best_gating_model_*.h5'),
        ], 'models')

        # Delete predictions and labels
        delete_files([
            os.path.join(output_paths['checkpoints'], '*predictions*.npy'),
            os.path.join(output_paths['checkpoints'], '*pred*.npy'),  # Catch pred_run* and combo_pred*
            os.path.join(output_paths['checkpoints'], '*labels*.npy'),
            os.path.join(output_paths['checkpoints'], '*label*.npy'),  # Catch true_label_run*
            os.path.join(output_paths['checkpoints'], 'training_progress_*.json'),
            os.path.join(output_paths['checkpoints'], 'gating_network_run_*_results.csv'),  # Gating results
        ], 'predictions')

        # Delete CSV results (but keep best_matching.csv - it's static patient-to-image mapping)
        delete_files([
            os.path.join(output_paths['csv'], '*.csv'),
            # Also delete CSVs created by main_original.py in results/ root
            os.path.join(result_dir, 'frequent_misclassifications_*.csv'),
            os.path.join(result_dir, 'modality_results_*.csv'),
            os.path.join(result_dir, 'gating_network_*.csv'),
            # NOTE: best_matching.csv is preserved - it's static patient-to-image mapping
        ], 'csv_results')

        # Delete patient splits
        delete_files([
            os.path.join(output_paths['checkpoints'], 'patient_split_*.npz'),
        ], 'patient_splits')

        # Delete TensorFlow cache
        delete_files([
            os.path.join(result_dir, 'tf_cache_*'),
            os.path.join(result_dir, 'tf_records/*'),
        ], 'tf_cache')

        # Delete progress files
        delete_files([
            os.path.join(output_paths['checkpoints'], 'progress_run*.json'),
            os.path.join(output_paths['checkpoints'], 'best_predictions_run*.npy'),
        ], 'progress_files')

    elif resume_mode == 'from_data':
        print("\n🔄 FROM DATA MODE: Keeping processed data, deleting models/predictions...")
        print("="*80)

        # Delete models, predictions, but keep processed data
        delete_files([
            os.path.join(output_paths['models'], '*.h5'),
            os.path.join(output_paths['models'], '*.ckpt*'),  # TF checkpoint format
        ], 'models')

        delete_files([
            os.path.join(output_paths['checkpoints'], '*predictions*.npy'),
            os.path.join(output_paths['checkpoints'], '*pred*.npy'),  # Catch pred_run* and combo_pred*
            os.path.join(output_paths['checkpoints'], '*labels*.npy'),
            os.path.join(output_paths['checkpoints'], '*label*.npy'),  # Catch true_label_run*
            os.path.join(output_paths['checkpoints'], 'patient_split_*.npz'),
            os.path.join(output_paths['checkpoints'], 'gating_network_run_*_results.csv'),  # Gating results
        ], 'predictions')

        delete_files([
            os.path.join(result_dir, 'tf_cache_*'),
        ], 'tf_cache')

    elif resume_mode == 'from_models':
        print("\n♻️  FROM MODELS MODE: Keeping models, deleting predictions...")
        print("="*80)

        # Keep models, delete predictions
        delete_files([
            os.path.join(output_paths['checkpoints'], '*predictions*.npy'),
            os.path.join(output_paths['checkpoints'], '*labels*.npy'),
        ], 'predictions')

        delete_files([
            os.path.join(output_paths['checkpoints'], 'progress_run*.json'),
        ], 'progress_files')

    elif resume_mode == 'from_predictions':
        print("\n⚡ FROM PREDICTIONS MODE: Keeping Block A predictions, deleting Block B gating models...")
        print("="*80)

        # Keep Block A predictions, delete gating network models
        delete_files([
            os.path.join(output_paths['models'], 'best_gating_model_*.h5'),
        ], 'models')

        delete_files([
            os.path.join(output_paths['checkpoints'], 'progress_run*.json'),
            os.path.join(output_paths['checkpoints'], 'best_predictions_run*.npy'),
        ], 'progress_files')

    elif resume_mode == 'auto':
        print("\n✨ AUTO RESUME MODE: Keeping all checkpoints, will resume from latest state...")
        print("="*80)
        # Don't delete anything, let the code auto-resume
        pass

    else:
        raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be one of: "
                        "'fresh', 'auto', 'from_data', 'from_models', 'from_predictions'")

    # Print statistics
    if resume_mode != 'auto':
        print("\nCleanup Statistics:")
        for key, count in deleted_stats.items():
            if count > 0:
                print(f"  {key.replace('_', ' ').title()}: {count} files deleted")
        print("="*80 + "\n")

    return deleted_stats

