"""
TensorFlow dataset creation and caching utilities.
Functions for creating optimized, cached datasets for training and validation.
"""

import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.utils.config import get_project_paths, get_data_paths, get_output_paths, CLASS_LABELS
from src.utils.verbosity import vprint
from src.data.image_processing import load_and_preprocess_image
from src.data.generative_augmentation_v2 import create_enhanced_augmentation_fn

# Get paths
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
output_paths = get_output_paths(result_dir)


def create_patient_folds(data, n_folds=3, random_state=42, max_imbalance=0.3):
    """
    Create k-fold cross-validation splits at patient level with class balance.

    Args:
        data: DataFrame with 'Patient#' and 'Healing Phase Abs' columns
        n_folds: Number of folds (default: 3)
        random_state: Random seed for reproducibility
        max_imbalance: Maximum allowed class distribution difference between folds (default: 0.3)

    Returns:
        List of tuples (train_patients, valid_patients) for each fold
    """
    np.random.seed(random_state)
    random.seed(random_state)

    # Convert labels if needed
    data = data.copy()
    if 'Healing Phase Abs' in data.columns:
        if data['Healing Phase Abs'].dtype == 'object':
            data['Healing Phase Abs'] = data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})

    # Group patients by their majority class
    patient_classes = {}
    unique_patients = data['Patient#'].unique()

    for patient in unique_patients:
        patient_data = data[data['Patient#'] == patient]
        # Assign patient to their majority class
        majority_class = patient_data['Healing Phase Abs'].mode()[0]
        patient_classes[patient] = majority_class

    # Stratify patients by class
    class_patients = {0: [], 1: [], 2: []}
    for patient, cls in patient_classes.items():
        class_patients[cls].append(patient)

    # Shuffle patients within each class
    for cls in class_patients:
        np.random.shuffle(class_patients[cls])

    # Create folds
    folds = [[] for _ in range(n_folds)]

    # Distribute patients from each class across folds
    for cls, patients in class_patients.items():
        patients_per_fold = len(patients) // n_folds
        remainder = len(patients) % n_folds

        start_idx = 0
        for fold_idx in range(n_folds):
            # Add one extra patient to some folds if there's remainder
            fold_size = patients_per_fold + (1 if fold_idx < remainder else 0)
            end_idx = start_idx + fold_size
            folds[fold_idx].extend(patients[start_idx:end_idx])
            start_idx = end_idx

    # Create train/valid splits for each fold
    fold_splits = []
    for fold_idx in range(n_folds):
        valid_patients = np.array(folds[fold_idx])
        train_patients = np.array([p for i, fold in enumerate(folds) if i != fold_idx for p in fold])

        # Validate the split
        train_data = data[data['Patient#'].isin(train_patients)]
        valid_data = data[data['Patient#'].isin(valid_patients)]

        # Check class balance
        train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
        valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)

        # Calculate max distribution difference
        max_diff = max(abs(train_dist.get(cls, 0) - valid_dist.get(cls, 0)) for cls in [0, 1, 2])

        # Check all classes present
        train_classes = set(train_data['Healing Phase Abs'].unique())
        valid_classes = set(valid_data['Healing Phase Abs'].unique())

        if len(train_classes) < 3 or len(valid_classes) < 3:
            print(f"Warning: Fold {fold_idx + 1} missing classes (train: {train_classes}, valid: {valid_classes})")
        elif max_diff > max_imbalance:
            print(f"Warning: Fold {fold_idx + 1} has class imbalance {max_diff:.3f} (threshold: {max_imbalance})")

        fold_splits.append((train_patients, valid_patients))

        vprint(f"Fold {fold_idx + 1}/{n_folds}: {len(train_patients, level=2)} train patients, {len(valid_patients)} valid patients")
        ordered_train = {i: train_dist.get(i, 0) for i in [0, 1, 2]}
        ordered_valid = {i: valid_dist.get(i, 0) for i in [0, 1, 2]}
        vprint(f"  Train dist: {{{', '.join([f'{k}: {v:.3f}' for k, v in ordered_train.items(, level=2)])}}}")
        vprint(f"  Valid dist: {{{', '.join([f'{k}: {v:.3f}' for k, v in ordered_valid.items(, level=2)])}}}")

    return fold_splits


def save_patient_split(run, train_patients, valid_patients, checkpoint_dir=None):
    """
    Save patient split for a specific run to ensure consistency across modality combinations.

    Args:
        run: Run number (0-indexed)
        train_patients: Array of patient numbers for training
        valid_patients: Array of patient numbers for validation
        checkpoint_dir: Directory to save the split (default: output_paths['checkpoints'])
    """
    if checkpoint_dir is None:
        checkpoint_dir = output_paths['checkpoints']

    os.makedirs(checkpoint_dir, exist_ok=True)
    split_file = os.path.join(checkpoint_dir, f'patient_split_run{run + 1}.npz')

    np.savez(split_file,
             train_patients=train_patients,
             valid_patients=valid_patients)
    print(f"Saved patient split for run {run + 1} to {split_file}")
    print(f"  Train patients: {len(train_patients)}, Valid patients: {len(valid_patients)}")


def load_patient_split(run, checkpoint_dir=None):
    """
    Load patient split for a specific run if it exists.

    Args:
        run: Run number (0-indexed)
        checkpoint_dir: Directory containing the split (default: output_paths['checkpoints'])

    Returns:
        Tuple of (train_patients, valid_patients) if file exists, else (None, None)
    """
    if checkpoint_dir is None:
        checkpoint_dir = output_paths['checkpoints']

    split_file = os.path.join(checkpoint_dir, f'patient_split_run{run + 1}.npz')

    if os.path.exists(split_file):
        data = np.load(split_file)
        train_patients = data['train_patients']
        valid_patients = data['valid_patients']
        print(f"Loaded existing patient split for run {run + 1} from {split_file}")
        print(f"  Train patients: {len(train_patients)}, Valid patients: {len(valid_patients)}")
        return train_patients, valid_patients

    return None, None


def create_cached_dataset(best_matching_df, selected_modalities, batch_size,
                         is_training=True, cache_dir=None, augmentation_fn=None, image_size=128):
    """
    Create a cached TF dataset optimized for training/validation with support for generative augmentation.

    Args:
        best_matching_df: DataFrame with matching data
        selected_modalities: List of selected modalities
        batch_size: Batch size
        is_training: Whether this is for training
        cache_dir: Directory for caching
        augmentation_fn: Custom augmentation function (including generative augmentations)
        image_size: Target image size for preprocessing (default: 128)
    """
    # Extract folder paths from data_paths for use in nested functions
    image_folder = data_paths['image_folder']
    depth_folder = data_paths['depth_folder']
    thermal_folder = data_paths['thermal_folder']
    thermal_rgb_folder = data_paths['thermal_rgb_folder']

    def process_single_sample(filename, bb_coords, modality_name):
        """Process a single image sample using py_function"""
        def _process_image(filename_tensor, bb_coords_tensor, modality_tensor):
            try:
                # Convert tensors to numpy/python types
                filename_str = filename_tensor.numpy().decode('utf-8')
                bb_coords_float = bb_coords_tensor.numpy()
                modality_str = modality_tensor.numpy().decode('utf-8')
                
                base_folders = {
                    'depth_rgb': image_folder,
                    'depth_map': depth_folder,
                    'thermal_rgb': thermal_rgb_folder,
                    'thermal_map': thermal_folder
                }
                
                img_path = os.path.join(base_folders[modality_str], filename_str)
                img_tensor = load_and_preprocess_image(
                    img_path, 
                    bb_coords_float,
                    modality_str,
                    target_size=(image_size, image_size),
                    augment=False
                )
                
                # Convert TensorFlow tensor to numpy array
                if isinstance(img_tensor, tf.Tensor):
                    img_array = img_tensor.numpy()
                else:
                    img_array = np.array(img_tensor)
                
                return img_array
            
            except Exception as e:
                print(f"Error in _process_image: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                return np.zeros((image_size, image_size, 3), dtype=np.float32)

        processed_image = tf.py_function(
            _process_image,
            [filename, bb_coords, modality_name],
            tf.float32
        )
        # Set the shape that was lost during py_function
        processed_image.set_shape((image_size, image_size, 3))
        return processed_image

    def load_and_preprocess_single_sample(row):
        features = {}
        # Handle image modalities
        for modality in [m for m in selected_modalities if m != 'metadata']:
            # Extract and validate bounding box coordinates
            if modality in ['depth_rgb', 'depth_map']:
                # Get all depth-related coordinates
                bb_coords = []
                for prefix in ['depth_']:
                    bb_coords.extend([
                        float(row[f'{prefix}xmin']),
                        float(row[f'{prefix}ymin']),
                        float(row[f'{prefix}xmax']),
                        float(row[f'{prefix}ymax'])
                    ])
                
                bb_coords = tf.stack(bb_coords)
            else:  # thermal modalities
                bb_coords = tf.stack([
                    float(row['thermal_xmin']),
                    float(row['thermal_ymin']),
                    float(row['thermal_xmax']),
                    float(row['thermal_ymax'])
                ])
            
            # Convert modality to tensor
            modality_tensor = tf.convert_to_tensor(modality, dtype=tf.string)
            
            # Process image
            img_tensor = process_single_sample(
                row[modality], 
                bb_coords,
                modality_tensor
            )
            
            features[f'{modality}_input'] = img_tensor

        # Handle metadata if selected
        if 'metadata' in selected_modalities:
            metadata_features = tf.stack([
                tf.cast(row['rf_prob_I'], tf.float32),
                tf.cast(row['rf_prob_P'], tf.float32),
                tf.cast(row['rf_prob_R'], tf.float32)
            ])
            features['metadata_input'] = metadata_features
        
        # Add sample identifiers for visualization only (not used in training)
        features['sample_id'] = tf.stack([
            tf.cast(row['Patient#'], tf.float32),
            tf.cast(row['Appt#'], tf.float32),
            tf.cast(row['DFU#'], tf.float32)
        ])
        
        # Extract label
        label = tf.cast(row['Healing Phase Abs'], tf.int32)
        label = tf.one_hot(label, depth=3)
        
        return features, label

    def df_to_dataset(dataframe):
        # Make a copy to avoid modifying the original
        df = dataframe.copy()
        tensor_slices = {}
        for col in df.columns:
            # Convert to appropriate numpy dtype
            if col in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                tensor_slices[col] = df[col].astype(str).values
                print(f"{col} type: {type(tensor_slices[col])} shape: {tensor_slices[col].shape}")
            elif col in ['Healing Phase Abs']:
                tensor_slices[col] = df[col].astype(np.int32).values
                print(f"{col} type: {type(tensor_slices[col])} shape: {tensor_slices[col].shape}")
            else:
                tensor_slices[col] = df[col].astype(np.float32).values
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
        return dataset

    # Initialize dataset from DataFrame
    dataset = df_to_dataset(best_matching_df)
    
    # Apply preprocessing to each sample
    dataset = dataset.map(
        load_and_preprocess_single_sample,
        num_parallel_calls=tf.data.AUTOTUNE
        # num_parallel_calls=4
    )
    
    # Calculate how many samples we need
    n_samples = len(best_matching_df)
    steps = np.ceil(n_samples / batch_size)
    k = int(steps * batch_size)  # Total number of samples needed
    
    # Cache the dataset with unique filename per modality combination
    modality_suffix = '_'.join(sorted(selected_modalities))  # e.g., "depth_rgb_metadata"
    cache_filename = f'tf_cache_train_{modality_suffix}' if is_training else f'tf_cache_valid_{modality_suffix}'

    # Use results/tf_records as default cache directory
    if cache_dir is None:
        cache_dir = os.path.join(result_dir, 'tf_records')

    os.makedirs(cache_dir, exist_ok=True)
    dataset = dataset.cache(os.path.join(cache_dir, cache_filename))
    
    with tf.device('/CPU:0'):
        pre_aug_dataset = dataset
        pre_aug_dataset = pre_aug_dataset.batch(batch_size)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000 if len(best_matching_df) > 1000 else len(best_matching_df), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Apply augmentation after batching
    if is_training:
        if augmentation_fn:
            # Use provided augmentation function (includes generative augmentations)
            dataset = dataset.map(
                augmentation_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
                # num_parallel_calls=4
                )
        # else:                                         #TODO: Add back default augmentations
        #     # Fall back to regular augmentation
        #     dataset = dataset.map(
        #         create_augmentation_fn(prob=0.25),
        #         num_parallel_calls=tf.data.AUTOTUNE
        #     )

    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.prefetch(2)
    return dataset, pre_aug_dataset, steps
# First, add this helper function near the start of your prepare_cached_datasets function
def check_split_validity(train_data, valid_data, max_ratio_diff=0.3, verbose=False):
    """
    Check if both splits contain all classes and have reasonable class distributions.
    
    Args:
        train_data: Training dataset
        valid_data: Validation dataset
        max_ratio_diff: Maximum allowed difference in class ratios between splits
        verbose: Whether to print detailed distribution information
        
    Returns:
        bool: True if split is valid, False otherwise
    """
    # Check if all classes present in both splits
    train_classes = set(train_data['Healing Phase Abs'].unique())
    valid_classes = set(valid_data['Healing Phase Abs'].unique())
    
    if len(train_classes) != 3 or len(valid_classes) != 3:
        if verbose:
            print(f"Missing classes - Train: {train_classes}, Valid: {valid_classes}")
        return False
    
    # Compare class distributions
    train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
    valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)
    
    if verbose:
        print("\nClass distributions:")
        print("Training:", dict(train_dist.round(3)))
        print("Validation:", dict(valid_dist.round(3)))
    
    # Check if class ratios are reasonably similar
    max_diff = 0
    for cls in [0, 1, 2]:
        train_ratio = train_dist.get(cls, 0)
        valid_ratio = valid_dist.get(cls, 0)
        diff = abs(train_ratio - valid_ratio)
        max_diff = max(max_diff, diff)
        if diff > max_ratio_diff:
            if verbose:
                print(f"Class {cls} ratio difference too high: {diff:.3f}")
            return False
    
    if verbose:
        print(f"Maximum ratio difference: {max_diff:.3f}")
    return True

def prepare_cached_datasets(data1, selected_modalities, train_patient_percentage=0.8,
                          batch_size=32, cache_dir=None, gen_manager=None, aug_config=None, run=0, max_split_diff=0.1, image_size=128,
                          train_patients=None, valid_patients=None, for_shape_inference=False):
    """
    Prepare cached datasets with proper metadata handling based on selected modalities.

    Args:
        max_split_diff: Maximum allowed class distribution difference between train/val (default 0.1)
                       Use higher values (e.g., 0.3) for small datasets with few patients
        image_size: Target image size for preprocessing (default: 128)
        train_patients: Optional pre-computed list of training patient IDs (for k-fold CV)
        valid_patients: Optional pre-computed list of validation patient IDs (for k-fold CV)
        for_shape_inference: If True, use quick split without messages (for metadata shape detection only)
    """
    random.seed(42 + run * (run + 3))
    tf.random.set_seed(42 + run * (run + 3))
    np.random.seed(42 + run * (run + 3))
    os.environ['PYTHONHASHSEED'] = f"{42 + run * (run + 3)}"
    # Create a deep copy of the data
    data = data1.copy(deep=True)
    data = data.reset_index(drop=True)
    
    # Then in your splitting code:
    # Convert labels once before the loop (not inside the loop!)
    if 'Healing Phase Abs' in data.columns:
        data['Healing Phase Abs'] = data['Healing Phase Abs'].astype(str)
        data['Healing Phase Abs'] = data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})

    # Use pre-computed patient splits if provided (k-fold CV), otherwise try to load/generate
    if train_patients is None or valid_patients is None:
        # Only try to load if not doing shape inference
        if not for_shape_inference:
            # Try to load existing patient split for this run to ensure consistency across modality combinations
            train_patients, valid_patients = load_patient_split(run)

    if train_patients is not None and valid_patients is not None:
        # Use the loaded/provided split
        if not for_shape_inference:
            vprint(f"Using consistent patient split across all modality combinations for run {run + 1}", level=2)
        train_data = data[data['Patient#'].isin(train_patients)]
        valid_data = data[data['Patient#'].isin(valid_patients)]

        # Display distributions (skip for shape inference)
        if not for_shape_inference:
            train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
            valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)
            ordered_train = {i: train_dist[i] if i in train_dist else 0 for i in [0, 1, 2]}
            ordered_valid = {i: valid_dist[i] if i in valid_dist else 0 for i in [0, 1, 2]}
            vprint("\nClass distributions:", level=2)
            vprint("Training:", {k: round(v, 3, level=2) for k, v in ordered_train.items()})
            vprint("Validation:", {k: round(v, 3, level=2) for k, v in ordered_valid.items()})
    else:
        # Generate a new split (first modality combination for this run)
        if not for_shape_inference:
            print(f"Generating new patient split for run {run + 1} (will be reused for all combinations)")
        max_retries = 2000
        best_split_diff = float('inf')
        best_split = None

        for attempt in range(max_retries):
            # Shuffle and split patients
            patient_numbers = sorted(data['Patient#'].unique())
            n_train_patients = int(len(patient_numbers) * train_patient_percentage)
            np.random.shuffle(patient_numbers)
            train_patients = patient_numbers[:n_train_patients]
            valid_patients = patient_numbers[n_train_patients:]

            # Split data
            train_data = data[data['Patient#'].isin(train_patients)]
            valid_data = data[data['Patient#'].isin(valid_patients)]

            # Calculate maximum distribution difference for this split
            train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
            valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)
            max_diff = max(abs(train_dist.get(cls, 0) - valid_dist.get(cls, 0)) for cls in [0, 1, 2])

            # Keep track of best split even if not perfect
            if max_diff < best_split_diff and len(set(train_data['Healing Phase Abs'].unique())) == 3 and len(set(valid_data['Healing Phase Abs'].unique())) == 3:
                best_split_diff = max_diff
                best_split = (train_data.copy(), valid_data.copy())

            # Check if split is valid
            if check_split_validity(train_data, valid_data, max_ratio_diff=max_split_diff):
                if not for_shape_inference:
                    print(f"Found valid split after {attempt + 1} attempts")
                    print("\nFinal class distributions:")
                    # Create ordered distributions
                    ordered_train = {i: train_dist[i] if i in train_dist else 0 for i in [0, 1, 2]}
                    ordered_valid = {i: valid_dist[i] if i in valid_dist else 0 for i in [0, 1, 2]}
                    vprint("Training:", {k: round(v, 3, level=2) for k, v in ordered_train.items()})
                    vprint("Validation:", {k: round(v, 3, level=2) for k, v in ordered_valid.items()})

                    # Save the split so all subsequent modality combinations use the same split
                    save_patient_split(run, train_patients, valid_patients)
                break

            if attempt == max_retries - 1:
                if not for_shape_inference:
                    print(f"Warning: Could not find optimal split after {max_retries} attempts.")
                    print(f"Using best found split with max difference of {best_split_diff:.3f}")
                if best_split is not None:
                    train_data, valid_data = best_split
                    train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
                    valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)

                    # Create ordered distributions
                    ordered_train = {i: train_dist[i] if i in train_dist else 0 for i in [0, 1, 2]}
                    ordered_valid = {i: valid_dist[i] if i in valid_dist else 0 for i in [0, 1, 2]}

                    if not for_shape_inference:
                        print("\nBest found class distributions:")
                        vprint("Training:", {k: round(v, 3, level=2) for k, v in ordered_train.items()})
                        vprint("Validation:", {k: round(v, 3, level=2) for k, v in ordered_valid.items()})

                        # Save the split so all subsequent modality combinations use the same split
                        save_patient_split(run, train_patients, valid_patients)
                else:
                    if not for_shape_inference:
                        print("No valid split found with all classes present in both sets")
                    raise ValueError("Could not create valid data split")
    
    # Determine columns to keep based on selected modalities
    if 'metadata' in selected_modalities:
        columns_to_keep = ['Healing Phase Abs']
    else:
        columns_to_keep = ['Patient#', 'Appt#', 'DFU#','Healing Phase Abs']
    
    # Add image and bounding box columns only once
    added_depth_bb = False
    added_thermal_bb = False
    
    for modality in selected_modalities:
        if modality == 'metadata':
            # Keep all metadata columns excluding image and bounding box columns
            metadata_columns = [col for col in data.columns if col not in [
                'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'
            ]]
            columns_to_keep.extend(metadata_columns)
        else:
            # Add image filename
            columns_to_keep.append(modality)
            
            # Add bounding box coordinates only once per type
            if modality in ['depth_rgb', 'depth_map'] and not added_depth_bb:
                columns_to_keep.extend(['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax'])
                added_depth_bb = True
            elif modality in ['thermal_rgb', 'thermal_map'] and not added_thermal_bb:
                columns_to_keep.extend(['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'])
                added_thermal_bb = True

    # Keep only necessary columns
    train_data = train_data[columns_to_keep].copy()
    valid_data = valid_data[columns_to_keep].copy()
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    
    # Apply sampling to training data only
    def apply_mixed_sampling_to_df(df, apply_sampling=True, mix=False):
        """
        Apply both random over and undersampling to balance classes with explicit ordering
        """
        alpha_unique_cases = df[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates().copy()
        X_alpha = df.drop('Healing Phase Abs', axis=1)
        y_alpha = df['Healing Phase Abs']
        X = df.drop('Healing Phase Abs', axis=1)
        y = df['Healing Phase Abs']
        resampled_df = df.copy()
        
        # Print original distribution with ordered classes
        counts = Counter(y_alpha)
        vprint("Original class distribution (ordered, level=2):")
        for class_idx in [0, 1, 2]:  # Explicit ordering
            print(f"Class {class_idx}: {counts[class_idx]}")
        # Calculate alpha values from original distribution
        total_samples = sum(counts.values())
        class_frequencies = {cls: count/total_samples for cls, count in counts.items()}
        median_freq = np.median(list(class_frequencies.values()))
        alpha_values = [median_freq/class_frequencies[i] for i in [0, 1, 2]]  # Keep ordered
        alpha_sum = sum(alpha_values)
        alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]

        vprint("\nCalculated alpha values from original distribution:", level=2)
        vprint(f"Alpha values (ordered, level=2) [I, P, R]: {[round(a, 3) for a in alpha_values]}")
        if not apply_sampling:
            # Return same format as sampling case
            resampled_df = df.copy()
            resampled_df = resampled_df.reset_index(drop=True)
            return resampled_df, alpha_values
        
        count_items = [(count, index) for index, count in counts.items()]
        count_items.sort()
        try:
            if not mix:
                OverSampleOnly
                
            del resampled_df
            # Calculate intermediate targets maintaining class order
            intermediate_target = {
                count_items[1][1]: count_items[1][0],          # Keep I class (0) as is
                count_items[2][1]: count_items[1][0],          # Reduce P class (1) to match I class
                count_items[0][1]: counts[2]           # Keep R class (2) as is
            }
            
            undersampler = RandomUnderSampler(
                sampling_strategy=intermediate_target,
                random_state=42 + run * (run + 3)
            )
            X_under, y_under = undersampler.fit_resample(X, y)
            
            # Print intermediate results with ordered classes
            under_counts = Counter(y_under)
            print("\nAfter undersampling (ordered):")
            for class_idx in [0, 1, 2]:
                print(f"Class {class_idx}: {under_counts[class_idx]}")
            
            # Set final targets
            final_target = {
                count_items[1][1]: count_items[1][0],          # Keep I class as is
                count_items[2][1]: count_items[1][0],          # Keep P class as reduced
                count_items[0][1]: count_items[1][0]           # Boost R class to match
            }
            
            oversampler = RandomOverSampler(
                sampling_strategy=final_target,
                random_state=42 + run * (run + 3)
            )
            X_resampled, y_resampled = oversampler.fit_resample(X_under, y_under)
            
            # Print final results with ordered classes
            final_counts = Counter(y_resampled)
            print("\nAfter oversampling (ordered):")
            for class_idx in [0, 1, 2]:
                print(f"Class {class_idx}: {final_counts[class_idx]}")
            
            # Reconstruct DataFrame
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Healing Phase Abs'] = y_resampled
            resampled_df = resampled_df.reset_index(drop=True)
        
            # Verify final order of classes
            print("\nVerifying final class distribution...")
            final_verify = Counter(resampled_df['Healing Phase Abs'])
            assert all(final_verify[i] == count_items[1][0] for i in [0, 1, 2]), "Classes not properly balanced"
            
            return resampled_df, alpha_values
            
        except Exception as e:
            print(f"Error in mixed sampling: {str(e)}")
            print("Falling back to simple random oversampling...")
            # Print original distribution with ordered classes
            counts = Counter(y_alpha)
            vprint("Original class distribution (ordered, level=2):")
            for class_idx in [0, 1, 2]:  # Explicit ordering
                print(f"Class {class_idx}: {counts[class_idx]}")
            # Calculate alpha values from original distribution
            total_samples = sum(counts.values())
            class_frequencies = {cls: count/total_samples for cls, count in counts.items()}
            median_freq = np.median(list(class_frequencies.values()))
            alpha_values = [median_freq/class_frequencies[i] for i in [0, 1, 2]]  # Keep ordered
            alpha_sum = sum(alpha_values)
            alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]
            
            oversampler = RandomOverSampler(random_state=42 + run * (run + 3))
            X_resampled, y_resampled = oversampler.fit_resample(X, y)
            
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Healing Phase Abs'] = y_resampled
            resampled_df = resampled_df.reset_index(drop=True)
            
            # Print final results with ordered classes
            final_counts = Counter(y_resampled)
            print("\nAfter oversampling (ordered):")
            for class_idx in [0, 1, 2]:
                print(f"Class {class_idx}: {final_counts[class_idx]}")
            
            return resampled_df, alpha_values
    if 'metadata' in selected_modalities:
        # Calculate class weights for Random Forest models
        unique_cases = train_data[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates().copy()
        print(f"\nUnique cases: {len(unique_cases)} (before oversampling)")
        
        # Create binary labels on unique cases
        unique_cases['label_bin1'] = (unique_cases['Healing Phase Abs'] > 0).astype(int)
        unique_cases['label_bin2'] = (unique_cases['Healing Phase Abs'] > 1).astype(int)
        
        # Print true class distributions
        vprint("\nTrue binary label distributions (unique cases, level=2):")
        vprint("Binary", unique_cases['label_bin1'].value_counts(, level=2))
        vprint("Binary", unique_cases['label_bin2'].value_counts(, level=2))
        
        # Calculate weights using only unique cases
        class_weights_binary1 = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=unique_cases['label_bin1']
        )
        class_weight_dict_binary1 = dict(zip([0, 1], class_weights_binary1))
        
        class_weights_binary2 = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=unique_cases['label_bin2']
        )
        class_weight_dict_binary2 = dict(zip([0, 1], class_weights_binary2))
        
        # print("\nClass weights (based on unique cases):")
        # print("Binary1:", class_weight_dict_binary1)
        # print("Binary2:", class_weight_dict_binary2)   
    else:
        class_weight_dict_binary1=None
        class_weight_dict_binary2=None
    
    train_data, alpha_values = apply_mixed_sampling_to_df(train_data, apply_sampling=False, mix=False)
    
    def preprocess_split(split_data, is_training=True, class_weight_dict_binary1=None, 
                            class_weight_dict_binary2=None, rf_model1=None, rf_model2=None, imputation_data=None):
            """Preprocess data with proper handling of metadata and image columns"""
            # Create a copy of the data
            split_data = split_data.copy()
            if imputation_data is not None:
                source_data = imputation_data.copy()
            else:
                source_data = split_data.copy()

            # Only process metadata if it's in selected modalities
            if 'metadata' in selected_modalities:
                # Identify columns for metadata processing
                image_related_columns = [
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'
                ]
                
                # Create a copy with only metadata columns
                metadata_df = split_data.drop(columns=[col for col in image_related_columns if col in split_data.columns])
                source_df = source_data.drop(columns=[col for col in image_related_columns if col in source_data.columns])
                
                # Feature engineering on metadata columns only
                metadata_df['BMI'] = metadata_df['Weight (Kg)'] / ((metadata_df['Height (cm)'] / 100) ** 2)
                source_df['BMI'] = source_df['Weight (Kg)'] / ((source_df['Height (cm)'] / 100) ** 2)
                metadata_df['Age above 60'] = (metadata_df['Age'] > 60).astype(int)
                source_df['Age above 60'] = (source_df['Age'] > 60).astype(int)
                metadata_df['Age Bin'] = pd.cut(metadata_df['Age'], bins=range(0, int(metadata_df['Age'].max()) + 20, 20), right=False, labels=range(len(range(0, int(metadata_df['Age'].max()) + 20, 20)) - 1))
                source_df['Age Bin'] = pd.cut(source_df['Age'], bins=range(0, int(source_df['Age'].max()) + 20, 20), right=False, labels=range(len(range(0, int(source_df['Age'].max()) + 20, 20)) - 1))
                metadata_df['Weight Bin'] = pd.cut(metadata_df['Weight (Kg)'], bins=range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20), right=False, labels=range(len(range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20)) - 1))
                source_df['Weight Bin'] = pd.cut(source_df['Weight (Kg)'], bins=range(0, int(source_df['Weight (Kg)'].max()) + 20, 20), right=False, labels=range(len(range(0, int(source_df['Weight (Kg)'].max()) + 20, 20)) - 1))
                metadata_df['Height Bin'] = pd.cut(metadata_df['Height (cm)'], bins=range(0, int(metadata_df['Height (cm)'].max()) + 10, 10), right=False, labels=range(len(range(0, int(metadata_df['Height (cm)'].max()) + 10, 10)) - 1))
                source_df['Height Bin'] = pd.cut(source_df['Height (cm)'], bins=range(0, int(source_df['Height (cm)'].max()) + 10, 10), right=False, labels=range(len(range(0, int(source_df['Height (cm)'].max()) + 10, 10)) - 1))
                
                # Handle categorical columns
                categorical_columns = ['Sex (F:0, M:1)', 'Side (Left:0, Right:1)', 'Foot Aspect', 'Odor', 'Type of Pain Grouped']
                for col in categorical_columns:
                    if col in metadata_df.columns:
                        metadata_df[col] = pd.Categorical(metadata_df[col]).codes
                        source_df[col] = pd.Categorical(source_df[col]).codes
                
                # Other categorical mappings
                categorical_mappings = {
                    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {'ankle': 4, 'Heel': 3, 'middle': 2, 'toes': 1, 'Hallux': 0},
                    'Dressing Grouped': {'NoDressing': 0, 'BandAid': 1, 'BasicDressing': 1, 'AbsorbantDressing': 2, 'Antiseptic': 3, 'AdvanceMethod': 4, 'other': 4},
                    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {'Serous': 0, 'Haemoserous': 1, 'Bloody': 2, 'Thick': 3}
                }
                
                for col, mapping in categorical_mappings.items():
                    if col in metadata_df.columns:
                        metadata_df[col] = metadata_df[col].map(mapping)
                        source_df[col] = source_df[col].map(mapping)
                # Remove unnecessary columns
                features_to_drop = [
                    'ID', 'Location', 'Healing Phase', 'Phase Confidence (%)', 'DFU#', 'Appt#',
                    'Appt Days', 'Type of Pain2', 'Type of Pain_Grouped2', 'Type of Pain', 
                    'Peri-Ulcer Temperature (°C)', 'Wound Centre Temperature (°C)', 'Dressing',
                    'Dressing Grouped', "No Offloading", "Offloading: Therapeutic Footwear", 
                    "Offloading: Scotcast Boot or RCW", "Offloading: Half Shoes or Sandals", 
                    "Offloading: Total Contact Cast", "Offloading: Crutches, Walkers or Wheelchairs", 
                    "Offloading Score"
                ]
                integer_columns = [
                    "Sex (F:0, M:1)", "Smoking","Alcohol Consumption", "Habits Score", "Type of Diabetes", "Heart Conditions", 
                    "Cancer History", "Sensory Peripheral", "Clinical Score", "Number of DFUs","Side (Left:0, Right:1)", "Foot Aspect", "Location", 
                    "Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)","No Toes Deformities", "Bunion", "Claw", "Hammer", "Charcot Arthropathy", 
                    "Flat (Pes Planus) Arch", "Abnormally High Arch","No Arch Deformities", "Foot Score", 
                    "Pain Level", "Type of Pain", "Type of Pain Grouped", "Type of Pain2","Type of Pain_Grouped2", "Wound Tunneling", 
                    "Exudate Amount (None:0,Minor,Medium,Severe:3)","Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)", "Odor", 
                    "No Peri-ulcer Conditions (False:0, True:1)", "Erythema at Peri-ulcer","Edema at Peri-ulcer", "Pale Colour at Peri-ulcer", "Maceration at Peri-ulcer", 
                    "Wound Score", "Dressing","Dressing Grouped", 
                    "No Foot Abnormalities", "Foot Hair Loss", "Foot Dry Skin","Foot Fissure Cracks", "Foot Callus", "Thickened Toenail", 
                    "Foot Fungal Nails", "Leg Score", "No Offloading","Offloading: Therapeutic Footwear", "Offloading: Scotcast Boot or RCW", 
                    "Offloading: Half Shoes or Sandals", "Offloading: Total Contact Cast","Offloading: Crutches, Walkers or Wheelchairs", "Offloading Score", "Healing Phase Abs", "Healing Phase Abs Regression", 'Age Bin', 'Age above 60', 'Weight Bin', 'Height Bin'
                ]
                metadata_df = metadata_df.drop(columns=[col for col in features_to_drop if col in metadata_df.columns])
                source_df = source_df.drop(columns=[col for col in features_to_drop if col in source_df.columns])
                # Impute missing values
                columns_to_impute = [col for col in metadata_df.columns if col not in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                                                                        'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                                                                        'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']+['Healing Phase Abs']]
                
                # numeric_columns = split_data.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=5)
                source_df[columns_to_impute] = imputer.fit_transform(source_df[columns_to_impute])
                metadata_df[columns_to_impute] = imputer.transform(metadata_df[columns_to_impute])
                
                for column in integer_columns:
                    if column in metadata_df.columns:
                        metadata_df[column] = metadata_df[column].astype(int)
                        source_df[column] = source_df[column].astype(int)
                    else:
                        continue
                
                # Scale features
                scaler = StandardScaler()
                source_df[columns_to_impute] = scaler.fit_transform(source_df[columns_to_impute])
                metadata_df[columns_to_impute] = scaler.transform(metadata_df[columns_to_impute])
                
                # Random Forest processing
                if is_training:
                    try:
                        import tensorflow_decision_forests as tfdf
                        print("Using TensorFlow Decision Forests")
                        
                        # Create models
                        rf_model1 = tfdf.keras.RandomForestModel(
                            num_trees=300,
                            task=tfdf.keras.Task.CLASSIFICATION,
                            random_seed=42 + run * (run + 3),
                            verbose=0
                        )
                        rf_model2 = tfdf.keras.RandomForestModel(
                            num_trees=300,
                            task=tfdf.keras.Task.CLASSIFICATION,
                            random_seed=42 + run * (run + 3),
                            verbose=0
                        )
                        
                        # Prepare features for RF
                        train_df = metadata_df.copy()
                        
                        # Create binary labels and verify their values
                        train_df['label_bin1'] = (train_df['Healing Phase Abs'] > 0).astype(int)
                        train_df['label_bin2'] = (train_df['Healing Phase Abs'] > 1).astype(int)
                        
                        # # Print value counts to verify
                        # print("\nLabel binary 1 distribution:", train_df['label_bin1'].value_counts())
                        # print("Label binary 2 distribution:", train_df['label_bin2'].value_counts())
                        # print("\nClass weights 1:", class_weight_dict_binary1)
                        # print("Class weights 2:", class_weight_dict_binary2)
                        
                        # Add weights with explicit mapping
                        train_df['weight1'] = train_df['label_bin1'].apply(lambda x: class_weight_dict_binary1[x])
                        train_df['weight2'] = train_df['label_bin2'].apply(lambda x: class_weight_dict_binary2[x])
                        
                        # # Print weight distribution to verify
                        # print("\nWeight1 unique values:", train_df['weight1'].unique())
                        # print("Weight2 unique values:", train_df['weight2'].unique())
                        
                        # Remove unnecessary columns
                        cols_to_drop = ['Patient#', 'Healing Phase Abs']
                        
                        # Create datasets
                        dataset1 = tfdf.keras.pd_dataframe_to_tf_dataset(
                            train_df.drop(columns=cols_to_drop + ['label_bin2', 'weight2']),
                            label='label_bin1',
                            weight='weight1'
                        )
                        
                        dataset2 = tfdf.keras.pd_dataframe_to_tf_dataset(
                            train_df.drop(columns=cols_to_drop + ['label_bin1', 'weight1']),
                            label='label_bin2',
                            weight='weight2'
                        )
                        
                        # Train models
                        rf_model1.fit(dataset1)
                        rf_model2.fit(dataset2)
                    except ImportError:
                        vprint("Using Scikit-learn RandomForestClassifier", level=2)
                        from sklearn.ensemble import RandomForestClassifier
                        rf_model1 = RandomForestClassifier(
                            n_estimators=300,
                            random_state=42 + run * (run + 3),
                            class_weight=class_weight_dict_binary1,
                            n_jobs=-1,
                            # max_features=None
                        )
                        rf_model2 = RandomForestClassifier(
                            n_estimators=300, #100,
                            random_state=42 + run * (run + 3),
                            class_weight=class_weight_dict_binary2,
                            n_jobs=-1,
                            # max_features=None
                        )
                        # Prepare features for RF
                        X = metadata_df.drop(['Patient#', 'Healing Phase Abs'], axis=1)
                        # y = split_data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})
                        y = metadata_df['Healing Phase Abs']
                        y_bin1 = (y > 0).astype(int)
                        y_bin2 = (y > 1).astype(int)
                        # Train RF models
                        rf_model1.fit(X, y_bin1)
                        rf_model2.fit(X, y_bin2)    
                try:
                    import tensorflow_decision_forests as tfdf
                    dataset1 = tfdf.keras.pd_dataframe_to_tf_dataset(
                        metadata_df.drop(['Patient#', 'Healing Phase Abs'], axis=1),
                        label=None  # No label needed for prediction
                    )
                    
                    # Get predictions
                    with tf.device('/CPU:0'):
                        pred1 = rf_model1.predict(dataset1)
                        pred2 = rf_model2.predict(dataset1)
                    # with tf.device('/CPU:0'):
                    #     prob1 = rf_predict_function(rf_model1, dataset1)
                    #     prob2 = rf_predict_function(rf_model2, dataset1)
                        # # Get probabilities for positive class (class 1)
                        prob1 = np.squeeze(pred1)
                        prob2 = np.squeeze(pred2)
                except ImportError:
                    dataset = metadata_df.drop(['Patient#', 'Healing Phase Abs'], axis=1)
                    # dataset_pd = tf_to_pd(dataset)
                    prob1 = rf_model1.predict_proba(dataset)[:, 1]
                    prob2 = rf_model2.predict_proba(dataset)[:, 1]
                
                # Calculate final probabilities
                prob_I = 1 - prob1
                prob_P = prob1 * (1 - prob2)
                prob_R = prob2
                
                # Store RF probabilities in the DataFrame
                split_data['rf_prob_I'] = prob_I
                split_data['rf_prob_P'] = prob_P
                split_data['rf_prob_R'] = prob_R
            
            metadata_columns = [col for col in split_data.columns if col not in [
                'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax',
                'rf_prob_I', 'rf_prob_P', 'rf_prob_R',
                'Patient#', 'Appt#', 'DFU#'
            ]]
            split_data = split_data.drop(columns=metadata_columns)
            
            return split_data, rf_model1, rf_model2

    # Preprocess both splits
    source_data = train_data.copy()
    del train_data
    train_data, rf_model1, rf_model2 = preprocess_split(source_data, is_training=True, class_weight_dict_binary1=class_weight_dict_binary1, class_weight_dict_binary2=class_weight_dict_binary2)
    valid_data, _, _ = preprocess_split(valid_data, is_training=False, rf_model1=rf_model1, rf_model2=rf_model2, imputation_data=source_data)
    del rf_model1, rf_model2
    
    # Create cached datasets
    train_dataset, pre_aug_dataset, steps_per_epoch = create_cached_dataset(
        train_data,
        selected_modalities,
        batch_size,
        is_training=True,
        cache_dir=cache_dir,  # Pass through the cache_dir parameter
        augmentation_fn=create_enhanced_augmentation_fn(gen_manager, aug_config) if gen_manager else None,
        image_size=image_size)

    valid_dataset, _, validation_steps = create_cached_dataset(
        valid_data,
        selected_modalities,
        batch_size,
        is_training=False,
        cache_dir=cache_dir,  # Pass through the cache_dir parameter
        augmentation_fn=None,
        image_size=image_size)
    
    return train_dataset, pre_aug_dataset, valid_dataset, steps_per_epoch, validation_steps, alpha_values
class BatchVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, modalities, freq=5, max_samples=5, run=1, save_dir='batch_visualizations'):
        """
        Args:
            dataset: TensorFlow dataset to visualize
            modalities: List of modality names
            freq: Frequency of epochs to create visualizations
            max_samples: Maximum number of samples to visualize
            save_dir: Directory to save visualizations
        """
        super(BatchVisualizationCallback, self).__init__()
        self.dataset = dataset
        self.modalities = modalities
        self.freq = freq
        self.max_samples = max_samples
        self.save_dir = save_dir
        self.run = run
        os.makedirs(save_dir, exist_ok=True)
        # Create modality abbreviation mapping
        self.modality_abbrev = {
            'metadata': 'md',
            'depth_rgb': 'dr',
            'depth_map': 'dm',
            'thermal_rgb': 'tr',
            'thermal_map': 'tm'
        }
    def _create_short_filename(self, epoch):
        """Create a shorter filename using modality abbreviations"""
        # Get abbreviated modalities string
        mod_str = '_'.join(self.modality_abbrev[m] for m in sorted(self.modalities))
        return f'e{epoch}_{mod_str}_{self.run}.png'
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            try:
                num_modalities = len(self.modalities)
                fig, axes = plt.subplots(self.max_samples, num_modalities, 
                                    figsize=(6*num_modalities, 5*self.max_samples))
                
                if self.max_samples == 1:
                    axes = axes.reshape(1, -1)
                elif num_modalities == 1:
                    axes = axes.reshape(-1, 1)
                
                label_map = {0: 'I', 1: 'P', 2: 'R'}
                
                # Get one batch from the dataset
                for batch in self.dataset.take(1):
                    batch_inputs, batch_labels = batch
                    
                    # Convert batch_labels to numpy if it's a tensor
                    if isinstance(batch_labels, tf.Tensor):
                        batch_labels = batch_labels.numpy()
                    
                    # Adjust max_samples if batch is smaller
                    actual_batch_size = batch_labels.shape[0]
                    num_samples = min(self.max_samples, actual_batch_size)
                    
                    for i in range(num_samples):
                        for j, modality in enumerate(self.modalities):
                            ax = axes[i, j]
                            input_key = f'{modality}_input'
                            
                            if input_key not in batch_inputs:
                                ax.text(0.5, 0.5, f"Missing modality: {modality}", 
                                    ha='center', va='center')
                                continue
                            
                            # Get the input data for this modality
                            modality_data = batch_inputs[input_key][i]
                            if isinstance(modality_data, tf.Tensor):
                                modality_data = modality_data.numpy()
                            
                            if modality == 'metadata':
                                # Plot RF probabilities
                                ax.clear()  # Clear previous plot
                                ax.bar(['I', 'P', 'R'], modality_data)
                                ax.set_ylim(0, 1)
                                ax.set_title('RF Probabilities')
                            else:
                                # Handle image data
                                ax.clear()  # Clear previous plot
                                if modality_data.ndim == 3 and modality_data.shape[2] == 3:
                                    if modality_data.max() <= 1.0:
                                        modality_data = (modality_data * 255).astype(np.uint8)
                                    ax.imshow(modality_data)
                                elif modality_data.ndim == 2 or (modality_data.ndim == 3 and modality_data.shape[2] == 1):
                                    ax.imshow(modality_data.squeeze(), cmap='gray')
                                else:
                                    ax.text(0.1, 0.5, f"Unsupported shape: {modality_data.shape}", 
                                        wrap=True, fontsize=8)
                                
                                ax.axis('off')
                            
                            if i == 0:
                                ax.set_title(modality)
                        
                        # Add label information
                        label_index = np.argmax(batch_labels[i])
                        label = label_map[label_index]
                        # Get model predictions
                        pred_batch = self.model.predict(batch_inputs, verbose=0)
                        pred_index = np.argmax(pred_batch[i])
                        pred_label = label_map[pred_index]
                        
                        # Get patient info from the passed identifiers
                        patient_info = ""
                        try:
                            ids = batch_inputs['sample_id'][i].numpy()
                            patient_info = f"\nP{int(ids[0]):03d}A{int(ids[1]):02d}D{int(ids[2]):d}"
                        except Exception as e:
                            print(f"Error getting patient info: {str(e)}")
                        
                        axes[i, 0].text(-0.1, -0.15, 
                                    f"Sample {i}\nTrue: {label}\nPred: {pred_label}{patient_info}", 
                                    transform=axes[i, 0].transAxes, 
                                    fontsize=8, 
                                    verticalalignment='top')
                    
                    break  # Only process one batch
                
                plt.suptitle(f'Epoch {epoch + 1}')
                plt.tight_layout()
                save_path = os.path.join(self.save_dir,  self._create_short_filename(epoch + 1))
                plt.savefig(save_path, bbox_inches='tight', dpi=400)
                plt.close(fig)
                
                print(f"\nBatch visualization saved for epoch {epoch + 1}")
            except Exception as e:
                print(f"Error during batch visualization: {str(e)}")
def visualize_dataset(train_dataset, selected_modalities, save_dir, 
                     max_samples_per_page=5, dataset_portion=100, 
                     dpi=400, max_batches=None, total_samples=None):
    """
    Visualize samples from a training dataset with multiple modalities.
    
    Args:
        train_dataset: TensorFlow dataset to visualize
        selected_modalities: List of modality names
        save_dir: Directory to save visualizations
        max_samples_per_page: Maximum number of samples per visualization page
        dataset_portion: Percentage of dataset to visualize (1-100)
        dpi: DPI for saved images
        max_batches: Optional maximum number of batches to process
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create modality abbreviation mapping
    modality_abbrev = {
        'metadata': 'md',
        'depth_rgb': 'dr',
        'depth_map': 'dm',
        'thermal_rgb': 'tr',
        'thermal_map': 'tm'
    }
    
    # Create shorter filename
    def create_filename(page_num):
        mod_str = '_'.join(modality_abbrev[m] for m in sorted(selected_modalities))
        return f'dataset_p{page_num}_{mod_str}.png'
    
    label_map = {0: 'I', 1: 'P', 2: 'R'}
    num_modalities = len(selected_modalities)
    processed_samples = 0
    page_num = 1
    
    # Initialize variables
    current_page_samples = []
    batch_count = 0
    
    # Get one batch to estimate total samples
    for first_batch in train_dataset.take(1):
        batch_size = first_batch[1].shape[0]
        break
    
    print(f"Processing dataset (batch size: {batch_size})")
    print(f"Will create pages with up to {max_samples_per_page} samples each")
    
    # Process the dataset
    for batch in train_dataset:
        if max_batches and batch_count >= max_batches:
            break
            
        batch_inputs, batch_labels = batch
        batch_size = batch_labels.shape[0]
        
        # Convert batch_labels to numpy if needed
        if isinstance(batch_labels, tf.Tensor):
            batch_labels = batch_labels.numpy()
        
        # Process each sample in the batch
        for i in range(batch_size):
            if total_samples and processed_samples >= total_samples:
                break
            # Skip samples based on dataset_portion
            if dataset_portion < 100:
                if np.random.rand() > dataset_portion / 100:
                    continue
            
            # Store sample data
            sample_data = {
                'inputs': {},
                'label': batch_labels[i],
                'sample_id': batch_inputs['sample_id'][i].numpy()
            }
            
            for modality in selected_modalities:
                input_key = f'{modality}_input'
                if input_key in batch_inputs:
                    modality_data = batch_inputs[input_key][i]
                    if isinstance(modality_data, tf.Tensor):
                        modality_data = modality_data.numpy()
                    sample_data['inputs'][modality] = modality_data
            
            current_page_samples.append(sample_data)
            processed_samples += 1

        
            # Create visualization when page is full
            if len(current_page_samples) >= max_samples_per_page:
                # Create figure
                fig, axes = plt.subplots(len(current_page_samples), num_modalities,
                                       figsize=(6*num_modalities, 5*len(current_page_samples)))
                
                # Handle single row/column cases
                if len(current_page_samples) == 1:
                    axes = axes.reshape(1, -1)
                elif num_modalities == 1:
                    axes = axes.reshape(-1, 1)
                
                # Plot each sample
                for row, sample in enumerate(current_page_samples):
                    for col, modality in enumerate(selected_modalities):
                        ax = axes[row, col]
                        
                        if modality not in sample['inputs']:
                            ax.text(0.5, 0.5, f"Missing modality: {modality}",
                                  ha='center', va='center')
                            continue
                        
                        modality_data = sample['inputs'][modality]
                        
                        if modality == 'metadata':
                            ax.clear()
                            ax.bar(['I', 'P', 'R'], modality_data)
                            ax.set_ylim(0, 1)
                            ax.set_title('RF Probabilities')
                        else:
                            ax.clear()
                            if modality_data.ndim == 3 and modality_data.shape[2] == 3:
                                if modality_data.max() <= 1.0:
                                    modality_data = (modality_data * 255).astype(np.uint8)
                                ax.imshow(modality_data)
                            elif modality_data.ndim == 2 or (modality_data.ndim == 3 and modality_data.shape[2] == 1):
                                ax.imshow(modality_data.squeeze(), cmap='gray')
                            else:
                                ax.text(0.1, 0.5, f"Unsupported shape: {modality_data.shape}",
                                      wrap=True, fontsize=8)
                            ax.axis('off')
                        
                        if row == 0:
                            ax.set_title(modality)
                    
                    # Add sample information
                    label_index = np.argmax(sample['label'])
                    label = label_map[label_index]
                    ids = sample['sample_id']
                    patient_info = f"P{int(ids[0]):03d}A{int(ids[1]):02d}D{int(ids[2]):d}"
                    
                    axes[row, 0].text(-0.1, -0.15,
                                    f"Sample {processed_samples-len(current_page_samples)+row+1}\n"
                                    f"Label: {label}\n{patient_info}",
                                    transform=axes[row, 0].transAxes,
                                    fontsize=8,
                                    verticalalignment='top')
                
                # Save figure
                plt.suptitle(f'Dataset Visualization - Page {page_num}')
                plt.tight_layout()
                save_path = os.path.join(save_dir, create_filename(page_num))
                plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
                plt.close(fig)
                
                print(f"Saved page {page_num} ({processed_samples} samples)")
                
                # Reset for next page
                current_page_samples = []
                page_num += 1
                
        if total_samples and processed_samples >= total_samples:
            break
        batch_count += 1
    
    # Save any remaining samples
    if current_page_samples:
        # Create figure for remaining samples
        fig, axes = plt.subplots(len(current_page_samples), num_modalities,
                               figsize=(6*num_modalities, 5*len(current_page_samples)))
        
        # Handle single row/column cases
        if len(current_page_samples) == 1:
            axes = axes.reshape(1, -1)
        elif num_modalities == 1:
            axes = axes.reshape(-1, 1)
                
        for row, sample in enumerate(current_page_samples):
            for col, modality in enumerate(selected_modalities):
                ax = axes[row, col]
                
                if modality not in sample['inputs']:
                    ax.text(0.5, 0.5, f"Missing modality: {modality}",
                          ha='center', va='center')
                    continue
                
                modality_data = sample['inputs'][modality]
                
                if modality == 'metadata':
                    ax.clear()
                    ax.bar(['I', 'P', 'R'], modality_data)
                    ax.set_ylim(0, 1)
                    ax.set_title('RF Probabilities')
                else:
                    ax.clear()
                    if modality_data.ndim == 3 and modality_data.shape[2] == 3:
                        if modality_data.max() <= 1.0:
                            modality_data = (modality_data * 255).astype(np.uint8)
                        ax.imshow(modality_data)
                    elif modality_data.ndim == 2 or (modality_data.ndim == 3 and modality_data.shape[2] == 1):
                        ax.imshow(modality_data.squeeze(), cmap='gray')
                    else:
                        ax.text(0.1, 0.5, f"Unsupported shape: {modality_data.shape}",
                              wrap=True, fontsize=8)
                    ax.axis('off')
                
                if row == 0:
                    ax.set_title(modality)
            
            # Add sample information
            label_index = np.argmax(sample['label'])
            label = label_map[label_index]
            ids = sample['sample_id']
            patient_info = f"P{int(ids[0]):03d}A{int(ids[1]):02d}D{int(ids[2]):d}"
            
            axes[row, 0].text(-0.1, -0.15,
                            f"Sample {processed_samples-len(current_page_samples)+row+1}\n"
                            f"Label: {label}\n{patient_info}",
                            transform=axes[row, 0].transAxes,
                            fontsize=8,
                            verticalalignment='top')
        
        # Save final figure
        plt.suptitle(f'Dataset Visualization - Page {page_num} (Final)')
        plt.tight_layout()
        save_path = os.path.join(save_dir, create_filename(page_num))
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        print(f"Saved final page {page_num} ({processed_samples} total samples)")
    
    print(f"\nVisualization complete! Created {page_num} pages with {processed_samples} samples")
    return processed_samples
class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir='training_plots', update_freq=1):
        """
        Args:
            save_dir: Directory to save the plots
            update_freq: Frequency of epochs to update plots
        """
        super(TrainingHistoryCallback, self).__init__()
        self.save_dir = save_dir
        self.update_freq = update_freq
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize history storage
        self.history = {
            'loss': [], 'accuracy': [], 'weighted_f1_score': [],
            'val_loss': [], 'val_accuracy': [], 'val_weighted_f1_score': []
        }

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Update history
            for metric in self.history.keys():
                if metric in logs:
                    self.history[metric].append(logs[metric])
            
            if (epoch + 1) % self.update_freq == 0:
                metrics = ['loss', 'accuracy', 'weighted_f1_score']
                
                for metric in metrics:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot training metrics
                    if len(self.history[metric]) > 0:
                        plt.plot(self.history[metric], label=f'Train', alpha=0.7)
                    if len(self.history[f'val_{metric}']) > 0:
                        plt.plot(self.history[f'val_{metric}'], label=f'Validation', 
                                linestyle='--', alpha=0.7)
                    
                    plt.title(f'Model {metric.replace("_", " ").title()} (Epoch {epoch + 1})')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric.replace('_', ' ').title())
                    plt.legend(loc='best')
                    plt.grid(True, alpha=0.3)
                    
                    save_path = os.path.join(self.save_dir, f'{metric}_history.png')
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
        except Exception as e:
            print(f"Error during training history plot: {str(e)}")
            
def plot_net_confusion_matrix(y_true, y_pred, save_dir='evaluation_plots'):
    """
    Plot confusion matrix and print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_dir: Directory to save the confusion matrix plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    present_classes = np.unique(y_true)
    present_class_names = [CLASS_LABELS[i] for i in present_classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_class_names, 
                yticklabels=present_class_names)
    plt.title('Net Confusion Matrix Across All Runs')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    save_path = os.path.join(save_dir, 'net_confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print("\nNet Confusion Matrix Results:")
    print(classification_report(y_true, y_pred, 
                              target_names=present_class_names, 
                              labels=present_classes))
    print(f"Net Cohen's Kappa: {cohen_kappa_score(y_true, y_pred, weights='quadratic'):.4f}")

