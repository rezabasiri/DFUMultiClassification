"""
Outlier Detection and Removal Utility

Detects and removes outliers from metadata using per-class Isolation Forest.
Integrated into main training pipeline for improved model performance.

Based on Phase 7 investigation: 15% outlier removal + 'combined' sampling
achieves Kappa 0.27 (+63% vs baseline).

See: agent_communication/fusion_fix/FUSION_FIX_GUIDE.md
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import Counter
import os
import shutil
from pathlib import Path

from src.utils.config import get_data_paths, get_project_paths
from src.utils.verbosity import vprint


def detect_outliers(contamination=0.15, random_state=42, force_recompute=False):
    """
    Detect outliers in metadata using per-class Isolation Forest.

    Args:
        contamination: Expected proportion of outliers (0.0-1.0), default 0.15
        random_state: Random seed for reproducibility
        force_recompute: If True, recompute even if cleaned dataset exists

    Returns:
        tuple: (cleaned_df, outlier_df, output_file)
            - cleaned_df: DataFrame with outliers removed
            - outlier_df: DataFrame of detected outliers
            - output_file: Path to saved cleaned dataset
    """
    # Check if already computed
    _, _, root = get_project_paths()
    root = Path(root)  # root is data directory (<project>/data)
    output_file = root / f"cleaned/metadata_cleaned_{int(contamination*100):02d}pct.csv"
    outlier_file = root / f"cleaned/outliers_{int(contamination*100):02d}pct.csv"

    if output_file.exists() and not force_recompute:
        vprint(f"Using existing cleaned dataset: {output_file}", level=1)
        cleaned_df = pd.read_csv(output_file)
        outlier_df = pd.read_csv(outlier_file) if outlier_file.exists() else None
        return cleaned_df, outlier_df, output_file

    vprint(f"Detecting outliers (contamination={contamination*100:.0f}%)...", level=1)

    # Load full dataset
    data_paths = get_data_paths(root)
    metadata_df = pd.read_csv(data_paths['csv_file'])
    vprint(f"  Loaded {len(metadata_df)} samples", level=2)

    # Get class distribution
    class_dist = Counter(metadata_df['Healing Phase Abs'])
    vprint("  Original class distribution:", level=2)
    for cls in ['I', 'P', 'R']:
        count = class_dist[cls]
        pct = count / len(metadata_df) * 100
        vprint(f"    {cls}: {count:3d} ({pct:5.1f}%)", level=2)

    # Prepare features for outlier detection (numeric only, exclude IDs/labels/paths)
    exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs', 'Healing Phase',
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
    feature_cols = [col for col in metadata_df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]
    X = metadata_df[feature_cols].fillna(metadata_df[feature_cols].median()).values
    y = metadata_df['Healing Phase Abs'].values

    vprint(f"  Using {len(feature_cols)} features", level=2)

    # Per-class outlier detection
    outlier_mask = np.zeros(len(metadata_df), dtype=bool)
    outlier_counts = {}

    for cls in ['I', 'P', 'R']:
        # Get samples for this class
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        X_cls = X[cls_mask]

        if len(X_cls) < 10:
            vprint(f"  Class {cls}: Skipping (too few samples: {len(X_cls)})", level=2)
            outlier_counts[cls] = 0
            continue

        # SAFETY: Reduce contamination for minority class R to preserve samples
        cls_contamination = contamination
        if cls == 'R' and len(X_cls) < 150:  # R class is minority
            cls_contamination = min(contamination, 0.10)
            if cls_contamination < contamination:
                vprint(f"  Class {cls}: Reducing contamination {contamination*100:.0f}% → {cls_contamination*100:.0f}% (minority class protection)", level=2)

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=cls_contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )

        predictions = iso_forest.fit_predict(X_cls)

        # -1 = outlier, 1 = inlier
        outliers_cls = (predictions == -1)
        n_outliers = outliers_cls.sum()

        # SAFETY CHECK: Never remove more than 20% of minority class
        max_allowed = int(len(X_cls) * 0.20)
        if cls == 'R' and n_outliers > max_allowed:
            vprint(f"  Class {cls}: Safety limit - would remove {n_outliers}, limiting to {max_allowed} (20% max)", level=2)
            # Keep only the top outliers by score
            scores = iso_forest.decision_function(X_cls)
            outlier_indices = np.argsort(scores)[:max_allowed]  # Most outlier-like
            outliers_cls = np.zeros(len(X_cls), dtype=bool)
            outliers_cls[outlier_indices] = True
            n_outliers = max_allowed

        outlier_counts[cls] = n_outliers

        # Mark outliers in global mask
        outlier_mask[cls_indices[outliers_cls]] = True

        vprint(f"  Class {cls}: Detected {n_outliers}/{len(X_cls)} outliers ({n_outliers/len(X_cls)*100:.1f}%)", level=2)

    # Create cleaned dataset
    cleaned_df = metadata_df[~outlier_mask].copy()
    outlier_df = metadata_df[outlier_mask][['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].copy()

    total_outliers = outlier_mask.sum()
    vprint(f"  Total outliers: {total_outliers}/{len(metadata_df)} ({total_outliers/len(metadata_df)*100:.1f}%)", level=1)
    vprint(f"  Cleaned dataset: {len(cleaned_df)} samples", level=1)

    # New class distribution
    cleaned_dist = Counter(cleaned_df['Healing Phase Abs'])
    vprint("  Cleaned class distribution:", level=2)
    for cls in ['I', 'P', 'R']:
        orig_count = class_dist[cls]
        new_count = cleaned_dist[cls]
        removed = orig_count - new_count
        vprint(f"    {cls}: {new_count:3d} (removed {removed}, {removed/orig_count*100:.1f}%)", level=2)

    # Save cleaned dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)
    outlier_df.to_csv(outlier_file, index=False)

    vprint(f"  Saved cleaned dataset: {output_file}", level=1)
    vprint(f"  Saved outlier list: {outlier_file}", level=2)

    return cleaned_df, outlier_df, output_file


def apply_cleaned_dataset(contamination=0.15, backup=True):
    """
    Apply cleaned dataset by filtering best_matching.csv to exclude outliers.

    Args:
        contamination: Contamination rate used for outlier detection
        backup: If True, backup original best_matching.csv before modifying

    Returns:
        bool: True if successful, False otherwise
    """
    _, _, root = get_project_paths()
    root = Path(root)  # root is data directory (<project>/data)
    project_root = root.parent

    # Paths
    best_matching_file = project_root / "results/best_matching.csv"
    best_matching_backup = project_root / "results/best_matching_original.csv"
    cleaned_file = root / f"cleaned/metadata_cleaned_{int(contamination*100):02d}pct.csv"

    if not cleaned_file.exists():
        vprint(f"Cleaned dataset not found: {cleaned_file}", level=0)
        vprint("Run outlier detection first", level=0)
        return False

    # Load cleaned metadata (contains Patient#, Appt#, DFU# of kept samples)
    cleaned_df = pd.read_csv(cleaned_file)

    # Backup original if needed
    if backup and not best_matching_backup.exists():
        vprint(f"Backing up original: {best_matching_file}", level=2)
        shutil.copy(best_matching_file, best_matching_backup)

    # Load original best_matching
    source_file = best_matching_backup if best_matching_backup.exists() else best_matching_file
    original_df = pd.read_csv(source_file)

    vprint(f"Applying cleaned dataset ({contamination*100:.0f}% outlier removal)...", level=1)
    vprint(f"  Original: {len(original_df)} samples", level=2)
    vprint(f"  Cleaned metadata: {len(cleaned_df)} samples", level=2)

    # Create key for matching
    cleaned_df['_key'] = (cleaned_df['Patient#'].astype(str) + '_' +
                          cleaned_df['Appt#'].astype(str) + '_' +
                          cleaned_df['DFU#'].astype(str))
    original_df['_key'] = (original_df['Patient#'].astype(str) + '_' +
                           original_df['Appt#'].astype(str) + '_' +
                           original_df['DFU#'].astype(str))

    # Filter to only include cleaned samples
    filtered_df = original_df[original_df['_key'].isin(cleaned_df['_key'])].copy()
    filtered_df = filtered_df.drop('_key', axis=1)

    vprint(f"  Filtered: {len(filtered_df)} samples", level=1)

    # Verify class distribution
    dist = Counter(filtered_df['Healing Phase Abs'])
    vprint(f"  Class distribution: I={dist['I']}, P={dist['P']}, R={dist['R']}", level=2)

    # Save filtered dataset
    filtered_df.to_csv(best_matching_file, index=False)
    vprint(f"  Applied cleaned dataset to: {best_matching_file}", level=1)

    return True


def restore_original_dataset():
    """
    Restore original best_matching.csv from backup.

    Returns:
        bool: True if restored, False if no backup exists
    """
    _, _, root = get_project_paths()
    root = Path(root)  # root is data directory (<project>/data)
    project_root = root.parent

    best_matching_file = project_root / "results/best_matching.csv"
    best_matching_backup = project_root / "results/best_matching_original.csv"

    if not best_matching_backup.exists():
        vprint("No backup found, dataset may already be original", level=1)
        return False

    vprint("Restoring original dataset...", level=1)
    shutil.copy(best_matching_backup, best_matching_file)
    vprint(f"  Restored: {best_matching_file}", level=1)

    return True


def check_metadata_in_modalities(modalities):
    """
    Check if metadata is included in any of the modality combinations.

    Args:
        modalities: Either a list of modalities or a list of combinations

    Returns:
        bool: True if metadata is present in any combination
    """
    if not modalities:
        return False

    # Handle single combination (list of strings)
    if isinstance(modalities[0], str):
        return 'metadata' in modalities

    # Handle multiple combinations (list of tuples/lists)
    for combo in modalities:
        if 'metadata' in combo:
            return True

    return False


# =============================================================================
# Multimodal Outlier Detection (Combination-Specific)
# =============================================================================


def get_combination_name(combination):
    """
    Generate a standardized name for a modality combination.

    Args:
        combination: Tuple or list of modality names

    Returns:
        str: Combination name (e.g., 'metadata_thermal_map')
    """
    return '_'.join(sorted(combination))


def load_cached_features(modality, cache_dir=None, image_size=None):
    """
    Load pre-computed features from cache.

    Cache filename includes backbone type to ensure correct features are loaded.
    Format: {modality}_features_{image_size}_{backbone}.npy

    Args:
        modality: Modality name ('metadata', 'thermal_map', etc.)
        cache_dir: Cache directory path (default: cache_outlier/)
        image_size: Image size used for cache (for image modalities only)

    Returns:
        numpy array: Cached features or None if not found
    """
    from src.utils.production_config import (
        IMAGE_SIZE as DEFAULT_IMAGE_SIZE,
        RGB_BACKBONE,
        MAP_BACKBONE
    )

    _, _, root = get_project_paths()
    root = Path(root)

    if cache_dir is None:
        cache_dir = root.parent / 'cache_outlier'

    # Use default image size if not specified
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZE

    # Determine backbone based on modality type
    if modality in ['depth_rgb', 'thermal_rgb']:
        backbone = RGB_BACKBONE
    elif modality in ['depth_map', 'thermal_map']:
        backbone = MAP_BACKBONE
    else:
        backbone = None  # metadata has no backbone

    # Cache filename includes image size and backbone for image modalities
    if modality == 'metadata':
        cache_file = cache_dir / f'{modality}_features.npy'
    else:
        cache_file = cache_dir / f'{modality}_features_{image_size}_{backbone}.npy'

    if not cache_file.exists():
        return None

    try:
        features = np.load(cache_file)
        return features
    except Exception as e:
        vprint(f"Error loading cache for {modality}: {e}", level=0)
        return None


def extract_features_on_the_fly(modality, best_matching_df, data_paths, image_size=32, batch_size=32, save_cache=True):
    """
    Extract features on-the-fly if cache is not available.
    Uses training pipeline architecture with current backbone configuration.

    Args:
        modality: Modality name
        best_matching_df: DataFrame with image paths and metadata
        data_paths: Dictionary of data paths
        image_size: Image size for feature extraction
        batch_size: Batch size for feature extraction (default: 32)
        save_cache: If True, save extracted features to cache for reuse (default: True)

    Returns:
        numpy array: Extracted features
    """
    import tensorflow as tf
    from src.models.builders import create_image_branch
    from src.data.image_processing import load_and_preprocess_image

    vprint(f"  Extracting {modality} features on-the-fly (cache not found)...", level=2)

    if modality == 'metadata':
        # Extract metadata features directly
        exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs', 'Healing Phase',
                        'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                        'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                        'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
        feature_cols = [col for col in best_matching_df.select_dtypes(include=[np.number]).columns
                        if col not in exclude_cols]
        X = best_matching_df[feature_cols].fillna(best_matching_df[feature_cols].median()).values
        return X.astype(np.float32)

    else:
        # Extract image features using training architecture
        # Import production config to check backbone type
        from src.utils.production_config import RGB_BACKBONE, MAP_BACKBONE

        # Create feature extractor
        input_shape = (image_size, image_size, 3)
        image_input, branch_output = create_image_branch(input_shape, modality)
        temp_model = tf.keras.Model(inputs=image_input, outputs=branch_output)

        # Determine which backbone is being used for this modality
        if modality in ['depth_rgb', 'thermal_rgb']:
            current_backbone = RGB_BACKBONE
        else:  # depth_map, thermal_map
            current_backbone = MAP_BACKBONE

        # Find appropriate feature extraction layer
        # For SimpleCNN: use GlobalAveragePooling2D
        # For EfficientNet: use first Dense layer (after built-in pooling)
        feature_layer = None

        if current_backbone == 'SimpleCNN':
            # Look for GlobalAveragePooling2D layer
            for layer in temp_model.layers:
                if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                    feature_layer = layer
                    vprint(f"  Using GlobalAveragePooling2D layer for {modality} features", level=2)
                    break
        else:
            # For EfficientNet backbones, use first Dense layer output
            # (EfficientNet has built-in pooling, so no separate GAP layer)
            for layer in temp_model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    feature_layer = layer
                    vprint(f"  Using Dense layer '{layer.name}' for {modality} features ({current_backbone})", level=2)
                    break

        if feature_layer is None:
            vprint(f"  Warning: Could not find suitable feature layer for {modality} ({current_backbone})", level=1)
            vprint(f"  Skipping outlier detection for {modality} - using identity features", level=1)
            # Return dummy features (will be ignored in multimodal detection)
            n_samples = len(best_matching_df)
            return np.zeros((n_samples, 1), dtype=np.float32)

        feature_extractor = tf.keras.Model(inputs=image_input, outputs=feature_layer.output)

        # Determine folder and bounding box columns
        if modality in ['depth_rgb', 'depth_map']:
            folder = data_paths['image_folder'] if modality == 'depth_rgb' else data_paths['depth_folder']
            bb_cols = ['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']
        elif modality in ['thermal_rgb', 'thermal_map']:
            folder = data_paths['thermal_rgb_folder'] if modality == 'thermal_rgb' else data_paths['thermal_folder']
            bb_cols = ['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']

        # Extract features
        n_samples = len(best_matching_df)
        feature_dim = feature_layer.output.shape[-1]
        all_features = np.zeros((n_samples, feature_dim), dtype=np.float32)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_images = []

            for idx in range(start_idx, end_idx):
                row = best_matching_df.iloc[idx]
                image_filename = row[modality]
                image_path = os.path.join(folder, image_filename)
                bb_coords = row[bb_cols].values

                try:
                    img_tensor = load_and_preprocess_image(
                        filepath=image_path,
                        bb_data=bb_coords,
                        modality=modality,
                        target_size=(image_size, image_size),
                        augment=False
                    )
                    batch_images.append(img_tensor.numpy())
                except Exception as e:
                    print(f"  [WARNING] Failed to load image for outlier detection (index {start_idx + j}): {type(e).__name__}: {e}", flush=True)
                    batch_images.append(np.zeros((image_size, image_size, 3), dtype=np.float32))

            batch_array = np.array(batch_images)
            batch_features = feature_extractor.predict(batch_array, verbose=0)
            all_features[start_idx:end_idx] = batch_features

        vprint(f"  Extracted {modality}: {all_features.shape[0]} × {all_features.shape[1]} features", level=2)

        # Save to cache for reuse if requested
        if save_cache:
            from src.utils.production_config import RGB_BACKBONE, MAP_BACKBONE
            _, _, root = get_project_paths()
            root = Path(root)
            cache_dir = root.parent / 'cache_outlier'
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Determine cache filename based on modality type
            if modality in ['depth_rgb', 'thermal_rgb']:
                backbone = RGB_BACKBONE
                cache_file = cache_dir / f'{modality}_features_{image_size}_{backbone}.npy'
            elif modality in ['depth_map', 'thermal_map']:
                backbone = MAP_BACKBONE
                cache_file = cache_dir / f'{modality}_features_{image_size}_{backbone}.npy'
            else:
                # Metadata - no backbone or image size needed
                cache_file = cache_dir / f'{modality}_features.npy'

            try:
                np.save(cache_file, all_features)
                vprint(f"  Saved to cache: {cache_file.name}", level=2)
            except Exception as e:
                vprint(f"  Warning: Could not save cache: {e}", level=2)

        return all_features


def detect_outliers_combination(combination, contamination=0.15, random_state=42,
                                 force_recompute=False, cache_dir=None,
                                 use_cache=True, image_size=32, batch_size=32):
    """
    Detect outliers for a specific modality combination using joint feature space.

    This function combines features from all modalities in the combination and runs
    per-class Isolation Forest on the joint feature space.

    HYBRID MODE: Tries to load from cache first. If cache doesn't exist and use_cache=True,
    extracts features on-the-fly using training pipeline architecture (with ImageNet weights).

    Args:
        combination: Tuple/list of modality names (e.g., ('metadata', 'thermal_map'))
        contamination: Expected proportion of outliers (0.0-1.0), default 0.15
        random_state: Random seed for reproducibility
        force_recompute: If True, recompute even if cleaned dataset exists
        cache_dir: Cache directory for pre-computed features
        use_cache: If True, try cache first; if False, always extract on-the-fly
        image_size: Image size for on-the-fly extraction (must match cache if using cache)
        batch_size: Batch size for on-the-fly extraction (default: 32)

    Returns:
        tuple: (cleaned_df, outlier_df, output_file)
            - cleaned_df: DataFrame with outliers removed
            - outlier_df: DataFrame of detected outliers
            - output_file: Path to saved cleaned dataset
    """
    from src.utils.production_config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE

    _, _, root = get_project_paths()
    root = Path(root)  # root is data directory (<project>/data)
    data_paths = get_data_paths(root)

    # Use default image size if not specified
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZE

    # Generate combination name
    combo_name = get_combination_name(combination)

    # Check if already computed
    output_file = root / f"cleaned/{combo_name}_{int(contamination*100):02d}pct.csv"
    outlier_file = root / f"cleaned/outliers_{combo_name}_{int(contamination*100):02d}pct.csv"

    if output_file.exists() and not force_recompute:
        vprint(f"Using existing cleaned dataset for {combo_name}: {output_file.name}", level=1)
        cleaned_df = pd.read_csv(output_file)
        outlier_df = pd.read_csv(outlier_file) if outlier_file.exists() else None
        return cleaned_df, outlier_df, output_file

    vprint(f"Detecting outliers for combination: {combo_name} (contamination={contamination*100:.0f}%)...", level=1)

    # Load best_matching dataset (contains all samples with metadata)
    best_matching_df = pd.read_csv(data_paths['best_matching_csv'])
    n_samples = len(best_matching_df)
    vprint(f"  Loaded {n_samples} samples", level=2)

    # Load and concatenate features for all modalities in combination
    all_features = []
    feature_dims = []

    for modality in combination:
        # Try to load cached features first
        features = None
        from_cache = False
        if use_cache:
            features = load_cached_features(modality, cache_dir, image_size)
            if features is not None:
                # Validate cached features match current dataset
                if len(features) != n_samples:
                    vprint(f"  Cache invalid for {modality}: {len(features)} samples vs {n_samples} expected (likely due to data_percentage or backbone change)", level=2)
                    vprint(f"  Discarding cache and extracting features on-the-fly...", level=2)
                    features = None  # Discard invalid cache
                else:
                    from_cache = True

        # Fall back to on-the-fly extraction if cache not available or invalid
        if features is None:
            if use_cache and not from_cache:
                vprint(f"  Cache not found for {modality} (image_size={image_size}), extracting on-the-fly...", level=2)
            features = extract_features_on_the_fly(modality, best_matching_df, data_paths, image_size, batch_size)

        if features is None:
            vprint(f"  Error: Failed to extract features for {modality}", level=0)
            return None, None, None

        if len(features) != n_samples:
            vprint(f"  Error: Feature count mismatch for {modality}: {len(features)} vs {n_samples} (this should not happen after cache validation)", level=0)
            return None, None, None

        all_features.append(features)
        feature_dims.append(features.shape[1])
        if from_cache:
            vprint(f"  Loaded {modality} from cache: {features.shape[1]} features", level=2)

    # Concatenate all features
    X = np.concatenate(all_features, axis=1)
    total_dims = X.shape[1]
    vprint(f"  Joint feature space: {n_samples} samples × {total_dims} features", level=2)

    # Get labels
    y = best_matching_df['Healing Phase Abs'].values

    # Get class distribution
    class_dist = Counter(y)
    vprint("  Original class distribution:", level=2)
    for cls in ['I', 'P', 'R']:
        count = class_dist[cls]
        pct = count / n_samples * 100
        vprint(f"    {cls}: {count:3d} ({pct:5.1f}%)", level=2)

    # Per-class outlier detection (same as metadata-only, but on joint features)
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_counts = {}

    for cls in ['I', 'P', 'R']:
        # Get samples for this class
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        X_cls = X[cls_mask]

        if len(X_cls) < 10:
            vprint(f"  Class {cls}: Skipping (too few samples: {len(X_cls)})", level=2)
            outlier_counts[cls] = 0
            continue

        # SAFETY: Reduce contamination for minority class R to preserve samples
        cls_contamination = contamination
        if cls == 'R' and len(X_cls) < 150:  # R class is minority
            cls_contamination = min(contamination, 0.10)
            if cls_contamination < contamination:
                vprint(f"  Class {cls}: Reducing contamination {contamination*100:.0f}% → {cls_contamination*100:.0f}% (minority class protection)", level=2)

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=cls_contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )

        predictions = iso_forest.fit_predict(X_cls)

        # -1 = outlier, 1 = inlier
        outliers_cls = (predictions == -1)
        n_outliers = outliers_cls.sum()

        # SAFETY CHECK: Never remove more than 20% of minority class
        max_allowed = int(len(X_cls) * 0.20)
        if cls == 'R' and n_outliers > max_allowed:
            vprint(f"  Class {cls}: Safety limit - would remove {n_outliers}, limiting to {max_allowed} (20% max)", level=2)
            # Keep only the top outliers by score
            scores = iso_forest.decision_function(X_cls)
            outlier_indices = np.argsort(scores)[:max_allowed]  # Most outlier-like
            outliers_cls = np.zeros(len(X_cls), dtype=bool)
            outliers_cls[outlier_indices] = True
            n_outliers = max_allowed

        outlier_counts[cls] = n_outliers

        # Mark outliers in global mask
        outlier_mask[cls_indices[outliers_cls]] = True

        vprint(f"  Class {cls}: Detected {n_outliers}/{len(X_cls)} outliers ({n_outliers/len(X_cls)*100:.1f}%)", level=2)

    # Create cleaned dataset
    cleaned_df = best_matching_df[~outlier_mask].copy()
    outlier_df = best_matching_df[outlier_mask][['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].copy()

    total_outliers = outlier_mask.sum()
    vprint(f"  Total outliers: {total_outliers}/{n_samples} ({total_outliers/n_samples*100:.1f}%)", level=1)
    vprint(f"  Cleaned dataset: {len(cleaned_df)} samples", level=1)

    # New class distribution
    cleaned_dist = Counter(cleaned_df['Healing Phase Abs'])
    vprint("  Cleaned class distribution:", level=2)
    for cls in ['I', 'P', 'R']:
        orig_count = class_dist[cls]
        new_count = cleaned_dist[cls]
        removed = orig_count - new_count
        vprint(f"    {cls}: {new_count:3d} (removed {removed}, {removed/orig_count*100:.1f}%)", level=2)

    # Save cleaned dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_file, index=False)
    outlier_df.to_csv(outlier_file, index=False)

    vprint(f"  Saved cleaned dataset: {output_file.name}", level=1)
    vprint(f"  Saved outlier list: {outlier_file.name}", level=2)

    return cleaned_df, outlier_df, output_file


def apply_cleaned_dataset_combination(combination, contamination=0.15, backup=True, modify_file=False):
    """
    Apply combination-specific cleaned dataset by filtering best_matching.csv.

    Uses depth_rgb filename as unique key (each image has unique filename based on
    patient/appt/dfu/angle/view combination per README.md naming convention).

    Args:
        combination: Tuple/list of modality names
        contamination: Contamination rate used for outlier detection
        backup: If True, backup original best_matching.csv before modifying (only if modify_file=True)
        modify_file: If True, modify best_matching.csv in place (legacy behavior).
                    If False (default), return filtered dataframe without modifying files.

    Returns:
        If modify_file=True: bool (True if successful, False otherwise)
        If modify_file=False: pd.DataFrame (filtered dataframe) or None if failed
    """
    _, _, root = get_project_paths()
    root = Path(root)  # root is data directory (<project>/data)
    project_root = root.parent

    # Generate combination name
    combo_name = get_combination_name(combination)

    # Paths
    best_matching_file = project_root / "results/best_matching.csv"
    best_matching_backup = project_root / "results/best_matching_original.csv"
    cleaned_file = root / f"cleaned/{combo_name}_{int(contamination*100):02d}pct.csv"

    if not cleaned_file.exists():
        vprint(f"Cleaned dataset not found for {combo_name}: {cleaned_file}", level=0)
        vprint(f"Run: detect_outliers_combination({combination}, contamination={contamination})", level=0)
        return False if modify_file else None

    # Load cleaned dataset (contains full row data including depth_rgb)
    cleaned_df = pd.read_csv(cleaned_file)

    # Backup original if needed (only if we're going to modify)
    if modify_file and backup and not best_matching_backup.exists():
        vprint(f"Backing up original: {best_matching_file.name}", level=2)
        shutil.copy(best_matching_file, best_matching_backup)

    # Load original best_matching (use backup if it exists to get full dataset)
    source_file = best_matching_backup if best_matching_backup.exists() else best_matching_file
    original_df = pd.read_csv(source_file)

    vprint(f"Applying cleaned dataset for {combo_name} ({contamination*100:.0f}% outlier removal)...", level=1)
    vprint(f"  Original: {len(original_df)} samples", level=2)
    vprint(f"  Cleaned: {len(cleaned_df)} samples", level=2)

    # Use depth_rgb as unique key (filename uniquely identifies each sample)
    # Filename format: {random_number}_P{patient#}{appt#}{dfu#}{B/A}{D/T}{R/M/L}{Z/W}.png
    # This is unique per image (includes angle R/M/L and view Z/W)
    cleaned_keys = set(cleaned_df['depth_rgb'].astype(str))
    filtered_df = original_df[original_df['depth_rgb'].astype(str).isin(cleaned_keys)].copy()

    vprint(f"  Filtered: {len(filtered_df)} samples", level=1)

    # Verify filtered count matches cleaned count
    if len(filtered_df) != len(cleaned_df):
        vprint(f"  Warning: Filtered count ({len(filtered_df)}) != Cleaned count ({len(cleaned_df)})", level=1)
        vprint(f"  This may indicate the cleaned file was generated from a different dataset version", level=2)

    # Verify class distribution
    dist = Counter(filtered_df['Healing Phase Abs'])
    vprint(f"  Class distribution: I={dist['I']}, P={dist['P']}, R={dist['R']}", level=2)

    if modify_file:
        # Legacy behavior: modify best_matching.csv in place
        filtered_df.to_csv(best_matching_file, index=False)
        vprint(f"  Applied cleaned dataset to: {best_matching_file.name}", level=1)
        return True
    else:
        # New behavior: return filtered dataframe without modifying files
        vprint(f"  Returning filtered dataframe (best_matching.csv unchanged)", level=2)
        return filtered_df
