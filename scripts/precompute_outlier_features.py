"""
Precompute Feature Cache for Multimodal Outlier Detection

Pre-computes and caches deep features for each modality to enable fast
combination-specific outlier detection. Uses the same architecture and
preprocessing as the main training pipeline.

Supports CPU, single-GPU, and multi-GPU modes for efficient processing.

Usage:
    # Pre-compute all features with multi-GPU (default)
    python scripts/precompute_outlier_features.py --image-size 32 --modalities all

    # Pre-compute with single GPU
    python scripts/precompute_outlier_features.py --modalities all --device-mode single

    # Pre-compute specific modalities with CPU only
    python scripts/precompute_outlier_features.py --modalities metadata thermal_map --device-mode cpu

    # Force recompute existing cache
    python scripts/precompute_outlier_features.py --modalities all --force
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_project_paths, get_data_paths
from src.utils.production_config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
from src.models.builders import create_image_branch
from src.data.image_processing import load_and_preprocess_image
from src.utils.gpu_config import setup_device_strategy


def create_feature_extractor(modality, image_size=32, strategy=None):
    """
    Create feature extractor using same architecture as training pipeline.

    Extracts features from GlobalAveragePooling2D layer (before projection layers)
    to get semantic representations from the convolutional layers.

    Args:
        modality: Image modality name ('thermal_map', 'depth_rgb', etc.)
        image_size: Input image size (must match training IMAGE_SIZE)
        strategy: TensorFlow distribution strategy for multi-GPU (optional)

    Returns:
        tuple: (feature_extractor_model, feature_dimension)
    """
    # Use strategy scope if provided (multi-GPU), otherwise default scope
    scope = strategy.scope() if strategy else tf.keras.utils.custom_object_scope({})

    with scope:
        # Create full image branch using training architecture
        input_shape = (image_size, image_size, 3)
        image_input, branch_output = create_image_branch(input_shape, modality)

        # Find GlobalAveragePooling2D layer
        # Build a temporary model to access layers
        temp_model = tf.keras.Model(inputs=image_input, outputs=branch_output)

        gap_layer = None
        for layer in temp_model.layers:
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                gap_layer = layer
                break

        if gap_layer is None:
            raise ValueError(f"Could not find GlobalAveragePooling2D layer for modality {modality}")

        # Create feature extractor that outputs from GAP layer
        feature_extractor = tf.keras.Model(
            inputs=image_input,
            outputs=gap_layer.output
        )

        # Get feature dimension
        feature_dim = gap_layer.output.shape[-1]

        print(f"  Created feature extractor for {modality}: {feature_dim} dims from {gap_layer.name}")

        return feature_extractor, feature_dim


def extract_metadata_features(metadata_df):
    """
    Extract metadata features (same as Phase 7 implementation).

    Args:
        metadata_df: DataFrame with metadata

    Returns:
        numpy array: (n_samples, n_features) metadata features
    """
    # Prepare features for outlier detection (numeric only, exclude IDs/labels/paths)
    exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs', 'Healing Phase',
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']

    feature_cols = [col for col in metadata_df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

    # Fill missing values with median
    X = metadata_df[feature_cols].fillna(metadata_df[feature_cols].median()).values

    print(f"  Extracted metadata: {X.shape[0]} samples × {X.shape[1]} features")

    return X.astype(np.float32)


def extract_image_features(modality, metadata_df, data_paths, image_size=32, batch_size=32, strategy=None):
    """
    Extract image features using training pipeline architecture and preprocessing.

    Args:
        modality: Image modality name
        metadata_df: DataFrame with image paths and bounding boxes
        data_paths: Dictionary of data paths
        image_size: Target image size (must match training)
        batch_size: Batch size for feature extraction
        strategy: TensorFlow distribution strategy for multi-GPU (optional)

    Returns:
        numpy array: (n_samples, feature_dim) image features
    """
    # Create feature extractor
    feature_extractor, feature_dim = create_feature_extractor(modality, image_size, strategy)

    # Determine image folder and bounding box columns based on modality
    if modality in ['depth_rgb', 'depth_map']:
        if modality == 'depth_rgb':
            folder = data_paths['image_folder']
        else:  # depth_map
            folder = data_paths['depth_folder']
        bb_cols = ['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']
    elif modality in ['thermal_rgb', 'thermal_map']:
        if modality == 'thermal_rgb':
            folder = data_paths['thermal_rgb_folder']
        else:  # thermal_map
            folder = data_paths['thermal_folder']
        bb_cols = ['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
    else:
        raise ValueError(f"Unknown image modality: {modality}")

    # Extract features for all samples
    n_samples = len(metadata_df)
    all_features = np.zeros((n_samples, feature_dim), dtype=np.float32)

    print(f"  Extracting features from {n_samples} images...")

    # Process in batches with progress bar
    for start_idx in tqdm(range(0, n_samples, batch_size), desc=f"  {modality}"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_images = []

        for idx in range(start_idx, end_idx):
            row = metadata_df.iloc[idx]

            # Get image path
            image_filename = row[modality]
            image_path = os.path.join(folder, image_filename)

            # Get bounding box
            bb_coords = row[bb_cols].values

            # Load and preprocess image (same as training pipeline)
            try:
                img_tensor = load_and_preprocess_image(
                    filepath=image_path,
                    bb_data=bb_coords,
                    modality=modality,
                    target_size=(image_size, image_size),
                    augment=False  # No augmentation for feature extraction
                )
                batch_images.append(img_tensor.numpy())
            except Exception as e:
                print(f"\n  Warning: Error loading {image_path}: {e}")
                # Use zero image as fallback
                batch_images.append(np.zeros((image_size, image_size, 3), dtype=np.float32))

        # Extract features for batch
        batch_array = np.array(batch_images)
        batch_features = feature_extractor.predict(batch_array, verbose=0)
        all_features[start_idx:end_idx] = batch_features

    print(f"  Extracted {modality}: {all_features.shape[0]} samples × {all_features.shape[1]} features")

    return all_features


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute feature cache for multimodal outlier detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-compute all features with multi-GPU (recommended)
  python scripts/precompute_outlier_features.py --modalities all --device-mode multi

  # Pre-compute specific modalities with single GPU
  python scripts/precompute_outlier_features.py --modalities metadata thermal_map --device-mode single

  # Custom image size
  python scripts/precompute_outlier_features.py --image-size 64 --modalities all

  # Force recompute
  python scripts/precompute_outlier_features.py --modalities all --force
        """
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f'Image size for feature extraction (default: {DEFAULT_IMAGE_SIZE})'
    )

    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['all'],
        choices=['all', 'metadata', 'thermal_map', 'depth_rgb', 'depth_map', 'thermal_rgb'],
        help='Modalities to cache (default: all)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )

    parser.add_argument(
        '--device-mode',
        type=str,
        choices=['cpu', 'single', 'multi'],
        default='multi',
        help='Device mode: cpu (CPU only), single (1 GPU), multi (all GPUs, default)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute even if cache exists'
    )

    args = parser.parse_args()

    # Setup GPU strategy
    print("="*80)
    print("Multimodal Outlier Detection - Feature Cache Builder")
    print("="*80)
    print(f"Device mode: {args.device_mode}")

    strategy = setup_device_strategy(args.device_mode)

    if strategy:
        print(f"Running on {strategy.num_replicas_in_sync} device(s)")
    else:
        print("Running on CPU")

    # Setup paths
    _, _, root = get_project_paths()
    data_paths = get_data_paths(root)
    cache_dir = root.parent / 'cache_outlier'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image size: {args.image_size}")
    print(f"Cache directory: {cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load metadata
    print("Loading dataset...")
    best_matching_file = data_paths['best_matching_csv']
    if not os.path.exists(best_matching_file):
        print(f"Error: best_matching.csv not found at {best_matching_file}")
        print("Please run the main training pipeline first to create this file.")
        return 1

    metadata_df = pd.read_csv(best_matching_file)
    print(f"  Loaded {len(metadata_df)} samples from best_matching.csv")
    print()

    # Determine which modalities to process
    if 'all' in args.modalities:
        modalities_to_cache = ['metadata', 'thermal_map', 'depth_rgb', 'depth_map', 'thermal_rgb']
    else:
        modalities_to_cache = args.modalities

    print(f"Processing {len(modalities_to_cache)} modalities: {', '.join(modalities_to_cache)}")
    print()

    # Extract and cache features for each modality
    for modality in modalities_to_cache:
        # Cache filename includes image size for image modalities
        if modality == 'metadata':
            cache_file = cache_dir / f'{modality}_features.npy'
            cache_name = f'{modality}_features.npy'
        else:
            cache_file = cache_dir / f'{modality}_features_{args.image_size}.npy'
            cache_name = f'{modality}_features_{args.image_size}.npy'

        # Check if cache exists
        if cache_file.exists() and not args.force:
            print(f"✓ {modality}: Cache exists, skipping (use --force to recompute)")
            cached = np.load(cache_file)
            print(f"  File: {cache_name}")
            print(f"  Shape: {cached.shape}, Size: {cached.nbytes / 1024 / 1024:.1f} MB")
            print()
            continue

        print(f"Processing {modality}...")

        # Extract features
        if modality == 'metadata':
            # Metadata extraction is CPU-only (NumPy operations)
            features = extract_metadata_features(metadata_df)
        else:
            # Image extraction can use GPU strategy
            features = extract_image_features(
                modality,
                metadata_df,
                data_paths,
                image_size=args.image_size,
                batch_size=args.batch_size,
                strategy=strategy
            )

        # Save to cache
        np.save(cache_file, features)
        print(f"  ✓ Saved to: {cache_name}")
        print(f"  Shape: {features.shape}, Size: {features.nbytes / 1024 / 1024:.1f} MB")
        print()

    print("="*80)
    print("Feature cache building complete!")
    print("="*80)
    print(f"\nCache location: {cache_dir}")
    print(f"Image size: {args.image_size}")
    print("\nCached files:")
    for modality in modalities_to_cache:
        # Use correct filename based on modality type
        if modality == 'metadata':
            cache_file = cache_dir / f'{modality}_features.npy'
            cache_name = f'{modality}_features.npy'
        else:
            cache_file = cache_dir / f'{modality}_features_{args.image_size}.npy'
            cache_name = f'{modality}_features_{args.image_size}.npy'

        if cache_file.exists():
            size_mb = cache_file.stat().st_size / 1024 / 1024
            print(f"  ✓ {cache_name} ({size_mb:.1f} MB)")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
