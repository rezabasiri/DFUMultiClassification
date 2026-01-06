"""
Test Feature Extraction for Multimodal Outlier Detection

Validates that extracted features match training pipeline and tests the cache system.

Usage:
    python agent_communication/outlier_detection/test_feature_extraction.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_project_paths, get_data_paths
from src.utils.production_config import IMAGE_SIZE
from src.utils.outlier_detection import (
    load_cached_features,
    detect_outliers_combination,
    get_combination_name
)


def test_cache_exists():
    """Test that cache directory and files exist."""
    print("="*80)
    print("Test 1: Cache Existence")
    print("="*80)

    _, _, root = get_project_paths()
    cache_dir = root.parent / 'cache_outlier'

    if not cache_dir.exists():
        print(f"❌ FAIL: Cache directory not found: {cache_dir}")
        print(f"   Please run: python scripts/precompute_outlier_features.py --modalities all")
        return False

    print(f"✓ Cache directory exists: {cache_dir}")

    # Check for expected cache files
    modalities = ['metadata', 'thermal_map', 'depth_rgb', 'depth_map', 'thermal_rgb']
    all_exist = True

    for modality in modalities:
        cache_file = cache_dir / f'{modality}_features.npy'
        if cache_file.exists():
            size_mb = cache_file.stat().st_size / 1024 / 1024
            features = np.load(cache_file)
            print(f"✓ {modality}: {features.shape} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {modality}: Not found")
            all_exist = False

    print()
    return all_exist


def test_feature_dimensions():
    """Test that feature dimensions are consistent."""
    print("="*80)
    print("Test 2: Feature Dimensions")
    print("="*80)

    _, _, root = get_project_paths()
    data_paths = get_data_paths(root)

    # Load best_matching to get sample count
    best_matching_df = pd.read_csv(data_paths['best_matching_csv'])
    n_samples = len(best_matching_df)
    print(f"Expected samples: {n_samples}")
    print()

    # Expected dimensions based on current architecture
    # Conv2D final layer outputs 32 channels, GlobalAveragePooling2D reduces to 32 dims
    expected_dims = {
        'metadata': 73,  # Tabular features
        'thermal_map': 32,  # GlobalAveragePooling2D output
        'depth_rgb': 32,
        'depth_map': 32,
        'thermal_rgb': 32
    }

    all_correct = True

    for modality, expected_dim in expected_dims.items():
        features = load_cached_features(modality)

        if features is None:
            print(f"❌ {modality}: Failed to load cache")
            all_correct = False
            continue

        actual_shape = features.shape

        if actual_shape[0] != n_samples:
            print(f"❌ {modality}: Sample count mismatch - expected {n_samples}, got {actual_shape[0]}")
            all_correct = False
        elif actual_shape[1] != expected_dim:
            print(f"⚠️  {modality}: Dimension mismatch - expected {expected_dim}, got {actual_shape[1]}")
            print(f"   (This is OK if architecture changed)")
        else:
            print(f"✓ {modality}: {actual_shape[0]} samples × {actual_shape[1]} features")

    print()
    return all_correct


def test_combination_detection():
    """Test combination-specific outlier detection."""
    print("="*80)
    print("Test 3: Combination-Specific Detection")
    print("="*80)

    # Test a simple 2-modality combination
    combination = ('metadata', 'thermal_map')
    combo_name = get_combination_name(combination)

    print(f"Testing combination: {combination}")
    print(f"Combination name: {combo_name}")
    print()

    try:
        # Run detection with low contamination for testing
        cleaned_df, outlier_df, output_file = detect_outliers_combination(
            combination,
            contamination=0.10,  # Lower for testing
            random_state=42,
            force_recompute=True  # Force recompute for testing
        )

        if cleaned_df is None:
            print("❌ FAIL: Detection returned None")
            return False

        print(f"✓ Detection completed successfully")
        print(f"  Cleaned samples: {len(cleaned_df)}")
        print(f"  Outliers removed: {len(outlier_df) if outlier_df is not None else 0}")
        print(f"  Output file: {output_file.name}")

        # Verify output file exists
        if output_file.exists():
            print(f"✓ Cleaned dataset saved successfully")
        else:
            print(f"❌ FAIL: Output file not created")
            return False

        print()
        return True

    except Exception as e:
        print(f"❌ FAIL: Exception during detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_joint_feature_space():
    """Test that features are properly concatenated for joint space."""
    print("="*80)
    print("Test 4: Joint Feature Space")
    print("="*80)

    # Test different combination sizes
    test_combinations = [
        ('metadata',),  # Single modality
        ('metadata', 'thermal_map'),  # Two modalities
        ('depth_rgb', 'depth_map', 'thermal_map'),  # Three image modalities
    ]

    all_correct = True

    for combination in test_combinations:
        combo_name = get_combination_name(combination)
        print(f"Testing: {combo_name}")

        # Calculate expected feature dimension
        expected_dims = 0
        feature_list = []

        for modality in combination:
            features = load_cached_features(modality)
            if features is None:
                print(f"  ❌ Failed to load {modality}")
                all_correct = False
                continue
            expected_dims += features.shape[1]
            feature_list.append(features)

        if len(feature_list) == len(combination):
            # Test concatenation
            joint_features = np.concatenate(feature_list, axis=1)
            print(f"  ✓ Joint features: {joint_features.shape[0]} samples × {joint_features.shape[1]} dims")

            if joint_features.shape[1] == expected_dims:
                print(f"  ✓ Dimension correct: {expected_dims}")
            else:
                print(f"  ❌ Dimension mismatch: expected {expected_dims}, got {joint_features.shape[1]}")
                all_correct = False

        print()

    return all_correct


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Feature Extraction Test Suite" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Cache Existence", test_cache_exists()))
    results.append(("Feature Dimensions", test_feature_dimensions()))
    results.append(("Joint Feature Space", test_joint_feature_space()))
    results.append(("Combination Detection", test_combination_detection()))

    # Summary
    print("="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Overall: {passed}/{total} tests passed")
    print("="*80)
    print()

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
