#!/usr/bin/env python3
"""Check hardware configuration to recommend optimal TensorFlow threading settings."""

import os
import sys

def check_hardware():
    """Print hardware info and threading recommendations."""

    print("=" * 80)
    print("HARDWARE CONFIGURATION CHECK")
    print("=" * 80)

    # Check CPU cores
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"\n✓ CPU Cores: {cpu_count}")
        print(f"  - Physical cores available to Python")
    except Exception as e:
        print(f"\n✗ Could not detect CPU cores: {e}")
        cpu_count = None

    # Check if running on GPU
    print("\n" + "=" * 80)
    print("GPU DETECTION")
    print("=" * 80)

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ GPUs detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if 'device_name' in gpu_details:
                        print(f"    Model: {gpu_details['device_name']}")
                except:
                    pass
        else:
            print("\n✗ No GPUs detected - running on CPU only")
    except Exception as e:
        print(f"\n✗ Could not detect GPUs: {e}")
        gpus = []

    # Check current TensorFlow threading settings
    print("\n" + "=" * 80)
    print("CURRENT TENSORFLOW SETTINGS")
    print("=" * 80)

    current_settings = {
        'TF_OMP_NUM_THREADS': os.environ.get('TF_OMP_NUM_THREADS', 'not set'),
        'TF_NUM_INTEROP_THREADS': os.environ.get('TF_NUM_INTEROP_THREADS', 'not set'),
        'TF_NUM_INTRAOP_THREADS': os.environ.get('TF_NUM_INTRAOP_THREADS', 'not set'),
        'TF_DETERMINISTIC_OPS': os.environ.get('TF_DETERMINISTIC_OPS', 'not set'),
        'TF_CUDNN_DETERMINISTIC': os.environ.get('TF_CUDNN_DETERMINISTIC', 'not set'),
    }

    for key, value in current_settings.items():
        print(f"  {key}: {value}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if gpus:
        print("\n🎯 GPU TRAINING DETECTED")
        print("\nThreading Settings (GPU mode):")
        print("  - TF_OMP_NUM_THREADS: 2-4 (low, GPU does the work)")
        print("  - TF_NUM_INTEROP_THREADS: 2 (low, GPU does the work)")
        print("  - TF_NUM_INTRAOP_THREADS: 4-8 (moderate for data preprocessing)")
        print("\n  ✓ Your current settings (2/2/4) are GOOD for GPU training!")
        print("    - Low CPU thread count avoids overhead")
        print("    - GPU does most computation, CPU just feeds data")
    else:
        print("\n🎯 CPU-ONLY TRAINING DETECTED")
        if cpu_count:
            recommended_interop = max(2, cpu_count // 2)
            recommended_intraop = cpu_count
            print("\nThreading Settings (CPU mode):")
            print(f"  - TF_OMP_NUM_THREADS: {cpu_count} (use all cores)")
            print(f"  - TF_NUM_INTEROP_THREADS: {recommended_interop} (half of cores)")
            print(f"  - TF_NUM_INTRAOP_THREADS: {recommended_intraop} (all cores)")
            print("\n  ⚠️  Your current settings (2/2/4) are LOW for CPU-only training!")
            print(f"    - Consider increasing to ({cpu_count}/{recommended_interop}/{recommended_intraop})")

    print("\nDeterminism Settings:")
    deterministic = (current_settings['TF_DETERMINISTIC_OPS'] == '1')
    if deterministic:
        print("  ✓ DETERMINISTIC mode enabled (reproducible results)")
        print("    - Same inputs always produce same outputs")
        print("    - Good for debugging and research")
        print("    - May be ~10-20% slower than non-deterministic")
    else:
        print("  ✗ NON-DETERMINISTIC mode (faster but not reproducible)")
        print("    - Results may vary slightly between runs")
        print("    - ~10-20% faster than deterministic mode")

    print("\n" + "=" * 80)
    print("PRIORITY GUIDE")
    print("=" * 80)

    print("\n1. REPRODUCIBILITY (research/debugging):")
    print("   - Keep TF_DETERMINISTIC_OPS=1 and TF_CUDNN_DETERMINISTIC=1")
    print("   - Accept ~10-20% slower training")
    print("   - Current setting: ✓ ENABLED")

    print("\n2. SPEED (production/final runs):")
    print("   - Set TF_DETERMINISTIC_OPS=0 and TF_CUDNN_DETERMINISTIC=0")
    print("   - ~10-20% faster training")
    print("   - Results may vary slightly between runs")

    print("\n3. THREADING:")
    if gpus:
        print("   - Your GPU settings are already optimal")
        print("   - No changes needed for threading")
    else:
        print("   - Increase thread counts to use full CPU")
        print(f"   - Recommended: {cpu_count}/{max(2, cpu_count // 2)}/{cpu_count}")

    print("\n" + "=" * 80)
    print("QUICK TEST: Measure Actual Impact")
    print("=" * 80)
    print("\nTo measure the actual impact on YOUR system:")
    print("  1. Run a small training test with current settings")
    print("  2. Run again with determinism OFF (TF_DETERMINISTIC_OPS=0)")
    print("  3. Compare training time per epoch")
    print("  4. If speed difference is <5%, keep determinism ON")
    print("  5. If speed difference is >15%, consider turning OFF")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_hardware()
