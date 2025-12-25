#!/usr/bin/env python3
"""
Test script for GPU configuration functionality.

Tests all device modes and validates GPU selection logic.
"""

import sys
import os

# Add project root to path
# Script is in agent_communication/gpu_setup/, need to go up 2 levels
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.utils.gpu_config import (
    get_gpu_info,
    filter_gpus,
    select_best_gpu,
    setup_device_strategy,
    print_gpu_memory_usage
)


def test_gpu_detection():
    """Test GPU detection and filtering."""
    print("\n" + "="*80)
    print("TEST 1: GPU DETECTION")
    print("="*80)

    gpus = get_gpu_info()

    if not gpus:
        print("❌ No GPUs detected")
        return False

    print(f"\n✅ Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        display_str = " [DISPLAY]" if gpu['is_display'] else ""
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_gb']:.1f} GB ({gpu['memory_mb']} MB)")
        print(f"    Display: {'Yes' if gpu['is_display'] else 'No'}{display_str}")

    return True


def test_gpu_filtering():
    """Test GPU filtering by memory and display status."""
    print("\n" + "="*80)
    print("TEST 2: GPU FILTERING")
    print("="*80)

    gpus = get_gpu_info()
    if not gpus:
        print("⚠️  Skipping (no GPUs)")
        return True

    # Test 1: Filter with default settings (8GB, exclude display)
    print("\nFilter 1: Default (>=8GB, exclude display)")
    filtered = filter_gpus(gpus, min_memory_gb=8.0, exclude_display=True)
    print(f"Result: {len(filtered)}/{len(gpus)} GPUs pass filter")

    # Test 2: Filter requiring 12GB
    print("\nFilter 2: High memory (>=12GB, exclude display)")
    filtered_12gb = filter_gpus(gpus, min_memory_gb=12.0, exclude_display=True)
    print(f"Result: {len(filtered_12gb)}/{len(gpus)} GPUs pass filter")

    # Test 3: Include display GPUs
    print("\nFilter 3: Include display (>=8GB, include display)")
    filtered_display = filter_gpus(gpus, min_memory_gb=8.0, exclude_display=False)
    print(f"Result: {len(filtered_display)}/{len(gpus)} GPUs pass filter")

    return True


def test_best_gpu_selection():
    """Test automatic best GPU selection."""
    print("\n" + "="*80)
    print("TEST 3: BEST GPU SELECTION")
    print("="*80)

    gpus = get_gpu_info()
    if not gpus:
        print("⚠️  Skipping (no GPUs)")
        return True

    filtered = filter_gpus(gpus, min_memory_gb=8.0, exclude_display=True)
    if not filtered:
        print("❌ No GPUs pass filter")
        return False

    best_id = select_best_gpu(filtered)
    best_gpu = next(g for g in filtered if g['id'] == best_id)

    print(f"\n✅ Best GPU selected:")
    print(f"  GPU {best_id}: {best_gpu['name']} ({best_gpu['memory_gb']:.1f} GB)")
    print(f"\n  Selection criteria:")
    print(f"  1. Most memory: {best_gpu['memory_gb']:.1f} GB")
    print(f"  2. Lowest ID (for consistency): {best_id}")

    return True


def test_device_strategies():
    """Test all device strategy modes."""
    print("\n" + "="*80)
    print("TEST 4: DEVICE STRATEGIES")
    print("="*80)

    test_modes = [
        ('cpu', {}, "CPU mode"),
        ('single', {}, "Single GPU (auto-select)"),
        ('multi', {}, "Multi-GPU (all available)"),
    ]

    gpus = get_gpu_info()
    if gpus and len(filter_gpus(gpus, min_memory_gb=8.0, exclude_display=True)) >= 2:
        # Only test custom mode if we have multiple GPUs
        available_ids = [g['id'] for g in filter_gpus(gpus, min_memory_gb=8.0, exclude_display=True)]
        custom_gpus = available_ids[:2]  # Select first 2 GPUs
        test_modes.append(('custom', {'custom_gpus': custom_gpus}, f"Custom GPUs {custom_gpus}"))

    for mode, kwargs, description in test_modes:
        print(f"\n{'─'*80}")
        print(f"Testing: {description}")
        print(f"{'─'*80}")

        try:
            strategy, selected_gpus = setup_device_strategy(
                mode=mode,
                min_memory_gb=8.0,
                exclude_display=True,
                verbose=False,
                **kwargs
            )

            print(f"✅ Success!")
            print(f"  Strategy type: {type(strategy).__name__}")
            print(f"  Selected GPUs: {selected_gpus if selected_gpus else 'CPU'}")
            print(f"  Num replicas: {strategy.num_replicas_in_sync}")

            # Cleanup
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)

        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    return True


def test_gpu_memory_usage():
    """Test GPU memory usage monitoring."""
    print("\n" + "="*80)
    print("TEST 5: GPU MEMORY USAGE")
    print("="*80)

    gpus = get_gpu_info()
    if not gpus:
        print("⚠️  Skipping (no GPUs)")
        return True

    print_gpu_memory_usage()
    return True


def main():
    """Run all GPU configuration tests."""
    print("\n" + "="*80)
    print("GPU CONFIGURATION TEST SUITE")
    print("="*80)

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("GPU Filtering", test_gpu_filtering),
        ("Best GPU Selection", test_best_gpu_selection),
        ("Device Strategies", test_device_strategies),
        ("GPU Memory Usage", test_gpu_memory_usage),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
