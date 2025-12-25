"""
GPU Configuration and Device Management

Provides utilities for:
- Detecting available GPUs
- Filtering GPUs by memory size and display status
- Setting up TensorFlow distribution strategies
- Managing CUDA_VISIBLE_DEVICES environment variable
"""

import os
import subprocess
import re
import tensorflow as tf
from typing import List, Dict, Tuple, Optional


def get_gpu_info() -> List[Dict[str, any]]:
    """
    Query GPU information using nvidia-smi.

    Returns:
        List of dicts with GPU info: [{'id': 0, 'name': 'RTX 5090', 'memory_mb': 32768, 'is_display': False}, ...]
        Returns empty list if no GPUs found or nvidia-smi fails.
    """
    try:
        # Query nvidia-smi for GPU information
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,display_active',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print(f"⚠️  nvidia-smi failed: {result.stderr}")
            return []

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue

            try:
                gpu_id = int(parts[0])
                name = parts[1]
                memory_mb = int(parts[2])
                is_display = parts[3].lower() == 'enabled'

                gpus.append({
                    'id': gpu_id,
                    'name': name,
                    'memory_mb': memory_mb,
                    'memory_gb': memory_mb / 1024,
                    'is_display': is_display
                })
            except (ValueError, IndexError) as e:
                print(f"⚠️  Failed to parse GPU info line: {line} ({e})")
                continue

        return gpus

    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - no NVIDIA GPUs available")
        return []
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi timeout - GPU detection failed")
        return []
    except Exception as e:
        print(f"⚠️  GPU detection error: {e}")
        return []


def filter_gpus(
    gpus: List[Dict[str, any]],
    min_memory_gb: float = 8.0,
    exclude_display: bool = True
) -> List[Dict[str, any]]:
    """
    Filter GPUs by memory size and display status.

    Args:
        gpus: List of GPU info dicts from get_gpu_info()
        min_memory_gb: Minimum memory in GB (default: 8.0)
        exclude_display: Whether to exclude GPUs with active displays (default: True)

    Returns:
        Filtered list of GPU info dicts
    """
    filtered = []

    for gpu in gpus:
        # Check memory requirement
        if gpu['memory_gb'] < min_memory_gb:
            print(f"  GPU {gpu['id']} ({gpu['name']}, {gpu['memory_gb']:.1f}GB): Excluded (< {min_memory_gb}GB)")
            continue

        # Check display status
        if exclude_display and gpu['is_display']:
            print(f"  GPU {gpu['id']} ({gpu['name']}, {gpu['memory_gb']:.1f}GB): Excluded (display GPU)")
            continue

        filtered.append(gpu)

    return filtered


def select_best_gpu(gpus: List[Dict[str, any]]) -> Optional[int]:
    """
    Select the best single GPU from filtered list.

    Selection criteria (in order):
    1. Most memory
    2. Lowest GPU ID (for consistency)

    Args:
        gpus: List of filtered GPU info dicts

    Returns:
        GPU ID (int) or None if no GPUs available
    """
    if not gpus:
        return None

    # Sort by memory (descending) then by ID (ascending)
    best_gpu = sorted(gpus, key=lambda g: (-g['memory_gb'], g['id']))[0]
    return best_gpu['id']


def setup_device_strategy(
    mode: str = 'single',
    custom_gpus: Optional[List[int]] = None,
    min_memory_gb: float = 8.0,
    exclude_display: bool = True,
    verbose: bool = True
) -> Tuple[tf.distribute.Strategy, List[int]]:
    """
    Set up TensorFlow distribution strategy based on device mode.

    Args:
        mode: Device mode - 'cpu', 'single', 'multi', or 'custom'
        custom_gpus: List of GPU IDs for 'custom' mode (e.g., [0, 1])
        min_memory_gb: Minimum GPU memory in GB (default: 8.0)
        exclude_display: Exclude GPUs with active displays (default: True)
        verbose: Print GPU selection info (default: True)

    Returns:
        Tuple of (tf.distribute.Strategy, list of selected GPU IDs)

    Raises:
        ValueError: If invalid mode or no suitable GPUs found
    """
    # Set environment variables for RTX 5090 compatibility (compute capability 12.0)
    # These need to be set before TensorFlow initializes CUDA
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Reduce TensorFlow logging
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN optimizations

    if verbose:
        print("\n" + "="*80)
        print(f"DEVICE CONFIGURATION (mode: {mode})")
        print("="*80)

    # CPU mode
    if mode == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if verbose:
            print("Using CPU only (all GPUs disabled)")
        return tf.distribute.get_strategy(), []

    # GPU modes - detect available GPUs
    all_gpus = get_gpu_info()

    if not all_gpus:
        print("❌ No GPUs detected - falling back to CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return tf.distribute.get_strategy(), []

    if verbose:
        print(f"\nDetected {len(all_gpus)} GPU(s):")
        for gpu in all_gpus:
            display_str = " (display)" if gpu['is_display'] else ""
            print(f"  GPU {gpu['id']}: {gpu['name']} - {gpu['memory_gb']:.1f}GB{display_str}")

    # Filter GPUs by criteria
    filtered_gpus = filter_gpus(all_gpus, min_memory_gb, exclude_display)

    if not filtered_gpus:
        raise ValueError(
            f"No suitable GPUs found (need >={min_memory_gb}GB, "
            f"exclude_display={exclude_display})"
        )

    # Select GPUs based on mode
    selected_gpu_ids = []

    if mode == 'single':
        best_gpu_id = select_best_gpu(filtered_gpus)
        if best_gpu_id is None:
            raise ValueError("No suitable GPU found for single GPU mode")
        selected_gpu_ids = [best_gpu_id]

        best_gpu = next(g for g in filtered_gpus if g['id'] == best_gpu_id)
        if verbose:
            print(f"\nSelected GPU {best_gpu_id}: {best_gpu['name']} ({best_gpu['memory_gb']:.1f}GB)")

    elif mode == 'multi':
        selected_gpu_ids = [gpu['id'] for gpu in filtered_gpus]

        if len(selected_gpu_ids) < 2:
            print(f"⚠️  Only 1 suitable GPU found - using single GPU mode instead")
            mode = 'single'

        if verbose:
            print(f"\nSelected {len(selected_gpu_ids)} GPU(s):")
            for gpu_id in selected_gpu_ids:
                gpu = next(g for g in filtered_gpus if g['id'] == gpu_id)
                print(f"  GPU {gpu_id}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")

    elif mode == 'custom':
        if custom_gpus is None or len(custom_gpus) == 0:
            raise ValueError("Custom mode requires --custom-gpus argument")

        # Validate custom GPU IDs
        available_ids = {gpu['id'] for gpu in filtered_gpus}
        invalid_ids = [gid for gid in custom_gpus if gid not in available_ids]

        if invalid_ids:
            raise ValueError(
                f"Invalid GPU IDs: {invalid_ids}. "
                f"Available GPUs (after filtering): {sorted(available_ids)}"
            )

        selected_gpu_ids = custom_gpus

        if verbose:
            print(f"\nCustom GPU selection ({len(selected_gpu_ids)} GPU(s)):")
            for gpu_id in selected_gpu_ids:
                gpu = next(g for g in all_gpus if g['id'] == gpu_id)
                print(f"  GPU {gpu_id}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")

    else:
        raise ValueError(f"Invalid device mode: {mode}")

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpu_ids))

    # Create TensorFlow distribution strategy
    if len(selected_gpu_ids) == 1:
        # Single GPU - use default strategy (no distribution)
        strategy = tf.distribute.get_strategy()
        if verbose:
            print(f"\nUsing default strategy (single GPU)")
    else:
        # Multi-GPU - use MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()
        if verbose:
            print(f"\nUsing MirroredStrategy ({len(selected_gpu_ids)} GPUs)")
            print(f"Effective batch size: {tf.distribute.get_strategy().num_replicas_in_sync}× global batch size")

    if verbose:
        print("="*80 + "\n")

    return strategy, selected_gpu_ids


def get_effective_batch_size(global_batch_size: int, strategy: tf.distribute.Strategy) -> int:
    """
    Calculate effective batch size per GPU replica.

    For MirroredStrategy, the global batch size is split across replicas.

    Args:
        global_batch_size: Total batch size across all GPUs
        strategy: TensorFlow distribution strategy

    Returns:
        Batch size per GPU replica
    """
    num_replicas = strategy.num_replicas_in_sync
    return global_batch_size // num_replicas


def print_gpu_memory_usage():
    """Print current GPU memory usage for all visible GPUs."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("\nGPU Memory Usage:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_id, used, total = parts[0], parts[1], parts[2]
                        print(f"  GPU {gpu_id}: {used}MB / {total}MB ({float(used)/float(total)*100:.1f}%)")
    except Exception:
        pass  # Silently fail if nvidia-smi unavailable
