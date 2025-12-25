# Multi-GPU Support

Enhanced `src/main.py` with flexible GPU configuration for CPU, single-GPU, multi-GPU, and custom GPU setups.

## Quick Start

```bash
# Default - auto-select best GPU (>=8GB, non-display)
python src/main.py --mode search

# CPU only
python src/main.py --mode search --device-mode cpu

# All available GPUs (MirroredStrategy)
python src/main.py --mode search --device-mode multi

# Specific GPUs
python src/main.py --mode search --device-mode custom --custom-gpus 0 1

# Require 12GB+ memory
python src/main.py --mode search --device-mode multi --min-gpu-memory 12.0
```

## Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--device-mode` | cpu, single, multi, custom | single | Device mode |
| `--custom-gpus` | List of ints | None | GPU IDs for custom mode |
| `--min-gpu-memory` | Float | 8.0 | Min GPU memory (GB) |
| `--include-display-gpus` | Flag | False | Include display GPUs |

## GPU Selection

**Automatic filtering**:
- Memory >= 8GB (configurable)
- Excludes display GPUs (prevents UI lag)

**Single mode**: Selects GPU with most memory, lowest ID
**Multi mode**: Uses TensorFlow MirroredStrategy, splits batch across GPUs
**Custom mode**: User-specified GPUs, applies filtering

## Testing

```bash
python scripts/test_gpu_config.py
```

## Implementation

- **Module**: `src/utils/gpu_config.py`
- **Detection**: nvidia-smi queries
- **Strategy**: TensorFlow MirroredStrategy for multi-GPU
- **Batch split**: Global batch / num_gpus per GPU

## Example Output

```
================================================================================
DEVICE CONFIGURATION (mode: multi)
================================================================================

Detected 3 GPU(s):
  GPU 0: NVIDIA RTX 5090 - 32.0GB
  GPU 1: NVIDIA TITAN Xp - 12.0GB
  GPU 2: NVIDIA GeForce GTX 1080 - 8.0GB (display)

  GPU 2 (GeForce GTX 1080, 8.0GB): Excluded (display GPU)

Selected 2 GPU(s):
  GPU 0: NVIDIA RTX 5090 (32.0GB)
  GPU 1: NVIDIA TITAN Xp (12.0GB)

Using MirroredStrategy (2 GPUs)
================================================================================

Device: GPUs [0, 1] (multi-GPU mode, MirroredStrategy)
  Replicas: 2× batch size distribution
Batch size: 16
  Per-GPU batch: 8 (16 / 2 GPUs)
```

## Troubleshooting

**"No suitable GPUs found"**:
- Lower memory: `--min-gpu-memory 4.0`
- Include display: `--include-display-gpus`
- Use CPU: `--device-mode cpu`

**"nvidia-smi not found"**: Install NVIDIA drivers or use CPU mode

**Multi-GPU not using all**: Check filtering (memory, display status) in startup output
