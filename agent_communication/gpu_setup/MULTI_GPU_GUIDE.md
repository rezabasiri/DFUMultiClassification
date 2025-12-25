# Multi-GPU Training Guide

## Quick Start

```bash
# Use all available GPUs (>=8GB, non-display)
python src/main.py --device-mode multi

# Use specific GPUs
python src/main.py --device-mode custom --custom-gpus 0 1

# Single GPU (auto-select best)
python src/main.py --device-mode single
```

## How It Works

**TensorFlow MirroredStrategy**: Data parallelism across GPUs
- Each GPU gets a replica of the model
- Global batch size is split evenly across GPUs
- Gradients are synchronized and averaged across GPUs

**Example with 2 GPUs and batch_size=128:**
- GPU 0: processes 64 samples
- GPU 1: processes 64 samples
- Gradients averaged → single model update

## Configuration

Batch size is set in `src/utils/production_config.py`:
```python
GLOBAL_BATCH_SIZE = 128  # Total across all GPUs
```

With 2 GPUs: each processes 64 samples per step
With 1 GPU: processes 128 samples per step

## GPU Selection Criteria

Automatic filtering (can be customized with flags):
1. Memory >= 8GB (default, use `--min-gpu-memory 12.0` for 12GB+)
2. Not a display GPU (use `--include-display-gpus` to override)
3. Best GPU = most memory, then lowest ID

## Strategy Scope

Multi-GPU requires wrapping operations in `strategy.scope()`:

```python
strategy = tf.distribute.get_strategy()

# Dataset creation
with strategy.scope():
    dataset = create_dataset(...)

# Model creation
with strategy.scope():
    model = create_model(...)
    model.compile(...)

# Training (automatically distributed)
model.fit(dataset, ...)
```

**Implementation locations:**
- `src/training/training_utils.py:926-927` - Dataset preprocessing
- `src/training/training_utils.py:947-959` - Master dataset creation
- `src/training/training_utils.py:1008-1021` - Dataset filtering
- `src/training/training_utils.py:1070-1080` - Model creation

## Performance Tips

1. **Batch Size**: Increase proportionally with GPU count
   - 1 GPU: 64-128
   - 2 GPUs: 128-256
   - 4 GPUs: 256-512

2. **Monitor GPU Utilization**: Use `nvidia-smi dmon` to check balance

3. **Memory Usage**: Multi-GPU needs model replicas, plan accordingly

## Troubleshooting

**CUDA errors with RTX 5090**: See `INSTALL_RTX5090_FIX.md`

**Uneven GPU usage**: Check if one GPU is display-active (use `--include-display-gpus` carefully)

**Out of memory**: Reduce `GLOBAL_BATCH_SIZE` in production_config.py

**Strategy not working**: Ensure `setup_device_strategy()` runs before TF operations
