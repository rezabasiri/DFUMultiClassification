# FID Computation Deadlock in Multi-GPU Training

## Problem Summary
Training hangs during FID (Fréchet Inception Distance) computation when using multi-GPU distributed training with HuggingFace Accelerate.

## Root Cause
The `torchmetrics.FrechetInceptionDistance.compute()` method **always synchronizes across all distributed processes**, even when only the main process (rank 0) is calling it. This creates a deadlock:

- **Rank 0 (main process)**: Calls `fid_metric.compute()` and waits for rank 1 to participate in synchronization
- **Rank 1 (non-main process)**: Already passed `accelerator.wait_for_everyone()` and is NOT calling `fid_metric.compute()`, so it's not participating in the synchronization

## Evidence from Debug Logs

```
# Rank 1 (non-main process) - passes barrier successfully:
[SYNC DEBUG] Non-main process: Val loss: 0.3001
[SYNC DEBUG] Non-main process about to wait at barrier
[SYNC DEBUG] Process 1 calling wait_for_everyone()...
[SYNC DEBUG] Process 1 passed wait_for_everyone()

# Rank 0 (main process) - hangs during FID computation:
[METRICS DEBUG] Main process about to call compute_all_metrics
[FID DEBUG] Calling fid_metric.compute()...
# <-- HANGS HERE, waiting for rank 1 to participate in dist sync
```

## Code Context

### FID Initialization (quality_metrics.py:72-76)
```python
self.fid_metric = FrechetInceptionDistance(
    feature=fid_feature,
    normalize=True,
    dist_sync_on_step=False  # This only affects .update(), NOT .compute()
).to(self.device)
```

### FID Computation (quality_metrics.py:152-153)
```python
print(f"[FID DEBUG] Calling fid_metric.compute()...")
fid_score = self.fid_metric.compute().item()  # <-- HANGS HERE
print(f"[FID DEBUG] FID computed successfully: {fid_score}")  # Never reached
```

### Training Loop Structure (train_lora_model.py:832-915)
```python
if reference_images is not None and (epoch + 1) % config['metrics']['compute_every_n_epochs'] == 0:
    if accelerator.is_main_process:  # Only rank 0 enters this block
        print("Computing quality metrics...")
        # ... CPU offloading ...
        val_metrics = quality_metrics.compute_all_metrics(...)  # Calls fid_metric.compute()
    else:
        print(f"[SYNC DEBUG] Non-main process: Val loss: {val_loss:.4f}")

    # Both processes reach this barrier
    accelerator.wait_for_everyone()  # Rank 1 passes, rank 0 never reaches here
```

## Attempted Solutions (Failed)

1. ✗ **`dist_sync_on_step=False`**: Only affects `.update()`, not `.compute()`
2. ✗ **`accelerator.wait_for_everyone()` after metrics**: Rank 0 never reaches the barrier because it's stuck in `.compute()`

## Potential Solutions

### Option 1: Make ALL processes compute FID (not just main process)
```python
# Both processes compute FID, but only main process uses the result
if reference_images is not None:
    # Generate samples on main process only
    if accelerator.is_main_process:
        generated_images = generate_validation_samples(...)

    # Broadcast generated_images to all processes
    if accelerator.num_processes > 1:
        generated_images = accelerator.gather(generated_images)  # Or broadcast

    # ALL processes compute FID (satisfies distributed sync requirement)
    val_metrics = quality_metrics.compute_all_metrics(...)

    # Only main process prints/saves
    if accelerator.is_main_process:
        print(f"Val FID: {val_metrics['fid']}")
```

### Option 2: Temporarily destroy/recreate process group
```python
if accelerator.is_main_process:
    # Temporarily disable distributed mode
    torch.distributed.destroy_process_group()
    val_metrics = quality_metrics.compute_all_metrics(...)
    # Recreate process group
    torch.distributed.init_process_group(...)
```
**Risk**: May break Accelerate's internal state

### Option 3: Use non-distributed FID computation
```python
# Create a separate QualityMetrics instance without distributed training
if accelerator.is_main_process:
    # Force metrics to not use distributed mode
    with torch.no_grad():
        # Manually compute FID without torchmetrics
        # Or monkey-patch torch.distributed.is_initialized() to return False
```

### Option 4: Compute FID in a separate subprocess
```python
if accelerator.is_main_process:
    # Launch FID computation in separate Python process (outside distributed group)
    import subprocess
    result = subprocess.run([
        "python", "compute_fid_standalone.py",
        "--gen_images", gen_path,
        "--real_images", real_path
    ])
```

## Question for Cloud Agent

**Which solution would you recommend for computing FID metrics in multi-GPU training with HuggingFace Accelerate?**

The key constraint is:
- FID computation is expensive (requires Inception model forward passes)
- We only want to compute it once per epoch on the main process
- `torchmetrics.FrechetInceptionDistance.compute()` requires ALL distributed processes to participate

**Additional context:**
- Framework: HuggingFace Accelerate for distributed training
- Model: Stable Diffusion XL 1.0 with LoRA
- GPUs: 2x RTX 4090 (24GB each)
- Current checkpoint strategy: Save best model based on val_fid (lower is better)

**Files involved:**
- `scripts/train_lora_model.py` (lines 832-915): Training loop with metrics
- `scripts/utils/quality_metrics.py` (lines 72-161): FID computation
- `configs/test_sdxl.yaml`: Configuration (32x32 resolution for fast testing)
