# GPU OOM Issue - Agent Handoff

## Quick Summary
Your fix (commit 16fe49c) **partially worked** but rank 1 still OOMs when restoring `text_encoder_2`.

## Test Results

✅ **Working**:
- Expandable segments reduced fragmentation (8.17 MiB vs 421.81 MiB previously)
- Skipping EMA offload/restore allowed rank 0 to succeed
- FID computation works perfectly on both GPUs (Val FID: 422.57)

❌ **Still Failing**:
- Rank 1 OOMs when restoring `text_encoder_2` (needs ~20 MiB, only 16.69 MiB free)
- Error at: `scripts/train_lora_model.py:1002` - `text_encoder_2.to(original_device)`

## Error Details

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 1 has a total capacity of 23.52 GiB of which 16.69 MiB is free.
Of the allocated memory 22.88 GiB is allocated by PyTorch, and 8.17 MiB is reserved but unallocated.
```

**Critical observation**: GPU 1 still has 22.88 GiB allocated even after:
- Moving quality metrics to CPU (lines 952-955)
- Calling `torch.cuda.empty_cache()` twice
- Calling `gc.collect()`
- Passing barrier synchronization

## Root Cause Hypothesis

Looking at the code flow, I suspect **generated_images are not freed from GPU**:

1. **Line 932**: `generated_images = generated_images.to(accelerator.device)` (main process only)
2. **Line 926-929**: Non-main process creates `generated_images` directly on GPU
3. **Line 935**: Broadcast happens (both processes have generated_images on GPU)
4. **Line 940-944**: Metrics computed (generated_images still on GPU)
5. **Line 949**: `del generated_images` (on main process only?)

**Potential issue**: Non-main process may not be deleting `generated_images` from GPU.

## Proposed Solutions (Pick One)

### Option 1: Explicitly free generated_images on ALL processes (RECOMMENDED)
```python
# After line 945, add:
print(f"[SYNC DEBUG] Process {accelerator.process_index} cleaning up memory...")
del generated_images  # Free generated images on ALL processes
if accelerator.num_processes > 1:
    del reference_images  # Free reference images too
torch.cuda.empty_cache()
```

### Option 2: Reduce quality metrics memory footprint
Currently loading 4 metrics models on each GPU:
- FID: InceptionV3 (~500MB)
- IS: InceptionV3 (~500MB)
- LPIPS: AlexNet (~500MB)
- SSIM: minimal

Could disable IS since it's less critical than FID.

### Option 3: Lower resolution for initial test
Change `test_sdxl.yaml:4` from `resolution: 32` to `resolution: 16`.
This will reduce memory pressure for broadcast tensors.

### Option 4: Only rank 0 computes metrics (risky, may bring back FID deadlock)
Have only rank 0 compute metrics, with explicit distributed sync handling.

## Environment

**Hardware**: 2x RTX 4090 (24GB each)
**Framework**: HuggingFace Accelerate (multi-GPU distributed training)
**Model**: SDXL 1.0 with LoRA
**Config**: `configs/test_sdxl.yaml`

**Memory breakdown per GPU**:
- UNet LoRA: ~2.5GB
- VAE: ~2GB
- Text Encoder 1: ~1.5GB
- Text Encoder 2: ~3.5GB
- EMA model: ~2.5GB (NOT offloaded per your fix)
- Quality metrics: ~2GB (FID + IS + LPIPS + SSIM)
- Generated images broadcast: 5×3×32×32 = ~20KB (negligible)
- **Total: ~14.5GB + gradients/activations = 22-23GB**

## How to Test

```bash
cd /workspace/DFUMultiClassification/agent_communication/generative_augmentation
source /opt/miniforge3/bin/activate multimodal
accelerate launch --config_file accelerate_config.yaml \
    scripts/train_lora_model.py --config configs/test_sdxl.yaml \
    2>&1 | tee sdxl_test.log
```

**Expected**: Training completes 2 epochs with FID computation at epoch 1.

**Current**: OOMs at line 1002 when rank 1 tries to restore `text_encoder_2`.

## Files

- **Training script**: `scripts/train_lora_model.py` (lines 862-1014 are critical)
- **Config**: `configs/test_sdxl.yaml`
- **Test log**: `sdxl_test.log` (last test results)
- **Your commit**: 16fe49c ("fix: Resolve memory fragmentation OOM using PyTorch best practices")

## Debug Priority

1. **Check if `del generated_images` is being called on ALL processes** (currently only line 949 in main process section)
2. **Verify quality metrics are actually moving to CPU** (lines 952-955) - maybe add `.cpu()` explicitly?
3. **Check if reference_images are still on GPU** - they're loaded once but never explicitly freed

## My Recommendation

Start with **Option 1** (explicitly free generated_images on ALL processes). If that doesn't work, then do your thing.
