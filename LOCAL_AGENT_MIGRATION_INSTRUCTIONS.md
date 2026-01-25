# SDXL Generative Augmentation Migration - Local Agent Instructions

## Overview

The SDXL-based generative augmentation system has been migrated from the experimental directory (`agent_communication/generative_augmentation`) to the main project structure. This guide provides step-by-step instructions for the local agent to complete the migration.

## What Has Been Done (Remote Agent)

✅ **Code Migration Completed:**
1. Utility files copied to `src/utils/`:
   - `checkpoint_utils.py` - Checkpoint loading/saving utilities
   - `generative_data_loader.py` - Data loading with bbox cropping
   - `quality_metrics.py` - FID, SSIM, LPIPS, Inception Score metrics
   - `generative_training_utils.py` - EMA, perceptual loss, early stopping
   - `full_sdxl_config.yaml` - Training configuration

2. New integration code created:
   - `src/data/generative_augmentation_sdxl.py` - SDXL-based augmentation (replaces v2)
   - Supports loading trained SDXL checkpoints
   - Generates images directly at target size (no resizing needed)
   - Maintains compatibility with existing `main.py` interface

## What Needs to Be Done (Local Agent)

### Task 1: Copy Checkpoint Files

**Source Location:**
```
agent_communication/generative_augmentation/checkpoints/full_sdxl/
├── checkpoint_epoch_0030.pt  (~10GB)
├── checkpoint_epoch_0035.pt  (~10GB)
└── checkpoint_history.json   (~1.3KB)
```

**Destination:**
```
src/models/sdxl_checkpoints/
├── checkpoint_epoch_0030.pt
├── checkpoint_epoch_0035.pt
└── checkpoint_history.json
```

**Commands:**
```bash
# Create checkpoint directory
mkdir -p /workspace/DFUMultiClassification/src/models/sdxl_checkpoints

# Copy checkpoint files (large files, will take time)
cp /workspace/DFUMultiClassification/agent_communication/generative_augmentation/checkpoints/full_sdxl/checkpoint_epoch_0030.pt \
   /workspace/DFUMultiClassification/src/models/sdxl_checkpoints/

cp /workspace/DFUMultiClassification/agent_communication/generative_augmentation/checkpoints/full_sdxl/checkpoint_epoch_0035.pt \
   /workspace/DFUMultiClassification/src/models/sdxl_checkpoints/

cp /workspace/DFUMultiClassification/agent_communication/generative_augmentation/checkpoints/full_sdxl/checkpoint_history.json \
   /workspace/DFUMultiClassification/src/models/sdxl_checkpoints/

# Create symlinks for easier access
cd /workspace/DFUMultiClassification/src/models/sdxl_checkpoints
ln -sf checkpoint_epoch_0035.pt checkpoint_latest.pt
ln -sf checkpoint_epoch_0030.pt checkpoint_best.pt  # Based on epoch 30 having better metrics

# Verify files
ls -lh /workspace/DFUMultiClassification/src/models/sdxl_checkpoints/
```

**Expected Output:**
```
-rw-r--r-- 1 root root  ~10G checkpoint_epoch_0030.pt
-rw-r--r-- 1 root root  ~10G checkpoint_epoch_0035.pt
-rw-r--r-- 1 root root  1.3K checkpoint_history.json
lrwxrwxrwx 1 root root    24 checkpoint_latest.pt -> checkpoint_epoch_0035.pt
lrwxrwxrwx 1 root root    24 checkpoint_best.pt -> checkpoint_epoch_0030.pt
```

### Task 2: Update main.py to Use New Generative Augmentation

**Location:** `src/main.py`

Find the import for generative augmentation (around line 30-50):
```python
from src.data.generative_augmentation_v2 import (
    AugmentationConfig,
    GenerativeAugmentationManager,
    create_enhanced_augmentation_fn,
    GenerativeAugmentationCallback
)
```

**Replace with:**
```python
from src.data.generative_augmentation_sdxl import (
    AugmentationConfig,
    GenerativeAugmentationManager,
    create_enhanced_augmentation_fn,
    GenerativeAugmentationCallback
)
```

Find where GenerativeAugmentationManager is instantiated (search for "GenerativeAugmentationManager"):
```python
# OLD (SD 1.5 LoRA):
gen_manager = GenerativeAugmentationManager(
    base_dir='path/to/old/models',
    config=aug_config
)
```

**Replace with:**
```python
# NEW (SDXL full fine-tuning):
gen_manager = GenerativeAugmentationManager(
    checkpoint_dir='src/models/sdxl_checkpoints',
    config=aug_config
)
```

### Task 3: Test the Integration

**Quick Test (5 minutes):**
```bash
cd /workspace/DFUMultiClassification
source /opt/miniforge3/bin/activate multimodal

# Run quick test with fresh data
python agent_communication/generative_augmentation/test_generative_aug.py --quick --fresh
```

**What to Expect:**
- Test should load SDXL checkpoint from new location
- Generate sample images for each phase (I, P, R)
- Compare baseline vs generative augmentation performance
- Should complete without errors

**Success Criteria:**
```
✓ Checkpoint loaded from: src/models/sdxl_checkpoints/checkpoint_best.pt
✓ Config loaded from: src/utils/full_sdxl_config.yaml
✓ SDXL pipeline loaded successfully
✓ Generated images for phases: I, P, R
✓ Test completed successfully
```

### Task 4: Verify Production Config

**Check:** `src/utils/production_config.py`

Ensure generative augmentation settings are correct:
```python
# Generative augmentation settings
USE_GENERATIVE_AUGMENTATION = True  # Enable/disable globally
GENERATIVE_AUG_PROB = 0.3  # 30% chance of using generative aug
GENERATIVE_AUG_MIX_RATIO = (0.2, 0.4)  # Replace 20-40% of batch
GENERATIVE_AUG_INFERENCE_STEPS = 50  # SDXL diffusion steps
GENERATIVE_AUG_BATCH_LIMIT = 8  # Max images per generation batch
GENERATIVE_AUG_MAX_MODELS = 1  # Only one model (SDXL) needed
GENERATIVE_AUG_PHASES = ['I', 'P', 'R']  # All wound phases
```

## Technical Details

### SDXL vs Old SD 1.5 System

| Aspect | Old (SD 1.5 LoRA) | New (SDXL Full) |
|--------|-------------------|-----------------|
| Model | Stable Diffusion 1.5 | SDXL 1.0 Base |
| Parameters | 11.5% (LoRA) | 100% (full fine-tune) |
| Resolution | 256×256 → resize | Direct at IMAGE_SIZE |
| Training Data | Generic prompts | Phase-specific, bbox-cropped |
| Checkpoint Size | ~500MB | ~10GB |
| Quality | FID ~300-400 | FID ~190-255 |

### Key Changes

1. **Direct Size Generation:**
   - Old: Generate at 256×256, resize to IMAGE_SIZE
   - New: Generate directly at IMAGE_SIZE (saves computation)

2. **Checkpoint Loading:**
   - Old: Load pretrained SD 1.5 + LoRA weights
   - New: Load SDXL base + trained UNet weights from checkpoint

3. **Phase-Specific Training:**
   - Old: Generic wound descriptions
   - New: Minimal prompts ("PHASE_I, diabetic foot ulcer, inflammatory phase wound")
   - Forces model to learn from training data, not pretrained knowledge

4. **Bbox-Aware:**
   - Old: Full images with background/feet
   - New: 100% bbox-cropped wound-only images

## Troubleshooting

### Issue: Checkpoint not found
```
Error: No checkpoint found in src/models/sdxl_checkpoints
```
**Solution:** Verify checkpoint files were copied correctly (Task 1)

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `GENERATIVE_AUG_BATCH_LIMIT` in `production_config.py`:
```python
GENERATIVE_AUG_BATCH_LIMIT = 4  # Reduce from 8 to 4
```

### Issue: Config file not found
```
FileNotFoundError: Config file not found near src/models/sdxl_checkpoints
```
**Solution:** Verify `full_sdxl_config.yaml` is in `src/utils/`:
```bash
ls -l /workspace/DFUMultiClassification/src/utils/full_sdxl_config.yaml
```

### Issue: Slow generation
```
Generation taking >30 seconds per batch
```
**Solution:** Reduce inference steps in `production_config.py`:
```python
GENERATIVE_AUG_INFERENCE_STEPS = 30  # Reduce from 50 to 30
```

## Performance Expectations

**Training Results (Epoch 30-38):**
- FID: 189-255 (target <50, still improving)
- SSIM: 0.80-0.90 (good, target >0.7) ✓
- LPIPS: 0.40-0.57 (moderate, target <0.3)
- Inception Score: 2.4-3.0 (good, target >2.0) ✓

**Generation Speed:**
- ~2-3 seconds per image at 256×256 (RTX 3090/A6000)
- ~5-7 seconds per image at 512×512
- Batch generation is more efficient

**Memory Usage:**
- SDXL pipeline: ~7-8GB VRAM
- Batch of 4 images: ~10-12GB VRAM
- Recommended: 24GB VRAM GPU

## Validation Checklist

After completing all tasks, verify:

- [ ] Checkpoint files copied (checkpoint_epoch_0030.pt, checkpoint_epoch_0035.pt)
- [ ] Symlinks created (checkpoint_best.pt, checkpoint_latest.pt)
- [ ] Config file accessible (src/utils/full_sdxl_config.yaml)
- [ ] main.py updated to import from generative_augmentation_sdxl
- [ ] GenerativeAugmentationManager points to new checkpoint_dir
- [ ] Quick test passes (`test_generative_aug.py --quick --fresh`)
- [ ] No errors in test output
- [ ] Generated images look realistic (not cartoonish)

## Next Steps

Once all validation checks pass:

1. **Test with full dataset:**
   ```bash
   python src/main.py --modality depth_rgb --use_gen_aug
   ```

2. **Monitor training:**
   - Check that generative augmentation is being used
   - Verify GPU memory usage is acceptable
   - Confirm no errors during generation

3. **Compare results:**
   - Run baseline without generative aug
   - Run with generative aug enabled
   - Compare final metrics (accuracy, F1, kappa)

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Verify all files are in correct locations
3. Check CUDA/GPU availability: `nvidia-smi`
4. Review error messages carefully

## Summary

This migration replaces the older SD 1.5 LoRA-based generative augmentation with a new SDXL full fine-tuning approach. The new system:
- Uses larger, more capable SDXL model
- Trained specifically on bbox-cropped wound images
- Generates phase-specific realistic wound images
- Integrates seamlessly with existing main.py
- Generates images directly at required size for efficiency

Expected improvement in augmentation quality should lead to better model generalization and performance on the wound classification task.
