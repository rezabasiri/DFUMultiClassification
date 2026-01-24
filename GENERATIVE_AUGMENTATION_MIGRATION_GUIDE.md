# Generative Augmentation Migration Guide

## Overview

This document describes the migration from the old generative augmentation system (`src/data/generative_augmentation_v2.py`) to the new SDXL-based system developed in `agent_communication/generative_augmentation`.

## Migration Status

**✅ COMPLETED (Remote Agent)**
- Code migrated to main project structure
- Utility files copied
- Integration code created
- Ready for local agent to copy checkpoints and test

**⏳ PENDING (Local Agent)**
- Copy checkpoint files (~20GB total)
- Update main.py import statements
- Run integration tests

**👉 See `LOCAL_AGENT_MIGRATION_INSTRUCTIONS.md` for detailed step-by-step instructions.**

---

## Summary of Changes

### Old System → New System

| Aspect | Old (v2) | New (SDXL) |
|--------|----------|------------|
| Model | SD 1.5 | SDXL 1.0 Base |
| Training | LoRA (11.5%) | Full (100%) |
| Resolution | 256×256 + resize | Direct at IMAGE_SIZE |
| Prompts | Generic descriptions | Minimal phase-specific |
| Data | Mixed sources | 100% bbox-cropped |
| Checkpoint | ~500MB | ~10GB |
| Quality (FID) | ~300-400 | ~190-255 |

### Files Migrated

**Remote Agent Actions (Completed):**

1. **Utility files** → `src/utils/`:
   - `checkpoint_utils.py` - Checkpoint management
   - `generative_data_loader.py` - Data loading with bbox
   - `quality_metrics.py` - FID, SSIM, LPIPS, IS
   - `generative_training_utils.py` - EMA, perceptual loss
   - `full_sdxl_config.yaml` - Training configuration

2. **Integration code** → `src/data/`:
   - `generative_augmentation_sdxl.py` - New SDXL-based system
   - Replaces `generative_augmentation_v2.py`
   - Maintains compatibility with existing interface

**Local Agent Actions (Pending):**

1. **Checkpoint files** → `src/models/sdxl_checkpoints/`:
   - `checkpoint_epoch_0030.pt` (~10GB) - Best by metrics
   - `checkpoint_epoch_0035.pt` (~10GB) - Latest
   - `checkpoint_history.json` (~1KB) - Training history

2. **Update main.py**:
   ```python
   # Change import from:
   from src.data.generative_augmentation_v2 import ...

   # To:
   from src.data.generative_augmentation_sdxl import ...

   # Update instantiation:
   gen_manager = GenerativeAugmentationManager(
       checkpoint_dir='src/models/sdxl_checkpoints',  # New parameter
       config=aug_config
   )
   ```

---

## Key Technical Improvements

### 1. Direct Size Generation
- **Old:** Generate at 256×256, then resize to IMAGE_SIZE
- **New:** Generate directly at IMAGE_SIZE (saves computation time)

### 2. Phase-Specific Training
**Old prompts (too detailed):**
```
"Close-up clinical photograph of diabetic foot ulcer wound in inflammatory phase,
acute open wound with redness erythema and swelling edema around wound margins,
early wound response showing tissue inflammation..."
```

**New prompts (minimal):**
```
"PHASE_I, diabetic foot ulcer, inflammatory phase wound"
```

**Why this matters:**
- Detailed prompts → model uses pretrained knowledge
- Minimal prompts → model MUST learn from training data
- Result: More realistic, dataset-specific wound generation

### 3. Bbox-Aware Training
- **Old:** Full images including background, feet, etc.
- **New:** 100% bbox-cropped wound-only images
- **Result:** Model focuses solely on wound characteristics

### 4. Full Fine-Tuning vs LoRA
- **Old:** 11.5% of parameters trainable (LoRA)
- **New:** 100% of parameters trainable (full fine-tuning)
- **Result:** Better domain adaptation for medical wound images

---

## Training Results

**Checkpoint Performance (Epochs 30-38):**
- **FID:** 189-255 (target <50, trending downward)
- **SSIM:** 0.80-0.90 ✓ (target >0.7, achieved)
- **LPIPS:** 0.40-0.57 (target <0.3, moderate)
- **Inception Score:** 2.4-3.0 ✓ (target >2.0, achieved)

**Phase-Specific Quality:**
```
Phase I (Inflammatory):  SSIM=0.84, LPIPS=0.47
Phase P (Proliferative): SSIM=0.85, LPIPS=0.45
Phase R (Remodeling):    SSIM=0.88, LPIPS=0.37
```

---

## Testing Procedure

After local agent completes migration:

```bash
# Quick test (5 minutes)
cd /workspace/DFUMultiClassification
python agent_communication/generative_augmentation/test_generative_aug.py --quick --fresh

# Full test (30 minutes)
python agent_communication/generative_augmentation/test_generative_aug.py --fresh

# Production training test
python src/main.py --modality depth_rgb --use_gen_aug --epochs 5
```

**Success Criteria:**
- ✓ Checkpoint loads without errors
- ✓ Images generate for all phases (I, P, R)
- ✓ Generation speed: <10 seconds per image
- ✓ Generated images look realistic (not cartoonish)
- ✓ Training runs without OOM errors

---

## Troubleshooting Reference

### Common Issues

**1. Checkpoint not found:**
```bash
# Verify checkpoint location
ls -lh /workspace/DFUMultiClassification/src/models/sdxl_checkpoints/
```

**2. CUDA OOM:**
```python
# Reduce batch limit in production_config.py
GENERATIVE_AUG_BATCH_LIMIT = 4  # Down from 8
```

**3. Slow generation:**
```python
# Reduce inference steps in production_config.py
GENERATIVE_AUG_INFERENCE_STEPS = 30  # Down from 50
```

---

## Next Steps

1. **Local agent:** Follow `LOCAL_AGENT_MIGRATION_INSTRUCTIONS.md`
2. **Test:** Run quick test to verify integration
3. **Validate:** Compare baseline vs generative aug performance
4. **Deploy:** Use in production training runs

---

## References

- **Experimental code:** `agent_communication/generative_augmentation/`
- **Training config:** `agent_communication/generative_augmentation/configs/full_sdxl.yaml`
- **Training logs:** `agent_communication/generative_augmentation/reports/full_sdxl_logs/`
- **Technical analysis:** `PROMPT_OVERFITTING_ISSUE.md` (why minimal prompts work better)
- **Local agent guide:** `LOCAL_AGENT_MIGRATION_INSTRUCTIONS.md` (detailed instructions)

---

## Summary

The new SDXL-based generative augmentation system represents a significant upgrade:
- **Better quality** through full fine-tuning instead of LoRA
- **More realistic** through minimal prompts and bbox-focused training
- **More efficient** through direct size generation
- **Easier integration** through compatible interface

Expected impact: Improved model generalization and classification performance through higher-quality augmented training data.
