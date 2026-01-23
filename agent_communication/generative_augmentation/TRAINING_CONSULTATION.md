# Training Generative Models - Technical Consultation

## Your Requirements
- ✅ Train Phase I and Phase R models at 128×128
- ✅ High quality is priority #1
- ✅ Utilize both GPUs
- ✅ Quantitative quality comparison with real dataset
- ✅ Use metrics for filtering during inference
- ✅ Configurable options, minimal hardcoding
- ✅ Resume capability
- ✅ Can use prompts or not (up to me)

---

## Key Decisions I Need Your Approval On

### **Decision 1: Base Model Architecture** ⭐⭐⭐⭐⭐

I recommend: **Stable Diffusion v2.1 Base**

**Options Comparison**:

| Model | Pros | Cons | Recommendation |
|-------|------|------|----------------|
| **SD v1.5** (current) | - Faster training<br>- Lower VRAM | - Lower quality<br>- Worse with medical images | ❌ Not recommended |
| **SD v2.1 Base** | - Better quality than v1.5<br>- Same VRAM as v1.5<br>- Better fine-tuning | - Slightly slower than v1.5 | ✅ **RECOMMENDED** |
| **SD v2.1 768** | - Even better quality<br>- Higher resolution | - 2× VRAM usage<br>- 2× slower training | ⚠️ Overkill for 128×128 |
| **SDXL** | - Best quality<br>- Latest architecture | - 3× VRAM usage<br>- 3-4× slower<br>- Might not fit on 2×4090 | ❌ Too heavy |

**My Recommendation**: **SD v2.1 Base**
- Sweet spot for quality vs compute
- Will fit on 2×4090 with 128×128 images
- Much better quality than v1.5
- Good fine-tuning characteristics

**Your Choice**: SD v2.1 Base / SD v1.5 / SD v2.1 768 / Other: ___________

---

### **Decision 2: Training Method** ⭐⭐⭐⭐⭐

I recommend: **LoRA (Low-Rank Adaptation)**

**Options Comparison**:

| Method | Trainable Params | Overfitting Risk | Training Speed | Quality | Storage |
|--------|------------------|------------------|----------------|---------|---------|
| **Full Fine-tuning** | 100% (860M) | Very High | Slow | Good if enough data | 3.4 GB/model |
| **LoRA** (rank=16) | 1-2% (~15M) | Low | Fast | Very Good | 60 MB/model |
| **DreamBooth** | 100% | High | Slow | Good for objects | 3.4 GB/model |
| **Textual Inversion** | <0.1% | Very Low | Very Fast | Lower quality | <1 MB/model |

**My Recommendation**: **LoRA with rank=16, alpha=32**
- Only 60 MB per trained model (vs 3.4 GB full fine-tune)
- Much less overfitting on small datasets (70-200 images per phase)
- Faster training (fewer parameters to update)
- Can stack multiple LoRAs if needed
- State-of-the-art for domain adaptation on small datasets

**LoRA Configuration**:
```yaml
rank: 16              # Higher rank = more capacity (8, 16, 32, 64)
alpha: 32             # Scaling factor (typically 2× rank)
target_modules:       # Which layers to adapt
  - to_q, to_k, to_v  # Attention layers
  - to_out.0          # Output projections
dropout: 0.1          # Regularization
```

**Your Choice**: LoRA (rank=16) / Full Fine-tuning / LoRA (rank=32) / Other: ___________

---

### **Decision 3: Prompting Strategy** ⭐⭐⭐⭐

I recommend: **Use prompts (class-conditional generation)**

**Options**:

**Option A: Prompt-based (RECOMMENDED)**
- Pros:
  - Can guide generation with medical terminology
  - Can emphasize phase-specific features
  - Better control over output
  - Can use negative prompts to avoid artifacts
- Cons:
  - Need to design good prompts
  - Slightly more complex

**Option B: Unconditional (no prompts)**
- Pros:
  - Simpler training code
  - Model learns pure image distribution
- Cons:
  - Less control over generation
  - Can't emphasize discriminative features
  - Harder to ensure phase-specific generation

**My Recommendation**: **Prompt-based with optimized medical prompts**

Example prompts:
```python
Phase I (Inflammatory):
  Positive: "Medical photograph of diabetic foot ulcer in inflammatory phase,
            showing redness, swelling, acute inflammation, wound edges,
            clinical documentation, high detail"
  Negative: "blurry, low quality, healthy skin, cartoon, artificial,
            watermark, text, smooth texture, unrealistic colors"

Phase R (Remodeling):
  Positive: "Medical photograph of diabetic foot ulcer in remodeling phase,
            showing wound contraction, epithelialization, scar tissue,
            mature healing, clinical documentation, high detail"
  Negative: "blurry, low quality, healthy skin, cartoon, artificial,
            watermark, text, smooth texture, unrealistic colors"
```

**Your Choice**: Prompts (recommended) / No prompts / Custom prompts: ___________

---

### **Decision 4: Quality Metrics for Filtering** ⭐⭐⭐⭐⭐

I recommend: **Combination of FID, SSIM, and LPIPS**

**Metrics to Implement**:

| Metric | What It Measures | Threshold | Use Case |
|--------|------------------|-----------|----------|
| **FID** | Distribution similarity to real images | < 50 (< 30 excellent) | Overall quality benchmark |
| **SSIM** | Structural similarity | > 0.70 (> 0.80 excellent) | Texture and structure matching |
| **LPIPS** | Perceptual similarity | < 0.30 (< 0.20 excellent) | Human-perceived quality |
| **IS** | Inception Score (diversity + quality) | > 2.0 (> 3.0 excellent) | Generation diversity |

**Filtering Strategy During Inference**:
```python
# Generate image
generated_img = pipeline(...)

# Check quality against real wound reference set
if SSIM(generated_img, real_wound_samples) > 0.70 and \
   LPIPS(generated_img, real_wound_samples) < 0.30:
    use_image()  # High quality
else:
    reject_image()  # Low quality
```

**Your Approval**: Use FID + SSIM + LPIPS / Different metrics / Add more: ___________

---

### **Decision 5: Training Configuration** ⭐⭐⭐⭐

I recommend these settings:

```yaml
# Model & Resolution
base_model: "stabilityai/stable-diffusion-2-1-base"
resolution: 128
training_method: "lora"

# LoRA Settings
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: ["to_q", "to_k", "to_v", "to_out.0"]

# Training
batch_size_per_gpu: 8          # Total batch = 16 (2 GPUs)
gradient_accumulation: 2       # Effective batch = 32
learning_rate: 1e-5
lr_scheduler: "cosine"
lr_warmup_steps: 100
max_epochs: 100
early_stopping_patience: 10

# Quality
inference_steps_during_training: 50    # For validation
inference_steps_production: 100         # For final use
guidance_scale: 12.0

# Data
train_val_split: 0.85                  # 85% train, 15% validation
data_augmentation: true                # Flip, rotate, color jitter

# Checkpointing
save_every_n_epochs: 5
keep_last_n_checkpoints: 3
save_best_only: false                  # Keep all for comparison

# Multi-GPU
mixed_precision: "fp16"
gradient_checkpointing: true
use_xformers: true                     # Memory-efficient attention
```

**Your Approval**: Looks good / Change batch size / Change LR / Other: ___________

---

### **Decision 6: Training Enhancements** ⭐⭐⭐

Optional enhancements I can add:

**A. Perceptual Loss** (RECOMMENDED)
```python
# Add perceptual loss using VGG features
total_loss = diffusion_loss + 0.1 * perceptual_loss(generated, real)
```
- Pros: Better visual quality, more realistic textures
- Cons: 10-15% slower training
- **Recommendation**: ✅ Include this

**B. EMA (Exponential Moving Average)** (RECOMMENDED)
- Keep moving average of model weights
- Use EMA weights for inference (smoother, better quality)
- Pros: Better generation quality, more stable
- Cons: Extra 3.4 GB VRAM
- **Recommendation**: ✅ Include this

**C. Image Quality Discriminator** (ADVANCED)
- Train a discriminator to distinguish real vs fake wounds
- Use adversarial loss to improve realism
- Pros: Can significantly improve realism
- Cons: 30% slower training, more complex, can be unstable
- **Recommendation**: ⚠️ Skip for first version, add later if needed

**Your Choice**: A + B (recommended) / A only / All three / None

---

### **Decision 7: Resume Capability Design** ⭐⭐⭐

**What to Save in Checkpoints**:
```python
checkpoint = {
    # Model state
    'unet_lora_state': unet.state_dict(),
    'ema_state': ema_model.state_dict(),      # If using EMA

    # Optimizer state
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': lr_scheduler.state_dict(),

    # Training state
    'epoch': current_epoch,
    'global_step': global_step,
    'best_fid': best_fid_score,
    'best_epoch': best_epoch,

    # Metrics history
    'train_losses': train_loss_history,
    'val_losses': val_loss_history,
    'fid_history': fid_scores,
    'ssim_history': ssim_scores,

    # Config
    'config': training_config,
    'random_state': {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'cuda': torch.cuda.get_rng_state_all(),
    }
}
```

**Resume Strategy**:
```bash
# Resume from latest checkpoint
python train_lora_model.py --resume latest

# Resume from specific checkpoint
python train_lora_model.py --resume checkpoints/epoch_50.pt

# Resume from best checkpoint (by FID)
python train_lora_model.py --resume best
```

**Your Approval**: Looks good / Add more state / Simplify: ___________

---

### **Decision 8: Code Structure** ⭐⭐⭐

**Proposed File Structure**:
```
agent_communication/generative_augmentation/
├── ACTION_PLANS.md                    # ✅ Created
├── TRAINING_CONSULTATION.md           # ✅ This file
├── configs/
│   ├── phase_I_config.yaml            # Phase I training config
│   ├── phase_R_config.yaml            # Phase R training config
│   └── quality_thresholds.yaml        # Filtering thresholds
├── scripts/
│   ├── train_lora_model.py            # 🔥 Main training script (300-400 lines)
│   ├── evaluate_quality.py            # 🔥 Compute FID/SSIM/LPIPS (150-200 lines)
│   ├── generate_samples.py            # 🔥 Generate test images (100 lines)
│   ├── compare_models.py              # 🔥 Compare multiple checkpoints (150 lines)
│   └── utils/
│       ├── quality_metrics.py         # 🔥 Metric implementations (200 lines)
│       ├── data_loader.py             # 🔥 Dataset class (150 lines)
│       ├── training_utils.py          # 🔥 Training helpers (200 lines)
│       └── checkpoint_utils.py        # 🔥 Save/load logic (100 lines)
├── checkpoints/                       # Model checkpoints
│   ├── phase_I/
│   └── phase_R/
├── generated_samples/                 # Test generations
│   ├── phase_I/
│   └── phase_R/
└── reports/
    ├── TRAINING_REPORT_phase_I.md     # Training logs
    ├── TRAINING_REPORT_phase_R.md
    └── QUALITY_COMPARISON.md          # Metric comparisons
```

**Estimated Code**:
- ~1,500 lines total (well-commented, configurable)
- ~300-400 lines main training script
- ~500 lines utilities
- ~200 lines quality metrics
- ~150 lines data loading
- ~100 lines evaluation/generation

**Your Approval**: Structure looks good / Prefer different organization: ___________

---

## My Recommendations Summary

| Decision | Recommendation | Confidence |
|----------|----------------|------------|
| 1. Base Model | Stable Diffusion v2.1 Base | ⭐⭐⭐⭐⭐ |
| 2. Training Method | LoRA (rank=16, alpha=32) | ⭐⭐⭐⭐⭐ |
| 3. Prompts | Yes, use medical prompts | ⭐⭐⭐⭐⭐ |
| 4. Quality Metrics | FID + SSIM + LPIPS | ⭐⭐⭐⭐⭐ |
| 5. Config | As shown above | ⭐⭐⭐⭐ |
| 6. Enhancements | Perceptual Loss + EMA | ⭐⭐⭐⭐ |
| 7. Resume | Full state saving | ⭐⭐⭐⭐⭐ |
| 8. Structure | As shown above | ⭐⭐⭐⭐ |

---

## Questions for You

1. **Do you approve using SD v2.1 Base?** (Better quality than current v1.5)
   - [ ] Yes, use SD v2.1 Base
   - [ ] No, use different model: ___________

2. **Do you approve LoRA training?** (Much less overfitting than full fine-tune)
   - [ ] Yes, LoRA rank=16
   - [ ] Yes, but different rank: ___________
   - [ ] No, use full fine-tuning

3. **Prompts or no prompts?**
   - [ ] Yes, use medical prompts (recommended)
   - [ ] No, unconditional generation

4. **Include perceptual loss?** (Better quality, 10-15% slower)
   - [ ] Yes, include it (recommended)
   - [ ] No, skip it

5. **Include EMA?** (Better quality, extra 3.4 GB VRAM)
   - [ ] Yes, include it (recommended)
   - [ ] No, skip it

6. **Any other requirements or preferences?**
   - ___________________________________________

---

## Next Steps After Your Approval

1. I'll write all the training code (~1,500 lines)
2. Create configuration files for Phase I and Phase R
3. Implement quality metrics (FID, SSIM, LPIPS)
4. Add resume capability
5. Create evaluation and comparison scripts
6. Document everything thoroughly

**Estimated Time to Code**: 4-6 hours for complete, production-ready implementation

---

**Please review and approve/modify these decisions, then I'll write the code!** 🚀
