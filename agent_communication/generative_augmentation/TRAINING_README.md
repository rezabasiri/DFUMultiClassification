# High-Quality Wound Image Generation Training System

Production-ready system for training Stable Diffusion models to generate diabetic foot ulcer (DFU) images using LoRA adaptation.

**Created**: 2026-01-19
**Version**: 1.0

---

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision diffusers transformers accelerate peft
pip install torchmetrics lpips pandas pyyaml tqdm pillow
pip install xformers  # Optional but recommended
```

### 2. Train Phase I Model (Inflammatory)

```bash
cd agent_communication/generative_augmentation
accelerate launch scripts/train_lora_model.py --config configs/phase_I_config.yaml
```

### 3. Train Phase R Model (Remodeling)

```bash
accelerate launch scripts/train_lora_model.py --config configs/phase_R_config.yaml
```

### 4. Evaluate Quality

```bash
python scripts/evaluate_quality.py \
    --checkpoint checkpoints/phase_I/checkpoint_best.pt \
    --config configs/phase_I_config.yaml \
    --num_samples 200
```

---

## 📁 File Structure

```
configs/
  ├── phase_I_config.yaml       # Phase I training config
  ├── phase_R_config.yaml       # Phase R training config
  └── quality_thresholds.yaml   # Quality filtering thresholds

scripts/
  ├── train_lora_model.py       # Main training script
  ├── evaluate_quality.py       # Quality evaluation
  ├── generate_samples.py       # Sample generation
  ├── compare_models.py         # Compare checkpoints
  └── utils/                     # Utility modules
      ├── quality_metrics.py    # FID, SSIM, LPIPS, IS
      ├── data_loader.py        # Dataset handling
      ├── training_utils.py     # Perceptual loss, EMA
      └── checkpoint_utils.py   # Checkpoint management

checkpoints/phase_I/            # Phase I model checkpoints
checkpoints/phase_R/            # Phase R model checkpoints
generated_samples/              # Test images
reports/                        # Training logs
```

---

## ⚙️ Key Features

- **Multi-GPU Training** - Automatic with Accelerate
- **LoRA Adaptation** - Only 60 MB per model vs 3.4 GB
- **Quality Enhancement** - Perceptual loss + EMA
- **Comprehensive Metrics** - FID, SSIM, LPIPS, Inception Score
- **Full Resume** - Save/restore from any checkpoint
- **Highly Configurable** - YAML configs, minimal hardcoding

---

## 🎓 Training Commands

**Basic Training:**
```bash
python scripts/train_lora_model.py --config configs/phase_I_config.yaml
```

**Resume from Latest:**
```bash
python scripts/train_lora_model.py --config configs/phase_I_config.yaml --resume latest
```

**Resume from Best:**
```bash
python scripts/train_lora_model.py --config configs/phase_I_config.yaml --resume best
```

**Multi-GPU (2 GPUs):**
```bash
accelerate launch scripts/train_lora_model.py --config configs/phase_I_config.yaml
```

---

## 📊 Expected Results

### Phase I (Inflammatory)
- **Training Time**: 8-12 hours on 2× RTX 4090
- **Target FID**: < 50 (excellent: < 30)
- **Target SSIM**: > 0.70 (excellent: > 0.80)
- **Target LPIPS**: < 0.30 (excellent: < 0.20)

### Phase R (Remodeling)
- **Training Time**: 8-12 hours on 2× RTX 4090
- **Target FID**: < 50 (excellent: < 30)
- **Target SSIM**: > 0.70 (excellent: > 0.80)
- **Target LPIPS**: < 0.30 (excellent: < 0.20)

---

## 🔧 Configuration Guide

### Key Parameters

**Model:**
```yaml
model:
  base_model: "stabilityai/stable-diffusion-2-1-base"
  resolution: 128
```

**LoRA:**
```yaml
lora:
  rank: 16        # Higher = more capacity
  alpha: 32       # Scaling factor (typically 2× rank)
  dropout: 0.1    # Regularization
```

**Training:**
```yaml
training:
  batch_size_per_gpu: 8
  learning_rate: 1.0e-5
  max_epochs: 100
  early_stopping:
    patience: 10
```

**Quality:**
```yaml
quality:
  perceptual_loss:
    enabled: true
    weight: 0.1
  ema:
    enabled: true
    decay: 0.9999
  guidance_scale: 12.0
  inference_steps_production: 100
```

---

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size_per_gpu` to 4 or 2
- Enable `gradient_checkpointing: true`
- Use `mixed_precision: "fp16"`
- Disable EMA temporarily

### Poor Quality
- Check you're using EMA weights (`use_ema_weights_for_inference: true`)
- Increase `inference_steps_production` to 150
- Increase `guidance_scale` to 15.0
- Train longer (50+ epochs)

### Slow Training
- Enable `use_xformers: true`
- Use `mixed_precision: "fp16"`
- Increase `batch_size_per_gpu` if you have VRAM
- Use multiple GPUs

---

## 📚 Scripts Reference

### train_lora_model.py

Main training script with multi-GPU support.

```bash
python scripts/train_lora_model.py \
    --config configs/phase_I_config.yaml \
    --resume latest
```

### evaluate_quality.py

Comprehensive quality evaluation.

```bash
python scripts/evaluate_quality.py \
    --checkpoint checkpoints/phase_I/checkpoint_best.pt \
    --config configs/phase_I_config.yaml \
    --num_samples 200 \
    --output reports/quality.json
```

### generate_samples.py

Generate sample images.

```bash
python scripts/generate_samples.py \
    --checkpoint checkpoints/phase_I/checkpoint_best.pt \
    --config configs/phase_I_config.yaml \
    --num_samples 50 \
    --grid
```

### compare_models.py

Compare multiple checkpoints.

```bash
python scripts/compare_models.py \
    --checkpoint_dir checkpoints/phase_I \
    --config configs/phase_I_config.yaml \
    --num_samples 100
```

---

## 💡 Tips for Best Quality

1. **Use LoRA** - Less overfitting than full fine-tuning
2. **Enable Perceptual Loss** - Better visual quality
3. **Enable EMA** - Smoother, more stable models
4. **Train Longer** - 50-100 epochs minimum
5. **High Inference Steps** - 100+ for production
6. **Appropriate Guidance** - 12-15 for medical images
7. **Good Prompts** - Phase-specific, descriptive
8. **Data Augmentation** - Flips, rotations during training

---

## 📖 Understanding Metrics

### FID (Fréchet Inception Distance)
- Measures distribution similarity to real images
- **Lower is better**
- < 30: Excellent
- < 50: Good
- < 100: Fair
- > 100: Poor

### SSIM (Structural Similarity Index)
- Measures structural similarity
- **Higher is better**
- > 0.80: Excellent
- > 0.70: Good
- > 0.60: Fair
- < 0.60: Poor

### LPIPS (Learned Perceptual Similarity)
- Measures perceptual similarity
- **Lower is better**
- < 0.20: Excellent
- < 0.30: Good
- < 0.40: Fair
- > 0.40: Poor

### Inception Score
- Measures quality and diversity
- **Higher is better**
- > 3.0: Excellent
- > 2.0: Good
- > 1.5: Fair
- < 1.5: Poor

---

## 🔬 Technical Details

### LoRA Advantages
- **60 MB** per model (vs 3.4 GB full model)
- **Much less overfitting** on small datasets
- **Faster training** (fewer parameters)
- **Stackable** (can combine multiple LoRAs)

### Perceptual Loss
- Uses VGG-16 conv3_3 features
- Ensures generated images have similar high-level features to real wounds
- Weight: 0.1× diffusion loss (configurable)

### EMA (Exponential Moving Average)
- Maintains smoothed version of model weights
- Decay: 0.9999 (very slow update)
- Used for inference (not training)
- Produces more stable, higher quality images

### Multi-GPU Training
- Automatic with Accelerate library
- Data parallelism across GPUs
- Linear speedup (2 GPUs ≈ 2× faster)
- Effective batch size = `batch_size_per_gpu × num_gpus × gradient_accumulation_steps`

---

## 🚀 Workflow Example

```bash
# 1. Train Phase I model
accelerate launch scripts/train_lora_model.py --config configs/phase_I_config.yaml

# 2. Evaluate quality (during or after training)
python scripts/evaluate_quality.py \
    --checkpoint checkpoints/phase_I/checkpoint_best.pt \
    --config configs/phase_I_config.yaml \
    --num_samples 200

# 3. Generate test samples
python scripts/generate_samples.py \
    --checkpoint checkpoints/phase_I/checkpoint_best.pt \
    --config configs/phase_I_config.yaml \
    --num_samples 100 \
    --grid

# 4. Compare checkpoints to find best
python scripts/compare_models.py \
    --checkpoint_dir checkpoints/phase_I \
    --config configs/phase_I_config.yaml

# 5. If quality is good (FID < 50, SSIM > 0.70), train Phase R
accelerate launch scripts/train_lora_model.py --config configs/phase_R_config.yaml

# 6. Repeat evaluation for Phase R
```

---

## 📝 Next Steps After Training

1. **Validate Quality** - Ensure FID < 50, SSIM > 0.70
2. **Visual Inspection** - Manually check generated images
3. **Update Inference Code** - Point to new checkpoints
4. **Re-run Augmentation Test** - Test with new models
5. **Compare Performance** - New vs old models

---

## 🤝 Support

For issues or questions:
1. Check `reports/` directory for training logs
2. Review this README's troubleshooting section
3. Check configuration files for typos
4. Ensure all dependencies installed correctly

---

**Created by**: Cloud Agent
**Date**: 2026-01-19
**Purpose**: High-quality generative model training for DFU classification improvement
