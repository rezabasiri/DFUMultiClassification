# Model Comparison: SDXL 1.0 vs SD 3.5 Medium

## Quick Test Protocol

**Goal**: Compare SDXL 1.0 and SD 3.5 Medium at 128×128, pick the winner

**Method**:
- Train both models for 3 epochs
- Generate 50 samples from each
- Compare quality metrics (FID, SSIM, LPIPS, IS)
- Pick winner based on best overall metrics

**Time**: ~1-2 hours total

## Running Comparison

### Automated (Recommended)
```bash
cd agent_communication/generative_augmentation
python scripts/compare_base_models.py
```

This runs both trainings sequentially and outputs comparison report.

### Manual (If Automated Fails)

**Train SDXL:**
```bash
python scripts/train_lora_model.py --config configs/test_sdxl.yaml
```

**Train SD3.5:**
```bash
python scripts/train_lora_model.py --config configs/test_sd3_medium.yaml
```

**Evaluate:**
```bash
python scripts/evaluate_quality.py \
  --checkpoint checkpoints/test_sdxl/checkpoint_epoch_003.pt \
  --config configs/test_sdxl.yaml \
  --num_samples 50

python scripts/evaluate_quality.py \
  --checkpoint checkpoints/test_sd3_medium/checkpoint_epoch_003.pt \
  --config configs/test_sd3_medium.yaml \
  --num_samples 50
```

## Expected Output

```
================================================================================
Quality Comparison Results
================================================================================

Metric               SDXL 1.0             SD 3.5 Medium        Winner
-------------------- -------------------- -------------------- ---------------
FID (lower better)   42.35                38.12                SD3.5
SSIM (higher better) 0.7234               0.7456               SD3.5
LPIPS (lower better) 0.2789               0.2534               SD3.5
IS (higher better)   2.34                 2.67                 SD3.5

--------------------------------------------------------------------------------
Overall Winner: ✓ SD 3.5 Medium (4/4 metrics)
--------------------------------------------------------------------------------
```

## After Comparison

Update production configs with winner:
- `configs/phase_I_config.yaml`
- `configs/phase_R_config.yaml`

Change `base_model` to winner's model ID.

## Notes

**SD 3.5 Medium Requirements**:
- Hugging Face login: `huggingface-cli login`
- Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
- May require different pipeline (StableDiffusion3Pipeline vs StableDiffusionPipeline)

**SDXL Requirements**:
- No gating, works immediately
- Well-tested, mature ecosystem
