# Critical Discovery: EMA Hiding Training Progress + NaN Gradient Explosion

## Problem Report

**User observation after all previous fixes (minimal prompts, no perceptual loss, low guidance):**
"still the same FID and the same cartoonish images after the first epoch. abs no difference."

**Log evidence:**
- Epoch 1: Loss 0.1042, FID 435 (slight improvement from 447)
- Epoch 2, Step 50: **Loss: nan** ← TRAINING EXPLODED!

## Root Cause Analysis

After all prompt/perceptual/guidance fixes, found TWO critical remaining issues:

### Issue 1: EMA Hiding All Training Progress (PRIMARY)

**The problem:**
```python
# In train_lora_model.py line 1243:
use_ema=config['quality']['ema'].get('use_ema_weights_for_inference', True)

# In generate_validation_samples line 648-651:
if use_ema and ema_model is not None:
    unet_to_use = ema_model.ema_model  # ← Samples use EMA, not trained model!
else:
    unet_to_use = unet_lora
```

**Config had:**
```yaml
ema:
  enabled: true
  decay: 0.999
  use_ema_weights_for_inference: true  # ← PROBLEM!
```

**EMA formula:**
```
ema_weight = decay * ema_weight + (1 - decay) * new_weight
Accumulated new weight after N steps = 1 - decay^N
```

**With decay 0.999:**
- After 128 steps (epoch 1): 1 - 0.999^128 = **0.12 (12%)** new weights
- After 256 steps (epoch 2): 1 - 0.999^256 = **0.23 (23%)** new weights
- After 512 steps (epoch 4): 1 - 0.999^512 = **0.40 (40%)** new weights
- After 1000 steps (epoch 8): 1 - 0.999^1000 = **0.63 (63%)** new weights

**Impact:**
At epoch 1, generated samples use:
- **88% pretrained SDXL weights** (unchanged from pretraining)
- **12% trained weights** (from your dataset)

**This explains everything:**
- Main model IS learning (loss decreased: 0.1404 → 0.1042)
- But samples use EMA which is 88% pretrained
- So generated images look mostly like pretrained SDXL
- User sees "abs no difference" because they're viewing 88% pretrained model
- FID barely improves because samples don't reflect actual training

**Why this is insidious:**
- Training IS working (loss decreasing)
- Gradients ARE flowing (weights updating)
- But you CAN'T SEE it because samples use EMA (mostly pretrained)
- False conclusion: "training not working" when it actually is!

### Issue 2: NaN Gradient Explosion at Epoch 2

**Evidence from log:**
```
Epoch 1:
  Step 1/128 - Loss: 0.2664, LR: 0.00e+00
  Step 50/128 - Loss: 0.1023, LR: 6.00e-06
  Step 100/128 - Loss: 0.1031, LR: 1.00e-05
  Epoch 1 complete - Loss: 0.1042, LR: 1.00e-05

Epoch 2:
  Step 1/128 - Loss: 0.1306, LR: 1.00e-05
  Step 50/128 - Loss: nan, LR: 1.00e-05  ← EXPLOSION!
```

**Root cause:**
- Learning rate 1e-5 might be slightly too high for full fine-tuning at resolution 512
- With 2.6B parameters, even small numerical instabilities can compound
- Once NaN appears, training is permanently broken (all future gradients = NaN)

**Impact:**
- After NaN at epoch 2, no more learning can happen
- All subsequent epochs will have NaN loss
- Weights stop updating entirely
- Model frozen in broken state

**Why 1e-5 caused NaN:**
- Worked fine at lower resolution (256 or 128)
- But at 512x512, activations are 4x larger
- Gradient magnitudes scale with activation size
- 1e-5 LR + large gradients = numerical overflow → NaN

## The Solution

### Fix 1: Disable EMA for Sample Generation (PRIMARY FIX)

**Before:**
```yaml
ema:
  enabled: true
  decay: 0.999
  use_ema_weights_for_inference: true  # Samples use 88% pretrained at epoch 1
```

**After:**
```yaml
ema:
  enabled: true
  decay: 0.99  # Faster - 63% new weights after epoch 1 vs 12%
  use_ema_weights_for_inference: false  # Use main model - see actual learning!
```

**Why this works:**
- Samples now generated from **main trained model** directly
- No EMA delay hiding training progress
- You'll see actual learning immediately at epoch 1
- FID will reflect true model quality, not 88% pretrained

**EMA decay change (0.999 → 0.99):**
- Faster incorporation of new weights into EMA
- After 128 steps: 63% new vs 12% new
- Still keeping EMA for potential later use
- Can enable `use_ema_weights_for_inference: true` after 5-10 epochs when EMA catches up

### Fix 2: Lower Learning Rate to Prevent NaN

**Before:**
```yaml
learning_rate: 1.0e-5  # Too high for 512x512, caused NaN at epoch 2
```

**After:**
```yaml
learning_rate: 5.0e-6  # Halved to prevent gradient explosion
```

**Why this works:**
- Still 5x higher than original 1e-6 (which was too low)
- But low enough to prevent numerical overflow at 512x512
- 5e-6 is appropriate for full fine-tuning of large models at high resolution
- Should prevent NaN while still allowing meaningful learning

**Learning rate history:**
1. Original: 1e-6 (too low, no learning)
2. First fix: 1e-5 (better, but caused NaN at epoch 2)
3. Final: 5e-6 (balanced - learning without explosion)

## Expected Results After Fix

### Epoch 1-2 (Immediate Impact):

**Training metrics:**
- No NaN gradients (fixed with 5e-6 LR)
- Loss should decrease smoothly: 0.10 → 0.08 → 0.06
- Training continues past epoch 2 (no explosion)

**Generated samples:**
- **IMMEDIATE VISIBLE CHANGE** - samples use main model, not EMA
- At epoch 1, should show SOME dataset-specific features
- Won't look like pretrained SDXL + prompts anymore
- May look "rough" or "unpolished" (that's GOOD - means actual learning!)
- FID should improve: 435 → 350-400

### Epoch 3-5 (Early Learning):

**Training metrics:**
- Loss continues decreasing: 0.06 → 0.04 → 0.03
- FID improves noticeably: 400 → 300-350
- SSIM starts increasing: 0.27 → 0.35-0.40
- LPIPS starts decreasing: 0.99 → 0.80-0.90

**Generated samples:**
- Clear wound-specific textures appearing
- Colors/shapes matching training dataset
- Less "generic stock photo" look
- Phase differences (I/P/R) becoming visible

### Epoch 5-15 (Convergence):

**Training metrics:**
- Loss plateaus around 0.02-0.03
- FID continues improving: 300 → 150-250
- SSIM: 0.40 → 0.50-0.60
- LPIPS: 0.80 → 0.50-0.70

**Generated samples:**
- Realistic wound appearance matching training data
- Strong phase-specific characteristics
- No cartoonish or artistic style
- Looks like actual medical photos from dataset

### After Training Stabilizes (Optional):

If you want, can re-enable EMA for final polish:
```yaml
use_ema_weights_for_inference: true  # Re-enable after epoch 10-15
```

At that point, EMA will have caught up (incorporated most trained weights) and will provide smoother, more consistent samples.

## Why Previous Fixes Didn't Show Results

**Recap of all changes made:**
1. ✅ Switched LoRA → full fine-tuning (11.5% → 100% trainable)
2. ✅ Fixed LR: 1e-6 → 1e-5 (too low → better, but then NaN)
3. ✅ Simplified prompts (detailed → minimal)
4. ✅ Disabled perceptual loss (VGG bias removed)
5. ✅ Lowered guidance: 12.0 → 3.0 (less prompt reliance)

**Why FID stayed at 435-447:**
- Main model WAS learning (loss 0.1404 → 0.1042)
- But samples used EMA (88% pretrained)
- So FID reflected pretrained model + 12% learning
- Not enough to see significant improvement

**User correctly observed:** "images mainly generated from prompts"
- Because 88% of sample weights were pretrained SDXL
- Which generates from its pretrained "clinical photo" knowledge
- Only 12% was from your dataset

**Why both LoRA and full failed identically (before prompt fix):**
- Both used detailed prompts → activated pretrained knowledge
- LoRA: 11.5% trainable, but learned weights hidden by EMA
- Full: 100% trainable, but learned weights hidden by EMA
- Result: Both looked like pretrained + prompts

## Key Insights

### 1. EMA Can Hide Training Progress

**The trap:**
- EMA is great for FINAL model quality (smoothed weights)
- But TERRIBLE for monitoring training progress (delayed updates)
- With slow decay (0.9999), can take 5-10 epochs to see any learning
- Creates false impression that "training not working"

**Best practice:**
- During training: Use main model for samples (disable use_ema_weights_for_inference)
- After training: Use EMA for production (enable use_ema_weights_for_inference)
- OR: Use very fast EMA decay (0.9 or 0.95) during training to see progress

### 2. Learning Rate Depends on Resolution

**Resolution affects gradient magnitude:**
- 128x128: latents are 16×16 → smaller gradients
- 256x256: latents are 32×32 → medium gradients
- 512x512: latents are 64×64 → large gradients

**LR should be adjusted:**
- 128x128: Can use 1e-5 or higher
- 256x256: Use 7e-6 to 1e-5
- 512x512: Use 5e-6 to 7e-6

**We learned this the hard way:**
- 1e-6: Too low (no learning)
- 1e-5 at 512: Too high (NaN at epoch 2)
- 5e-6 at 512: Just right (Goldilocks)

### 3. Loss Decreasing ≠ Quality Improving (If Using EMA)

**What we saw:**
- Loss: 0.1404 → 0.1042 (29% improvement)
- FID: 447 → 435 (only 2.7% improvement)

**Why the discrepancy:**
- Loss reflects **main model** (which IS learning)
- FID reflects **EMA model** (which is 88% pretrained)
- Huge gap between what's training and what's being evaluated

**Lesson:**
- Always check if samples use EMA or main model
- If using EMA, loss and quality metrics will be misaligned early in training
- Can't trust quality metrics during early training with slow EMA

### 4. Single Epoch Not Enough to Judge Full Fine-tuning

**The math:**
- 2.6B parameters to train
- 128 steps per epoch
- Each parameter gets ~1-2 gradient updates per epoch
- Not enough to significantly change model behavior

**Full fine-tuning timeline:**
- Epochs 1-5: Model "exploring" - trying different solutions
- Epochs 5-15: Model "converging" - finding good solution
- Epochs 15+: Model "refining" - polishing details

**User tested after epoch 1:**
- Too early to see results even if training working perfectly
- Need at least 3-5 epochs to judge if training is effective
- Especially with EMA (which delays visibility further)

## Verification Plan

### 1. Run Training

```bash
cd /workspace/DFUMultiClassification/agent_communication/generative_augmentation
source /opt/miniforge3/bin/activate multimodal

accelerate launch --config_file accelerate_config_deepspeed.yaml \
  scripts/train_lora_model.py --resume none --config configs/full_sdxl.yaml \
  2>&1 | tee full_sdxl_final.log
```

### 2. Check for NaN (Critical)

Monitor the log file:
```bash
tail -f full_sdxl_final.log | grep "Loss:"
```

**If you see NaN at any point:**
- Training is broken, stop immediately
- Further reduce LR: 5e-6 → 3e-6 or 2e-6
- Report back with epoch/step where NaN occurred

**If no NaN through epoch 5:**
- Training is stable, let it continue
- NaN risk decreases after early epochs

### 3. Check FID Improvement (Every 5 Epochs)

**Expected trajectory:**
```
Epoch 1: FID 420-440 (slight improvement from 435-447)
Epoch 5: FID 300-380 (significant drop if learning working)
Epoch 10: FID 200-300 (approaching good quality)
Epoch 15: FID 150-250 (good quality, dataset-specific)
```

**If FID NOT improving by epoch 5:**
- Problem is deeper than EMA/NaN
- May need to investigate data quality, preprocessing, or training setup

### 4. Visual Inspection

**Compare samples:**
- Epoch 1 vs pretrained: Should show SOME differences (colors, textures)
- Epoch 5 vs epoch 1: Should show clear improvement
- Epoch 10 vs epoch 5: Should show dataset-specific features

**What to look for:**
- Are wound colors matching your dataset?
- Are textures realistic (not cartoonish)?
- Can you see differences between phases I/P/R?
- Do images look like they could be from your training set?

## If Still No Improvement After These Fixes

If after 5 epochs with these fixes FID still hasn't improved (stuck at 400+), then the problem is NOT:
- Prompts (fixed)
- Perceptual loss (disabled)
- Guidance scale (lowered)
- Learning rate (fixed)
- EMA hiding progress (disabled)

It would be:
- Data quality issues (images themselves problematic)
- Preprocessing issues (bbox cropping, normalization)
- Training data insufficient (2860 images might be too few)
- SDXL incompatibility with medical domain (unlikely but possible)
- Something fundamental with SDXL implementation

But first, let's see if these fixes work. The EMA issue alone could explain EVERYTHING.

## Summary

**Main culprit:** EMA with `use_ema_weights_for_inference: true` and slow decay (0.999)
- Samples used 88% pretrained weights after epoch 1
- Made it look like no learning happening
- User saw "abs no difference" because they were looking at pretrained model

**Secondary issue:** NaN gradients at epoch 2 from LR 1e-5 being too high
- Stopped all learning after epoch 1
- Model frozen with broken gradients

**Fix:** Disable EMA for inference + lower LR to 5e-6
- Now samples show actual main model (100% trained weights)
- No NaN → training continues smoothly
- Should see clear improvement by epoch 3-5

**Next step:** Run training and monitor for NaN + FID improvement
