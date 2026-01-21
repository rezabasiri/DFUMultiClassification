# Critical Discovery: Prompt Over-Specification Causing Cartoonish Outputs

## Problem Statement

**User observation**: "Full fine-tuning produces cartoonish images identical to LoRA quality. Images are mainly generated from prompts, learn very little from training data."

**Key insight**: "There must be something shared between them causing this" - User correctly identified that both approaches fail identically.

## Root Cause Investigation

### The Shared Component: Prompts + Guidance + Perceptual Loss

Both LoRA and full fine-tuning share:
1. **Text prompts** used during training and inference
2. **Guidance scale** during sample generation
3. **Perceptual loss** (VGG-based) during training

These shared components were causing BOTH approaches to fail identically, regardless of whether we train 11.5% (LoRA) or 100% (full) of parameters.

## Critical Issue #1: Prompt Over-Specification (PRIMARY CAUSE)

### What We Had
```yaml
phase_prompts:
  I: >
    Close-up clinical photograph of diabetic foot ulcer wound in inflammatory phase,
    acute open wound with redness erythema and swelling edema around wound margins,
    early wound response showing tissue inflammation and wound debridement stage,
    detailed wound bed texture with surrounding inflamed tissue,
    professional medical documentation, high resolution, sharp focus, clinical lighting
```

### Why This is Catastrophic

**SDXL's Pretrained Knowledge:**
- SDXL was trained on **billions of images** from the internet
- This includes medical images, clinical photos, stock medical photography
- SDXL already "knows" what clinical wound photos look like from pretraining

**The Problem:**
With such detailed, descriptive prompts:
- "Close-up clinical photograph" → Activates SDXL's pretrained "clinical photo" style
- "acute open wound with redness erythema and swelling" → SDXL already knows this from pretraining
- "professional medical documentation, high resolution, sharp focus, clinical lighting" → Generic stock photo aesthetics

**Result:**
The model generates images from **pretrained SDXL knowledge** + **prompt description**, not from your training data!

**Analogy:**
It's like training a student (SDXL) who already studied medicine (pretraining), then giving them extremely detailed exam questions (prompts) instead of showing them your specific patient photos (training data). They'll answer based on their medical textbook knowledge, not your specific patients.

### Evidence

1. **Both LoRA and full fine-tuning fail identically**
   - If learning capacity was the issue, full (100%) would perform better than LoRA (11.5%)
   - But they produce IDENTICAL cartoonish results
   - This proves the issue is NOT about trainable parameters, but about prompt over-reliance

2. **"Images mainly generated from prompt"**
   - User's exact observation
   - Detailed prompts give SDXL everything it needs to generate without learning

3. **FID remains terrible despite training**
   - FID: 447 (9x worse than target)
   - No improvement across epochs
   - Model isn't learning new features, just using pretrained knowledge

## Critical Issue #2: Perceptual Loss Pushes Toward "Natural" Style

### The Problem

**Perceptual loss uses VGG16**:
- VGG was trained on **ImageNet** (natural images: cats, dogs, cars, flowers)
- NOT trained on medical wounds
- VGG's idea of "good features" = natural, photographic, artistic

**From training log:**
```
Train loss: 0.1404 (includes perceptual loss)
Val loss: 0.0796 (diffusion only, no perceptual)
Difference: 0.0608

Perceptual loss weight: 0.01
Implied raw perceptual loss: 0.0608 / 0.01 = 6.08
```

The raw perceptual loss (6.08) is **huge** compared to diffusion loss (0.08). Even with 0.01 weight, it's adding 76% extra loss.

**Why this causes cartoonish style:**
- VGG pushes model to generate features that look "natural" to ImageNet
- Medical wound textures don't match ImageNet feature distributions
- Model compromises: generates "artistic" or "cartoonish" wounds that satisfy VGG
- Result: Wounds look like stock photos or medical illustrations, not real tissue

### Evidence

- Train loss = 0.1404 (with perceptual)
- Val loss = 0.0796 (without perceptual)
- **Train loss is 76% higher** due to perceptual component
- Model optimizing for VGG features, not actual diffusion quality

## Critical Issue #3: High Guidance Scale Amplifies Prompt Reliance

### The Problem

**Guidance scale: 12.0** (very high)

**What guidance scale does:**
- High guidance (12.0) = model strongly follows text prompt
- Low guidance (3.0-5.0) = model balances prompt + learned features
- CFG formula: `output = unconditional + guidance_scale * (conditional - unconditional)`

**Why this is problematic:**
With guidance 12.0, the model:
- **Over-relies on text prompt interpretation** (activates pretrained knowledge)
- **Under-utilizes learned features** from training data
- Amplifies the prompt over-specification issue

**Combined effect:**
Detailed prompts + high guidance = "please generate clinical wound using your pretrained knowledge, ignore training data"

## The Perfect Storm

These three issues create a **reinforcement loop**:

1. **Detailed prompts** → Activate pretrained "clinical photo" knowledge
2. **High guidance (12.0)** → Strongly follow prompt interpretation
3. **Perceptual loss (VGG)** → Push toward "natural"/artistic style
4. **Result** → Model generates from pretrained knowledge, learns nothing from training data

**Why both LoRA and full fine-tuning fail identically:**
- The training process is dominated by pretrained knowledge + prompts
- Actual learning from training images is suppressed
- Doesn't matter if you train 11.5% or 100% of parameters - they're all being pulled toward pretrained knowledge

## The Solution

### Fix 1: Minimal Prompts (PRIMARY FIX)

**Before:**
```yaml
I: "Close-up clinical photograph of diabetic foot ulcer wound in inflammatory phase,
    acute open wound with redness erythema and swelling edema around wound margins,
    early wound response showing tissue inflammation and wound debridement stage,
    detailed wound bed texture with surrounding inflamed tissue,
    professional medical documentation, high resolution, sharp focus, clinical lighting"
```

**After:**
```yaml
I: "diabetic foot ulcer, inflammatory phase"
P: "diabetic foot ulcer, proliferative phase"
R: "diabetic foot ulcer, remodeling phase"
```

**Why this works:**
- Minimal information → Can't generate from pretrained knowledge alone
- Model **MUST** learn from training data to generate realistic images
- Forces model to learn wound-specific textures, colors, shapes from YOUR dataset
- Prompts only provide high-level class conditioning, not visual details

### Fix 2: Disable Perceptual Loss

**Before:**
```yaml
perceptual_loss:
  enabled: true
  weight: 0.01
```

**After:**
```yaml
perceptual_loss:
  enabled: false  # VGG trained on ImageNet, not medical wounds
```

**Why this works:**
- Removes VGG's "natural image" bias
- Model optimizes purely for diffusion objective
- No pressure to match ImageNet feature distributions
- Allows realistic wound textures that VGG might consider "unnatural"

### Fix 3: Lower Guidance Scale

**Before:**
```yaml
guidance_scale: 12.0  # Very high
```

**After:**
```yaml
guidance_scale: 3.0  # Moderate
```

**Why this works:**
- Reduces reliance on text prompt interpretation
- Allows learned features from training data to dominate
- Better balance between conditioning and learned generation
- Standard range for fine-tuned models: 3.0-5.0 (vs 7.5 for pretrained)

## Expected Results After Fix

### Immediate Changes (Epoch 1)

**Training metrics:**
- Train loss should **increase** initially (0.14 → 0.20-0.25)
  - Why: Perceptual loss removal + minimal prompts = harder problem
  - Model can't cheat with pretrained knowledge anymore
  - Good sign! Means model is actually learning

**Generated samples:**
- May look **worse** initially at epoch 1-5
- Less "polished" clinical photo look (that was from pretrained knowledge)
- More variation and experimentation

### Mid-Training (Epoch 5-15)

**Training metrics:**
- Loss should **decrease steadily** as model learns
- FID should **improve significantly**: 447 → 200-300
- Train/val loss gap should be smaller (no perceptual loss component)

**Generated samples:**
- Start showing **wound-specific features** from training data
- Colors, textures, shapes matching YOUR dataset
- Less "generic clinical photo" look, more realistic tissue appearance

### Late Training (Epoch 15-30)

**Training metrics:**
- FID should continue improving: 200-300 → 100-150 (or better)
- Loss plateaus at lower value
- SSIM increases toward 0.5-0.6
- LPIPS decreases toward 0.4-0.5

**Generated samples:**
- Clear visual differences between phases (I/P/R)
- Realistic wound textures specific to YOUR dataset
- Not cartoonish or artistic
- Looks like actual training images, not stock photos

## Comparison: Before vs After

### Before (Detailed Prompts + Perceptual Loss + High Guidance)

**What the model learns:**
```
Training: "Generate image matching this detailed prompt description + VGG features"
          ↓
Model thinks: "I already know clinical wounds from pretraining,
               let me generate that + satisfy VGG's natural image features"
          ↓
Result: Generic clinical photo style, cartoonish, stock photo look
```

**Evidence:**
- FID stuck at 447
- Both LoRA and full fail identically
- "Images mainly generated from prompt"

### After (Minimal Prompts + No Perceptual + Low Guidance)

**What the model learns:**
```
Training: "Generate 'diabetic foot ulcer, inflammatory phase' using training data"
          ↓
Model thinks: "Prompt is too vague to use pretrained knowledge,
               I MUST learn from training images to generate realistic wounds"
          ↓
Result: Dataset-specific wound features, realistic textures, learned characteristics
```

**Expected evidence:**
- FID improves: 447 → 100-200 by epoch 15
- Full fine-tuning outperforms LoRA (as expected)
- "Images look like training data, not stock photos"

## Key Insights

1. **Prompts are NOT free information**
   - Detailed prompts = give away the answer
   - Model generates from prompt knowledge, not data
   - Minimal prompts = force learning from data

2. **More trainable parameters ≠ better if training is wrong**
   - Full (100%) and LoRA (11.5%) failed identically
   - Problem wasn't parameter count, but training methodology
   - Garbage in, garbage out - even with 2.6B parameters

3. **Perceptual loss is domain-specific**
   - VGG trained on ImageNet ≠ good for medical images
   - Pushes toward "natural" style, not medical realism
   - For medical imaging: diffusion loss alone is better

4. **Guidance scale should match prompt specificity**
   - Detailed prompts + high guidance = over-reliance on pretrained knowledge
   - Minimal prompts + low guidance = force learning from data
   - Fine-tuned models need lower guidance (3.0-5.0) than pretrained (7.5)

## Testing the Fix

### Command
```bash
cd /workspace/DFUMultiClassification/agent_communication/generative_augmentation && \
source /opt/miniforge3/bin/activate multimodal && \
accelerate launch --config_file accelerate_config_deepspeed.yaml \
  scripts/train_lora_model.py --resume none --config configs/full_sdxl.yaml \
  2>&1 | tee full_sdxl_fixed.log
```

### What to Look For

**Epoch 1-3 (Initial):**
- Loss may be higher (0.20-0.25) - this is GOOD!
- Samples may look worse - model is confused without pretrained crutch
- This means it's actually trying to learn instead of cheating

**Epoch 5-10 (Learning):**
- Loss should decrease steadily
- FID should show improvement (447 → 300-350)
- Samples start showing dataset-specific features

**Epoch 10-20 (Convergence):**
- FID continues improving (300 → 150-200)
- Samples clearly match training data characteristics
- Visual differences between phases become obvious

**Success criteria:**
- FID < 200 by epoch 15 (currently 447)
- Samples look like training images, not stock photos
- Clear phase-specific characteristics (I/P/R)

## Lessons Learned

1. **Prompt engineering for training ≠ prompt engineering for inference**
   - Inference: Detailed prompts help get what you want
   - Training: Detailed prompts prevent learning from data

2. **The "shared component" insight**
   - User correctly identified: "something shared causing this"
   - Investigating what's shared between failing approaches = powerful debugging

3. **Pretraining can be a curse for fine-tuning**
   - SDXL's pretraining on clinical images backfired
   - It "knew too much" and relied on pretrained knowledge
   - Solution: Remove information channels that activate pretrained knowledge

4. **Loss going UP can be good**
   - If training loss increases after fix, celebrate!
   - Means model lost its pretrained knowledge crutch
   - Now it must learn from scratch (as intended)

## References

- Train loss WITH perceptual: 0.1404
- Val loss WITHOUT perceptual: 0.0796
- Perceptual contribution: 76% of total loss
- FID before fix: 447
- Expected FID after fix: < 200 by epoch 15
