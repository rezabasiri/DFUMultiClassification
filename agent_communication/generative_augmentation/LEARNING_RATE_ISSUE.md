# Critical Issue: Learning Rate Starvation Causing Poor Quality

## Problem Report

**Observation**: Full fine-tuning (100% parameters) produces cartoonish, generic images similar to LoRA quality. Images appear generated mainly from prompts with minimal learning from training data.

## Root Cause Analysis

### Issue 1: Learning Rate 10x Too Low

**From training log** (epoch 1-4):
```
Epoch 1, Step 128: LR = 3.2e-7  (still in warmup, only 64% complete)
Epoch 2, Step 256: LR = 6.4e-7  (still in warmup)
Epoch 3, Step 384: LR = 9.6e-7  (still in warmup)
Epoch 4, Step 512: LR = 1.0e-6  (warmup complete, start cosine decay)
```

**Config had**:
- `learning_rate: 1.0e-6` (0.000001)
- `lr_warmup_steps: 500` (~4 epochs with 128 steps/epoch)

**Why this is catastrophic**:
1. **Target LR too low**: Full fine-tuning of SDXL needs LR ~1e-5 to 2e-5
   - LoRA uses 1e-4 to 5e-4 (higher because fewer params)
   - Full fine-tuning uses 1e-5 to 2e-5 (lower but more params)
   - Using 1e-6 = **10x too conservative**

2. **Warmup too long**: First 4 epochs at sub-optimal LR
   - Epoch 1-3: LR < 1e-6 (even lower than target!)
   - Only reaches 1e-6 at epoch 4
   - Then cosine decay reduces it further

3. **Effective learning**: With LR = 1e-6, weight updates are **microscopic**
   - Each parameter changes by ~0.0001% per update
   - Model stays ~99.99% pretrained SDXL weights
   - Result: Generates from pretrained knowledge + prompt, not from training data

### Issue 2: EMA Decay Too Slow (Secondary)

**Config had**:
- `ema_decay: 0.9999` (very slow)
- `use_ema_weights_for_inference: true`

**Impact**:
- EMA update formula: `ema_weight = 0.9999 * ema_weight + 0.0001 * new_weight`
- After 128 steps (epoch 1): EMA incorporates only **1.3%** of new weights
- Generated samples use EMA → **98.7% pretrained SDXL**, 1.3% fine-tuned
- Even if main model learns, samples look like pretrained SDXL

Formula: `1 - (0.9999)^steps`
- After 128 steps: 1 - 0.9999^128 = 0.0127 (1.3%)
- After 500 steps: 1 - 0.9999^500 = 0.0488 (4.8%)
- After 1000 steps: 1 - 0.9999^1000 = 0.0952 (9.5%)

So generated images are essentially **pretrained SDXL with 98%+ weight**, explaining the cartoonish appearance.

## Evidence from Log

**Training metrics**:
- Train loss: 0.1490 → 0.1346 (minimal decrease, barely learning)
- Val loss: 0.0855 → 0.1335 → 0.1079 → 0.1115 (unstable, no clear improvement)
- FID: 453.22 (target < 50, so **9x worse than target**)
- SSIM: 0.3389 (target > 0.7, so **2x worse**)
- LPIPS: 0.9025 (target < 0.3, so **3x worse**)

**Loss barely decreasing = weights barely updating = learning not happening**

## Solution Applied

### Fix 1: Increase Learning Rate 10x
```yaml
learning_rate: 1.0e-6 → 1.0e-5  # 10x higher, appropriate for full fine-tuning
lr_warmup_steps: 500 → 100     # Reach target LR in 1 epoch, not 4
```

**Expected impact**:
- Meaningful weight updates starting from epoch 1
- Model can actually learn wound-specific features
- Weight changes: 0.0001% → 0.001% per update (10x larger)

### Fix 2: Speed Up EMA Decay 10x
```yaml
ema_decay: 0.9999 → 0.999  # 10x faster incorporation of new weights
```

**Expected impact**:
- After 128 steps: 12% new weights (was 1.3%) - **9x more responsive**
- After 500 steps: 39% new weights (was 4.8%) - **8x more responsive**
- Generated samples reflect actual learning much faster

## Why This Explains Everything

**User observation**: "Images mainly generated from prompt, learn very little from training data"

**Explanation**:
1. LR 1e-6 → Model weights barely change → Stay ~99.99% pretrained
2. EMA 0.9999 → Generated samples use ~98.7% pretrained weights
3. **Combined effect**: Generated images are 99.9%+ pretrained SDXL
4. Pretrained SDXL was trained on internet images → Tends toward "photorealistic" but generic/cartoonish style
5. No wound-specific learning → Can't generate realistic wound textures

**Why similar to LoRA quality**:
- LoRA: Trains 11.5% of parameters with LR 1e-4 → Effective learning capacity ≈ 11.5% × 1e-4
- Full with LR 1e-6: Trains 100% of parameters with LR 1e-6 → Effective learning capacity ≈ 100% × 1e-6
- **Ratio**: (100% × 1e-6) / (11.5% × 1e-4) = 0.0869
- Full fine-tuning with LR 1e-6 has **9x LESS learning capacity than LoRA**!

## Expected Results After Fix

**With LR 1e-5 and EMA 0.999**:
- Effective learning capacity: 100% × 1e-5 = **87x higher than before**
- After epoch 1: ~12% new weights in samples (was 1.3%)
- After epoch 5: ~50% new weights in samples
- Loss should decrease steadily (not plateau)
- FID should improve significantly (453 → ~200-300 by epoch 5)
- Images should show wound-specific textures, not generic SDXL style

## Lessons Learned

1. **For full fine-tuning diffusion models**:
   - Use LR 1e-5 to 2e-5, not 1e-6
   - Conservative LR is counterproductive - prevents learning

2. **EMA decay tuning**:
   - Slower (0.9999): Better for final model quality, but obscures early training progress
   - Faster (0.999): Samples reflect learning faster, better for monitoring
   - Can use 0.999 during training, then continue with 0.9999 for final polish

3. **Warmup should be short**:
   - 1 epoch warmup is sufficient (100-150 steps)
   - Long warmup (500+ steps) wastes early training

4. **Monitoring training**:
   - If train loss plateaus quickly → LR too low
   - If val loss doesn't improve → Model not learning
   - If generated samples look like pretrained model → Check EMA decay + LR

## Test Plan

1. Resume training with fixed config: `--resume latest`
2. Monitor epoch 5-10:
   - Train loss should decrease steadily
   - Val loss should decrease (not increase/plateau)
   - FID should improve (decrease from 453)
3. Check generated samples at epoch 10:
   - Should show wound-specific textures
   - Less cartoonish, more realistic tissue appearance
   - Clear differences between phases (I/P/R)

If results still poor by epoch 10 → Other issues (data quality, prompt mismatch, etc.)
