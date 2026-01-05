# Fusion Architecture Analysis - ROOT CAUSE IDENTIFIED

## Date: 2026-01-05

## Critical Finding: Fixed-Weight Fusion with NO Trainable Layer

### Current Fusion Architecture (Metadata + 1 Image)

**Location:** `src/models/builders.py` lines 316-348

```python
elif len(selected_modalities) == 2:
    if has_metadata:
        # MULTI-MODAL (2): Metadata + 1 Image
        vprint("Model: Metadata + 1 image - two-stage fine-tuning with pre-trained image", level=2)

        # Get RF probabilities (already optimal - Kappa 0.254, proper probabilities)
        rf_probs = branches[metadata_idx]  # Shape: (batch, 3)

        # Get image classifier output (also softmax probabilities)
        image_probs = branches[image_idx]  # Shape: (batch, 3)

        # CRITICAL: Fixed weights (NO trainable parameters!)
        rf_weight = 0.70
        image_weight = 0.30
        vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)

        # Compute weighted average with FIXED weights
        weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
        weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)

        # Sum weighted predictions (always sums to 1.0)
        output = Add(name='output')([weighted_rf, weighted_image])
```

### Analysis

**Design Philosophy:**
- Uses **FIXED weights** (70% RF, 30% image)
- **NO trainable fusion layer** - just a weighted sum
- Rationale: "RF already optimal" (Kappa 0.254)

**Implications:**

1. **Stage 1 (Frozen Image):**
   - Image branch is frozen
   - RF branch is just predictions (not trainable)
   - Fusion is fixed weights
   - **Result: 0 trainable parameters!** ⚠️
   - Training does nothing - model cannot learn

2. **Stage 2 (Unfrozen Image):**
   - Image branch unfrozen with very low LR (1e-6)
   - Can only improve image features slightly
   - Fusion weights STILL fixed at 70/30
   - Cannot learn optimal fusion ratio

### Why This Might Fail at 128x128

**Hypothesis:** At higher resolutions:
1. Image branch needs more capacity (more parameters)
2. Simple CNN (current backbone) insufficient for 128x128
3. Pre-training gets worse Kappa (0.097 vs 0.094 at 32x32)
4. Fixed 30% weight to a weak image model hurts overall performance
5. Cannot adapt fusion ratio to compensate for weak image branch

**Why 32x32 Might Work:**
- Smaller image → simpler patterns
- Simple CNN sufficient
- 30% weight to decent image model helps
- RF (70%) carries most of the performance

### Evidence from Initial 100% Data Test (Before Kill)

**Fold 3 Results (32x32, 100% data):**
```
Pre-training thermal_map: Kappa 0.0947
Stage 1 (frozen):         Kappa -0.0478  ❌ NEGATIVE!
Stage 2 (fine-tune):      Kappa -0.0398  ❌ NEGATIVE!
```

**Analysis:**
- Even Stage 2 got NEGATIVE Kappa!
- Something is fundamentally broken
- This contradicts baseline 32x32 result (Kappa 0.316)
- Need to investigate what changed

### Potential Root Causes

1. **Architecture Issue:**
   - Fixed fusion weights might be wrong ratio
   - Need trainable fusion layer to learn optimal weights
   - Different image sizes need different fusion ratios

2. **Training Issue:**
   - Stage 1 with 0 trainable params is pointless
   - Very low LR (1e-6) in Stage 2 might prevent learning
   - Pre-trained weights not loading correctly?

3. **Data Issue:**
   - 3-fold CV vs 1-fold in baseline?
   - Different random seed?
   - Cache corruption?

### Questions for Cloud Agent

1. **Why did baseline 32x32 work (Kappa 0.316) but current test fails (Kappa -0.04)?**
   - What changed between runs?
   - Was baseline also using fixed fusion weights?

2. **Should fusion have trainable weights?**
   - Current design uses fixed 70/30 ratio
   - Would trainable fusion layer help?
   - Can we make fusion weights learnable?

3. **Is Stage 1 necessary?**
   - If 0 trainable params, why run Stage 1?
   - Could we skip directly to Stage 2?

4. **What's the optimal fusion architecture for different image sizes?**
   - Should fusion ratio be 70/30 for all sizes?
   - Should we use a Dense layer instead of fixed weights?

### Recommended Next Steps

1. **Complete current 50% tests** (32, 64, 128) to see degradation pattern
2. **Investigate baseline discrepancy** - why did 32x32 work before?
3. **Consider architecture modifications:**
   - Add trainable fusion layer (Dense layer after concatenation)
   - Make fusion weights learnable parameters
   - Use attention mechanism instead of fixed weights
4. **Test different backbones** (EfficientNet) if simple CNN insufficient

### Code Locations to Review

- **Fusion architecture:** `src/models/builders.py:316-348`
- **Stage 1 training:** `src/training/training_utils.py:1385-1450`
- **Stage 2 training:** `src/training/training_utils.py:1450-1520`
- **Pre-training:** `src/training/training_utils.py:1140-1260`
