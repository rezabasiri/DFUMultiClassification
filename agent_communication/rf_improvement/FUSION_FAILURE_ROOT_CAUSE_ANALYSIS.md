# FUSION CATASTROPHIC FAILURE - ROOT CAUSE ANALYSIS

**Date**: 2026-01-04
**Analyzed by**: GPU Agent (Independent Analysis)
**User Report**: Metadata-only (Kappa 0.254) ✅ | Fusion (Kappa 0.022) ❌ | thermal_map-only (Kappa 0.145) ✅

---

## EXECUTIVE SUMMARY

**CRITICAL FINDING**: The fusion architecture has **THREE FUNDAMENTAL FLAWS** that cause catastrophic degradation:

1. ❌ **Asymmetric Input Problem**: RF branch outputs BatchNorm-transformed values (NOT probabilities), while image branch outputs proper probabilities
2. ❌ **No Quality Preservation Constraint**: Dense(3) fusion layer can learn ANY 6→3 mapping with 21 trainable weights - no guarantee to preserve RF quality
3. ❌ **BatchNorm Destroys Probability Structure**: RF probabilities (sum to 1) → BatchNorm → arbitrary values (can be negative, don't sum to 1)

**Result**: The fusion layer learns to degrade/ignore high-quality RF predictions (Kappa 0.254) in favor of weaker image predictions (Kappa 0.145).

---

## DETAILED ARCHITECTURE ANALYSIS

### What Metadata-Only Does (WORKS - Kappa 0.254)

**File**: `src/models/builders.py:304-310`

```python
if has_metadata:
    vprint("Model: Metadata-only - using RF predictions directly (no Dense layer)", level=2)
    output = Activation('softmax', name='output')(branches[0])
```

**Flow**:
```
RF probabilities [p_I, p_P, p_R] (sum to 1.0)
    ↓
Lambda(cast to float32)
    ↓
BatchNormalization → transforms: (x - μ) / σ  [NOT probabilities anymore!]
    ↓
Activation('softmax') → re-normalizes to probabilities (sum to 1.0)
    ↓
Output: [p'_I, p'_P, p'_R] ✅
```

**Why it works**: The softmax re-normalizes BatchNorm output back into valid probabilities. Minimal transformation, RF quality preserved.

---

### What Fusion Does (FAILS - Kappa 0.022)

**File**: `src/models/builders.py:315-337`

```python
elif len(selected_modalities) == 2:
    if has_metadata:
        vprint("Model: Metadata + 1 image - preserving RF quality in fusion", level=2)

        rf_probs = branches[metadata_idx]  # ⚠️ BatchNorm output, NOT probabilities!

        image_idx = 1 - metadata_idx
        image_features = branches[image_idx]  # 64-dim feature vector
        image_probs = Dense(3, activation='softmax')(image_features)  # ✅ Proper probabilities

        merged = concatenate([rf_probs, image_probs])  # ⚠️ [3 BatchNorm values, 3 probabilities]

        fusion = Dense(3, activation='softmax', name='output')(merged)  # ❌ Unconstrained 6→3 mapping
        output = fusion
```

**Flow for RF branch**:
```
RF probabilities [0.7, 0.2, 0.1] (sum to 1.0)
    ↓
BatchNormalization → e.g., [1.2, -0.3, -0.9]  ❌ NOT probabilities!
    ↓
Concatenate with image_probs → [1.2, -0.3, -0.9, 0.4, 0.5, 0.1]
    ↓
Dense(3, softmax) with 21 weights → Can learn ANY transformation
    ↓
Output: Degraded predictions (Kappa 0.022) ❌
```

**Flow for Image branch**:
```
thermal_map image
    ↓
Conv2D layers → GAP → Dense(512) → ... → Dense(64)
    ↓
64-dimensional feature vector
    ↓
Dense(3, activation='softmax') → [0.4, 0.5, 0.1] ✅ Proper probabilities
    ↓
Concatenate with rf_probs
```

---

## THE THREE CRITICAL FLAWS

### FLAW #1: Asymmetric Inputs to Fusion Layer

**Concatenated Input**:
- **First 3 values** (RF): BatchNorm-transformed, arbitrary scale, can be negative, do NOT sum to 1
- **Last 3 values** (Image): Softmax probabilities, range [0, 1], sum to 1

**Problem**: The fusion Dense layer receives incompatible representations:
- RF values are in "BatchNorm space" (mean=0, variance=1 per batch)
- Image values are in "probability space" (sum=1, non-negative)

**Impact**: The network cannot learn a consistent fusion strategy because inputs are on completely different scales and have different mathematical properties.

---

### FLAW #2: Unconstrained Dense Layer

**Current Implementation**:
```python
fusion = Dense(3, activation='softmax')(merged)  # 6 inputs → 3 outputs
```

**Trainable Weights**: 6×3 + 3 = **21 parameters**

**What the comment claims**:
> "Learn fusion weights (lightweight - just learns α and (1-α))"

**Reality**: This is FALSE. The Dense layer can learn:
- **Any 6→3 linear transformation** + bias
- **No constraint** to preserve RF quality
- **No constraint** to combine as weighted average
- Can set RF weights to 0 (ignore RF completely)
- Can set RF weights to negative (reverse RF predictions)
- Can amplify noise from weaker modality

**Example of what Dense layer might learn**:
```
Output_class_I = 0.1*rf[0] + 0.8*image[0] + 0.2*image[1] - 0.3*rf[2] + bias
Output_class_P = -0.2*rf[0] + 0.5*image[1] + 0.4*rf[1] + 0.1*image[2] + bias
Output_class_R = 0.3*rf[2] + 0.6*image[2] - 0.1*rf[0] + 0.2*image[0] + bias
```

This is NOT a weighted average of RF and image predictions!

---

### FLAW #3: BatchNorm Destroys RF Probability Structure

**What RF produces** (from `dataset_utils.py`):
```python
# Ordinal RF outputs calibrated probabilities
rf_prob_I = (1 - prob_PorR)  # P(Inflammation)
rf_prob_P = prob_PorR * (1 - prob_R_given_PorR)  # P(Proliferation)
rf_prob_R = prob_PorR * prob_R_given_PorR  # P(Remodeling)
# These sum to 1.0 by construction
```

**What BatchNormalization does**:
```python
x_norm = (x - batch_mean) / batch_std
```

**Example**:
```
Input batch:
Sample 1: [0.7, 0.2, 0.1]  (sum = 1.0)
Sample 2: [0.5, 0.3, 0.2]  (sum = 1.0)
Sample 3: [0.6, 0.3, 0.1]  (sum = 1.0)

Batch mean: [0.6, 0.267, 0.133]
Batch std: [0.082, 0.047, 0.047]

After BatchNorm:
Sample 1: [1.22, -1.43, -0.71]  ❌ Negative values! Don't sum to 1!
Sample 2: [-1.22, 0.70, 1.43]   ❌ Completely different structure!
Sample 3: [0.00, 0.70, -0.71]   ❌ No longer probabilities!
```

**Impact**: The RF probabilities lose their meaning as calibrated predictions. They become arbitrary normalized values.

---

## WHY METADATA-ONLY WORKS BUT FUSION FAILS

### Metadata-Only Success

```python
output = Activation('softmax')(branches[0])
```

**Key**: Softmax RE-NORMALIZES the BatchNorm output back into probabilities.

**Example**:
```
RF probs: [0.7, 0.2, 0.1]
    ↓ BatchNorm
[1.22, -1.43, -0.71]
    ↓ Softmax
[0.72, 0.05, 0.23]  ← Still captures RF's preference for class I (0.72 highest)
```

The transformation preserves the RANKING of probabilities (argmax usually unchanged).

---

### Fusion Failure

```python
merged = concatenate([rf_probs, image_probs])  # [BatchNorm values, Softmax probs]
fusion = Dense(3, softmax)(merged)
```

**Problem 1**: RF values are NOT re-normalized before fusion
**Problem 2**: Dense layer can learn to ignore RF entirely
**Problem 3**: Training optimizes fusion layer to minimize loss, which may mean relying more on image predictions if they're more consistent (even if lower quality)

**Why fusion might prefer image over RF**:
- RF values after BatchNorm are unstable (vary by batch statistics)
- Image probabilities are stable (always normalized)
- Dense layer learns: "Image predictions are more predictable" → weights them higher
- Result: Ignores high-quality RF (Kappa 0.254) in favor of consistent-but-weaker image (Kappa 0.145)

---

## EVIDENCE FROM TEST RESULTS

### TEST 1: Metadata-Only ✅
- **Kappa**: 0.254 ± 0.125
- **Architecture**: RF → BatchNorm → Softmax
- **Trainable weights**: 2 (just BatchNorm gamma/beta)
- **Message**: "using RF predictions directly (no Dense layer)"

**Analysis**: Minimal transformation preserves RF quality.

---

### TEST 2: Metadata + thermal_map ❌
- **Kappa**: 0.022 ± 0.103 (91% WORSE than metadata alone!)
- **Architecture**: [RF → BatchNorm, thermal → Dense(3)] → Concatenate → Dense(3)
- **Trainable weights**: Thousands (image branch) + 21 (fusion)
- **Message**: "preserving RF quality in fusion"

**Analysis**: Message is WRONG. Architecture does NOT preserve RF quality.

**Performance breakdown**:
- vs Metadata-only: -91% (0.022 vs 0.254)
- vs thermal_map-only: -85% (0.022 vs 0.145)
- **Fusion is worse than BOTH individual modalities**

This is the **opposite of expected behavior**. Fusion should combine strengths, not amplify weaknesses.

---

### TEST 3: thermal_map-Only ✅
- **Kappa**: 0.145 ± 0.026
- **Architecture**: thermal → Dense layers → Dense(3, softmax)
- **Message**: Standard image classification

**Analysis**: Works as expected for image-only classification.

---

## HYPOTHESIS: Why Fusion Performs So Poorly

### Hypothesis 1: Dense Layer Ignores RF (Most Likely)

The fusion Dense layer learns to:
- Set RF input weights ≈ 0
- Rely primarily on image predictions
- Result: Performance ≈ image-only (0.145) or worse

**Why "or worse"?**: The Dense layer has 21 parameters trying to learn from small dataset → overfitting, unstable gradients.

---

### Hypothesis 2: BatchNorm Variance Across Batches

RF values after BatchNorm depend on batch statistics:
- Different batches → different BatchNorm transformations
- Same RF probs [0.7, 0.2, 0.1] become different values in different batches
- Fusion layer cannot learn consistent pattern from RF
- Ignores RF as "noisy input"

---

### Hypothesis 3: Training Instability

Concatenating [3 BatchNorm values, 3 probabilities] creates:
- Gradient flow issues (different scales)
- Optimization challenges (incompatible inputs)
- Convergence to poor local minimum

---

## RECOMMENDED FIXES

### FIX #1: Re-Normalize RF Before Fusion (RECOMMENDED)

```python
elif len(selected_modalities) == 2:
    if has_metadata:
        # Get RF probabilities and RE-NORMALIZE
        rf_probs_raw = branches[metadata_idx]  # BatchNorm output
        rf_probs = Activation('softmax', name='rf_normalized')(rf_probs_raw)  # ✅ Now proper probabilities

        # Get image probabilities
        image_idx = 1 - metadata_idx
        image_features = branches[image_idx]
        image_probs = Dense(3, activation='softmax', name='image_classifier')(image_features)

        # Now both are proper probabilities [0, 1] summing to 1
        merged = concatenate([rf_probs, image_probs], name='concat_predictions')

        # Fusion layer - still unconstrained, but at least inputs are symmetric
        output = Dense(3, activation='softmax', name='output')(merged)
```

**Benefits**:
- Both inputs are now probabilities (symmetric)
- RF quality preserved in rf_probs
- Fusion layer receives compatible inputs

**Limitations**: Dense layer still unconstrained (can ignore RF)

---

### FIX #2: Constrained Weighted Average Fusion (BETTER)

```python
elif len(selected_modalities) == 2:
    if has_metadata:
        # Get RF probabilities (re-normalized)
        rf_probs_raw = branches[metadata_idx]
        rf_probs = Activation('softmax', name='rf_normalized')(rf_probs_raw)

        # Get image probabilities
        image_idx = 1 - metadata_idx
        image_features = branches[image_idx]
        image_probs = Dense(3, activation='softmax', name='image_classifier')(image_features)

        # Learn a SINGLE weight α ∈ [0, 1]
        # Use both predictions to estimate confidence
        combined_for_weight = concatenate([rf_probs, image_probs])
        alpha = Dense(1, activation='sigmoid', name='fusion_weight')(combined_for_weight)

        # Weighted average: α * RF + (1-α) * Image
        # This GUARANTEES RF quality is preserved when α ≈ 1
        weighted_rf = Multiply(name='weighted_rf')([rf_probs, alpha])
        one_minus_alpha = Lambda(lambda x: 1 - x, name='one_minus_alpha')(alpha)
        weighted_image = Multiply(name='weighted_image')([image_probs, one_minus_alpha])

        # Final output is weighted sum (already probabilities, just normalize)
        output = Add(name='output')([weighted_rf, weighted_image])
```

**Benefits**:
- **CONSTRAINED**: Can only learn weighted average, not arbitrary transformation
- **RF preservation**: If α=1, output = RF exactly (Kappa 0.254)
- **Graceful degradation**: If α=0.8, output = 0.8×RF + 0.2×Image
- Only **1 trainable weight** for fusion (vs 21) → less overfitting

---

### FIX #3: Remove BatchNorm from Metadata Branch (BEST)

Since RF already outputs calibrated probabilities, BatchNorm is unnecessary and harmful.

```python
def create_metadata_branch(input_shape, index):
    metadata_input = Input(shape=input_shape, name=f'metadata_input')
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name=f'metadata_cast_{index}')(metadata_input)
    # ❌ REMOVE BatchNormalization - RF probs are already calibrated!
    # x = BatchNormalization(name=f'metadata_BN_{index}')(x)
    return metadata_input, x
```

Then in fusion:
```python
elif len(selected_modalities) == 2:
    if has_metadata:
        rf_probs = branches[metadata_idx]  # ✅ Already proper probabilities!

        image_features = branches[image_idx]
        image_probs = Dense(3, activation='softmax')(image_features)

        # Use FIX #2 constrained weighted average
        # ...
```

**Benefits**:
- **Simplest fix**: Just remove harmful BatchNorm
- **RF probabilities preserved** throughout
- **No re-normalization needed**
- **Symmetric inputs** to fusion

---

## COMPARISON OF FIXES

| Fix | Complexity | RF Preservation | Fusion Constraint | Expected Kappa |
|-----|------------|-----------------|-------------------|----------------|
| **Current** | Simple | ❌ No | ❌ No | 0.022 (FAIL) |
| **Fix #1** | Low | ⚠️ Partial | ❌ No | 0.10-0.15? |
| **Fix #2** | Medium | ✅ Yes | ✅ Yes | 0.20-0.28 |
| **Fix #3** | Low | ✅ Yes | Depends | 0.18-0.25 |
| **Fix #2 + #3** | Medium | ✅✅ Yes | ✅ Yes | **0.25-0.30** ⭐ |

---

## RECOMMENDED ACTION PLAN

### IMMEDIATE: Apply Fix #2 + Fix #3 Combined

**Step 1**: Remove BatchNorm from metadata branch
**Step 2**: Implement constrained weighted average fusion
**Step 3**: Re-run TEST 2 (metadata + thermal_map)

**Expected result**: Kappa 0.25-0.30 (better than either modality alone)

### Validation

After fix, TEST 2 should show:
- ✅ Kappa > 0.25 (better than metadata-only 0.254)
- ✅ Kappa > 0.145 (better than thermal_map-only)
- ✅ Fusion weight α ≈ 0.6-0.8 (RF contributes more than image)
- ✅ Message: "preserving RF quality in fusion" is actually TRUE

---

## SUMMARY

**Root Cause**: Multi-modal fusion has three critical architecture flaws:
1. BatchNorm destroys RF probability structure
2. Asymmetric inputs (BatchNorm values vs probabilities)
3. Unconstrained Dense layer can ignore/degrade RF quality

**Evidence**: Fusion (Kappa 0.022) is 91% worse than metadata-only (0.254) and 85% worse than image-only (0.145)

**Solution**: Remove BatchNorm from metadata branch + use constrained weighted average fusion

**Expected Impact**: Kappa improvement from 0.022 → 0.25-0.30 (+1,000% improvement!)

---

**Analysis completed**: 2026-01-04
**Confidence**: HIGH - Architecture flaws are clear and measurable
**Recommendation**: Apply Fix #2 + #3 and re-validate
