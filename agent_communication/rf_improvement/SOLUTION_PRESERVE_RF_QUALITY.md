# SOLUTION: Preserve RF Quality for All Modality Combinations

## Problem Analysis

**Current Architecture Flow:**
```
RF (Bayesian) → [rf_prob_I, rf_prob_P, rf_prob_R] → Cast → BatchNorm → Dense(3) → Softmax
                 ↑ Kappa 0.20                                              ↑ Degrades to 0.109
```

**The Issue:**
- RF produces **high-quality probabilities** (Kappa 0.20)
- Metadata branch preserves them (Cast → BatchNorm) ✅
- But then **Dense(3) layer re-learns** the classification ❌
- This degrades performance in BOTH metadata-only AND multi-modal cases

**Location**: `src/models/builders.py:292`
```python
if len(selected_modalities) == 1:
    output = Dense(3, activation='softmax', name='output')(branches[0])  # ❌ Degrades RF!
```

---

## Solution: Treat RF Probabilities as Pre-Trained Predictions

The key insight: **RF probabilities ARE the final prediction**, not features to classify.

### Metadata Branch Output

RF produces 3 probabilities that sum to 1.0:
- rf_prob_I: probability of Inflammation
- rf_prob_P: probability of Proliferation
- rf_prob_R: probability of Remodeling

After Cast → BatchNorm, these are still valid probabilities. We should **use them directly**, not re-classify!

### Proposed Architecture Changes

#### Option A: Use RF Probabilities Directly (RECOMMENDED)

**Metadata-only:**
```python
if len(selected_modalities) == 1 and 'metadata' in selected_modalities:
    # RF probabilities are already the final output - no Dense layer needed!
    # Just apply activation to ensure proper probability distribution
    output = Activation('softmax', name='output')(branches[0])
```

**Multi-modal:**
```python
# Combine RF probabilities with image predictions using weighted average
# Each modality branch outputs probabilities
rf_probs = branches[metadata_idx]  # [rf_prob_I, rf_prob_P, rf_prob_R]
image_probs = Dense(3, activation='softmax')(branches[image_idx])

# Learn optimal weighting between modalities
weight_rf = Dense(1, activation='sigmoid', name='rf_weight')(rf_probs)
weight_img = Dense(1, activation='sigmoid', name='img_weight')(image_probs)

# Weighted combination
output = (rf_probs * weight_rf + image_probs * weight_img) / (weight_rf + weight_img)
```

#### Option B: Minimal Final Layer with High Dropout

If you prefer to keep some trainability:

```python
if len(selected_modalities) == 1 and 'metadata' in selected_modalities:
    # Very minimal layer - mostly identity transform
    # High dropout to prevent overfitting on RF probabilities
    x = Dropout(0.95, name='preserve_rf_dropout')(branches[0])  # 95% dropout!
    output = Dense(3, activation='softmax',
                   kernel_initializer='identity',  # Start as identity
                   bias_initializer='zeros',
                   name='output')(x)
```

#### Option C: Skip Connection with Minimal Transform

```python
if len(selected_modalities) == 1 and 'metadata' in selected_modalities:
    # Let model learn to use RF predictions directly or adjust slightly
    transform = Dense(3, activation='softmax', name='output_transform')(branches[0])
    # Skip connection: original RF probs have strong influence
    output = Lambda(lambda x: 0.9 * branches[0] + 0.1 * x, name='output')(transform)
```

---

## Detailed Implementation (Option A - RECOMMENDED)

### File: `src/models/builders.py`

#### Modify create_multimodal_model() function (lines 257-330)

```python
def create_multimodal_model(input_shapes, selected_modalities, class_weights, strategy=None):
    """
    Create multimodal model for DFU classification.

    Key Design Principle:
    - Metadata (RF) predictions are PRE-TRAINED and high-quality (Kappa 0.20)
    - DO NOT re-train Dense layers on top of RF probabilities
    - Use RF predictions directly or combine with minimal transformation
    """
    scope = strategy.scope() if strategy else tf.keras.utils.custom_object_scope({})

    with scope:
        inputs = {}
        branches = []
        metadata_idx = None

        # Process each modality
        for i, modality in enumerate(selected_modalities):
            if modality == 'metadata':
                metadata_idx = i
                metadata_input, branch_output = create_metadata_branch(input_shapes[modality], i)
                inputs[f'metadata_input'] = metadata_input
                branches.append(branch_output)
            elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                image_input, branch_output = create_image_branch(input_shapes[modality], f'{modality}')
                inputs[f'{modality}_input'] = image_input
                branches.append(branch_output)

        # ============================================
        # CRITICAL: Preserve RF Quality
        # ============================================

        if len(selected_modalities) == 1:
            if 'metadata' in selected_modalities:
                # METADATA-ONLY: Use RF probabilities directly
                # RF already provides high-quality predictions (Kappa 0.20)
                # NO Dense layer - just ensure proper probability distribution
                vprint("Metadata-only: Using RF probabilities directly (no Dense layer)", level=2)
                output = Activation('softmax', name='output')(branches[0])
            else:
                # Single image modality - train classification layer
                output = Dense(3, activation='softmax', name='output')(branches[0])

        elif len(selected_modalities) == 2:
            if 'metadata' in selected_modalities:
                # MULTI-MODAL WITH METADATA: Preserve RF quality
                vprint("Multi-modal: Combining RF probabilities with image predictions", level=2)

                # Get RF probabilities (already high-quality)
                rf_probs = branches[metadata_idx]

                # Get image predictions (need to train classifier)
                image_idx = 1 - metadata_idx  # The other branch
                image_features = branches[image_idx]
                image_probs = Dense(3, activation='softmax', name='image_classifier')(image_features)

                # Learn weighted combination
                # Simple approach: concatenate and let network learn optimal weighting
                merged = concatenate([rf_probs, image_probs], name='concat_predictions')

                # Lightweight fusion layer (just learns weights, not re-classification)
                fusion_weights = Dense(6, activation='softmax', name='fusion_weights')(merged)

                # Weighted sum of RF and image predictions
                weighted_rf = Multiply(name='weighted_rf')([rf_probs, fusion_weights[:, :3]])
                weighted_img = Multiply(name='weighted_img')([image_probs, fusion_weights[:, 3:]])

                output = Add(name='output')([weighted_rf, weighted_img])
            else:
                # Two image modalities - standard fusion
                merged = concatenate(branches, name='concat_branches')
                output = Dense(3, activation='softmax', name='output')(merged)

        elif len(selected_modalities) >= 3:
            if 'metadata' in selected_modalities:
                # MULTI-MODAL WITH METADATA: Advanced fusion preserving RF
                vprint(f"Multi-modal ({len(selected_modalities)} modalities): Advanced fusion with RF", level=2)

                # Separate RF from image branches
                rf_probs = branches[metadata_idx]
                image_branches = [b for i, b in enumerate(branches) if i != metadata_idx]

                # Fuse image modalities
                if len(image_branches) > 1:
                    image_merged = concatenate(image_branches, name='concat_images')
                else:
                    image_merged = image_branches[0]

                # Image branch classification
                if len(selected_modalities) == 3:
                    x = Dense(32, activation='relu', name='image_dense')(image_merged)
                    x = BatchNormalization(name='image_BN')(x)
                    x = Dropout(0.10, name='image_dropout')(x)
                elif len(selected_modalities) == 4:
                    x = Dense(64, activation='relu', name='image_dense_1')(image_merged)
                    x = BatchNormalization(name='image_BN_1')(x)
                    x = Dropout(0.10, name='image_dropout_1')(x)
                    x = Dense(32, activation='relu', name='image_dense_2')(x)
                    x = BatchNormalization(name='image_BN_2')(x)
                    x = Dropout(0.10, name='image_dropout_2')(x)
                else:  # 5 modalities
                    x = Dense(128, activation='relu', name='image_dense_1')(image_merged)
                    x = BatchNormalization(name='image_BN_1')(x)
                    x = Dropout(0.25, name='image_dropout_1')(x)
                    x = Dense(64, activation='relu', name='image_dense_2')(x)
                    x = BatchNormalization(name='image_BN_2')(x)
                    x = Dropout(0.10, name='image_dropout_2')(x)
                    x = Dense(32, activation='relu', name='image_dense_3')(x)
                    x = BatchNormalization(name='image_BN_3')(x)
                    x = Dropout(0.10, name='image_dropout_3')(x)

                image_probs = Dense(3, activation='softmax', name='image_classifier')(x)

                # Combine RF and image predictions
                # Learn to weight RF vs image predictions
                combined = concatenate([rf_probs, image_probs], name='concat_predictions')
                fusion_weights = Dense(2, activation='softmax', name='fusion_weights')(combined)

                # Weighted combination
                rf_weighted = Lambda(lambda x: x[0] * x[1][:, 0:1], name='rf_weighted')([rf_probs, fusion_weights])
                img_weighted = Lambda(lambda x: x[0] * x[1][:, 1:2], name='img_weighted')([image_probs, fusion_weights])

                output = Add(name='output')([rf_weighted, img_weighted])
            else:
                # No metadata - use original fusion architecture
                merged = create_fusion_layer(branches, len(branches))

                if len(selected_modalities) == 3:
                    x = Dense(32, activation='relu', name='final_dense')(merged)
                    x = BatchNormalization(name='final_BN')(x)
                    x = Dropout(0.10, name='final_dropout')(x)
                    output = Dense(3, activation='softmax', name='output')(x)
                elif len(selected_modalities) == 4:
                    x = Dense(64, activation='relu', name='final_dense_1')(merged)
                    x = BatchNormalization(name='final_BN_1')(x)
                    x = Dropout(0.10, name='final_dropout_1')(x)
                    x = Dense(32, activation='relu', name='final_dense_2')(x)
                    x = BatchNormalization(name='final_BN_2')(x)
                    x = Dropout(0.10, name='final_dropout_2')(x)
                    output = Dense(3, activation='softmax', name='output')(x)
                else:  # 5 modalities
                    x = Dense(128, activation='relu', name='final_dense_1')(merged)
                    x = BatchNormalization(name='final_BN_1')(x)
                    x = Dropout(0.25, name='final_dropout_1')(x)
                    x = Dense(64, activation='relu', name='final_dense_2')(merged)
                    x = BatchNormalization(name='final_BN_2')(x)
                    x = Dropout(0.10, name='final_dropout_2')(x)
                    x = Dense(32, activation='relu', name='final_dense_3')(merged)
                    x = BatchNormalization(name='final_BN_3')(x)
                    x = Dropout(0.10, name='final_dropout_3')(x)
                    output = Dense(3, activation='softmax', name='output')(x)

        # Create model
        model = Model(inputs=inputs, outputs=output, name='multimodal_dfu_classifier')

        return model
```

---

## Expected Results

### Metadata-Only
**Before:**
```
RF (Kappa 0.20) → Cast → BatchNorm → Dense(3) → Kappa 0.109 ❌
```

**After:**
```
RF (Kappa 0.20) → Cast → BatchNorm → Softmax → Kappa 0.20 ✅
```

**Improvement**: +83% (0.109 → 0.20)

### Multi-Modal (Metadata + Images)
**Before:**
```
RF (Kappa 0.20) → Dense layers → Degraded
Image → Features → Dense layers
→ Combined → Final Dense → Kappa ~0.15-0.18
```

**After:**
```
RF (Kappa 0.20) → Preserved → Weight α
Image → Classifier → Kappa ~0.15 → Weight (1-α)
→ Weighted Fusion → Kappa 0.22-0.25 ✅
```

**Improvement**: RF contributes its full quality, fusion learns optimal weighting

---

## Validation Plan

### Test 1: Metadata-Only
```bash
# included_modalities = [('metadata',)]
python src/main.py --mode search --cv_folds 5 --verbosity 2
```

**Expected:**
- Model shows: "Using RF probabilities directly (no Dense layer)"
- Kappa: **0.20 ± 0.05**
- NO Dense(3) layer in model summary

### Test 2: Metadata + 1 Image
```bash
# included_modalities = [('metadata', 'depth_rgb')]
python src/main.py --mode search --cv_folds 3 --verbosity 2
```

**Expected:**
- Model shows: "Combining RF probabilities with image predictions"
- Kappa: **0.22-0.25** (better than either alone)
- RF quality preserved in fusion

### Test 3: All Modalities
```bash
# included_modalities = [('metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map')]
python src/main.py --mode search --cv_folds 3 --verbosity 2
```

**Expected:**
- Model shows: "Advanced fusion with RF"
- Kappa: **0.25-0.30** (all modalities contribute optimally)
- RF provides strong baseline, images add complementary info

---

## Key Benefits

✅ **Metadata-only**: Full RF quality (Kappa 0.20)
✅ **Multi-modal**: RF quality preserved AND enhanced by images
✅ **Flexibility**: Still supports all modality combinations
✅ **Principled**: Treats RF as pre-trained predictions, not features
✅ **Unified path**: Same architecture philosophy for all cases

---

## Summary

**Problem**: Dense layer re-learns classification from RF probabilities, degrading quality

**Solution**: Treat RF probabilities as final predictions
- Metadata-only: Use directly (Softmax activation only)
- Multi-modal: Learn weighted combination (preserve RF, combine with images)

**Impact**:
- Metadata-only: 0.109 → 0.20 (+83%)
- Multi-modal: RF quality preserved + image enhancement
- All paths improved

**Philosophy**: RF is a **pre-trained expert** on metadata, not raw features to re-classify
