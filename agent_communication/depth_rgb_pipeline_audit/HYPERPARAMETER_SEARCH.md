# Depth RGB Hyperparameter Search Plan

## Goal
Find the best architecture + training configuration for `depth_rgb` standalone to maximize **val_cohen_kappa** (currently ~0.24 across 3 folds).

## Current Baseline
- **Backbone**: EfficientNetB0 (frozen), head-only training
- **Head**: Dense(128, relu) + BN + Dropout(0.3) + Dense(3, softmax)
- **Loss**: Focal loss (gamma=2.0, alpha=frequency-based, ordinal_weight=0.0)
- **LR**: 1e-3, ReduceLROnPlateau(factor=0.5, patience=7), EarlyStopping(patience=15)
- **Epochs**: 50 (Stage 1 only, no fine-tuning)
- **Augmentation**: General augmentation ON (brightness, contrast, saturation, noise)
- **Batch size**: 64
- **Image size**: 256x256
- **Val Kappa**: ~0.24 (3-fold average)

## Search Dimensions (12 axes, ~40 configs)

### 1. Backbone Choice (3 options)
| Config | Backbone | Rationale |
|--------|----------|-----------|
| SimpleCNN | 4-layer CNN from scratch | No transfer learning overhead, fewer params |
| EfficientNetB0 | 4M params, ImageNet | Current default, good transfer |
| EfficientNetB2 | 9M params, ImageNet | More capacity, risk of overfitting |

### 2. Freeze Strategy (3 options)
| Config | Strategy | Rationale |
|--------|----------|-----------|
| frozen | Backbone fully frozen | Current default, safe with ~2K images |
| partial_unfreeze | Unfreeze top 20% of backbone | Middle ground |
| full_unfreeze | Fully trainable, very low LR | Maximum adaptation |

### 3. Head Architecture (4 options)
| Config | Head | Rationale |
|--------|------|-----------|
| small | Dense(64) + BN + Drop(0.3) | Fewer params, more regularization |
| medium | Dense(128) + BN + Drop(0.3) | Current default |
| large | Dense(256) + BN + Drop(0.3) | More capacity |
| two_layer | Dense(256) + Dense(64) + BN + Drop | Deeper head |

### 4. Dropout Rate (3 options)
- 0.2, 0.3 (current), 0.5

### 5. Learning Rate (3 options)
- 5e-4, 1e-3 (current), 3e-3

### 6. Loss Function (3 options)
| Config | Loss | Rationale |
|--------|------|-----------|
| focal_g2 | Focal gamma=2.0 | Current default |
| focal_g3 | Focal gamma=3.0 | Harder example focus |
| cce | Categorical crossentropy | Simpler, no focal weighting |

### 7. Alpha Normalization Sum (2 options)
- 3.0 (current), 1.0 (standard)

### 8. Augmentation On/Off (2 options)
- ON (current), OFF

### 9. Label Smoothing (2 options)
- 0.0 (current), 0.1

### 10. Batch Size (2 options)
- 32, 64 (current)

### 11. Image Size (2 options)
- 224 (EfficientNet native), 256 (current)

### 12. Epochs / Patience
- Stage1 50 (current) vs 100 (longer training)

## Search Strategy

**NOT full grid search** (would be 3x3x4x3x3x3x2x2x2x2x2 = 15,552 configs).

Instead: **Sequential elimination** in 5 rounds:

1. **Round 1 — Backbone + Freeze** (9 configs): Test 3 backbones x 3 freeze strategies with default head
2. **Round 2 — Head Architecture** (4 configs): Best backbone/freeze from R1 + 4 head options
3. **Round 3 — Loss + Regularization** (6 configs): Best from R2 + loss/dropout/label-smoothing combos
4. **Round 4 — Training Dynamics** (4 configs): Best from R3 + LR/batch-size/epochs combos
5. **Round 5 — Augmentation + Image Size** (4 configs): Best from R4 + aug/image-size combos

**Total: ~27-30 configs**, each runs 1 fold only (fold 1) for speed. Final best config runs all 3 folds.

## Output
- CSV with all results: `results/depth_rgb_search_results.csv`
- Best config summary printed to stdout
- Best config's 3-fold kappa at the end
