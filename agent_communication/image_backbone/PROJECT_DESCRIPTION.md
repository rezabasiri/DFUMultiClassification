# Image Backbone Comparison for DFU Classification

**Project:** Testing different CNN backbones for RGB and Map image branches
**Status:** 🚧 IN PROGRESS
**Date:** January 2026

---

## Mission

Compare simple CNN backbones vs EfficientNet variants to find optimal image feature extractors for:
- **RGB images** (depth_rgb, thermal_rgb)
- **Map images** (depth_map, thermal_map)

---

## Current Baseline

**RGB Branch:** Simple CNN
- 4 Conv2D layers: 256 → 128 → 64 → 32 filters
- GlobalAveragePooling2D
- Dense projection: 512 → 256 → 128 → 64
- Modular attention

**Map Branch:** Simple CNN
- 3 Conv2D layers: 128 → 64 → 32 filters
- GlobalAveragePooling2D
- Dense projection: 512 → 256 → 128 → 64
- Modular attention

**Current Performance** (metadata + thermal_map, 32x32, 15% outlier removal):
- Kappa: 0.2976 ± 0.08
- Accuracy: 0.5561
- F1 Macro: 0.4937

---

## Backbones to Test

### For RGB Images (depth_rgb, thermal_rgb)
1. ✅ **SimpleCNN** (baseline) - current implementation
2. 🔄 **EfficientNetB0** - lightweight, 5.3M params
3. 🔄 **EfficientNetB1** - slightly larger, 7.8M params
4. 🔄 **EfficientNetB3** - medium size, 12M params

### For Map Images (depth_map, thermal_map)
1. ✅ **SimpleCNN** (baseline) - current implementation
2. 🔄 **EfficientNetB0** - lightweight
3. 🔄 **EfficientNetB1** - slightly larger

**Why EfficientNet?**
- Efficient scaling (depth, width, resolution)
- Pre-trained on ImageNet (transfer learning)
- Good performance/size trade-off
- Already partially implemented (commented out in code)

---

## Test Configuration

### Fixed Parameters
- **Image size:** 32x32 (optimal from Phase 7)
- **Data percentage:** 30% (for quick testing)
- **Modality combination:** metadata + thermal_map
- **Outlier removal:** 15% (enabled)
- **Augmentation:** Disabled
- **Mode:** Single GPU, fresh resume

### What Changes
- RGB backbone (4 variants)
- Map backbone (3 variants)
- **Total tests:** 4 (RGB) × 3 (Map) = 12 combinations

### Test Matrix

| Test | RGB Backbone | Map Backbone | Expected Runtime |
|------|--------------|--------------|------------------|
| 1 | SimpleCNN | SimpleCNN | ~10 min (baseline) |
| 2 | SimpleCNN | EfficientNetB0 | ~12 min |
| 3 | SimpleCNN | EfficientNetB1 | ~12 min |
| 4 | EfficientNetB0 | SimpleCNN | ~12 min |
| 5 | EfficientNetB0 | EfficientNetB0 | ~15 min |
| 6 | EfficientNetB0 | EfficientNetB1 | ~15 min |
| 7 | EfficientNetB1 | SimpleCNN | ~12 min |
| 8 | EfficientNetB1 | EfficientNetB0 | ~15 min |
| 9 | EfficientNetB1 | EfficientNetB1 | ~15 min |
| 10 | EfficientNetB3 | SimpleCNN | ~15 min |
| 11 | EfficientNetB3 | EfficientNetB0 | ~18 min |
| 12 | EfficientNetB3 | EfficientNetB1 | ~18 min |

**Total estimated time:** ~3 hours

---

## Implementation Approach

### 1. Configuration System
Add to `production_config.py`:
```python
# Image backbone selection
RGB_BACKBONE = 'SimpleCNN'  # Options: 'SimpleCNN', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB3'
MAP_BACKBONE = 'SimpleCNN'  # Options: 'SimpleCNN', 'EfficientNetB0', 'EfficientNetB1'
```

### 2. Modify create_image_branch()
In `src/models/builders.py`, add logic to select backbone based on config:
```python
def create_image_branch(input_shape, modality):
    from src.utils.production_config import RGB_BACKBONE, MAP_BACKBONE

    image_input = Input(shape=input_shape, name=f'{modality}_input')

    # Select backbone based on modality type
    if modality in ['depth_rgb', 'thermal_rgb']:
        backbone = RGB_BACKBONE
    else:
        backbone = MAP_BACKBONE

    # Create feature extractor
    if backbone == 'SimpleCNN':
        x = create_simple_cnn(image_input, modality)
    elif backbone.startswith('EfficientNet'):
        x = create_efficientnet_branch(image_input, modality, backbone)

    # ... rest of projection layers and attention ...
```

### 3. Automated Test Script
`agent_communication/image_backbone/test_backbones.py`:
- Loops through all backbone combinations
- Updates production_config.py
- Runs training with `--data-percentage 30`
- Collects results (Kappa, accuracy, F1, runtime)
- Generates comparison report

---

## Success Metrics

**Primary:** Cohen's Kappa
**Secondary:** Accuracy, F1 Macro, Training time, Model size

**Success Criteria:**
- Improvement > 3% Kappa → worth adopting
- Improvement 1-3% Kappa → marginal, consider complexity
- Improvement < 1% Kappa → not worth complexity

**Current Baseline:** 0.2976 Kappa (100% data)
**30% Data Baseline:** TBD (need to measure first)

---

## Files in This Folder

### Documentation
- `PROJECT_DESCRIPTION.md` - This file, experiment overview
- `RUN_BACKBONE_TESTS.txt` - Step-by-step instructions for local agent

### Test Script
- `test_backbones.py` - Automated test runner (run all 12 combinations)

### Results
- `BACKBONE_RESULTS.txt` - Test results (generated after running tests)

## Code Modified in Project

- `src/utils/production_config.py` - Added RGB_BACKBONE and MAP_BACKBONE flags
- `src/models/builders.py` - Refactored to support configurable backbones

---

## Expected Outcomes

### Hypothesis
EfficientNet should outperform SimpleCNN due to:
- Better feature extraction (deeper, wider network)
- Transfer learning from ImageNet
- Optimized architecture design

### Realistic Expectations
- **Conservative:** +2-5% Kappa improvement
- **Optimistic:** +5-10% Kappa improvement
- **Model size:** Larger (5-12M params vs ~1M for SimpleCNN)
- **Training time:** 20-50% longer

### Trade-offs
If EfficientNet improves Kappa by <3%, may not be worth:
- Increased model size
- Slower inference
- More complex maintenance

---

**Next Step:** Run automated test script and analyze results
