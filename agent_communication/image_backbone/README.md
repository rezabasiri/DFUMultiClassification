# Image Backbone Comparison

**Mission:** Find optimal CNN backbone for RGB and map image branches
**Status:** 🚧 READY TO TEST
**Date:** January 2026

---

## Quick Start

Run automated tests (all 12 combinations):
```bash
python agent_communication/image_backbone/test_backbones.py
```

**Time:** ~3 hours (12 tests × 15 min)
**Output:** `BACKBONE_RESULTS.txt`

---

## What's Being Tested

### Backbones
- **RGB images** (depth_rgb, thermal_rgb): SimpleCNN, EfficientNetB0, B1, B3
- **Map images** (depth_map, thermal_map): SimpleCNN, EfficientNetB0, B1
- **Total:** 4 × 3 = 12 combinations

### Current Baseline
- **Architecture:** SimpleCNN (4-layer RGB, 3-layer map)
- **Performance (100% data):** Kappa 0.2976
- **Expected (30% data):** Kappa 0.24-0.28

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Data | 30% (quick testing) |
| Image size | 32×32 |
| Outlier removal | 15% (enabled) |
| Augmentation | Disabled |
| Device | Single GPU |
| Resume mode | Fresh |

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `README.md` | This file, quick reference |
| `PROJECT_DESCRIPTION.md` | Detailed experiment overview |
| `RUN_BACKBONE_TESTS.txt` | Step-by-step instructions |
| `test_backbones.py` | Automated test script |
| `BACKBONE_RESULTS.txt` | Results (after running tests) |

---

## Code Changes Made

### Configuration (`src/utils/production_config.py`)
```python
RGB_BACKBONE = 'SimpleCNN'  # Options: SimpleCNN, EfficientNetB0, B1, B3
MAP_BACKBONE = 'SimpleCNN'  # Options: SimpleCNN, EfficientNetB0, B1
```

### Model Builder (`src/models/builders.py`)
Refactored `create_image_branch()` to support:
- `create_simple_cnn_rgb()` - Baseline 4-layer CNN
- `create_simple_cnn_map()` - Baseline 3-layer CNN
- `create_efficientnet_branch()` - EfficientNetB0/B1/B3 with ImageNet weights
- Automatic selection based on modality type and config

---

## Expected Results

### Best Case
- EfficientNet provides +5-10% Kappa improvement
- Worth adopting despite larger model size

### Realistic Case
- EfficientNet provides +2-5% Kappa improvement
- Consider trade-offs (model size, inference speed)

### Worst Case
- EfficientNet provides <1% Kappa improvement
- Not worth complexity, keep SimpleCNN

---

## Success Criteria

| Improvement | Decision |
|-------------|----------|
| > 3% Kappa | ✅ Adopt EfficientNet |
| 1-3% Kappa | 🤔 Marginal, evaluate trade-offs |
| < 1% Kappa | ❌ Keep SimpleCNN |

---

## Manual Testing (if automated script fails)

For each combination:

1. Edit `src/utils/production_config.py`:
   ```python
   RGB_BACKBONE = 'EfficientNetB0'  # or SimpleCNN, B1, B3
   MAP_BACKBONE = 'SimpleCNN'       # or EfficientNetB0, B1
   ```

2. Run training:
   ```bash
   python src/main.py --mode search --device-mode single \
     --resume-mode fresh --data-percentage 30
   ```

3. Record: Kappa, accuracy, F1 macro, runtime

4. Repeat for all 12 combinations

---

## Next Steps After Testing

1. Analyze results in `BACKBONE_RESULTS.txt`
2. If improvement > 3%, update production config to use best backbone
3. Rerun with 100% data to confirm improvement holds
4. If improvement < 3%, keep SimpleCNN

---

**See:** `RUN_BACKBONE_TESTS.txt` for detailed instructions
