# Multimodal Outlier Detection for DFU Classification

## Mission

Implement **combination-specific outlier detection** that detects outliers in the joint feature space of each modality combination being tested, using features extracted from the actual training pipeline models.

---

## Problem Context

### Background from Phase 1-7 Investigation

**Previous work (metadata-only outlier removal):**
- Phase 7 investigation: 15% metadata outlier removal → Kappa 0.27 (+63% vs baseline 0.1664)
- Used Isolation Forest on metadata features only (73 tabular features)
- **Limitation:** Only works when metadata is present, ignores image-based outliers

**New requirement:**
- Detect outliers for ANY modality combination (with or without metadata)
- Examples:
  - `depth_rgb + depth_map` → detect outliers in joint visual space
  - `metadata + thermal_map` → detect outliers in joint clinical+visual space
  - `thermal_map` → detect outliers in thermal-only space

---

## Key Requirements

### 1. Combination-Specific Detection
- **NOT** per-modality: Don't detect outliers in `depth_rgb` and `depth_map` separately
- **YES** per-combination: Detect outliers in combined `depth_rgb + depth_map` feature space
- Each combination gets its own outlier detection in joint feature space

### 2. Works Without Metadata
- Must work for image-only combinations: `thermal_map`, `depth_rgb + depth_map`, etc.
- Metadata is optional, not required

### 3. Uses Training Pipeline Models
- Extract features using **same architecture** as main.py training
- Use **same preprocessing** (`load_and_preprocess_image`)
- Use **same model backbones** (`create_image_branch`)
- Ensures consistency and semantic meaningfulness

---

## Technical Approach

### Feature Extraction Strategy (Hybrid)

**For each modality:**

1. **Metadata:**
   - Extract tabular features (73 features)
   - Same as Phase 7 implementation

2. **Image modalities** (thermal_map, depth_rgb, depth_map, thermal_rgb):
   - Load image using training pipeline preprocessing
   - Create feature extractor from `create_image_branch()` architecture
   - Extract from global pooling layer (1280-2048 dims)
   - Use pre-trained ImageNet weights (or trained checkpoint if available)

**For each combination:**
- Concatenate features from all modalities in that combination
- Run per-class Isolation Forest on joint feature space
- Save combination-specific cleaned dataset

**Example dimensions:**
- `metadata`: 73 features
- `thermal_map`: 1280 features (EfficientNet global pooling)
- `metadata + thermal_map`: 73 + 1280 = 1353 features
- `depth_rgb + depth_map`: 1280 + 1280 = 2560 features

---

## Implementation Components

### 1. Cache System (`cache_outlier/`)

**Purpose:** Pre-compute and cache deep features for each modality

**Structure:**
```
cache_outlier/
├── metadata_features.npy          # (3107, 73)
├── thermal_map_features.npy       # (3107, 1280)
├── depth_rgb_features.npy         # (3107, 1280)
├── depth_map_features.npy         # (3107, 1280)
└── thermal_rgb_features.npy       # (3107, 1280)
```

**Benefits:**
- Extract once, reuse for all combinations
- Fast combination-specific detection
- Consistent features across runs

### 2. Standalone Cache Builder (`scripts/precompute_outlier_features.py`)

**Purpose:** Pre-compute feature cache before training

**Workflow:**
1. Load best_matching.csv
2. For each image modality:
   - Create feature extractor using `create_image_branch()` (same as training)
   - Load images using `load_and_preprocess_image()` (same preprocessing)
   - Extract features from global pooling layer
   - Save to `cache_outlier/{modality}_features.npy`
3. For metadata:
   - Extract tabular features
   - Save to `cache_outlier/metadata_features.npy`

**Usage:**
```bash
# Pre-compute all features (run once)
python scripts/precompute_outlier_features.py --image-size 32 --modalities all

# Pre-compute specific modalities
python scripts/precompute_outlier_features.py --modalities metadata thermal_map
```

### 3. Per-Combination Detection (`src/utils/outlier_detection.py`)

**Updated functions:**
- `extract_modality_features()`: Load from cache or extract on-the-fly
- `detect_outliers_combination()`: Detect outliers for specific combination
- Uses cached features for speed

**Workflow:**
1. For combination `('metadata', 'thermal_map')`:
   - Load `metadata_features.npy` (73 dims)
   - Load `thermal_map_features.npy` (1280 dims)
   - Concatenate → 1353 dims
   - Run Isolation Forest per-class on joint space
   - Save to `data/cleaned/metadata_thermal_map_15pct.csv`

### 4. Main.py Integration

**CLI flags:**
```bash
--outlier-removal              # Enable (default: True)
--no-outlier-removal           # Disable
--outlier-contamination 0.15   # Contamination rate (default: 0.15)
--outlier-modalities auto      # Auto-detect from combination (default)
```

**Auto-detection:**
- For each combination being tested, automatically detect and apply outliers
- Uses modalities in that specific combination
- No manual specification needed

---

## Relevant Context from Previous Work

### Phase 7 Investigation (Metadata-Only)

**Key findings:**
- 15% outlier removal optimal (0.2714 Kappa, 97% of seed 789 target)
- Per-class Isolation Forest important (protects minority class R)
- Minority class R: max 10% contamination, hard limit 20% removal
- Contamination range: 0.05 (conservative) to 0.20 (aggressive)

**Code patterns to preserve:**
```python
# Per-class detection
for cls in ['I', 'P', 'R']:
    cls_mask = (y == cls)
    X_cls = X[cls_mask]

    # Safety for minority class R
    cls_contamination = contamination
    if cls == 'R' and len(X_cls) < 150:
        cls_contamination = min(contamination, 0.10)

    # Safety check: max 20% removal for R
    max_allowed = int(len(X_cls) * 0.20)
    if cls == 'R' and n_outliers > max_allowed:
        n_outliers = max_allowed
```

### Training Pipeline Architecture

**Image branch** (`src/models/builders.py`):
- Uses transfer learning (EfficientNet or ResNet)
- Pre-trained ImageNet weights
- Global pooling layer before final dense layers
- Input: `(image_size, image_size, 3)`
- Output: `(1280,)` or `(2048,)` depending on backbone

**Preprocessing** (`src/data/image_processing.py`):
- `load_and_preprocess_image(path, bb_coords, modality, target_size, augment)`
- Handles bounding box cropping
- Normalizes to [0, 1] or [-1, 1] depending on backbone
- Resizes to target_size (typically 32x32 or 128x128)

### Production Configuration

**From `src/utils/production_config.py`:**
```python
IMAGE_SIZE = 32  # Optimal for fusion (Phase 7 finding)
SAMPLING_STRATEGY = 'combined'  # Undersample P + oversample R to middle
RANDOM_SEED = 42  # For reproducibility
```

---

## File Locations

### Created/Modified Files

**New:**
- `agent_communication/outlier_detection/PROJECT_DESCRIPTION.md` (this file)
- `agent_communication/outlier_detection/test_feature_extraction.py` (test script)
- `scripts/precompute_outlier_features.py` (cache builder)
- `cache_outlier/` (feature cache directory)

**Modified:**
- `src/utils/outlier_detection.py` (add sophisticated feature extraction)
- `src/main.py` (integrate per-combination detection)

**Reference:**
- `agent_communication/fusion_fix/FUSION_FIX_GUIDE.md` (Phase 7 results)
- `agent_communication/fusion_fix/scripts_production/detect_outliers.py` (original metadata-only)

---

## Expected Results

### Performance Goals

**Baseline (metadata-only, Phase 7):**
- 15% metadata outlier removal → Kappa 0.27 ± 0.08

**With multimodal outlier detection:**
- Image-only combinations: Expected 5-15% improvement (catches visual outliers)
- Fusion combinations: Expected 10-20% improvement (catches both clinical and visual outliers)
- Goal: Kappa 0.28-0.32 for fusion with multimodal outlier removal

### Cache Performance

**Cache size estimates:**
- metadata_features.npy: ~1.7 MB (3107 samples × 73 features × 8 bytes)
- thermal_map_features.npy: ~31 MB (3107 × 1280 × 8 bytes)
- Total for all 5 modalities: ~125 MB

**Speed:**
- First run: ~2-3 min to compute all features (one-time cost)
- Subsequent runs: <1 sec to load from cache
- Per-combination detection: ~1-2 sec (fast, cached features)

---

## Testing Strategy

### 1. Feature Extraction Test
```bash
# Test that extracted features match training pipeline
python agent_communication/outlier_detection/test_feature_extraction.py
```

**Validates:**
- Same preprocessing as training
- Same architecture as training
- Correct feature dimensions
- Reproducibility

### 2. Cache Building Test
```bash
# Build cache for all modalities
python scripts/precompute_outlier_features.py --modalities all --image-size 32
```

**Validates:**
- All modalities processed
- Cache files created
- Correct shapes

### 3. Integration Test
```bash
# Test on single combination
python src/main.py --mode search --cv_folds 1 --outlier-contamination 0.15
# In production_config.py: INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

**Validates:**
- Outliers detected for combination
- Cleaned dataset applied
- Training completes successfully

---

## Common Issues and Solutions

### Issue: Features don't match training pipeline
**Solution:** Use `create_image_branch()` directly, extract from exact layer name

### Issue: Cache files too large
**Solution:** Use float32 instead of float64 (halves size)

### Issue: Different image sizes
**Solution:** Cache must match training IMAGE_SIZE in production_config.py

### Issue: Missing trained checkpoints
**Solution:** Hybrid approach falls back to ImageNet pre-trained weights (still semantic)

---

## Quick Reference

### Commands

```bash
# 1. Pre-compute features (one-time setup)
python scripts/precompute_outlier_features.py --image-size 32 --modalities all

# 2. Run training with multimodal outlier removal (default: enabled)
python src/main.py --mode search --cv_folds 3 --verbosity 2

# 3. Disable outlier removal (test baseline)
python src/main.py --mode search --no-outlier-removal

# 4. Custom contamination
python src/main.py --mode search --outlier-contamination 0.10
```

### Key Parameters

- `contamination`: 0.05 (conservative) to 0.20 (aggressive), default 0.15
- `image_size`: Must match production_config.py IMAGE_SIZE (typically 32)
- `cache_dir`: `cache_outlier/` (feature cache)
- `cleaned_dir`: `data/cleaned/` (cleaned datasets)

---

## Success Criteria

✅ Feature extraction uses training pipeline architecture
✅ Cache system works (fast, reusable)
✅ Per-combination detection functional
✅ Works for any modality combination (with/without metadata)
✅ Performance improvement over metadata-only baseline
✅ Integrated into main.py with CLI flags
✅ Documentation complete and concise

---

**Status:** Implementation in progress
**Last updated:** 2026-01-06
**Related:** Phase 1-7 fusion investigation (agent_communication/fusion_fix/)
