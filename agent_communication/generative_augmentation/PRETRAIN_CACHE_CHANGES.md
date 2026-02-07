# Pre-train Cache & Config Changes (2026-02-07)

## Overview

Three changes were made:
1. **Pre-trained depth_rgb weights caching** - saves ~50 min on subsequent `--resume_mode auto` runs
2. **GENERATIVE_AUG_PROB**: 0.05 → 0.10
3. **OUTLIER_CONTAMINATION**: 0.15 → 0.20

---

## How to Revert

### Quick revert (config values only)

In `src/utils/production_config.py`:
- Line 88: `OUTLIER_CONTAMINATION = 0.20` → `0.15`
- Line 103: `GENERATIVE_AUG_PROB = 0.10` → `0.05`

### Full revert (all cache logic)

Apply changes below in order.

---

## File 1: `src/utils/production_config.py`

### Change A: OUTLIER_CONTAMINATION (line 88)
**Before:** `OUTLIER_CONTAMINATION = 0.15`
**After:** `OUTLIER_CONTAMINATION = 0.20`

### Change B: GENERATIVE_AUG_PROB (line 103)
**Before:** `GENERATIVE_AUG_PROB = 0.05`
**After:** `GENERATIVE_AUG_PROB = 0.10`

---

## File 2: `src/utils/config.py`

### Change C: Pretrain cache cleanup in fresh mode (line ~191)

**Before:**
```python
# Delete TensorFlow cache
delete_files([
    os.path.join(result_dir, 'tf_cache_*'),
    os.path.join(result_dir, 'tf_records/*'),
], 'tf_cache')
```

**After:**
```python
# Delete TensorFlow cache and pretrain cache
delete_files([
    os.path.join(result_dir, 'tf_cache_*'),
    os.path.join(result_dir, 'tf_records/*'),
    os.path.join(result_dir, 'pretrain_cache', '*.ckpt*'),
], 'tf_cache')
```

**To revert:** Remove the `os.path.join(result_dir, 'pretrain_cache', '*.ckpt*'),` line and change comment back.

---

## File 3: `src/training/training_utils.py`

### Change D: Added imports (lines 35-36)

**Before:**
```python
    GENERATIVE_AUG_MODEL_PATH, GENERATIVE_AUG_VERSION,
    GENERATIVE_AUG_SDXL_MODEL_PATH
)
```

**After:**
```python
    GENERATIVE_AUG_MODEL_PATH, GENERATIVE_AUG_VERSION,
    GENERATIVE_AUG_SDXL_MODEL_PATH,
    OUTLIER_CONTAMINATION, GENERATIVE_AUG_PROB
)
```

**To revert:** Remove `,\n    OUTLIER_CONTAMINATION, GENERATIVE_AUG_PROB` from the import.

---

### Change E: Module-level variable + helper function (after line 63)

**Added** (between `logs_path = ...` and `class EpochMemoryCallback`):
```python
pretrain_cache_dir = os.path.join(result_dir, 'pretrain_cache')


def _get_pretrain_cache_path(image_modality, fold_num):
    """Get config-aware cache path for pre-trained image weights.

    Cache key includes all settings that affect pre-training data/model.
    If any of these change, a different cache file is used automatically.
    """
    import hashlib
    cache_key = (f"{image_modality}_fold{fold_num}_img{IMAGE_SIZE}_data{DATA_PERCENTAGE}"
                 f"_outlier{OUTLIER_CONTAMINATION}"
                 f"_genaug{USE_GENERATIVE_AUGMENTATION}_{GENERATIVE_AUG_PROB}")
    config_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    os.makedirs(pretrain_cache_dir, exist_ok=True)
    return os.path.join(pretrain_cache_dir, f'pretrain_{image_modality}_fold{fold_num}_{config_hash}.ckpt')
```

**To revert:** Delete the `pretrain_cache_dir = ...` line, the blank lines, and the entire `_get_pretrain_cache_path` function.

---

### Change F: Restructured pre-training flow (lines ~1247-1285)

**Before** (original structure):
```python
                                    fusion_use_pretrained = False
                            else:
                                # AUTOMATIC PRE-TRAINING: Train image-only model inline
                                vprint("=" * 80, level=1)
                                vprint(f"AUTOMATIC PRE-TRAINING: {image_modality} weights not found", level=1)
                                vprint(f"  Training {image_modality}-only model first (same data split)...", level=1)
                                vprint("=" * 80, level=1)

                                try:
```

**After** (cache check + conditional auto-pretrain):
```python
                                    fusion_use_pretrained = False

                            # Check pretrain cache before training from scratch
                            if not fusion_use_pretrained:
                                pretrain_cache_path = _get_pretrain_cache_path(image_modality, run+1)
                                cache_ckpt, cache_fmt = find_checkpoint_for_loading(pretrain_cache_path)

                                if cache_fmt is not None:
                                    vprint("=" * 80, level=1)
                                    vprint(f"CACHED PRE-TRAINING: Loading {image_modality} weights from cache", level=1)
                                    vprint(f"  Cache: {pretrain_cache_path}", level=2)
                                    vprint("=" * 80, level=1)
                                    try:
                                        temp_model = create_multimodal_model(input_shapes, [image_modality], None)
                                        temp_model.load_weights(cache_ckpt)
                                        for layer in temp_model.layers:
                                            if image_modality in layer.name or layer.name == 'output':
                                                try:
                                                    fusion_layer = model.get_layer(layer.name)
                                                    fusion_layer.set_weights(layer.get_weights())
                                                except:
                                                    continue
                                        vprint(f"  STAGE 1: Freezing {image_modality} branch...", level=2)
                                        for layer in model.layers:
                                            if image_modality in layer.name or 'image_classifier' in layer.name:
                                                layer.trainable = False
                                        del temp_model
                                        fusion_use_pretrained = True
                                        vprint(f"  Loaded from cache - skipping ~17min pre-training!", level=1)
                                    except Exception as e:
                                        vprint(f"  Cache load failed: {e}. Will train from scratch.", level=1)

                            # AUTOMATIC PRE-TRAINING: Train image-only model inline
                            if not fusion_use_pretrained:
                                vprint("=" * 80, level=1)
                                vprint(f"AUTOMATIC PRE-TRAINING: {image_modality} weights not found", level=1)
                                vprint(f"  Training {image_modality}-only model first (same data split)...", level=1)
                                vprint("=" * 80, level=1)

                                try:
```

**To revert:** Replace the entire block (from `# Check pretrain cache` through the `try:`) with the original `else:` block shown above.

---

### Change G: Cache save after pre-training (lines ~1399-1405)

**Added** (right before `del pretrain_model, ...`):
```python
                                    # Save pre-trained weights to cache for future runs
                                    try:
                                        pretrain_cache_path = _get_pretrain_cache_path(image_modality, run+1)
                                        pretrain_model.save_weights(pretrain_cache_path)
                                        vprint(f"  Saved pre-trained weights to cache: {pretrain_cache_path}", level=2)
                                    except Exception as e:
                                        vprint(f"  Warning: Could not save to pretrain cache: {e}", level=2)
```

**To revert:** Delete these 7 lines.

---

## Cache Behavior Summary

| Resume Mode | Cache Used? | Cache Cleared? | Cache Saved? |
|-------------|-------------|----------------|--------------|
| `--resume_mode fresh` | No (cleared first) | Yes | Yes (for next run) |
| `--resume_mode auto` | Yes (if exists) | No | Yes (on miss) |

- **Cache directory**: `results/pretrain_cache/`
- **Cache key**: `{modality}_fold{N}_img{size}_data{pct}_outlier{cont}_genaug{bool}_{prob}`
- **Hash**: MD5 of cache key, first 12 chars
- **Example file**: `pretrain_depth_rgb_fold1_a1b2c3d4e5f6.ckpt`
- **Auto-invalidation**: Any config change (image size, data%, outlier%, gen aug settings) produces a different hash → cache miss → retrain

## To delete cache manually

```bash
rm -rf results/pretrain_cache/
```
