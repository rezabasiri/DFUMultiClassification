# Generative Augmentation Effectiveness Study

**Mission:** Determine effectiveness of Stable Diffusion-based generative augmentation
**Status:** 🚧 INVESTIGATION
**Date:** January 17, 2026

---

## Objective

Evaluate impact of generative augmentation on DFU classification performance and identify optimal configuration.

## How It Works

### Architecture
- **Fine-tuned Stable Diffusion models** trained per modality and healing phase
- **Model location**: `results/GenerativeAug_Models/` (43 GB)
- **Organization**: `{modality}_{phase}/` subdirectories
  - Modalities: `rgb`, `thermal_map`, `depth_map`
  - Phases: `I` (Inflammatory), `P` (Proliferative), `R` (Remodeling)
  - Example: `rgb_I/`, `thermal_map_P/`, etc.

### Implementation
- `GenerativeAugmentationManager` loads models from base directory
- Uses `StableDiffusionPipeline` from diffusers library
- LRU cache limits GPU memory (max 3 models loaded)
- Prompt generator adds medical context
- **Current status**: Disabled in production config

---

## Investigation TODO

- [x] Verify model files exist in `results/GenerativeAug_Models/models_5_7/`
- [x] Check which modality/phase combinations have trained models (12 found, 9 expected + 3 extras)
- [x] Document model specifications (Diffusers v0.31.0, 4GB each, 48GB total)
- [x] Review current augmentation config in production_config.py
- [x] Identify what needs to be enabled for testing

**Configuration Location:** `src/utils/production_config.py` lines 70-79
- Set `USE_GENERATIVE_AUGMENTATION = True` to enable
- Adjust `GENERATIVE_AUG_PROB`, `GENERATIVE_AUG_MIX_RATIO`, `GENERATIVE_AUG_INFERENCE_STEPS` as needed
- Model path: Hardcoded in code (needs update to new location)

## Testing

**Status:** ✅ Ready to test

**For Local Agent:** See **README.md** for quick start instructions

**Test Configuration:**
- Modalities: metadata, depth_rgb, thermal_map, depth_map (fixed)
- Test 1: Baseline (USE_GENERATIVE_AUGMENTATION=False)
- Test 2: With gen aug (USE_GENERATIVE_AUGMENTATION=True, depth_rgb only)
- Automatically generates comparison report

---

## Files

| File | Purpose |
|------|---------|
| **README.md** | Quick start guide for local agent |
| **test_generative_aug.py** | Automated test script |
| **INVESTIGATION_NOTES.md** | Technical code review and configuration details |
| **MODEL_INSPECTION_REPORT.txt** | Model specifications (48 GB, 12 variants) |
| **gengen_test.log** | Live test log (generated during test, synced to git) |
| **GENGEN_PROGRESS.json** | Progress tracker (generated during test, resumable) |
| **GENGEN_REPORT.txt** | Final effectiveness report (generated after test) |
