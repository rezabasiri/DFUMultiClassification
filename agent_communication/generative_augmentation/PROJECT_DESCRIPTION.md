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
- **Model location**: `Codes/MultimodalClassification/ImageGeneration/models_5_7/`
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

- [ ] Verify model files exist in `ImageGeneration/models_5_7/`
- [ ] Check which modality/phase combinations have trained models
- [ ] Document model specifications (steps, resolution, etc.)
- [ ] Review current augmentation config in production_config.py
- [ ] Identify what needs to be enabled for testing

## Testing TODO

- [ ] Design comparison matrix:
  - Baseline (no generative aug)
  - Generative aug ON (all modalities)
  - Per-modality effectiveness (rgb only, thermal only, etc.)
  - Mix ratio optimization
- [ ] Create test script similar to `test_backbones.py`
- [ ] Run controlled experiments with same data/config
- [ ] Track metrics: Kappa, F1, accuracy, runtime overhead

---

## Key Questions

1. Do the trained Stable Diffusion models still exist?
2. What's the quality of generated samples?
3. Does generative aug help all modalities equally?
4. What's the computational cost vs performance gain?
5. Optimal mix ratio between real and synthetic samples?

---

## Files in This Folder

| File | Purpose |
|------|---------|
| **PROJECT_DESCRIPTION.md** | This document |
| **INVESTIGATION_NOTES.md** | Detailed findings from code review |
| **model_check.txt** | Model availability check results |
