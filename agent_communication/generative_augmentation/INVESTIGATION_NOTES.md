# Generative Augmentation - Code Investigation

**Date:** January 17, 2026
**Investigator:** Claude
**Source Files:** `src/data/generative_augmentation_v2.py`, `src/main_original.py`

---

## System Architecture

### Core Components

1. **GenerativeAugmentationManager** (`generative_augmentation_v2.py`)
   - Manages Stable Diffusion model loading and caching
   - Constructor: `__init__(self, base_dir, config, device=None)`
   - Uses LRU cache to limit loaded models (max 3 simultaneous)
   - Loads models from: `base_dir / "{modality}_{phase}"`

2. **AugmentationConfig** (`generative_augmentation_v2.py`)
   - Per-modality configuration class
   - Key settings:
     - `enabled`: Boolean to turn on/off per modality
     - `inference_steps`: Number of diffusion steps
     - `prob`: Probability of applying augmentation
     - `mix_ratio_range`: Range for mixing real/synthetic samples

### Model Organization

**Base Directory** (from `main_original.py` line 2817):
```
Original: Codes/MultimodalClassification/ImageGeneration/models_5_7/
Current:  results/GenerativeAug_Models/
```

**Subdirectory Structure**:
```
models_5_7/
├── rgb_I/              # RGB Inflammatory
├── rgb_P/              # RGB Proliferative
├── rgb_R/              # RGB Remodeling
├── thermal_map_I/      # Thermal map Inflammatory
├── thermal_map_P/      # Thermal map Proliferative
├── thermal_map_R/      # Thermal map Remodeling
├── depth_map_I/        # Depth map Inflammatory
├── depth_map_P/        # Depth map Proliferative
└── depth_map_R/        # Depth map Remodeling
```

Each subdirectory contains a fine-tuned Stable Diffusion model.

---

## Model Loading Process

**Code snippet from `generative_augmentation_v2.py`:**
```python
pipeline = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to(self.device)
```

**Key Details:**
- Uses FP16 for memory efficiency
- Safety checker disabled (medical images, not general use)
- Models loaded on-demand to GPU
- LRU cache evicts least-recently-used models when limit reached

---

## Current Configuration Status

**Location:** `src/utils/production_config.py`
- No explicit generative augmentation flags found in current config
- Likely controlled through modality-specific config passed to manager
- Based on code review, currently **DISABLED** for all modalities

**Original Usage** (from `main_original.py`):
```python
gen_manager = GenerativeAugmentationManager(
    base_dir=os.path.join(directory, 'Codes/MultimodalClassification/ImageGeneration/models_5_7'),
    config=aug_config
)
```

---

## How Augmentation Works

1. **During Training:**
   - For each batch, check if augmentation enabled for modality
   - With probability `prob`, generate synthetic sample
   - Mix synthetic with real using `mix_ratio_range`
   - Apply to training data only (not validation)

2. **Prompt Generation:**
   - System includes medical context in prompts
   - Tailored to healing phase (Inflammatory/Proliferative/Remodeling)
   - Guides diffusion model to generate relevant samples

3. **Memory Management:**
   - Max 3 models in GPU memory simultaneously
   - Automatically unloads least-used models
   - Prevents OOM errors during multi-modality training

---

## Next Steps for Testing

1. **Verify Model Availability:**
   ```bash
   # Check if base directory exists
   ls -la Codes/MultimodalClassification/ImageGeneration/models_5_7/

   # List available models
   find Codes/MultimodalClassification/ImageGeneration/models_5_7/ -type d -maxdepth 1
   ```

2. **Enable in Configuration:**
   - Add generative augmentation config to production_config.py
   - Set per-modality enabled flags
   - Configure inference_steps, prob, mix_ratio_range

3. **Design Experiment:**
   - Baseline run (gen aug OFF)
   - Full run (gen aug ON for all modalities)
   - Ablation study (one modality at a time)
   - Parameter sweep (mix ratios, probabilities)

---

## Questions to Resolve

- [ ] Do model files exist at expected location?
- [ ] What format are the models? (diffusers, safetensors, ckpt?)
- [ ] How many training images were used to fine-tune?
- [ ] What resolution were models trained at?
- [ ] Are models class-conditional or unconditional?
