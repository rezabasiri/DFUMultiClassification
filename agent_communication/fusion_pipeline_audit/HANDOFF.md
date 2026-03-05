# Fusion Pipeline Audit — Handoff Document

## Project Overview

DFU (Diabetic Foot Ulcer) healing phase classification — 3 classes: Inflammatory (I), Proliferative (P), Remodeling (R). Multimodal fusion of metadata (RF) + depth_rgb (EfficientNet) + thermal_map (EfficientNet) with cross-modal attention.

**Environment:** Python 3.11, TensorFlow 2.18.1, 2 GPUs, conda env `multimodal`

## Dataset

- Source: `best_matching.csv` — ensures uniform samples across all modalities
- Total: 3108 samples (all modalities present)
- Patient-wise, class-stratified 5-fold CV (`create_patient_folds(data, n_folds=5, random_state=42)`)
- No core data filtering in fusion (disabled intentionally)

## Standalone Audit Results (Pre-Fusion)

Each image modality was independently optimized via sequential elimination hparam search.

### depth_rgb — Selected: #1 R4_lr1e3_b64_e100 (TOP5_3)
- Log: `agent_communication/depth_rgb_pipeline_audit/logs/depth_rgb_hparam_search_20260225_193121.md`
- Config: EfficientNetB2, head=[256,64], dropout=0.3, bn=True, focal(gamma=2.0, alpha_sum=3.0), lr=0.001, plateau, adam, epochs=100+30ft(20%), aug=True, mixup=False, img_size=256
- 5-fold kappa: 0.3144, 0.3265, 0.2217, 0.2676, 0.1967 → **Mean=0.2654**
- No data filtering, 3108 samples

### thermal_map — Selected: BASELINE (EfficientNetB0 frozen)
- Log: `agent_communication/thermal_map_pipeline_audit/logs/thermal_map_hparam_search_20260225_193401.md`
- Config: EfficientNetB0, head=[128], dropout=0.3, bn=True, focal(gamma=2.0, alpha_sum=3.0), lr=0.001, plateau, adam(wd=0.0001), epochs=50+30ft(20%), aug=True, mixup=False(0.2), img_size=256
- 5-fold kappa: 0.3636, 0.4255, 0.4133, 0.4915, 0.4343 → **Mean=0.4257**
- No data filtering, 3108 samples
- Note: An earlier run (`thermal_map_hparam_search_20260225_014800.md`) used data filtering (2072 samples) and found R6_ft_top20_50ep (EfficientNetB2, head=[256], adamw, img128) as #1, but we chose the BASELINE from the no-filtering run for consistency with fusion's full dataset.

### depth_map — Not used in fusion
- Standalone baseline kappa ~0.1067 (very low signal)
- Config stored in `_STANDALONE_CONFIGS` dict but depth_map is NOT in `image_modalities`

## Fusion Audit Script

**File:** `agent_communication/fusion_pipeline_audit/fusion_hparam_search.py`

### How to Run
```bash
# Fresh start (backs up old CSV):
python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --fresh

# Resume from existing results:
python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py
```

### Architecture

**Pre-fusion stage (per image modality):**
1. Build standalone model using `build_pretrain_model()` — exact same architecture as standalone audits: Input → EfficientNet backbone → Dense head → BN → Dropout → softmax(3)
2. Stage 1: Train with backbone frozen (head-only)
3. Stage 2: Partial unfreeze (top 20%) + lower LR fine-tuning
4. Save backbone weights, cache per (modality, fold_idx) to avoid retraining
5. POST-EVAL kappa printed for sanity check against standalone values

**Fusion stage:**
1. Build fusion model via `create_multimodal_model()` from `src/models/builders.py` — different architecture (projection heads, ConfidenceBasedMetadataAttention, cross-modal attention)
2. Transfer pre-trained backbone weights into fusion model
3. Freeze backbones, train fusion layers only
4. Optional Stage 2: partial unfreeze of fusion backbones

### Data Pipeline Overrides for Pre-Training

The pre-training loop temporarily overrides production_config globals to match standalone settings:
- `RGB_BACKBONE` / `MAP_BACKBONE` — controls data normalization ([0,255] for EfficientNet)
- `USE_GENERAL_AUGMENTATION` — on/off per modality
- Image size, augmentation config, cache directory — all per-modality from `_STANDALONE_CONFIGS`

### Search Structure

- **Baseline:** 5-fold CV with default fusion params
- **Screening (R1-R7):** 5-fold split, train fold 0 only, pick best per round
  - R1: Fusion LR (6 configs)
  - R2: Epochs/patience (4 configs)
  - R3: Metadata confidence scaling (6 configs)
  - R4: Cross-modal attention dim/L2 (4 configs)
  - R5: Attention asymmetry (4 configs)
  - R6: RF hyperparameters (6 configs)
  - R7: Stage 2 fine-tuning (4 configs)
- **Final:** Top 3 from screening + baseline, full 5-fold CV, report mean kappa

### Key Config Details (FusionSearchConfig defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| stage1_lr | 1e-4 | Fusion training LR |
| stage1_epochs | 200 | With early stopping |
| early_stop_patience | 20 | Monitor val_cohen_kappa |
| label_smoothing | 0.1 | Fusion loss |
| alpha_sum | 3.0 | Focal loss class weights |
| meta_min/max_scale | 1.5/3.0 | ConfidenceBasedMetadataAttention |
| fusion_query_dim | 64 | Cross-modal attention |
| fusion_query_l2 | 0.001 | L2 on attention layers |
| meta/image_query_scale | 0.8/1.5 | Asymmetric attention |
| stage2_epochs | 0 | Off by default |

### Weight Transfer

Only backbone (EfficientNet) weights transfer from pre-training to fusion model. Head/projection layer names differ between standalone and fusion architectures, so head weights are NOT transferred — fusion model learns its own projection heads from scratch.

### Resume Support

Results append to `fusion_search_results.csv`. On resume, completed config names are skipped. Pre-trained weights are cached in `_pretrain_cache` dict (in-memory, per session).

## Output Files

- `agent_communication/fusion_pipeline_audit/fusion_search_results.csv` — all results
- `agent_communication/fusion_pipeline_audit/fusion_best_config.json` — winner config
- `agent_communication/fusion_pipeline_audit/logs/fusion_hparam_search_*.log` — full logs

## Known Issues & Design Decisions

1. **Pre-training kappas won't exactly match standalone** — stochastic training, different augmentation cache. Expected to be in same ballpark (within ~0.05).

2. **Backbone weight transfer by order** — fusion model backbones matched to pre-trained by position in `image_modalities` list, not by name. Works because `create_multimodal_model` creates branches in the same order.

3. **Fusion model uses `create_multimodal_model`** which adds projection layers (`{modality}_projection`), `ConfidenceBasedMetadataAttention`, and cross-modal attention — different from standalone's simple `head_dense → BN → Dropout → softmax`.

4. **No depth_map in fusion** — only `["depth_rgb", "thermal_map"]` + metadata.

## File Map

```
agent_communication/
├── fusion_pipeline_audit/
│   ├── fusion_hparam_search.py          # Main script
│   ├── fusion_search_results.csv        # Results CSV
│   ├── fusion_best_config.json          # Best config output
│   ├── HANDOFF.md                       # This document
│   └── logs/                            # Run logs
├── depth_rgb_pipeline_audit/
│   ├── depth_rgb_hparam_search.py       # Standalone depth_rgb audit
│   └── logs/
├── thermal_map_pipeline_audit/
│   ├── thermal_map_hparam_search.py     # Standalone thermal_map audit
│   └── logs/
└── depth_map_pipeline_audit/
    ├── depth_map_hparam_search.py       # Standalone depth_map audit (not used in fusion)
    └── logs/
```

## Source Code Dependencies

- `src/models/builders.py` — `create_multimodal_model()`, `ConfidenceBasedMetadataAttention`, `create_fusion_layer`
- `src/data/dataset_utils.py` — `create_patient_folds()`, `prepare_cached_datasets()`
- `src/data/image_processing.py` — `prepare_dataset()`
- `src/utils/production_config.py` — `RGB_BACKBONE`, `MAP_BACKBONE`, `USE_GENERAL_AUGMENTATION`, RF params
- `src/models/losses.py` — `get_focal_ordinal_loss()`
- `src/utils/gpu_config.py` — `setup_device_strategy()`
