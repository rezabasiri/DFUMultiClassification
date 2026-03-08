# Fusion Pre-Training vs Standalone: 5-Fold Comparison

## Description

- **Standalone**: Each image modality trained independently using its best config
  - depth_rgb: BASELINE (EfficientNetB0 frozen, head=[128], lr=0.001, 50+30ft epochs)
  - thermal_map: B3_R6_ft_top50_50ep (EfficientNetB2 frozen, head=[128,32], lr=0.003, 50+50ft epochs)
- **Fusion Pre-train**: Per-modality pre-training inside the fusion pipeline using _STANDALONE_CONFIGS, before the fusion layer
- **Fusion Post-Fusion**: The fused 3-modality output (metadata RF + depth_rgb + thermal_map) after the attention fusion layer (16 trainable params, stage2_epochs=0)

## Depth RGB — BASELINE (EfficientNetB0 frozen)

| Fold | Standalone Kappa | Fusion Pre-train Kappa | Fusion Post-Fusion Kappa |
|------|-----------------|----------------------|------------------------|
| 1    | 0.2078          | 0.1722               | 0.2710                 |
| 2    | 0.3223          | 0.2351               | 0.2149                 |
| 3    | 0.3314          | 0.2444               | 0.2704                 |
| 4    | 0.3642          | 0.3202               | 0.2435                 |
| 5    | 0.2457          | 0.2153               | 0.2711                 |
| **Mean** | **0.2943 +/- 0.058** | **0.2374 +/- 0.050** | **0.2542 +/- 0.022** |

## Thermal Map — B3_R6_ft_top50_50ep (EfficientNetB2 frozen)

| Fold | Standalone Kappa | Fusion Pre-train Kappa | Fusion Post-Fusion Kappa |
|------|-----------------|----------------------|------------------------|
| 1    | 0.3401          | 0.3176               | 0.2710                 |
| 2    | 0.2873          | 0.2047               | 0.2149                 |
| 3    | 0.2554          | 0.2638               | 0.2704                 |
| 4    | 0.3311          | 0.2801               | 0.2435                 |
| 5    | 0.2988          | 0.2243               | 0.2711                 |
| **Mean** | **0.3025 +/- 0.031** | **0.2581 +/- 0.040** | **0.2542 +/- 0.022** |

## Stage 2 Fine-tuning During Fusion Pre-training

| Fold | depth_rgb S1 | depth_rgb S2 | S2 used? | thermal_map S1 | thermal_map S2 | S2 used? |
|------|-------------|-------------|----------|---------------|---------------|----------|
| 1    | 0.1722      | 0.1431      | No       | 0.3006        | 0.3171        | Yes      |
| 2    | 0.2346      | 0.1877      | No       | 0.2047        | 0.1951        | No       |
| 3    | 0.2444      | 0.1604      | No       | 0.2507        | 0.2638        | Yes      |
| 4    | 0.3239      | 0.2413      | No       | 0.2800        | 0.2827        | Yes      |
| 5    | 0.2152      | 0.1322      | No       | 0.2243        | 0.2132        | No       |

## Summary

| Metric         | SA depth_rgb | Fusion PT depth_rgb | SA thermal_map | Fusion PT thermal_map | Fusion (combined) |
|----------------|-------------|--------------------|--------------|-----------------------|-------------------|
| **Mean Kappa** | 0.2943      | 0.2374             | 0.3025       | 0.2581                | 0.2542            |
| **Std**        | 0.058       | 0.050              | 0.031        | 0.040                 | 0.022             |

SA = Standalone, PT = Pre-train

## Config Mismatches Found and Fixed

1. Backbone construction: fusion used temp+copy 2-model approach, standalone uses direct `weights='imagenet'` (FIXED in fusion)
2. No explicit seed reset before pre-training model build (FIXED in fusion)
3. Fusion used separate cache paths (`fusion_pretrain_*`) instead of standalone cache paths (`{modality}_*_pretrained_*`) (FIXED in fusion)

## Bug Found in Standalone Scripts

`pick_best()` in both standalone scripts did not save/restore `finetune_lr` — the result dict omitted it, and the SearchConfig reconstruction used the default (`1e-5`). This means R6 configs with non-default `finetune_lr` (e.g. thermal_map `ft_top50_50ep` used `5e-6` in R6 screening) had their `finetune_lr` silently reset to `1e-5` during 5-fold CV. FIXED by adding `finetune_lr` to both the result dict and `pick_best()` reconstruction in both standalone scripts.

## Source Files

- Fusion results: `fusion_search_results.csv`
- Fusion log: `logs/fusion_hparam_search_20260306_040149.log`
- Depth RGB standalone: `../depth_rgb_pipeline_audit/logs/depth_rgb_hparam_search_20260305_003922.log`
- Thermal Map standalone: `../thermal_map_pipeline_audit/logs/thermal_map_hparam_search_20260305_134816.log`
