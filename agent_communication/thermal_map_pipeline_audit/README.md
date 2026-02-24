# Thermal Map Pipeline Audit

Hyperparameter search for the **thermal_map** standalone modality pathway.

## Run

```bash
# Resume (default — skips completed configs):
python agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py

# Fresh start (backs up previous results):
python agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py --fresh
```

## Search Rounds

| Round | What it tests | Configs |
|-------|--------------|---------|
| R1 | Backbone + freeze strategy | 8 |
| R2 | Head architecture (32–256 units) | 5 |
| R3 | Loss, regularization, alpha sweep | 12 |
| R4 | LR, batch, schedule, optimizer | 8 |
| R5 | Augmentation + image size (128/256) | 4 |
| R6 | Fine-tuning depth/duration | 4 |
| Top 5 | 5-fold CV of best 5 configs | 25 |
| Baseline | EfficientNetB0 frozen 5-fold | 5 |

## Map-Specific Adjustments (vs depth_rgb)

- **SimpleCNN**: 3 layers (128→64→32) instead of 4 — maps encode sensor intensity, less feature complexity
- **Data pipeline**: Overrides `MAP_BACKBONE` (not `RGB_BACKBONE`) for correct normalization
- **Augmentation**: Spatial transforms + mild sensor noise (no color jitter — maps have no color)
- **Image sizes**: 128 and 256 (maps have less fine detail than RGB; 384 not tested)
- **Bbox**: Fixed +30px margin applied automatically by data pipeline for thermal_map

## Outputs

- `thermal_map_search_results.csv` — all results
- `thermal_map_best_config.json` — best config from 5-fold validation
- `logs/thermal_map_hparam_search_*.log` — timestamped run logs
- TF cache records: `results/search_cache/thermal_map_*/`
