# Fusion Pipeline Audit

Hyperparameter search for **fusion-specific** parameters — things that only activate when
combining 2+ modalities (metadata + image). Standalone modality parameters (backbone, head,
loss, augmentation) are kept fixed at their optimized values from the standalone audits.

## Prerequisites

Standalone audits must be completed first:
- `agent_communication/depth_rgb_pipeline_audit/`
- `agent_communication/depth_map_pipeline_audit/`
- `agent_communication/thermal_map_pipeline_audit/`

Pre-trained standalone checkpoints must exist for weight transfer into fusion models.

## What This Searches (vs Standalone)

| Parameter | Standalone? | Fusion? | Description |
|-----------|------------|---------|-------------|
| Backbone, head, loss | Yes | Fixed | Optimized per-modality in standalone audits |
| STAGE1_LR | No | **Yes** | Fusion training LR (lower than pretrain LR) |
| Fusion epochs/patience | No | **Yes** | How long to train fusion layers |
| Metadata confidence scaling | No | **Yes** | min_scale/max_scale in ConfidenceBasedMetadataAttention |
| Cross-modal attention dim/L2 | No | **Yes** | fusion_query Dense layer config |
| Attention asymmetry | No | **Yes** | Different scaling for meta->image vs image->meta |
| RF hyperparameters | No | **Yes** | n_estimators, max_depth, feature_selection_k |
| Stage 2 fine-tuning in fusion | Partially | **Yes** | Whether backbone unfreezing helps in fusion context |

## Search Rounds (34 configs + top-3 5-fold CV)

| Round | Focus | Configs |
|-------|-------|---------|
| R1 | Fusion learning rate (STAGE1_LR) | 6 |
| R2 | Training epochs + patience | 4 |
| R3 | Metadata confidence scaling | 6 |
| R4 | Cross-modal attention (dim, L2) | 4 |
| R5 | Attention asymmetry (meta vs image) | 4 |
| R6 | RF hyperparameters | 6 |
| R7 | Stage 2 fine-tuning | 4 |
| Top 3 | 5-fold CV of best 3 configs | 15 |
| **Total** | Sequential elimination | **49** |

## Usage

```bash
# Resume (default -- skips completed configs):
python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py

# Fresh start (backs up previous results):
python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --fresh

# Test with a different image modality:
python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --image-modality thermal_map
```

## Output Files

- `fusion_search_results.csv` — All configs tested with metrics
- `fusion_best_config.json` — Best configuration from 5-fold CV
- `logs/fusion_hparam_search_*.log` — Timestamped execution logs
