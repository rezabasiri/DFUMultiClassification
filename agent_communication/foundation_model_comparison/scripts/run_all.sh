#!/bin/bash
# Run all foundation model comparison experiments.
# Results saved to agent_communication/foundation_model_comparison/results/
#
# Prerequisites:
#   pip install torch torchvision open_clip_torch transformers
#
# Usage:
#   bash run_all.sh           # run everything
#   bash run_all.sh dinov2    # run only DINOv2
#   bash run_all.sh biomedclip # run only BiomedCLIP

set -e
cd "$(dirname "$0")"

FILTER="${1:-all}"

echo "============================================================"
echo "Foundation Model Comparison for DFU Healing Phase Classification"
echo "============================================================"
echo ""

# ---- DINOv2 experiments ----
if [ "$FILTER" = "all" ] || [ "$FILTER" = "dinov2" ]; then
    echo ">>> DINOv2: All 3 modalities (frozen features + logistic regression)"
    python run_dinov2.py --modalities depth_rgb depth_map thermal_map --classifier logreg

    echo ""
    echo ">>> DINOv2: All 3 modalities + metadata"
    python run_dinov2.py --modalities depth_rgb depth_map thermal_map --include_metadata --classifier logreg

    echo ""
    echo ">>> DINOv2: RGB only"
    python run_dinov2.py --modalities depth_rgb --classifier logreg

    echo ""
    echo ">>> DINOv2: All 3 modalities + MLP head"
    python run_dinov2.py --modalities depth_rgb depth_map thermal_map --classifier mlp

    echo ""
    echo ">>> DINOv2: All 3 modalities + metadata + MLP head"
    python run_dinov2.py --modalities depth_rgb depth_map thermal_map --include_metadata --classifier mlp
fi

# ---- BiomedCLIP experiments ----
if [ "$FILTER" = "all" ] || [ "$FILTER" = "biomedclip" ]; then
    echo ""
    echo ">>> BiomedCLIP: All 3 modalities (frozen features + logistic regression)"
    python run_biomedclip.py --modalities depth_rgb depth_map thermal_map --classifier logreg

    echo ""
    echo ">>> BiomedCLIP: All 3 modalities + text metadata"
    python run_biomedclip.py --modalities depth_rgb depth_map thermal_map --include_metadata --classifier logreg

    echo ""
    echo ">>> BiomedCLIP: All 3 modalities + numeric metadata"
    python run_biomedclip.py --modalities depth_rgb depth_map thermal_map --include_metadata_numeric --classifier logreg

    echo ""
    echo ">>> BiomedCLIP: RGB only"
    python run_biomedclip.py --modalities depth_rgb --classifier logreg

    echo ""
    echo ">>> BiomedCLIP: All 3 modalities + numeric metadata + MLP head"
    python run_biomedclip.py --modalities depth_rgb depth_map thermal_map --include_metadata_numeric --classifier mlp
fi

echo ""
echo "============================================================"
echo "All experiments complete. Results in ../results/"
echo "============================================================"
