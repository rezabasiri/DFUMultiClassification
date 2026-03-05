#!/bin/bash
set -e

echo "=========================================="
echo " Sequential Standalone Hparam Searches"
echo "=========================================="

echo ""
echo "[1/2] depth_rgb (--fresh)"
echo "------------------------------------------"
python agent_communication/depth_rgb_pipeline_audit/depth_rgb_hparam_search.py --fresh

echo ""
echo "[2/2] thermal_map (--fresh)"
echo "------------------------------------------"
python agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py --fresh

echo ""
echo "=========================================="
echo " Both searches complete."
echo "=========================================="
