#!/bin/bash
# Quick test script for fusion investigation
# Usage: ./quick_test.sh [32|64|128]

if [ -z "$1" ]; then
    echo "Usage: ./quick_test.sh [32|64|128]"
    exit 1
fi

IMAGE_SIZE=$1
LOG_FILE="agent_communication/fusion_fix/run_fusion_${IMAGE_SIZE}x${IMAGE_SIZE}_50pct.txt"

echo "=========================================="
echo "Testing fusion with IMAGE_SIZE=${IMAGE_SIZE}"
echo "Log will be saved to: ${LOG_FILE}"
echo "=========================================="
echo ""
echo "STEP 1: Update production_config.py"
echo "  Set IMAGE_SIZE = ${IMAGE_SIZE}"
echo "  Set DATA_PERCENTAGE = 50.0"
echo ""
read -p "Press Enter after updating config (or Ctrl+C to cancel)..."

echo ""
echo "STEP 2: Running training..."
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Test complete! Results saved to:"
echo "  $LOG_FILE"
echo ""
echo "Key results:"
grep -A 2 "Pre-training completed!" "$LOG_FILE" | tail -1
grep "Total model trainable weights:" "$LOG_FILE" | head -1
grep "Stage 1 completed" "$LOG_FILE"
grep "Final fusion:" "$LOG_FILE"
grep "Cohen's Kappa:" "$LOG_FILE" | tail -1
echo "=========================================="
