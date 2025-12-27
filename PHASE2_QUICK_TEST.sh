#!/bin/bash
# Quick Test Script for Phase 2 Fixes
# Run this after applying code fixes from PHASE2_CODE_FIXES.py

set -e  # Exit on error

echo "=========================================="
echo "Phase 2 Fix Verification & Testing"
echo "=========================================="
echo ""

# Activate environment
echo "Step 1: Activating environment..."
source /opt/miniforge3/bin/activate multimodal
echo "✓ Environment activated"
echo ""

# Navigate to project
echo "Step 2: Navigating to project..."
cd /workspace/DFUMultiClassification
echo "✓ In project directory: $(pwd)"
echo ""

# Check current configuration
echo "Step 3: Checking current configuration..."
python -c "from src.utils.production_config import GLOBAL_BATCH_SIZE, IMAGE_SIZE; print(f'Current GLOBAL_BATCH_SIZE: {GLOBAL_BATCH_SIZE}'); print(f'Current IMAGE_SIZE: {IMAGE_SIZE}')"
echo ""

# Verify fixes are applied
echo "Step 4: Verifying fixes are applied..."
echo "Checking for _calculate_phase2_batch_size method..."
if grep -q "_calculate_phase2_batch_size" scripts/auto_polish_dataset_v2.py; then
    echo "✓ Fix #1 (batch size calculator) is present"
else
    echo "✗ Fix #1 NOT FOUND - apply FIX_1 from PHASE2_CODE_FIXES.py"
    exit 1
fi

echo "Checking for batch size adjustment in config override..."
if grep -q "adjusted_batch_size = self._calculate_phase2_batch_size()" scripts/auto_polish_dataset_v2.py; then
    echo "✓ Fix #2 (batch size override) is present"
else
    echo "✗ Fix #2 NOT FOUND - apply FIX_2 from PHASE2_CODE_FIXES.py"
    exit 1
fi

echo "Checking for error logging..."
if grep -q "TRAINING SUBPROCESS FAILED" scripts/auto_polish_dataset_v2.py; then
    echo "✓ Fix #3 (error logging) is present"
else
    echo "✗ Fix #3 NOT FOUND - apply FIX_3 from PHASE2_CODE_FIXES.py"
    exit 1
fi
echo ""

# Create log directory if it doesn't exist
mkdir -p results/misclassifications_saved

# Test 1: Metadata only (should work)
echo "=========================================="
echo "TEST 1: Metadata Only (Baseline)"
echo "=========================================="
echo "This should succeed since metadata is small"
echo "Running 3 evaluations with multi-GPU..."
echo ""

python scripts/auto_polish_dataset_v2.py \
    --phase2-modalities metadata \
    --phase2-only \
    --n-evaluations 3 \
    --device-mode multi \
    2>&1 | tee results/misclassifications_saved/test1_metadata_only.log

echo ""
echo "✓ Test 1 completed"
echo "Check results/misclassifications_saved/test1_metadata_only.log for details"
echo ""

# Test 2: Two modalities (metadata + one image)
echo "=========================================="
echo "TEST 2: Two Modalities (metadata + depth_rgb)"
echo "=========================================="
echo "Testing batch size adjustment with 1 image modality"
echo "Running 3 evaluations with multi-GPU..."
echo ""

python scripts/auto_polish_dataset_v2.py \
    --phase2-modalities metadata+depth_rgb \
    --phase2-only \
    --n-evaluations 3 \
    --device-mode multi \
    2>&1 | tee results/misclassifications_saved/test2_two_modalities.log

echo ""
echo "✓ Test 2 completed"
echo "Check results/misclassifications_saved/test2_two_modalities.log for details"
echo ""

# Test 3: Four modalities (original problem)
echo "=========================================="
echo "TEST 3: Four Modalities (Original Problem)"
echo "=========================================="
echo "Testing batch size adjustment with 3 image modalities"
echo "Running 3 evaluations with multi-GPU..."
echo ""

python scripts/auto_polish_dataset_v2.py \
    --phase2-modalities metadata+depth_rgb+depth_map+thermal_map \
    --phase2-only \
    --n-evaluations 3 \
    --device-mode multi \
    2>&1 | tee results/misclassifications_saved/test3_four_modalities.log

echo ""
echo "✓ Test 3 completed"
echo "Check results/misclassifications_saved/test3_four_modalities.log for details"
echo ""

# Summary
echo "=========================================="
echo "ALL TESTS COMPLETED"
echo "=========================================="
echo ""
echo "Log files saved to results/misclassifications_saved/"
ls -lh results/misclassifications_saved/test*.log
echo ""
echo "To view logs:"
echo "  cat results/misclassifications_saved/test1_metadata_only.log"
echo "  cat results/misclassifications_saved/test2_two_modalities.log"
echo "  cat results/misclassifications_saved/test3_four_modalities.log"
echo ""
echo "To search for errors:"
echo "  grep -i 'error\\|fail\\|❌' results/misclassifications_saved/test*.log"
echo ""
echo "To check for success:"
echo "  grep -i '✅' results/misclassifications_saved/test*.log"
echo ""
echo "Next steps:"
echo "1. Review the log files"
echo "2. If all tests passed, run full optimization with 20 evaluations"
echo "3. Push log files to repo for remote agent review"
echo ""
