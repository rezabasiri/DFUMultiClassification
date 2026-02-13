#!/bin/bash
# Quick test to measure determinism performance impact

echo "=========================================================================="
echo "DETERMINISM SPEED TEST"
echo "=========================================================================="
echo ""
echo "This will run a short training test twice:"
echo "  1. With determinism ON (current settings)"
echo "  2. With determinism OFF"
echo ""
echo "You'll see the actual speed difference on YOUR hardware."
echo ""

# Test 1: Deterministic (current)
echo "Test 1: DETERMINISTIC mode (current settings)"
echo "----------------------------------------------------------------------"
time python3 src/main.py --data_percentage 10 --device-mode multi --verbosity 1 --resume_mode fresh --n-epochs 3 2>&1 | grep -E "(Epoch|total time)"

echo ""
echo ""

# Test 2: Non-deterministic
echo "Test 2: NON-DETERMINISTIC mode (faster)"
echo "----------------------------------------------------------------------"
export TF_DETERMINISTIC_OPS="0"
export TF_CUDNN_DETERMINISTIC="0"
time python3 src/main.py --data_percentage 10 --device-mode multi --verbosity 1 --resume_mode fresh --n-epochs 3 2>&1 | grep -E "(Epoch|total time)"

echo ""
echo "=========================================================================="
echo "Compare the training times above to see the actual speed difference"
echo "=========================================================================="
