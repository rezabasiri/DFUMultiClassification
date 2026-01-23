#!/bin/bash
# Run all diagnostic tests in sequence

echo "=========================================="
echo "METADATA DIAGNOSTIC TEST SUITE"
echo "=========================================="
echo ""

FAILED=0

run_test() {
    echo ""
    echo "Running $1..."
    echo "------------------------------------------"
    python "$1"
    if [ $? -ne 0 ]; then
        echo "❌ FAILED: $1"
        FAILED=1
        return 1
    fi
    echo "✅ PASSED: $1"
    return 0
}

# Run tests in order
run_test "test_1_data_loading.py" || exit 1
run_test "test_2_feature_engineering.py" || exit 1
run_test "test_3_preprocessing.py" || exit 1
run_test "test_4_rf_training.py" || exit 1
run_test "test_5_end_to_end.py" || exit 1

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "ALL TESTS PASSED ✅"
else
    echo "SOME TESTS FAILED ❌"
fi
echo "=========================================="
