#!/usr/bin/env python3
"""DEBUG 5: Test if focal loss applies alpha correctly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from src.models.losses import get_focal_ordinal_loss

# Force CPU
tf.config.set_visible_devices([], 'GPU')
np.random.seed(42)
tf.random.set_seed(42)

def test_focal_loss_alpha():
    """Test if focal loss properly applies per-class alpha weights"""
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 5: FOCAL LOSS ALPHA WEIGHTING TEST")
    log("="*80)

    # Create test data
    # Scenario: 3 samples, each belongs to a different class
    y_true = tf.constant([
        [1, 0, 0],  # Class 0 (I)
        [0, 1, 0],  # Class 1 (P)
        [0, 0, 1],  # Class 2 (R)
    ], dtype=tf.float32)

    # Model predictions (all predict class 1 strongly)
    y_pred = tf.constant([
        [0.1, 0.8, 0.1],  # Predicts class 1, but true is 0
        [0.1, 0.8, 0.1],  # Predicts class 1, true is 1 - CORRECT
        [0.1, 0.8, 0.1],  # Predicts class 1, but true is 2
    ], dtype=tf.float32)

    log("\nTest scenario:")
    log("  3 samples, each from different class")
    log("  All predictions favor class 1 (P)")
    log("  True classes: [0, 1, 2]")
    log("  Predicted class: [1, 1, 1]")

    # Test with equal alpha (no weighting)
    log("\n" + "="*80)
    log("TEST 1: Equal alpha values [1, 1, 1]")
    log("="*80)
    alpha_equal = [1.0, 1.0, 1.0]
    loss_fn_equal = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.0, gamma=2.0, alpha=alpha_equal)
    loss_equal = loss_fn_equal(y_true, y_pred)
    log(f"  Loss per sample: {loss_equal.numpy()}")
    log(f"  Mean loss: {tf.reduce_mean(loss_equal).numpy():.4f}")

    # Test with inverse frequency alpha (class 2 should get higher weight)
    log("\n" + "="*80)
    log("TEST 2: Inverse frequency alpha (e.g., [3.48, 1.65, 9.27])")
    log("="*80)
    log("  Expected: Class 2 (R) errors should have MUCH higher loss")
    alpha_weighted = [3.48, 1.65, 9.27]  # Example: class 2 is rare, gets 9.27x weight
    loss_fn_weighted = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.0, gamma=2.0, alpha=alpha_weighted)
    loss_weighted = loss_fn_weighted(y_true, y_pred)
    log(f"  Loss per sample: {loss_weighted.numpy()}")
    log(f"  Mean loss: {tf.reduce_mean(loss_weighted).numpy():.4f}")

    # Analysis
    log("\n" + "="*80)
    log("ANALYSIS")
    log("="*80)

    loss_equal_np = loss_equal.numpy()
    loss_weighted_np = loss_weighted.numpy()

    log(f"Sample 0 (Class I, alpha={alpha_weighted[0]:.2f}): {loss_equal_np[0]:.4f} -> {loss_weighted_np[0]:.4f} ({loss_weighted_np[0]/loss_equal_np[0]:.2f}x)")
    log(f"Sample 1 (Class P, alpha={alpha_weighted[1]:.2f}): {loss_equal_np[1]:.4f} -> {loss_weighted_np[1]:.4f} ({loss_weighted_np[1]/loss_equal_np[1]:.2f}x)")
    log(f"Sample 2 (Class R, alpha={alpha_weighted[2]:.2f}): {loss_equal_np[2]:.4f} -> {loss_weighted_np[2]:.4f} ({loss_weighted_np[2]/loss_equal_np[2]:.2f}x)")

    # Check if weighting is proportional to alpha
    expected_ratio_0 = alpha_weighted[0] / alpha_equal[0]  # Should be 3.48
    expected_ratio_1 = alpha_weighted[1] / alpha_equal[1]  # Should be 1.65
    expected_ratio_2 = alpha_weighted[2] / alpha_equal[2]  # Should be 9.27

    actual_ratio_0 = loss_weighted_np[0] / loss_equal_np[0]
    actual_ratio_1 = loss_weighted_np[1] / loss_equal_np[1]
    actual_ratio_2 = loss_weighted_np[2] / loss_equal_np[2]

    log("\nExpected vs Actual ratios:")
    log(f"  Class 0: Expected {expected_ratio_0:.2f}x, Actual {actual_ratio_0:.2f}x")
    log(f"  Class 1: Expected {expected_ratio_1:.2f}x, Actual {actual_ratio_1:.2f}x")
    log(f"  Class 2: Expected {expected_ratio_2:.2f}x, Actual {actual_ratio_2:.2f}x")

    # Verdict
    tolerance = 0.1  # 10% tolerance
    ratios_match = (
        abs(actual_ratio_0 - expected_ratio_0) < tolerance and
        abs(actual_ratio_1 - expected_ratio_1) < tolerance and
        abs(actual_ratio_2 - expected_ratio_2) < tolerance
    )

    log("\n" + "="*80)
    if ratios_match:
        log("✅ PASS: Focal loss applies alpha weights correctly")
        log("="*80)
    else:
        log("❌ FAIL: Focal loss does NOT apply alpha correctly")
        log("="*80)
        log("\nThis means the class weights are not being applied to the loss!")
        log("The model cannot learn to prioritize minority classes.")

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'results_05_focal_loss_test.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    log(f"\nResults saved to: {output_file}")
    return ratios_match

if __name__ == '__main__':
    try:
        success = test_focal_loss_alpha()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
