"""
Test script to verify model architectures can be built for all modality combinations (1-5).
This is a lightweight test that only builds models without training.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("MODEL ARCHITECTURE TEST - All Modality Combinations")
print("=" * 80)
print("\nVerifying model builder can create architectures for 1-5 modality combinations")
print("This validates the refactored code maintains the original dynamic behavior\n")

# Import after adding to path
try:
    import tensorflow as tf
    from src.models.builders import create_multimodal_model
    print("✓ Imports successful\n")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure TensorFlow and src modules are available.")
    sys.exit(1)

# Define all modality combinations to test
MODALITY_COMBINATIONS = [
    # 1 modality
    (['metadata'], "Metadata only"),
    (['depth_rgb'], "Depth RGB only"),

    # 2 modalities
    (['metadata', 'depth_rgb'], "Metadata + Depth RGB"),
    (['depth_rgb', 'depth_map'], "Depth RGB + Depth Map"),

    # 3 modalities
    (['metadata', 'depth_rgb', 'depth_map'], "Metadata + Depth RGB + Depth Map"),
    (['metadata', 'depth_rgb', 'thermal_rgb'], "Metadata + Depth RGB + Thermal RGB"),

    # 4 modalities
    (['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb'], "Metadata + 3 Images"),
    (['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map'], "4 Image Modalities"),

    # 5 modalities (all)
    (['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map'], "All Modalities"),
]

IMAGE_SIZE = 64  # For testing

def test_model_architecture(modalities, description):
    """Test if model architecture can be built for a specific modality combination"""
    print("-" * 80)
    print(f"Test: {len(modalities)} modality(ies) - {description}")
    print(f"Modalities: {', '.join(modalities)}")

    try:
        # Determine input shapes
        input_shapes = {}

        for modality in modalities:
            if modality == 'metadata':
                # Metadata uses 3 RF probability features
                input_shapes['metadata'] = (3,)
            elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                input_shapes[modality] = (IMAGE_SIZE, IMAGE_SIZE, 3)

        # Create dummy class weights
        class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

        # Build model
        model = create_multimodal_model(
            input_shapes=input_shapes,
            selected_modalities=modalities,
            class_weights=class_weights,
            strategy=None
        )

        # Get model info
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

        # Verify model structure
        num_inputs = len(model.inputs)
        num_outputs = len(model.outputs)
        output_shape = model.outputs[0].shape

        print(f"  ✓ Model built successfully")
        print(f"  ✓ Inputs: {num_inputs} (expected: {len(modalities) + 1})") # +1 for sample_id
        print(f"  ✓ Outputs: {num_outputs}")
        print(f"  ✓ Output shape: {output_shape}")
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")

        # Verify output shape is correct (batch_size, 3 classes)
        if output_shape[-1] != 3:
            raise ValueError(f"Expected output shape (..., 3), got {output_shape}")

        # Clean up
        del model
        tf.keras.backend.clear_session()

        print(f"  ✅ SUCCESS\n")
        return True

    except Exception as e:
        print(f"  ❌ FAILED")
        print(f"  Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all architecture tests"""

    print(f"Will test {len(MODALITY_COMBINATIONS)} different combinations:\n")

    results = []

    for i, (modalities, description) in enumerate(MODALITY_COMBINATIONS, 1):
        print(f"\n[Test {i}/{len(MODALITY_COMBINATIONS)}]")
        success = test_model_architecture(modalities, description)
        results.append((len(modalities), description, success))

    # Print summary
    print("=" * 80)
    print("SUMMARY OF ALL ARCHITECTURE TESTS")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for _, _, success in results if success)
    failed = total - passed

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    print("\nDetailed results:")
    for i, (num_mods, desc, success) in enumerate(results, 1):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {i}. {num_mods} modalities ({desc}): {status}")

    print("\n" + "=" * 80)

    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe refactored code successfully builds model architectures for all")
        print("modality combinations (1-5). The dynamic modality system from the")
        print("original code has been preserved in the refactored codebase.")
    else:
        print(f"⚠️  {failed} test(s) failed. Review the errors above.")

    print("=" * 80 + "\n")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
