#!/usr/bin/env python3
"""
Quick test to verify all imports work without errors.
"""

print("Testing imports...")

print("1. Testing TensorFlow...")
import tensorflow as tf
print(f"   ✓ TensorFlow {tf.__version__}")
print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")

print("\n2. Testing PyTorch...")
import torch
print(f"   ✓ PyTorch {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

print("\n3. Testing transformers...")
import transformers
print(f"   ✓ transformers {transformers.__version__}")

print("\n4. Testing diffusers...")
import diffusers
print(f"   ✓ diffusers {diffusers.__version__}")

print("\n5. Testing generative augmentation imports...")
try:
    from src.data.generative_augmentation_v2 import create_enhanced_augmentation_fn
    print("   ✓ generative_augmentation_v2 imported successfully")
except Exception as e:
    print(f"   ✗ Error importing generative_augmentation_v2: {e}")

print("\n6. Testing main imports...")
try:
    from src.utils.verbosity import vprint, set_verbosity
    print("   ✓ verbosity module imported successfully")
except Exception as e:
    print(f"   ✗ Error importing verbosity: {e}")

print("\n✅ All critical imports successful!")
print("\nYou can now run your training with:")
print("  python -m src.main --verbosity 3 ...")
