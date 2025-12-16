"""
Debug and memory management utilities for TensorFlow/Keras.
"""

import os
import gc
import tensorflow as tf

# Environment configuration for GPU garbage collection
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'


def clear_gpu_memory():
    """Clear GPU memory and reset Keras session."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.backend.clear_session()
            gc.collect()
        except RuntimeError as e:
            print('clear_gpu_memory Warning:', e)


def reset_keras():
    """Reset Keras backend and default TensorFlow graph."""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()


def clear_cuda_memory():
    """Clear CUDA memory if GPU is available."""
    if 'cuda' in tf.test.gpu_device_name():
        tf.keras.backend.clear_session()
        gc.collect()
