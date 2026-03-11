"""Mixup augmentation for TensorFlow datasets.

Implements the mixup training strategy from Zhang et al. (2018) — creates
virtual training examples by linearly interpolating between random pairs
of samples.  Acts as a strong regularizer for small datasets by smoothing
the decision boundary.

Used by the main training pipeline (training_utils.py) when per-modality
config has use_mixup=True (e.g. thermal_map, depth_map).
"""

import tensorflow as tf


def apply_mixup(features, labels, alpha=0.2):
    """Mixup augmentation — blends pairs of samples within a batch.

    Args:
        features: Dict of input tensors (e.g. {'thermal_map_input': ..., 'metadata_input': ...}).
        labels: One-hot encoded label tensor, shape (batch, num_classes).
        alpha: Controls mixing strength.
            alpha <= 0.3: lam = max(u1, u2) — conservative, keeps samples close to originals.
            alpha > 0.3:  lam = (u1+u2)/2   — more uniform blending.
            Default 0.2 (conservative).

    Returns:
        (mixed_features, mixed_labels) — same structure as inputs.
    """
    batch_size = tf.shape(list(features.values())[0])[0]

    u1 = tf.random.uniform([], 0.0, 1.0)
    u2 = tf.random.uniform([], 0.0, 1.0)
    lam = tf.maximum(u1, u2) if alpha <= 0.3 else (u1 + u2) / 2.0

    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_features = {}
    for key, val in features.items():
        shuffled = tf.gather(val, indices)
        mixed_features[key] = lam * val + (1.0 - lam) * shuffled

    shuffled_labels = tf.gather(labels, indices)
    mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels

    return mixed_features, mixed_labels
