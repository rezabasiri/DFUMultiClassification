"""
Custom loss functions for DFU classification.
Includes weighted, ordinal, and focal losses for handling class imbalance and ordinal relationships.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

class WeightedF1Score(tf.keras.metrics.Metric):
    """
    Weighted F1 Score metric using alpha values (inverse frequency weights).

    This metric calculates F1 score for each class and weights them using
    the provided alpha values (inverse frequency normalized to sum=3).
    """
    def __init__(self, alpha_values=None, name='weighted_f1_score', **kwargs):
        super(WeightedF1Score, self).__init__(name=name, **kwargs)
        if alpha_values is None:
            alpha_values = [1.0, 1.0, 1.0]  # Default to equal weights
        self.alpha_values = tf.constant(alpha_values, dtype=tf.float32)
        # Normalize alpha values to sum=3 if not already
        alpha_sum = tf.reduce_sum(self.alpha_values)
        self.alpha_values = self.alpha_values / alpha_sum * 3.0

        self.true_positives = self.add_weight(name='tp', shape=(3,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(3,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(3,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_cls = tf.argmax(y_true, axis=1)
        y_pred_cls = tf.argmax(y_pred, axis=1)

        # Calculate confusion matrix elements for each class using vectorized operations
        # This approach works correctly with distributed strategies
        tp_batch = []
        fp_batch = []
        fn_batch = []

        for class_id in range(3):
            y_true_class = tf.cast(tf.equal(y_true_cls, class_id), tf.float32)
            y_pred_class = tf.cast(tf.equal(y_pred_cls, class_id), tf.float32)

            tp = tf.reduce_sum(y_true_class * y_pred_class)
            fp = tf.reduce_sum((1 - y_true_class) * y_pred_class)
            fn = tf.reduce_sum(y_true_class * (1 - y_pred_class))

            tp_batch.append(tp)
            fp_batch.append(fp)
            fn_batch.append(fn)

        # Stack and add to weights (works with distributed strategy)
        self.true_positives.assign_add(tf.stack(tp_batch))
        self.false_positives.assign_add(tf.stack(fp_batch))
        self.false_negatives.assign_add(tf.stack(fn_batch))

    def result(self):
        # Calculate F1 for each class
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-7)

        # Weight by alpha values and normalize by sum of alpha values
        weighted_f1 = tf.reduce_sum(f1_per_class * self.alpha_values) / tf.reduce_sum(self.alpha_values)

        return weighted_f1

    def reset_state(self):
        self.true_positives.assign(tf.zeros((3,)))
        self.false_positives.assign(tf.zeros((3,)))
        self.false_negatives.assign(tf.zeros((3,)))

def weighted_f1_score(y_true, y_pred):
    """
    Legacy weighted F1 score function (uses support-based weighting).
    Kept for backward compatibility.
    """
    y_true_cls = tf.argmax(y_true, axis=1)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    conf_matrix = tf.math.confusion_matrix(y_true_cls, y_pred_cls, num_classes=3)
    conf_matrix = tf.cast(conf_matrix, tf.float32)
    true_positives = tf.linalg.diag_part(conf_matrix)
    predicted_positives = tf.reduce_sum(conf_matrix, axis=0)
    actual_positives = tf.reduce_sum(conf_matrix, axis=1)
    precision = true_positives / (predicted_positives + 1e-7)
    recall = true_positives / (actual_positives + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    weights = actual_positives / tf.reduce_sum(actual_positives)
    weighted_f1 = tf.reduce_sum(f1 * weights)

    return weighted_f1
def weighted_ordinal_crossentropy(y_true, y_pred, num_classes=3, ordinal_weight=0.5):
    # Standard categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # Ordinal penalty
    true_class = tf.argmax(y_true, axis=-1)
    pred_class = tf.argmax(y_pred, axis=-1)
    ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))
    # Combine losses
    total_loss = cce + ordinal_weight * ordinal_penalty
    return total_loss

# Wrapper function to set hyperparameters
def get_weighted_ordinal_crossentropy(num_classes=3, ordinal_weight=0.5):
    def loss(y_true, y_pred):
        return tf.cast(weighted_ordinal_crossentropy(y_true, y_pred, num_classes, ordinal_weight), tf.float32)
    return loss
def get_weighted_ordinal_crossentropyF1(num_classes=3, ordinal_weight=0.5, f1_weight=0.1):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Weighted ordinal cross-entropy
        wocce = weighted_ordinal_crossentropy(y_true, y_pred, num_classes, ordinal_weight)
        # Weighted F1 score
        wf1score = weighted_f1_score(y_true, y_pred)
        # Combine losses
        total_loss = wocce - f1_weight * tf.math.log(wf1score + 1e-7)
        # Ensure the loss is finite
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.zeros_like(total_loss))
        
        return total_loss
    return loss
def focal_ordinal_loss(y_true, y_pred, num_classes=3, ordinal_weight=0.5, gamma=3.0, alpha=[0.598, 0.315, 1.597]):
    # Clip prediction values to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Focal loss
    cross_entropy = -y_true * tf.math.log(y_pred)
    focal_weight = alpha * tf.math.pow(1 - y_pred, gamma)
    focal_loss = focal_weight * cross_entropy
    
    # Sum over classes
    focal_loss = tf.reduce_sum(focal_loss, axis=-1)
    
    # Ordinal penalty
    true_class = tf.argmax(y_true, axis=-1)
    pred_class = tf.argmax(y_pred, axis=-1)
    ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))
    
    # Combine losses
    total_loss = focal_loss + ordinal_weight * ordinal_penalty
    return total_loss
# Wrapper function to set hyperparameters
def get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.5, gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        return focal_ordinal_loss(y_true, y_pred, num_classes, ordinal_weight, gamma, alpha)
    return loss
# def get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.5, gamma=2.0, alpha=[1,1,1]):
#     return lambda y_true, y_pred: focal_ordinal_loss(y_true, y_pred, num_classes, ordinal_weight, gamma, alpha)
