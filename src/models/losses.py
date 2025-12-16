"""
Custom loss functions for DFU classification.
Includes weighted, ordinal, and focal losses for handling class imbalance and ordinal relationships.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_f1_score(y_true, y_pred):
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
