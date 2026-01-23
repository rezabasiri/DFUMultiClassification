import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate, GlobalAveragePooling2D, Multiply, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import random
import numpy as np
import pandas as pd


from src.utils.config import get_project_paths, RANDOM_SEED

# Set random seeds
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Get paths from centralized config
directory, result_dir, root = get_project_paths()

# Image processing parameters
image_size = 128

#%% Model Generators
def create_image_branch(input_shape, modality):
    image_input = Input(shape=(image_size, image_size, 3), name=f'{modality}_input')
    if modality in ['depth_rgb', 'thermal_rgb']:
        with tf.keras.mixed_precision.Policy('mixed_float16'):
            if os.path.exists(os.path.join(directory, "local_weights/efficientnetb0_notop.h5")):
                base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=image_input)
                base_model.load_weights(os.path.join(directory, "local_weights/efficientnetb0_notop.h5"))
            else: 
                base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_tensor=image_input)
        base_model.trainable = False
        for layer in base_model.layers:
            layer._name = f'{modality}_{layer.name}'
        x = base_model.output
        x = GlobalAveragePooling2D(name=f'{modality}_RGB_GAP2D')(x)
    else:
        x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', use_bias=False, name=f'{modality}_Conv2D_1')(image_input)
        x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN1')(x)
        x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', use_bias=False, name=f'{modality}_Conv2D_2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN2')(x)
        x = GlobalAveragePooling2D(name=f'{modality}_GAP2D')(x)
        
    # Projection layer to ensure consistent dimensionality across modalities
    x = Dense(256, activation='relu', use_bias=False, name=f'{modality}_projection1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj1')(x)
    x = tf.keras.layers.Dropout(0.25, name=f'{modality}_dropout_1')(x)
    x = Dense(128, activation='relu', use_bias=False, name=f'{modality}_projection2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj2')(x)
    x = tf.keras.layers.Dropout(0.15, name=f'{modality}_dropout_2')(x)
    x = Dense(64, activation='relu', use_bias=False, name=f'{modality}_projection3')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj3')(x)
    x = tf.keras.layers.Dropout(0.15, name=f'{modality}_DP_proj3')(x)
    
    # Self-attention mechanism
    attention_scores = Dense(64, activation='sigmoid', use_bias=False, name=f'{modality}_attention')(x)
    attention_output = tf.multiply(x, attention_scores, name=f'{modality}_attention_mul')
    
    # Apply modular attention
    modular_attention = OptimizedModularAttention(name=f'{modality}_modular_attention')
    attention_output = modular_attention(x)
        
    return image_input, attention_output

def create_metadata_branch(input_shape, index):
    """Metadata branch with MINIMAL processing (preserves RF probabilities)

    Design Philosophy (from main_original.py):
    - RF produces 3 probabilities (rf_prob_I, rf_prob_P, rf_prob_R)
    - These should pass through with minimal transformation
    - Preserves probability information for fusion with other modalities
    - Prevents overfitting on just 3 input values

    Architecture: Cast → BatchNorm (that's it!)
    - NO dense layers (they cause overfitting)
    - NO dropout (unnecessary for 3 probabilities)
    - NO attention (minimal trainable weights)
    """
    metadata_input = Input(shape=input_shape, name=f'metadata_input')

    # Add minimal processing - preserve RF probability information
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name=f'metadata_cast_{index}')(metadata_input)
    x = tf.keras.layers.BatchNormalization(name=f'metadata_BN_{index}')(x)

    # REMOVED extensive layers that were causing overfitting:
    # - Dense(256) → BN → Dropout(0.25)  [REMOVED]
    # - Dense(128) → BN → Dropout(0.15)  [REMOVED]
    # - Dense(64) → BN                   [REMOVED]
    # - Dense(64) → BN                   [REMOVED]
    # - Modular Attention                [REMOVED]

    return metadata_input, x
class OptimizedModularAttention(Layer):
    """Optimized version of ModularAttention"""
    def __init__(self, **kwargs):
        super(OptimizedModularAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='modality_attention',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            dtype='float32'
        )
        super(OptimizedModularAttention, self).build(input_shape)
        
    def call(self, inputs):
        normalized_weights = tf.nn.softmax(self.attention_weights)
        return inputs * tf.expand_dims(normalized_weights, 0)
def create_fusion_layer(branches, num_branches):
    """Optimized version of fusion layer focusing on essential cross-modal attention"""
    if num_branches > 1:
        # Simplified cross-modal attention
        attention_outputs = []
        for i, branch in enumerate(branches):
            # Create attention queries
            query = Dense(64, activation='tanh', use_bias=False, name=f'fusion_query_{i}')(branch)
            
            # Calculate attention with other branches
            attended_branches = []
            for j, other_branch in enumerate(branches):
                if i != j:
                    # Simplified attention mechanism
                    score = tf.reduce_sum(tf.multiply(query, other_branch), axis=-1, keepdims=True)
                    weight = tf.nn.sigmoid(score)  # Simplified weighting
                    attended = tf.multiply(other_branch, weight)
                    attended_branches.append(attended)
            
            if attended_branches:
                # Combine attended features
                attended_sum = tf.add_n(attended_branches)
                attention_outputs.append(tf.add(branch, attended_sum))
            else:
                attention_outputs.append(branch)
        
        return concatenate(attention_outputs, name='modal_fusion')
    else:
        return branches[0]
def create_multimodal_model(input_shapes, selected_modalities, class_weights, strategy):
    with strategy.scope():
        inputs = {}
        branches = []
        
        # Process each modality
        for i, modality in enumerate(selected_modalities):
            if modality == 'metadata':
                metadata_input, branch_output = create_metadata_branch(input_shapes[modality], i)
                inputs[f'metadata_input'] = metadata_input
                branches.append(branch_output)
            elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                image_input, branch_output = create_image_branch(input_shapes[modality], f'{modality}_{i}')
                inputs[f'{modality}_input'] = image_input
                branch_output = tf.keras.layers.Dropout(0.25, name=f'{modality}_dropout_{i}')(branch_output)
                branches.append(branch_output)
        
        # Add sample_id input but don't connect it to the model
        inputs['sample_id'] = Input(shape=(3,), name='sample_id')
        
        # Apply cross-modal fusion
        fused = create_fusion_layer(branches, len(branches))
        # if len(branches) > 1:
        #     merged = concatenate(branches, name='concat_branches')
        # else:
        #     merged = branches[0]
        
        # Final classification layers
        x = Dense(256, activation='relu', use_bias=False, name='final_dense_1')(fused)
        x = tf.keras.layers.BatchNormalization(name='final_BN_1')(x)
        x = tf.keras.layers.Dropout(0.25, name='final_dropout_1')(x)
        x = Dense(128, activation='relu', use_bias=False, name='final_dense_2')(fused)
        x = tf.keras.layers.BatchNormalization(name='final_BN_2')(x)
        x = tf.keras.layers.Dropout(0.15, name='final_dropout_2')(x)
        x = Dense(64, activation='relu', use_bias=False, name='final_dense_3')(fused)
        x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
        x = Dense(32, activation='relu', use_bias=False, name='final_dense_3')(fused)
        x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
        output = Dense(3, activation='softmax', name='output')(x)

        model = Model(inputs=inputs, outputs=output)
        
        # Use the same loss function as before
        # loss = get_weighted_ordinal_crossentropyF1(num_classes=3, ordinal_weight=0.5, f1_weight=1.0)
        loss = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.5, gamma=3.0, alpha=[0.598, 0.315, 1.597])
        model.compile(optimizer=Adam(learning_rate=5e-3, clipnorm=1.0), 
                     loss=loss, 
                     metrics=['accuracy', weighted_f1_score])
        
        return model
#%% Loss Functions
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