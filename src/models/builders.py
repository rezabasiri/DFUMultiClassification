"""
Model architecture builders for multimodal DFU classification.
Functions for creating image branches, metadata branches, fusion layers, and complete models.
"""

import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, concatenate, Concatenate, GlobalAveragePooling2D,
    Multiply, Layer, BatchNormalization, Dropout, Lambda, GlobalAveragePooling1D,
    Flatten, Add, Attention, LayerNormalization, Reshape, MultiHeadAttention, Activation
)
from tensorflow.keras.models import Model
import numpy as np

from src.utils.config import get_project_paths, CLASS_LABELS
from src.utils.verbosity import vprint

# Get paths
directory, result_dir, root = get_project_paths()

# Import IMAGE_SIZE and backbone configs from production config
from src.utils.production_config import IMAGE_SIZE, RGB_BACKBONE, MAP_BACKBONE

def create_simple_cnn_rgb(image_input, modality):
    """Simple CNN for RGB images (4 conv layers)"""
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_0')(image_input)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_2')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_3')(x)
    x = GlobalAveragePooling2D(name=f'{modality}_GAP2D')(x)
    return x

def create_simple_cnn_map(image_input, modality):
    """Simple CNN for map images (3 conv layers)"""
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_1')(image_input)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_2')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', name=f'{modality}_Conv2D_3')(x)
    x = GlobalAveragePooling2D(name=f'{modality}_GAP2D')(x)
    return x

def create_efficientnet_branch(image_input, modality, backbone_name):
    """Create EfficientNet branch with specified variant and unique layer names"""
    # Map backbone name to Keras application
    backbone_map = {
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'EfficientNetB1': tf.keras.applications.EfficientNetB1,
        'EfficientNetB3': tf.keras.applications.EfficientNetB3,
    }

    EfficientNetClass = backbone_map[backbone_name]

    # Try to load local weights, fall back to ImageNet
    local_weights_path = os.path.join(directory, f"local_weights/{backbone_name.lower()}_notop.h5")
    if os.path.exists(local_weights_path):
        vprint(f"Loading {backbone_name} from local weights", level=2)
        # Create base model WITHOUT input_tensor to avoid layer name conflicts
        # Note: EfficientNet wrappers don't support 'name' parameter
        # We'll rename layers after creation instead
        base_model = EfficientNetClass(
            weights=None,
            include_top=False,
            pooling='avg'
        )
        base_model.load_weights(local_weights_path)
    else:
        vprint(f"Loading {backbone_name} from ImageNet", level=2)
        # Create base model WITHOUT input_tensor to avoid layer name conflicts
        # Note: EfficientNet wrappers don't support 'name' parameter
        # We'll rename layers after creation instead
        base_model = EfficientNetClass(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )

    base_model.trainable = True

    # Rename the base model itself to avoid conflicts when same backbone is used for multiple modalities
    base_model._name = f'{modality}_{backbone_name.lower()}'

    # Rename ALL layers (including nested ones) to avoid conflicts between modalities
    # This is critical when using same backbone for multiple modalities
    for layer in base_model.layers:
        layer._name = f'{modality}_{layer.name}'

    # Connect the input manually
    x = base_model(image_input)

    vprint(f"{modality} using {backbone_name}: {len(base_model.trainable_weights)} trainable weights", level=2)

    return x

def create_image_branch(input_shape, modality):
    """Create image branch with configurable backbone"""
    vprint(f"\nCreating image branch for {modality}", level=2)

    image_input = Input(shape=input_shape, name=f'{modality}_input')

    # Select backbone based on modality type and config
    if modality in ['depth_rgb', 'thermal_rgb']:
        backbone = RGB_BACKBONE
        is_rgb = True
    else:  # depth_map, thermal_map
        backbone = MAP_BACKBONE
        is_rgb = False

    vprint(f"{modality} using backbone: {backbone}", level=2)

    # Create feature extractor based on backbone
    if backbone == 'SimpleCNN':
        if is_rgb:
            x = create_simple_cnn_rgb(image_input, modality)
        else:
            x = create_simple_cnn_map(image_input, modality)
    elif backbone.startswith('EfficientNet'):
        x = create_efficientnet_branch(image_input, modality, backbone)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # x = Dense(64, activation='relu', name=f'{modality}_projection3')(x)
        
    # # Projection layer to ensure consistent dimensionality across modalities
    x = Dense(512, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001),  kernel_initializer='he_normal', name=f'{modality}_projection512')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj512')(x)
    x = tf.keras.layers.Dropout(0.15, name=f'{modality}_dropout_512')(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),  kernel_initializer='he_normal', name=f'{modality}_projection1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj1')(x)
    x = tf.keras.layers.Dropout(0.10, name=f'{modality}_dropout_1')(x)
    x = Dense(128, activation='relu',  kernel_initializer='he_normal', name=f'{modality}_projection2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj2')(x)
    # x = tf.keras.layers.Dropout(0.10, name=f'{modality}_dropout_2')(x)
    x = Dense(64, activation='relu',  kernel_initializer='he_normal', name=f'{modality}_projection3')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj3')(x)
    # x = tf.keras.layers.Dropout(0.10, name=f'{modality}_DP_proj3')(x)
    
    # Apply modular attention
    modular_attention = OptimizedModularAttention(name=f'{modality}_modular_attention')
    attention_output = modular_attention(x)
    
    
    return image_input, attention_output
    # return image_input, x

class ConfidenceBasedMetadataAttention(Layer):
    """Attention mechanism that scales based on metadata confidence"""
    def __init__(self, min_scale=1.5, max_scale=3.0, **kwargs):
        super(ConfidenceBasedMetadataAttention, self).__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def build(self, input_shape):
        # Calculate feature dimension (total dim minus RF probs)
        self.feature_dim = input_shape[-1] - 3
        
        # Learnable attention weights for feature dimension
        self.attention_weights = self.add_weight(
            name='metadata_attention',
            shape=(self.feature_dim,),  # Only for features, not RF probs
            initializer='ones',
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            dtype='float32'
        )
        super(ConfidenceBasedMetadataAttention, self).build(input_shape)
    
    def compute_confidence_score(self, probabilities):
        """Calculate confidence score from RF probabilities"""
        max_prob = tf.reduce_max(probabilities, axis=-1, keepdims=True)
        entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10), axis=-1, keepdims=True)
        normalized_entropy = entropy / tf.math.log(3.0)
        confidence = max_prob * (1 - normalized_entropy)
        scaled_confidence = self.min_scale + (self.max_scale - self.min_scale) * confidence
        return scaled_confidence
        
    def call(self, inputs):
        # Split inputs into RF probs and features
        rf_probs = inputs[:, :3]  # First 3 values are RF probabilities
        features = inputs[:, 3:]   # Rest are processed features
        
        # Compute confidence-based scaling
        confidence_scale = self.compute_confidence_score(rf_probs)
        
        # Apply learnable attention to features only
        normalized_weights = tf.nn.softmax(self.attention_weights)
        scaled_features = features * tf.expand_dims(normalized_weights, 0)
        
        # Apply confidence scaling to attended features
        return scaled_features * confidence_scale
    
def create_metadata_branch(input_shape, index):
    """
    Metadata branch - minimal processing to preserve RF quality.

    CRITICAL: RF produces calibrated probabilities (Kappa ~0.20).
    BatchNormalization was destroying probability structure (negative values, wrong scale).
    Just cast to float32 - that's it!
    """
    metadata_input = Input(shape=input_shape, name=f'metadata_input')

    # ONLY cast to float32 - RF probabilities are already calibrated
    # DO NOT use BatchNormalization - it destroys probability structure
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name=f'metadata_cast_{index}')(metadata_input)

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
    """Updated fusion layer with improved metadata scaling and skip connections"""
    if num_branches > 1:
        attention_outputs = []
        has_metadata = any('metadata' in branch.name for branch in branches)
        metadata_branch = None

        if has_metadata:
            metadata_branch = next((branch for branch in branches if 'metadata' in branch.name), None)

        for i, branch in enumerate(branches):
            # Apply skip connection before attention mechanism
            original_branch = branch  # Store the original branch for skip connection

            query = Dense(64, activation='tanh', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f'fusion_query_{i}')(branch)

            attended_branches = []

            for j, other_branch in enumerate(branches):
                if i != j:
                    interaction = tf.multiply(query, other_branch)
                    score = Dense(1, activation='sigmoid', name=f'fusion_score_{i}_{j}')(interaction)

                    # Improved metadata scaling based on confidence
                    if has_metadata and metadata_branch is not None:
                        if branch is metadata_branch:
                            # Metadata querying other modalities, scale down attention
                            weight = tf.nn.sigmoid(score * 0.8)  # Reduced scaling factor
                        elif other_branch is metadata_branch:
                            # Other modalities querying metadata, scale up attention
                            weight = tf.nn.sigmoid(score * 1.5)  # Increased scaling factor
                        else:
                            weight = tf.nn.sigmoid(score)
                    else:
                        weight = tf.nn.sigmoid(score)

                    attended = tf.multiply(other_branch, weight)
                    attended_branches.append(attended)

            if attended_branches:
                attended_sum = tf.add_n(attended_branches)
                attention_outputs.append(tf.add(original_branch, attended_sum))  # Add the original branch for skip connection
            else:
                attention_outputs.append(branch)

        return concatenate(attention_outputs, name='modal_fusion')
    else:
        return branches[0]
# Add visualization callback
class MetadataConfidenceCallback(tf.keras.callbacks.Callback):
    """Callback to monitor metadata confidence and influence when metadata is present"""
    def __init__(self, selected_modalities, log_dir='metadata_confidence_logs'):
        super(MetadataConfidenceCallback, self).__init__()
        self.has_metadata = 'metadata' in selected_modalities
        if self.has_metadata:
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.confidence_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if not self.has_metadata:
            return  # Skip if no metadata
            
        metadata_layer = next(
            (layer for layer in self.model.layers 
             if isinstance(layer, ConfidenceBasedMetadataAttention)),
            None
        )
        
        if metadata_layer and hasattr(metadata_layer, 'last_confidence'):
            avg_confidence = np.mean(metadata_layer.last_confidence)
            self.confidence_history.append(avg_confidence)
            
            if (epoch + 1) % 10 == 0:  # Plot every 10 epochs
                plt.figure(figsize=(10, 5))
                plt.plot(self.confidence_history)
                plt.title('Metadata Confidence History')
                plt.xlabel('Epoch')
                plt.ylabel('Average Confidence Score')
                plt.savefig(os.path.join(self.log_dir, f'metadata_confidence_epoch_{epoch+1}.png'))
                plt.close()
def create_multimodal_model(input_shapes, selected_modalities, class_weights, strategy=None):
    """
    Create multimodal model for DFU classification.

    Key Design Principle:
    - Metadata (RF) branch provides PRE-TRAINED high-quality predictions (Kappa ~0.20)
    - DO NOT re-train Dense layers on top of RF probabilities (degrades to Kappa 0.109)
    - For metadata-only: Use RF predictions directly (Softmax activation only)
    - For multi-modal: Preserve RF quality, learn weighted combination with images

    Args:
        input_shapes: Dictionary of input shapes for each modality
        selected_modalities: List of modalities to use
        class_weights: Class weights for handling imbalance
        strategy: Optional TensorFlow distribution strategy for multi-GPU training
    """
    # Use strategy scope if provided (multi-GPU), otherwise use default scope (single-GPU/CPU)
    scope = strategy.scope() if strategy else tf.keras.utils.custom_object_scope({})

    with scope:
        inputs = {}
        branches = []
        metadata_idx = None

        # Process each modality and track metadata branch
        for i, modality in enumerate(selected_modalities):
            if modality == 'metadata':
                metadata_idx = i
                metadata_input, branch_output = create_metadata_branch(input_shapes[modality], i)
                inputs[f'metadata_input'] = metadata_input
                branches.append(branch_output)
            elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                image_input, branch_output = create_image_branch(input_shapes[modality], f'{modality}')
                inputs[f'{modality}_input'] = image_input
                branches.append(branch_output)

        # Note: sample_id is not added to model inputs - it's tracked externally
        # Keras 3 (TF 2.16+) requires all inputs to be connected to outputs

        # ============================================
        # CRITICAL FIX: Preserve RF Quality
        # ============================================
        # RF probabilities are pre-trained predictions (Kappa 0.20), not features!
        # Do NOT add Dense layers that re-learn classification (degrades to 0.109)

        has_metadata = metadata_idx is not None

        if len(selected_modalities) == 1:
            if has_metadata:
                # METADATA-ONLY: Use RF probabilities directly
                # NO Dense layer - RF already provides optimal predictions
                from src.utils.verbosity import vprint
                vprint("Model: Metadata-only - using RF predictions directly (no Dense layer)", level=2)
                output = Activation('softmax', name='output')(branches[0])
            else:
                # Single image modality - train classifier
                output = Dense(3, activation='softmax', name='output')(branches[0])

        elif len(selected_modalities) == 2:
            if has_metadata:
                # MULTI-MODAL (2): Metadata + 1 Image
                # CRITICAL: Image branch MUST use pre-trained weights from standalone training!
                # Training image branch in fusion mode causes catastrophic overfitting
                from src.utils.verbosity import vprint
                vprint("Model: Metadata + 1 image - two-stage fine-tuning with pre-trained image", level=2)

                # Get RF probabilities (already optimal - Kappa 0.254, proper probabilities)
                rf_probs = branches[metadata_idx]

                # Get image features and classify
                image_idx = 1 - metadata_idx  # The other branch
                image_features = branches[image_idx]
                image_classifier = Dense(3, activation='softmax', name='image_classifier')
                image_probs = image_classifier(image_features)

                # FIXED WEIGHTED AVERAGE FUSION
                # Use FIXED α = 0.70: output = 0.70*RF + 0.30*Image
                # This GUARANTEES RF quality dominates while allowing image contribution

                # Create fixed weights as constants (not trainable)
                rf_weight = 0.70  # RF contributes 70% (since RF has Kappa 0.254)
                image_weight = 0.30  # Image contributes 30%

                vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)

                # Compute weighted average with FIXED weights
                weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
                weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)

                # Sum weighted predictions (always sums to 1.0)
                output = Add(name='output')([weighted_rf, weighted_image])
            else:
                # Two image modalities - original architecture
                merged = concatenate(branches, name='concat_branches')
                output = Dense(3, activation='softmax', name='output')(merged)

        elif len(selected_modalities) == 3:
            if has_metadata:
                # MULTI-MODAL (3): Metadata + 2 Images
                from src.utils.verbosity import vprint
                vprint("Model: Metadata + 2 images - constrained weighted fusion preserving RF quality", level=2)

                # Get RF probabilities (proper probabilities, no BatchNorm)
                rf_probs = branches[metadata_idx]

                # Get image branches and fuse them
                image_branches = [b for i, b in enumerate(branches) if i != metadata_idx]
                image_merged = concatenate(image_branches, name='concat_images')

                # Image processing
                x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='image_dense')(image_merged)
                x = tf.keras.layers.BatchNormalization(name='image_BN')(x)
                x = tf.keras.layers.Dropout(0.10, name='image_dropout')(x)
                image_probs = Dense(3, activation='softmax', name='image_classifier')(x)

                # FIXED weighted average fusion
                rf_weight = 0.70
                image_weight = 0.30
                vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)
                weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
                weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)
                output = Add(name='output')([weighted_rf, weighted_image])
            else:
                # Three image modalities - original architecture
                merged = concatenate(branches, name='concat_branches')
                x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(merged)
                x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
                x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
                output = Dense(3, activation='softmax', name='output')(x)

        elif len(selected_modalities) == 4:
            if has_metadata:
                # MULTI-MODAL (4): Metadata + 3 Images
                from src.utils.verbosity import vprint
                vprint("Model: Metadata + 3 images - constrained weighted fusion preserving RF quality", level=2)

                # Get RF probabilities (proper probabilities, no BatchNorm)
                rf_probs = branches[metadata_idx]

                # Get image branches and fuse them
                image_branches = [b for i, b in enumerate(branches) if i != metadata_idx]
                image_merged = concatenate(image_branches, name='concat_images')

                # Image processing
                x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='image_dense_1')(image_merged)
                x = tf.keras.layers.BatchNormalization(name='image_BN_1')(x)
                x = tf.keras.layers.Dropout(0.10, name='image_dropout_1')(x)
                x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='image_dense_2')(x)
                x = tf.keras.layers.BatchNormalization(name='image_BN_2')(x)
                x = tf.keras.layers.Dropout(0.10, name='image_dropout_2')(x)
                image_probs = Dense(3, activation='softmax', name='image_classifier')(x)

                # FIXED weighted average fusion
                rf_weight = 0.70
                image_weight = 0.30
                vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)
                weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
                weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)
                output = Add(name='output')([weighted_rf, weighted_image])
            else:
                # Four image modalities - original architecture
                merged = concatenate(branches, name='concat_branches')
                x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_3')(merged)
                x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
                x = tf.keras.layers.Dropout(0.10, name='final_dropout_3')(x)
                x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(x)
                x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
                x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
                output = Dense(3, activation='softmax', name='output')(x)

        elif len(selected_modalities) == 5:
            if has_metadata:
                # MULTI-MODAL (5): Metadata + 4 Images
                from src.utils.verbosity import vprint
                vprint("Model: Metadata + 4 images - constrained weighted fusion preserving RF quality", level=2)

                # Get RF probabilities (proper probabilities, no BatchNorm)
                rf_probs = branches[metadata_idx]

                # Get image branches and fuse them
                image_branches = [b for i, b in enumerate(branches) if i != metadata_idx]
                image_merged = concatenate(image_branches, name='concat_images')

                # Image processing
                x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='image_dense_1')(image_merged)
                x = tf.keras.layers.BatchNormalization(name='image_BN_1')(x)
                x = tf.keras.layers.Dropout(0.25, name='image_dropout_1')(x)
                x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='image_dense_2')(x)
                x = tf.keras.layers.BatchNormalization(name='image_BN_2')(x)
                x = tf.keras.layers.Dropout(0.10, name='image_dropout_2')(x)
                x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='image_dense_3')(x)
                x = tf.keras.layers.BatchNormalization(name='image_BN_3')(x)
                x = tf.keras.layers.Dropout(0.10, name='image_dropout_3')(x)
                image_probs = Dense(3, activation='softmax', name='image_classifier')(x)

                # FIXED weighted average fusion
                rf_weight = 0.70
                image_weight = 0.30
                vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)
                weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
                weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)
                output = Add(name='output')([weighted_rf, weighted_image])
            else:
                # Five image modalities - original architecture
                merged = concatenate(branches, name='concat_branches')
                x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_2')(merged)
                x = tf.keras.layers.BatchNormalization(name='final_BN_2')(x)
                x = tf.keras.layers.Dropout(0.25, name='final_dropout_2')(x)
                x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_3')(x)
                x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
                x = tf.keras.layers.Dropout(0.10, name='final_dropout_3')(x)
                x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(x)
                x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
                x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
                output = Dense(3, activation='softmax', name='output')(x)

        model = Model(inputs=inputs, outputs=output)

        return model 
