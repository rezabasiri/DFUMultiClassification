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
from src.utils.production_config import IMAGE_SIZE, RGB_BACKBONE, MAP_BACKBONE, get_modality_config

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
    """Create EfficientNet branch with specified variant and unique layer names

    Keras 3 Compatibility Note:
    - In Keras 3, model/layer names must be globally unique within a Model graph
    - When using the same backbone for multiple modalities, we create separate instances
      with modality-specific names to avoid conflicts
    - CRITICAL: The backbone is called directly (not wrapped in Lambda) so its weights
      are visible to the parent model's optimizer and actually train
    """
    # Cache to store loaded base models (shared weights, separate instances per modality)
    if not hasattr(create_efficientnet_branch, '_model_cache'):
        create_efficientnet_branch._model_cache = {}

    # Map backbone name to Keras application
    backbone_map = {
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'EfficientNetB1': tf.keras.applications.EfficientNetB1,
        'EfficientNetB2': tf.keras.applications.EfficientNetB2,
        'EfficientNetB3': tf.keras.applications.EfficientNetB3,
    }

    EfficientNetClass = backbone_map[backbone_name]

    # Create a unique model instance for this modality
    # Even if we load the same backbone, each modality gets its own instance
    cache_key = f"{modality}_{backbone_name}"

    if cache_key not in create_efficientnet_branch._model_cache:
        # Use modality-specific name to avoid Keras 3 name conflicts
        model_name = f'{modality}_{backbone_name.lower()}'

        # Try to load local weights, fall back to ImageNet
        local_weights_path = os.path.join(directory, f"local_weights/{backbone_name.lower()}_notop.h5")
        if os.path.exists(local_weights_path):
            vprint(f"Loading {backbone_name} from local weights", level=2)
            base_model = EfficientNetClass(
                weights=None,
                include_top=False,
                pooling='avg',
                name=model_name
            )
            base_model.load_weights(local_weights_path)
        else:
            vprint(f"Loading {backbone_name} from ImageNet", level=2)
            # Load with default name first (Keras uses model name in weights download URL)
            # then create a new instance with modality-specific name and transfer weights
            _temp_model = EfficientNetClass(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            base_model = EfficientNetClass(
                weights=None,
                include_top=False,
                pooling='avg',
                name=model_name
            )
            base_model.set_weights(_temp_model.get_weights())
            del _temp_model

        # Cache this model instance for this modality
        create_efficientnet_branch._model_cache[cache_key] = base_model
    else:
        base_model = create_efficientnet_branch._model_cache[cache_key]

    base_model.trainable = True

    # Call backbone directly — exposes all backbone weights to the parent model's optimizer
    # (Lambda wrapping hid backbone weights, causing only projection head to train)
    x = base_model(image_input)

    vprint(f"{modality} using {backbone_name}: {len(base_model.trainable_weights)} trainable weights", level=2)

    return x


def create_generic_backbone_branch(image_input, modality, backbone_name):
    """Create a pretrained backbone branch for non-EfficientNet architectures.

    Supports DenseNet121, ResNet50V2, MobileNetV3Large. These require explicit
    preprocess_input (unlike EfficientNet which has built-in Rescaling).
    Data pipeline delivers [0,255] so preprocessing is applied inside the model.
    """
    if not hasattr(create_generic_backbone_branch, '_model_cache'):
        create_generic_backbone_branch._model_cache = {}

    backbone_map = {
        'DenseNet121': tf.keras.applications.DenseNet121,
        'ResNet50V2': tf.keras.applications.ResNet50V2,
        'MobileNetV3Large': tf.keras.applications.MobileNetV3Large,
    }

    preprocess_map = {
        'DenseNet121': tf.keras.applications.densenet.preprocess_input,
        'ResNet50V2': tf.keras.applications.resnet_v2.preprocess_input,
        'MobileNetV3Large': tf.keras.applications.mobilenet_v3.preprocess_input,
    }

    BackboneClass = backbone_map[backbone_name]
    preprocess_fn = preprocess_map[backbone_name]

    cache_key = f"{modality}_{backbone_name}"

    if cache_key not in create_generic_backbone_branch._model_cache:
        model_name = f'{modality}_{backbone_name.lower()}'

        local_weights_path = os.path.join(directory, f"local_weights/{backbone_name.lower()}_notop.h5")
        if os.path.exists(local_weights_path):
            vprint(f"Loading {backbone_name} from local weights", level=2)
            base_model = BackboneClass(
                weights=None, include_top=False, pooling='avg', name=model_name)
            base_model.load_weights(local_weights_path)
        else:
            vprint(f"Loading {backbone_name} from ImageNet", level=2)
            _temp_model = BackboneClass(
                weights='imagenet', include_top=False, pooling='avg')
            base_model = BackboneClass(
                weights=None, include_top=False, pooling='avg', name=model_name)
            base_model.set_weights(_temp_model.get_weights())
            del _temp_model

        create_generic_backbone_branch._model_cache[cache_key] = base_model
    else:
        base_model = create_generic_backbone_branch._model_cache[cache_key]

    base_model.trainable = True

    # Apply preprocessing (DenseNet/ResNet expect specific normalization, not [0,255])
    from tensorflow.keras.layers import Lambda
    x = Lambda(lambda img: preprocess_fn(img),
               name=f'{modality}_preprocess')(image_input)
    x = base_model(x)

    vprint(f"{modality} using {backbone_name}: {len(base_model.trainable_weights)} trainable weights", level=2)

    return x


def create_image_branch(input_shape, modality):
    """Create image branch with per-modality backbone and projection head.

    Each modality uses its own validated hyperparameters from MODALITY_CONFIGS
    (backbone, head_units, head_l2).  See production_config.py for values.
    """
    mod_cfg = get_modality_config(modality)
    backbone = mod_cfg['backbone']
    head_units = mod_cfg['head_units']
    head_l2 = mod_cfg['head_l2']
    is_rgb = modality in ['depth_rgb', 'thermal_rgb']

    vprint(f"\nCreating image branch for {modality}", level=2)
    vprint(f"{modality} using backbone: {backbone}, head: {head_units}, l2={head_l2}", level=2)

    image_input = Input(shape=input_shape, name=f'{modality}_input')

    # Create feature extractor based on backbone
    if backbone == 'SimpleCNN':
        if is_rgb:
            x = create_simple_cnn_rgb(image_input, modality)
        else:
            x = create_simple_cnn_map(image_input, modality)
    elif backbone.startswith('EfficientNet'):
        x = create_efficientnet_branch(image_input, modality, backbone)
    elif backbone in ('DenseNet121', 'ResNet50V2', 'MobileNetV3Large'):
        x = create_generic_backbone_branch(image_input, modality, backbone)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Per-modality projection head (supports single int or list of ints)
    l2_reg = tf.keras.regularizers.l2(head_l2) if head_l2 > 0 else None
    units_list = head_units if isinstance(head_units, (list, tuple)) else [head_units]
    for i, units in enumerate(units_list):
        suffix = f'_{i}' if len(units_list) > 1 else ''
        x = Dense(units, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=l2_reg, name=f'{modality}_projection{suffix}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj{suffix}')(x)
        x = tf.keras.layers.Dropout(0.3, name=f'{modality}_dropout{suffix}')(x)

    return image_input, x

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

    CRITICAL: RF produces calibrated probabilities.
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
                # METADATA-ONLY: Use RF probabilities directly (already sum to 1.0)
                # NO Dense layer, NO softmax — RF probabilities are pre-normalized.
                # Softmax on valid probabilities distorts them (sharpens peaks, dampens tails).
                from src.utils.verbosity import vprint
                vprint("Model: Metadata-only - using RF predictions directly (no Dense layer)", level=2)
                output = Lambda(lambda x: tf.identity(x), name='output')(branches[0])
            else:
                # Single image modality - train classifier
                output = Dense(3, activation='softmax', name='output', dtype='float32')(branches[0])

        elif len(selected_modalities) >= 2 and has_metadata:
            # FEATURE_CONCAT FUSION (validated by fusion hparam search)
            # RF probs (3) + image features (N) → Dense(3)
            # Much more expressive than prob_concat (RF probs + image probs → Dense(3))
            from src.utils.verbosity import vprint
            n_images = len(selected_modalities) - 1
            vprint(f"Model: Metadata + {n_images} image(s) - feature_concat fusion", level=2)

            rf_probs = branches[metadata_idx]
            image_branches = [b for i, b in enumerate(branches) if i != metadata_idx]

            if len(image_branches) > 1:
                image_features = concatenate(image_branches, name='concat_image_features')
            else:
                image_features = image_branches[0]

            # Feature-level fusion: concat RF probs with raw image features
            # (not image probs — preserves richer feature representation)
            fused = concatenate([rf_probs, image_features], name='fusion_concat')
            output = Dense(3, activation='softmax', name='output', dtype='float32',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))(fused)

        elif len(selected_modalities) == 2:
            # Two image modalities (no metadata)
            merged = concatenate(branches, name='concat_branches')
            output = Dense(3, activation='softmax', name='output', dtype='float32')(merged)

        elif len(selected_modalities) == 3:
            # Three image modalities (no metadata)
            merged = concatenate(branches, name='concat_branches')
            output = Dense(3, activation='softmax', name='output', dtype='float32')(merged)

        elif len(selected_modalities) == 4:
            # Four image modalities (no metadata)
            merged = concatenate(branches, name='concat_branches')
            x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_3')(merged)
            x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_3')(x)
            x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(x)
            x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
            output = Dense(3, activation='softmax', name='output', dtype='float32')(x)

        elif len(selected_modalities) >= 5:
            # Five+ image modalities (no metadata)
            merged = concatenate(branches, name='concat_branches')
            output = Dense(3, activation='softmax', name='output', dtype='float32')(merged)

        model = Model(inputs=inputs, outputs=output)

        return model 
