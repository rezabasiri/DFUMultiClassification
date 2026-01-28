#!/usr/bin/env python3
"""
Main training script for DFU Multi-Classification.
Multimodal deep learning for Diabetic Foot Ulcer healing phase classification.
"""

#%% Import Libraries and Configure Environment
import os
import sys
import warnings

# Suppress sklearn FutureWarnings about deprecated internal methods (sklearn 1.6.x issue)
# These are sklearn's internal deprecation warnings, not issues with our code
warnings.filterwarnings('ignore', message='.*BaseEstimator._check_n_features.*')
warnings.filterwarnings('ignore', message='.*BaseEstimator._check_feature_names.*')

# Suppress TensorFlow C++ logging (must be set before importing TensorFlow)
# This will be overridden later based on verbosity level, but default to minimal logging
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress INFO and WARNING messages (keep errors only)

# XLA JIT compilation control (must be set BEFORE importing TensorFlow)
# Import the setting from production_config without importing TensorFlow
from src.utils.production_config import DISABLE_XLA_JIT
if DISABLE_XLA_JIT:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    print("[CONFIG] XLA JIT disabled (fast startup for testing)")
else:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    print("[CONFIG] XLA JIT enabled (slow first step, faster training)")

# GPU configuration will be set up later via argparse and gpu_config module
# DO NOT set CUDA_VISIBLE_DEVICES here - it's handled dynamically based on --device-mode
import glob
import pandas as pd
import numpy as np
import re
import random
import logging
import pickle
import argparse
import math
import csv
import itertools
from collections import Counter
import gc

# Add project root to path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Scikit-learn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# PIL
from PIL import Image

# SHAP
import shap

# Project imports - organized by module
from src.utils.config import get_project_paths, get_data_paths, get_output_paths, CLASS_LABELS, RANDOM_SEED
from src.utils.production_config import *  # Import all production configuration parameters
from src.utils.debug import clear_gpu_memory, reset_keras, clear_cuda_memory
from src.utils.verbosity import set_verbosity, vprint, get_verbosity, init_progress_bar, update_progress, close_progress
from src.utils.gpu_config import setup_device_strategy, get_effective_batch_size, print_gpu_memory_usage
from src.utils.outlier_detection import (
    detect_outliers, apply_cleaned_dataset, restore_original_dataset, check_metadata_in_modalities,
    detect_outliers_combination, apply_cleaned_dataset_combination
)
from src.data.image_processing import (
    extract_info_from_filename, find_file_match, find_best_alternative,
    create_best_matching_dataset, prepare_dataset, preprocess_image_data,
    adjust_bounding_box, load_and_preprocess_image, create_default_image,
    is_valid_bounding_box
)
from src.data.dataset_utils import (
    create_cached_dataset, check_split_validity, prepare_cached_datasets,
    visualize_dataset, plot_net_confusion_matrix
)
from src.data.generative_augmentation_sdxl import (
    GenerativeAugmentationManager,
    GenerativeAugmentationCallback,
    create_enhanced_augmentation_fn,
    AugmentationConfig
)
from src.models.builders import (
    create_image_branch, create_metadata_branch, create_fusion_layer,
    create_multimodal_model
)
from src.models.losses import (
    weighted_f1_score, weighted_ordinal_crossentropy,
    get_weighted_ordinal_crossentropy, get_weighted_ordinal_crossentropyF1,
    focal_ordinal_loss, get_focal_ordinal_loss
)
from src.training.training_utils import (
    clean_up_training_resources, create_checkpoint_filename,
    analyze_modality_contributions, average_attention_values,
    cross_validation_manual_split, correct_and_validate_predictions,
    save_run_results, save_run_metrics, save_gating_results,
    save_aggregated_results, save_run_predictions, load_run_predictions,
    get_completed_configs_for_run, load_aggregated_predictions,
    save_aggregated_predictions, is_run_complete, WeightedAccuracy,
    save_combination_predictions, load_combination_predictions
)
from src.evaluation.metrics import filter_frequent_misclassifications, track_misclassifications

#%% Environment Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.basicConfig(level=logging.INFO)

# Configure GPU memory growth - MOVED to after setup_device_strategy() is called
# This is now done in setup_device_strategy() to avoid initializing TF before CUDA_VISIBLE_DEVICES is set
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         vprint(str(e), level=1)

# Set random seeds for reproducibility (non-TensorFlow only - TF seed set later)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# TensorFlow configuration - MOVED to after setup_device_strategy() to avoid early GPU init
# apply_environment_config()  # Apply threading and determinism settings
# tf.random.set_seed(RANDOM_SEED)
# tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# Distributed strategy - DO NOT initialize here! It will be set by setup_device_strategy() in main()
# Setting it here would initialize TensorFlow before CUDA_VISIBLE_DEVICES is configured
strategy = None  # Will be set in main() after parsing args

#%% Path Configuration
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
output_paths = get_output_paths(result_dir)

ck_path = output_paths['checkpoints']
csv_path = output_paths['csv']
models_path = output_paths['models']
misclass_path = output_paths['misclassifications']
vis_path = output_paths['visualizations']
logs_path = output_paths['logs']

# Data paths
image_folder = data_paths['image_folder']
depth_folder = data_paths['depth_folder']
thermal_folder = data_paths['thermal_folder']
thermal_rgb_folder = data_paths['thermal_rgb_folder']
csv_file = data_paths['csv_file']
depth_bb_file = data_paths['bb_depth_csv']
thermal_bb_file = data_paths['bb_thermal_csv']

# Load bounding box data
depth_bb = pd.read_csv(depth_bb_file)
thermal_bb = pd.read_csv(thermal_bb_file)

#%% Training Parameters - using production_config values
image_size = IMAGE_SIZE
global_batch_size = GLOBAL_BATCH_SIZE
# batch_size will be calculated in main() after strategy is set up
# batch_size = global_batch_size // strategy.num_replicas_in_sync
n_epochs = N_EPOCHS

#%% Main Function Section
#%% Main Function
def create_focal_ordinal_loss_with_params(ordinal_weight, gamma, alpha):
    """Create a focal ordinal loss function with specified parameters"""
    vprint("\nInitializing focal ordinal loss:", level=2)
    vprint(f"Ordinal weight: {ordinal_weight}", level=2)
    vprint(f"Gamma: {gamma}", level=2)
    vprint(f"Alpha: {alpha}", level=2)

    def loss_fn(y_true, y_pred):
        # Print shapes for debugging (only at verbosity level 2)
        if get_verbosity() == 2:
            print(f"\nLoss function shapes:")
            print(f"y_true shape: {tf.shape(y_true)}")
            print(f"y_pred shape: {tf.shape(y_pred)}")

        # Clip prediction values to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Convert alpha to tensor with proper shape
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

        # Focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.expand_dims(alpha_tensor, 0) * tf.math.pow(1 - y_pred, gamma)
        focal_loss = focal_weight * cross_entropy

        # Print intermediate values for first sample (only at verbosity level 2)
        if get_verbosity() == 2:
            print("\nIntermediate values (first sample):")
            print(f"Cross entropy: {cross_entropy[0]}")
            print(f"Focal weight: {focal_weight[0]}")
            print(f"Focal loss: {focal_loss[0]}")

        # Sum over classes
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)

        # Ordinal penalty
        true_class = tf.argmax(y_true, axis=-1)
        pred_class = tf.argmax(y_pred, axis=-1)
        ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))

        # Print ordinal penalty information (only at verbosity level 2)
        if get_verbosity() == 2:
            print(f"Ordinal penalty (first 5 samples): {ordinal_penalty[:5]}")

        # Combine losses
        total_loss = focal_loss + ordinal_weight * ordinal_penalty

        # Print final loss (only at verbosity level 2)
        if get_verbosity() == 2:
            print(f"Final loss (first 5 samples): {total_loss[:5]}")
        return total_loss

    return loss_fn

def create_attention_visualization_callback(val_data, val_labels, save_dir='attention_weights'):
    """Enhanced callback with explicit validation data"""
    os.makedirs(save_dir, exist_ok=True)
    
    class AttentionVisualizerCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data, val_labels, save_dir):
            super().__init__()
            self.save_dir = save_dir
            self.class_attention_history = {
                'I': [], 'P': [], 'R': []
            }
            self.val_data = val_data
            self.val_labels = val_labels


        def on_train_begin(self, logs=None):
            """Store validation data at the start of training"""
            if hasattr(self.model, 'validation_data'):
                self.val_data = self.model.validation_data[0]
                self.val_labels = self.model.validation_data[1]
            # else:
            #     print("Warning: No validation data found in model")

        def analyze_class_specific_attention(self, model_weights, class_weights, predictions, true_labels):
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(true_labels, axis=1)
            
            # Analyze attention patterns for each class
            class_analysis = {}
            for class_idx, class_name in enumerate(['I', 'P', 'R']):
                # Get samples for this class
                class_mask = true_classes == class_idx
                correct_mask = (pred_classes == class_idx) & class_mask
                
                if np.any(class_mask):
                    class_analysis[class_name] = {
                        'model_attention': {
                            'all': np.mean(model_weights[class_mask], axis=(0,1)),
                            'correct': np.mean(model_weights[correct_mask], axis=(0,1)) if np.any(correct_mask) else None
                        },
                        'class_attention': {
                            'all': np.mean(class_weights[class_mask], axis=(0,1)),
                            'correct': np.mean(class_weights[correct_mask], axis=(0,1)) if np.any(correct_mask) else None
                        },
                        'accuracy': np.mean(correct_mask[class_mask]) if np.any(class_mask) else 0
                    }
            
            return class_analysis

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % ATTENTION_VIS_FREQUENCY == 0:
                vprint(f"\nProcessing attention visualization for epoch {epoch + 1}", level=2)
                
                attention_layer = None
                for layer in self.model.layers:
                    if isinstance(layer, DualLevelAttentionLayer):
                        attention_layer = layer
                        break
                
                if attention_layer:
                    try:
                        attention_values = attention_layer.get_attention_values()
                        
                        if attention_values['model_attention'] is None:
                            vprint("Warning: model_attention is None", level=2)
                            return
                        if attention_values['class_attention'] is None:
                            vprint("Warning: class_attention is None", level=2)
                            return
                            
                        model_weights = attention_values['model_attention']
                        class_weights = attention_values['class_attention']
                        
                        # Average over batch and heads if needed
                        Avg_model_weights = np.mean(model_weights, axis=(0, 1))
                        Avg_class_weights = np.mean(class_weights, axis=(0, 1))
                        
                        # Define global min and max values for consistent scaling - from production_config
                        model_vmin = ATTENTION_MODEL_VMIN
                        model_vmax = ATTENTION_MODEL_VMAX
                        class_vmin = ATTENTION_CLASS_VMIN
                        class_vmax = ATTENTION_CLASS_VMAX
                        # Create model name mapping
                        model_names = {
                            '1': 'Metadata',
                            '2': 'RGB',
                            '3': 'Depth',
                            '4': 'Thermal',
                            '5': 'Thermal+Depth',
                            '6': 'Metadata+RGB+Depth',
                            '7': 'Metadata+RGB+Thermal',
                            '8': 'Metadata+Thermal+Depth',
                            '9': 'RGB+Thermal+Depth'
                        }
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Plot model attention with fixed scale
                        sns.heatmap(Avg_model_weights, ax=ax1, cmap='viridis',
                                    vmin=model_vmin, vmax=model_vmax,
                                    xticklabels=range(1, Avg_model_weights.shape[1] + 1),
                                    yticklabels=range(1, Avg_model_weights.shape[0] + 1),
                                    cbar_kws={'label': 'Attention Weight'})
                        ax1.set_title('Model-Level Attention', fontsize=14)
                        ax1.set_xlabel('Target Models', fontsize=14)
                        ax1.set_ylabel('Source Models', fontsize=14)
                        ax1.tick_params(axis='both', which='major', labelsize=14)
                        
                        # Add legend for model names
                        legend_elements = [f"{k}: {v}" for k, v in model_names.items()]
                        ax1.text(1.3, 0.5, '\n'.join(legend_elements),
                                transform=ax1.transAxes, fontsize=14,
                                verticalalignment='center')

                        # Plot class attention with fixed scale
                        sns.heatmap(Avg_class_weights, ax=ax2, cmap='viridis',
                                    vmin=class_vmin, vmax=class_vmax,
                                    xticklabels=['I', 'P', 'R'],
                                    yticklabels=['I', 'P', 'R'],
                                    cbar_kws={'label': 'Attention Weight'})
                        ax2.set_title('Class-Level Attention', fontsize=14)
                        ax2.set_xlabel('Target Classes', fontsize=14)
                        ax2.set_ylabel('Source Classes', fontsize=14)
                        ax2.tick_params(axis='both', which='major', labelsize=14)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.save_dir, f'Avg_attention_weights_epoch_{epoch+1}.png'))
                        plt.close()
                        
                        if self.val_data is None or self.val_labels is None:
                            vprint("No validation data available. Trying to get from model...", level=2)
                            if hasattr(self.model, 'validation_data'):
                                self.val_data = self.model.validation_data[0]
                                self.val_labels = self.model.validation_data[1]
                            else:
                                vprint("Error: Cannot access validation data", level=1)
                                return
                        
                        # Get predictions
                        val_predictions = self.model.predict(self.val_data, verbose=0)
                        val_labels = self.val_labels
                        
                        # Analyze per-class attention
                        class_analysis = self.analyze_class_specific_attention(
                            model_weights, class_weights, val_predictions, val_labels
                        )
                        
                        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
                        for idx, class_name in enumerate(['I', 'P', 'R']):
                            if class_name in class_analysis:
                                analysis = class_analysis[class_name]
                                
                                # Plot model attention for this class with fixed scale
                                sns.heatmap(analysis['model_attention']['all'].reshape(-1, model_weights.shape[-1]), 
                                            ax=axes[idx,0], cmap='viridis',
                                            vmin=model_vmin, vmax=model_vmax,
                                            xticklabels=range(1, model_weights.shape[-1] + 1),
                                            yticklabels=range(1, model_weights.shape[-1] + 1),
                                            cbar_kws={'label': 'Attention Weight'})
                                axes[idx,0].set_title(f'{class_name} Class Model Attention', fontsize=14)
                                axes[idx,0].tick_params(axis='both', which='major', labelsize=14)
                                
                                # Add legend for the first plot only
                                if idx == 0:
                                    legend_elements = [f"{k}: {v}" for k, v in model_names.items()]
                                    axes[idx,0].text(1.3, 0.5, '\n'.join(legend_elements),
                                                transform=axes[idx,0].transAxes, fontsize=14,
                                                verticalalignment='center')
                                
                                # Plot class attention for this class with fixed scale
                                sns.heatmap(analysis['class_attention']['all'].reshape(-1, class_weights.shape[-1]),
                                            ax=axes[idx,1], cmap='viridis',
                                            vmin=class_vmin, vmax=class_vmax,
                                            xticklabels=['I', 'P', 'R'],
                                            yticklabels=['I', 'P', 'R'],
                                            cbar_kws={'label': 'Attention Weight'})
                                axes[idx,1].set_title(f'{class_name} Class Attention Pattern', fontsize=14)
                                axes[idx,1].tick_params(axis='both', which='major', labelsize=14)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.save_dir, f'class_attention_epoch_{epoch+1}.png'))
                        plt.close()
                        
                        # Print analysis
                        # print(f"\nEpoch {epoch + 1} Per-Class Attention Analysis:")
                        for class_name in ['I', 'P', 'R']:
                            if class_name in class_analysis:
                                analysis = class_analysis[class_name]
                                # print(f"\n{class_name} Class:")
                                # print(f"Accuracy: {analysis['accuracy']:.4f}")
                                # print(f"Average model attention: {np.mean(analysis['model_attention']['all']):.4f}")
                                # print(f"Average class attention: {np.mean(analysis['class_attention']['all']):.4f}")
                                # if analysis['model_attention']['correct'] is not None:
                                #     print("Attention patterns for correct predictions:")
                                #     print(f"Model attention: {np.mean(analysis['model_attention']['correct']):.4f}")
                                #     print(f"Class attention: {np.mean(analysis['class_attention']['correct']):.4f}")
                
                    except Exception as e:
                        vprint(f"Error in attention visualization: {str(e)}", level=1)
                        if get_verbosity() == 2:
                            print("Detailed error information:")
                            import traceback
                            traceback.print_exc()

                            # Print debug information
                            print("\nDebug Information:")
                            print(f"Attention layer found: {attention_layer is not None}")
                            if attention_layer:
                                print("Attention values available:")
                                values = attention_layer.get_attention_values()
                                print(f"Model attention: {'Yes' if values['model_attention'] is not None else 'No'}")
                                print(f"Class attention: {'Yes' if values['class_attention'] is not None else 'No'}")
                else:
                    vprint("Warning: Could not find attention layer in model", level=2)
    
    return AttentionVisualizerCallback(val_data, val_labels, save_dir)
            
from tensorflow.keras import layers, models

class BestModelAttentionCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_callback, monitor='val_weighted_accuracy', mode='max'):
        super().__init__()
        self.base_callback = base_callback
        self.monitor = monitor
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        
    def set_model(self, model):
        super().set_model(model)
        # Also set the model for the base callback
        self.base_callback.model = model
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'max':
            improved = current > self.best
        else:
            improved = current < self.best
            
        if improved:
            self.best = current
            # Only call the attention visualization when metric improves
            self.base_callback.on_epoch_end(epoch, logs)
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        # Store the arguments as instance attributes
        self._input_dim = input_dim  # using _input_dim to avoid conflict
        self._dropout_rate = dropout_rate
        
        # Create layers in __init__
        self.dense1 = tf.keras.layers.Dense(input_dim, activation=None)
        self.dense2 = tf.keras.layers.Dense(input_dim, activation=None)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.activation = tf.keras.layers.ReLU()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self._input_dim,
            "dropout_rate": self._dropout_rate,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=True):
        # First sub-block
        x = self.norm1(inputs)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        
        # Second sub-block
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        
        # Add residual connection
        return inputs + x

class DualLevelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.key_dim = key_dim
        
        # Store numpy arrays directly
        self.model_attention_values = None
        self.class_attention_values = None
        self.model_attention_scores = None
        self.class_attention_scores = None
        
        # Add trainable temperature parameter
        self.temperature = tf.Variable(0.1, trainable=True, name='attention_temperature')
        
        # Add separate attention for different purposes
        self.model_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        
        self.class_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, #num_heads//2,  # Fewer heads for class attention or fix values becuase calsses dont change
            key_dim=32,       # Larger key dimension
            dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        
        # Add gating mechanism
        self.gate = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "num_classes": self.num_classes,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def _store_attention(self, attention_weights, is_model_attention=True):
        """Convert attention weights to numpy immediately"""
        def convert_to_numpy(x):
            result = x.numpy()
            if is_model_attention:
                self.model_attention_values = result
            else:
                self.class_attention_values = result
            return x  # Return original tensor to maintain graph

        return tf.py_function(convert_to_numpy, [attention_weights], attention_weights.dtype)

    def call(self, inputs, training=True):
        # Scale inputs by learned temperature
        scaled_inputs = inputs / tf.math.maximum(self.temperature, 0.01)
        
        # Model-level attention
        model_attended, model_weights = self.model_attention(
            query=scaled_inputs,
            value=scaled_inputs,
            key=scaled_inputs,
            training=training,
            return_attention_scores=True
        )
        # Store model attention weights
        _ = self._store_attention(model_weights, is_model_attention=True)
        self.model_attention_scores = model_weights
        
        # Dynamic gating
        gate_value = self.gate(tf.reduce_mean(model_attended, axis=1, keepdims=True))
        model_attended = model_attended * gate_value
        
        # Class-level attention with different scaling
        transposed = tf.transpose(model_attended, perm=[0, 2, 1])
        class_attended, class_weights = self.class_attention(
            query=transposed,
            value=transposed,
            key=transposed,
            training=training,
            return_attention_scores=True
        )
        # Store class attention weights
        _ = self._store_attention(class_weights, is_model_attention=False)
        self.class_attention_scores = class_weights
        
        output = tf.transpose(class_attended, perm=[0, 2, 1])
        
        # Final processing
        return tf.reduce_mean(output, axis=1)
    def get_attention_scores(self):
        """Check if attention scores are available"""
        if self.model_attention_scores is None or self.class_attention_scores is None:
            vprint("Warning: One or both attention scores are None", level=2)
            if self.model_attention_scores is None:
                vprint("Model attention scores are None", level=2)
            if self.class_attention_scores is None:
                vprint("Class attention scores are None", level=2)
            return {'model_attention': None, 'class_attention': None}
        
        return {
            'model_attention': self.model_attention_scores,
            'class_attention': self.class_attention_scores
        }
    def get_attention_values(self):
        """Return stored numpy attention values"""
        return {
            'model_attention': self.model_attention_values,
            'class_attention': self.class_attention_values
        }

def attention_entropy_loss(attention_scores):
    def calculate_entropy(scores):
        # Normalize scores
        scores = tf.nn.softmax(scores, axis=-1)
        epsilon = ENTROPY_EPSILON
        scores = tf.clip_by_value(scores, epsilon, 1.0)
        
        # Calculate entropy
        entropy = -tf.reduce_sum(scores * tf.math.log(scores), axis=-1)
        
        # Calculate standard deviation manually
        mean = tf.reduce_mean(scores, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(scores - mean), axis=-1)
        diversity = tf.sqrt(variance + epsilon)  # Add epsilon to avoid sqrt(0)
        
        return tf.reduce_mean(entropy) + tf.reduce_mean(diversity)
    
    model_entropy = calculate_entropy(attention_scores['model_attention'])
    class_entropy = calculate_entropy(attention_scores['class_attention'])

    # Balance between model and class attention - from production_config
    return ENTROPY_MODEL_WEIGHT * model_entropy + ENTROPY_CLASS_WEIGHT * class_entropy

def custom_loss(model):
    def loss_function(y_true, y_pred):
        # Base classification loss
        base_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        )
        
        attention_layer = next((layer for layer in model.layers 
                              if isinstance(layer, DualLevelAttentionLayer)), None)
        
        if attention_layer is not None:
            attention_scores = attention_layer.get_attention_scores()
            if all(v is not None for v in attention_scores.values()):
                entropy_loss = attention_entropy_loss(attention_scores)

                # Dynamic entropy weight based on training progress - from production_config
                entropy_weight = ENTROPY_LOSS_WEIGHT * (1.0 - tf.exp(-base_loss))
                
                total_loss = base_loss - entropy_weight * entropy_loss
                
                # tf.print("Loss components:", {
                #     'base_loss': base_loss,
                #     'entropy_weight': entropy_weight,
                #     'entropy_loss': entropy_loss,
                #     'total_loss': total_loss,
                #     'temperature': attention_layer.temperature
                # })
                return total_loss
        
        return base_loss
    
    return loss_function
class DynamicLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self,
                 initial_lr=LR_SCHEDULE_INITIAL_LR,
                 min_lr=LR_SCHEDULE_MIN_LR,
                 exploration_epochs=LR_SCHEDULE_EXPLORATION_EPOCHS,
                 cycle_length=LR_SCHEDULE_CYCLE_LENGTH,
                 cycle_multiplier=LR_SCHEDULE_CYCLE_MULTIPLIER):
        super(DynamicLRSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.exploration_epochs = exploration_epochs
        self.cycle_length = cycle_length
        self.cycle_multiplier = cycle_multiplier
        
        # Internal state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.current_cycle = 0
        self.last_restart = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.exploration_epochs:
            # Exploration phase: higher learning rate
            lr = self.initial_lr
        else:
            # Calculate position in current cycle
            cycle_epoch = epoch - self.last_restart
            if cycle_epoch >= self.cycle_length:
                # Start new cycle
                self.current_cycle += 1
                self.last_restart = epoch
                cycle_epoch = 0
                # Increase cycle length
                self.cycle_length = int(self.cycle_length * self.cycle_multiplier)
            
            # Cosine annealing within cycle
            progress = cycle_epoch / self.cycle_length
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * progress))
            
            # Calculate learning rate
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
            
            # Adjust based on loss improvement
            if logs and 'loss' in logs:
                current_loss = logs['loss']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    # If loss hasn't improved for a while, reduce max learning rate - from production_config
                    if self.patience_counter >= LR_SCHEDULE_PATIENCE_THRESHOLD:
                        self.initial_lr *= LR_SCHEDULE_DECAY_FACTOR
                        self.patience_counter = 0
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Log the learning rate for monitoring
        if logs is None:
            logs = {}
        logs['lr'] = lr
def ImprovedGatingNetwork(num_models=16, num_classes=3):
    input_layer = tf.keras.layers.Input(shape=(num_models, num_classes))

    # Main attention and processing - using production_config values
    dual_attention_output = DualLevelAttentionLayer(
        num_heads=max(8, num_models + GATING_NUM_HEADS_MULTIPLIER),
        key_dim=max(16, GATING_KEY_DIM_MULTIPLIER * (num_models + 1)),
        num_classes=num_classes
    )(input_layer)
    
    # Final residual block and prediction
    final_residual = ResidualBlock(num_classes)(dual_attention_output)
    predictions = tf.nn.softmax(final_residual)  # Direct softmax since dimensions already match
    
    return tf.keras.Model(inputs=input_layer, outputs=predictions)
def train_model_combination(train_data, val_data, train_labels, val_labels):
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Validate input data
    if train_data is None or len(train_data) == 0:
        raise ValueError("Train data is empty")
    if train_labels is None or len(train_labels) == 0:
        raise ValueError("Train labels is empty")

    # print(f"\nTraining model combination with {len(train_data[0])} models")
    if len(train_labels.shape) == 1:
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=3)

    # Get class indices from labels
    train_label_indices = np.argmax(train_labels, axis=1)
    unique_classes = np.unique(train_label_indices)

    vprint(f"Debug train_model_combination: train_labels shape={train_labels.shape}, indices shape={train_label_indices.shape}", level=2)
    vprint(f"Debug: unique_classes={unique_classes}, train_label_indices[:10]={train_label_indices[:10]}", level=2)

    # Handle case where not all classes are present
    if len(unique_classes) == 0:
        raise ValueError("No classes found in training labels")

    # Calculate class weights (only for classes present)
    try:
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_label_indices)
    except Exception as e:
        vprint(f"Error in compute_class_weight: {e}", level=1)
        vprint(f"  unique_classes type={type(unique_classes)}, shape={unique_classes.shape}", level=1)
        vprint(f"  train_label_indices type={type(train_label_indices)}, shape={train_label_indices.shape}", level=1)
        raise
    class_weights_dict = {int(c): w for c, w in zip(unique_classes, class_weights)}
    # Fill in missing classes with weight 1.0
    for i in range(3):
        if i not in class_weights_dict:
            class_weights_dict[i] = 1.0

    # print(f"Class weights for Gating Network: {class_weights_dict}")
    tf.keras.backend.clear_session()
    counts = Counter(train_label_indices)
    total_samples = sum(counts.values())
    class_frequencies = {cls: count/total_samples for cls, count in counts.items()}
    median_freq = np.median(list(class_frequencies.values()))
    # Handle missing classes in frequency calculation
    alpha_values = []
    for i in [0, 1, 2]:
        if i in class_frequencies:
            alpha_values.append(median_freq/class_frequencies[i])
        else:
            alpha_values.append(1.0)  # Default weight for missing class
    alpha_sum = sum(alpha_values)
    alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]


    weighted_acc = WeightedAccuracy(alpha_values=class_weights, name='weighted_accuracy')
    
    # with strategy.scope():
    # Build the model
    model = ImprovedGatingNetwork(len(train_data[0]), 3)
    model.compile(
        optimizer=Adam(learning_rate=GATING_LEARNING_RATE),
        loss=custom_loss(model),
        metrics=['accuracy', weighted_acc]
    )
    # # Create distributed datasets #TODO Distribute the second training for multigpu processing
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    # # Set proper batch size
    # global_batch_size = batch_size * strategy.num_replicas_in_sync
    # train_dataset = train_dataset.batch(global_batch_size)
    # val_dataset = val_dataset.batch(global_batch_size)
    
    # # Distribute datasets
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    # Setup callbacks - using production_config values
    callbacks2 = [
        # DynamicLRSchedule can be enabled with custom parameters
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=GATING_REDUCE_LR_FACTOR,
            patience=GATING_REDUCE_LR_PATIENCE,
            min_lr=GATING_REDUCE_LR_MIN_LR,
            min_delta=GATING_REDUCE_LR_MIN_DELTA,
            verbose=0,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_weighted_accuracy',
            patience=GATING_EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=GATING_EARLY_STOP_VERBOSE,
            min_delta=GATING_EARLY_STOP_MIN_DELTA,
            mode='max'
        )
    ]
    # Create a custom training history object that includes learning rate
    class CustomHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None and hasattr(self.model.optimizer, 'lr'):
                logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
    callbacks2.append(create_attention_visualization_callback(val_data=val_data, val_labels=val_labels, save_dir=os.path.join(result_dir, 'attention_analysis')))
    # Add custom history to callbacks
    callbacks2.append(CustomHistory())
    
    # Add deterministic data handling - using production_config values
    # Use smaller batch size if data is too small
    actual_batch_size = min(GATING_BATCH_SIZE, len(train_labels))
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    dataset = dataset.batch(actual_batch_size, drop_remainder=False)  # Changed to False for small datasets
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(len(val_labels), drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Train the model - using production_config values
    history = model.fit(
        dataset,
        class_weight=class_weights_dict,
        epochs=GATING_EPOCHS,
        # batch_size is set in dataset.batch() above
        validation_data=val_dataset,
        callbacks=callbacks2,
        shuffle=True,
        verbose=GATING_VERBOSE
    )

    # Predict outputs
    predictions = model.predict(train_data, verbose=0)
    val_predictions = model.predict(val_data, verbose=0)
    
    # Calculate validation accuracy
    accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(val_labels, axis=1))
    kappa = cohen_kappa_score(np.argmax(val_labels, axis=1), np.argmax(val_predictions, axis=1), weights='quadratic')
    val_weighted_acc = np.max(history.history['val_weighted_accuracy'])
    # print(f"Final predictions shape: {predictions.shape}")
    # print(f"Validation accuracy: {accuracy:.4f}, Cohen Kappa Score: {kappa:.4f}")

    return val_predictions, history.history['loss'][-1], accuracy, kappa, model, val_weighted_acc
import json
from multiprocessing import Pool
from functools import partial
# Create a function that processes a single trial
def process_trial(trial, all_combinations, completed_combinations, train_predictions_list, 
                 valid_predictions_list, train_labels, valid_labels, run_number, max_tries_min):
    selected_indices = all_combinations[trial]
    selected_key = tuple(sorted(selected_indices))
    current_model_count = len(selected_indices)
    
    # Count how many combinations already tried for this number of models
    existing_combinations_count = sum(1 for combo in completed_combinations if len(combo) == current_model_count)
    remaining_tries = max_tries_min - existing_combinations_count
    # If already completed all tries for this model count, skip
    if remaining_tries <= 0:
        return None
    # If this specific combination was already tried, skip
    if selected_key in completed_combinations:
        return None
    # Keep track of how many new combinations processed for this model count
    new_combinations_count = sum(1 for combo in completed_combinations if len(combo) == current_model_count and combo not in completed_combinations)     
    # Skip if processed enough new combinations to reach max_tries_min
    if new_combinations_count >= remaining_tries:
        return None
    
    train_predictions_selected = [train_predictions_list[i] for i in selected_indices]
    valid_predictions_selected = [valid_predictions_list[i] for i in selected_indices]
    
    try:
        train_data_truncated, train_labels_truncated = correct_and_validate_predictions(
            train_predictions_selected, train_labels, "train")
        val_data_truncated, val_labels_truncated = correct_and_validate_predictions(
            valid_predictions_selected, valid_labels, "valid")
        
        train_data_sliced = np.stack(train_data_truncated, axis=1).astype(np.float32)
        val_data_sliced = np.stack(val_data_truncated, axis=1).astype(np.float32)
        
        predictions, loss, accuracy, kappa, _, val_weighted_acc = train_model_combination(train_data_sliced, val_data_sliced, train_labels_truncated, val_labels_truncated)
        
        save_progress(run_number, completed_combination=selected_key)
        
        return {
            'selected_indices': selected_indices,
            'predictions': predictions,
            'loss': loss,
            'accuracy': accuracy,
            'kappa': kappa,
            'weighted_accuracy': val_weighted_acc,
            'selected_key': selected_key
        }
    except Exception as e:
        vprint(f"Error in trial {trial}: {str(e)}", level=1)
        return None
def train_gating_network(train_predictions_list, valid_predictions_list, train_labels, valid_labels, run_number, find_optimal=True, min_models=2, max_tries=100):
    tf.config.experimental.enable_op_determinism()
    vprint("\nInitializing gating network training...", level=1)
    vprint(f"Number of models: {len(train_predictions_list)}", level=2)
    vprint(f"Shape of first model predictions: {train_predictions_list[0].shape}", level=2)
    vprint(f"Shape of true labels: {train_labels.shape}", level=2)

    # Skip gating network training if there's only 1 model (nothing to combine)
    if len(train_predictions_list) < 2:
        vprint(f"Skipping gating network: only {len(train_predictions_list)} model(s), need at least 2 to combine", level=1)
        # Return the single model's predictions directly
        return valid_predictions_list[0], valid_labels
    excluded_models = set()
    excluded_temp = set()
    
    # Load previous progress
    progress, best_predictions = load_progress(run_number)
    best_accuracy = progress['best_metrics']['accuracy']
    best_weighted_accuracy = progress['best_metrics']['weighted_accuracy']
    best_loss = progress['best_metrics']['loss']
    best_kappa = progress['best_metrics']['kappa']
    best_combination = progress.get('best_combination', None)
    completed_combinations = set(tuple(comb) for comb in progress['completed_combinations'])
    if best_combination:
        excluded_models = set(range(len(train_predictions_list))) - set(best_combination)
        vprint(f"Excluded models from previous time code was run: {excluded_models}", level=2)

    # Ensure predictions are in float32
    train_predictions_list = [np.array(p, dtype=np.float32) for p in train_predictions_list]
    valid_predictions_list = [np.array(p, dtype=np.float32) for p in valid_predictions_list]

    # Train with all models first (only if not done before)
    all_models_key = tuple(range(len(train_predictions_list)))
    if all_models_key not in completed_combinations:
        vprint("\nTraining with all models...", level=1)
        train_data_truncated, train_labels_truncated = correct_and_validate_predictions(train_predictions_list, train_labels, "train")
        val_data_truncated, val_labels_truncated = correct_and_validate_predictions(valid_predictions_list, valid_labels, "valid")

        train_data = np.stack(train_data_truncated, axis=1).astype(np.float32)
        val_data = np.stack(val_data_truncated, axis=1).astype(np.float32)

        vprint(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels_truncated.shape}", level=2)
        vprint(f"Val data shape: {val_data.shape}, Val labels shape: {val_labels_truncated.shape}", level=2)
        train_label_indices = np.argmax(train_labels_truncated, axis=1) if len(train_labels_truncated.shape) > 1 else train_labels_truncated
        vprint(f"Train classes: {sorted(np.unique(train_label_indices).astype(int))}", level=2)

        predictions, loss, accuracy, kappa, model, val_weighted_acc = train_model_combination(train_data, val_data, train_labels_truncated, val_labels_truncated) 
        if best_accuracy and accuracy+val_weighted_acc >= best_accuracy+best_weighted_accuracy:
            best_accuracy = accuracy
            best_weighted_accuracy = val_weighted_acc
            best_loss = loss
            best_kappa = kappa
            best_predictions = predictions
            best_combination = list(range(len(train_predictions_list)))
            best_model = model
            best_model.save(os.path.join(models_path, f'best_gating_model_run{run_number}.h5'))
        elif not best_accuracy:
            best_accuracy = accuracy
            best_weighted_accuracy = val_weighted_acc
            best_loss = loss
            best_kappa = kappa
            best_predictions = predictions
            best_combination = list(range(len(train_predictions_list)))
            best_model = model
            best_model.save(os.path.join(models_path, f'best_gating_model_run{run_number}.h5'))
            
        # Save progress
        save_progress(run_number, completed_combination=all_models_key,best_metrics={'accuracy': best_accuracy, 'weighted_accuracy': best_weighted_accuracy ,'loss': best_loss, 'kappa': best_kappa},best_combination=best_combination,best_predictions=best_predictions)
        vprint(f"Initial Accuracy: {best_accuracy:.4f}, Initial Weighted Accuracy: {best_weighted_accuracy:.4f}, Loss: {best_loss:.4f}, kappa: {best_kappa:.4f}", level=1)
        write_save_best_combo_results(best_predictions, best_accuracy, best_weighted_accuracy, best_loss, valid_labels, best_combination, excluded_models, result_dir, run_number)
    
    # Search for the optimal combination of models
    if find_optimal:
        vprint("\nSearching for the optimal model combination...", level=1)
        available_models = set(range(len(train_predictions_list)))
        # excluded_models = set()
        available_indices = list(available_models)
        # Manually exclude models - from production_config
        for i in SEARCH_EXCLUDED_MODELS:
            if i in available_indices:
                available_indices.remove(i)
        num_models = len(available_indices) - 1 - 0  # Start with all models except 1
        if num_models == SEARCH_MIN_MODELS:
            steps = [SEARCH_MIN_MODELS]
        else:
            steps = list(np.sort(range(min_models, num_models, max(int(SEARCH_STEP_SIZE_FRACTION * num_models), 1)))) + [len(available_indices) - 1]
        vprint(f"Explornig Steps: {steps}", level=2)
        while num_models >= min_models:
            # Progressively exclude models
            # available_indices = list(available_models - excluded_models)

            if len(available_indices) < num_models:
                if steps and len(steps) > 1:
                    steps = steps[:-1]
                num_models = len(available_indices)  # Adjust num_models to match available models
                if num_models < min_models:
                    break
                continue
            vprint(f"\nTrying combinations with {num_models} models...", level=2)
            vprint(f"Available indices: {available_indices}", level=2)
            from itertools import combinations
            import random
            all_combinations = list(combinations(available_indices, num_models))
            random.shuffle(all_combinations)
            max_tries_min = min(max_tries, len(all_combinations))
            vprint(f"Max tries: {max_tries_min}", level=2)

            from tqdm import tqdm

            # Parallel processing - from production_config
            num_processes = min(SEARCH_NUM_PROCESSES, os.cpu_count() or 1)
            with Pool(processes=num_processes) as pool:
                process_func = partial(process_trial, 
                                    all_combinations=all_combinations,
                                    completed_combinations=completed_combinations,
                                    train_predictions_list=train_predictions_list,
                                    valid_predictions_list=valid_predictions_list,
                                    train_labels=train_labels,
                                    valid_labels=valid_labels,
                                    run_number=run_number,
                                    max_tries_min=max_tries_min,
                                    )
                
                for result in tqdm(pool.imap(process_func, range(max_tries_min)), 
                                total=max_tries_min,
                                desc=f"Trying combinations with {num_models} models"):
                    if result is not None:
                        # If this is better than our best so far
                        if best_accuracy and result['accuracy']+result['weighted_accuracy'] >= best_accuracy+best_weighted_accuracy:
                            # Retrain the model with these indices (to get a fresh model)
                            train_predictions_selected = [train_predictions_list[i] for i in result['selected_indices']]
                            valid_predictions_selected = [valid_predictions_list[i] for i in result['selected_indices']]
                            
                            train_data_truncated, train_labels_truncated = correct_and_validate_predictions(
                                train_predictions_selected, train_labels, "train")
                            val_data_truncated, val_labels_truncated = correct_and_validate_predictions(
                                valid_predictions_selected, valid_labels, "valid")
                            
                            train_data_sliced = np.stack(train_data_truncated, axis=1).astype(np.float32)
                            val_data_sliced = np.stack(val_data_truncated, axis=1).astype(np.float32)
                            
                            # Retrain to get the model
                            _, _, _, _, best_model, _ = train_model_combination(train_data_sliced, val_data_sliced, train_labels_truncated, val_labels_truncated)
                            
                            excluded_temp = excluded_models.copy()
                            vprint(f"\nNew best combination found!", level=1)
                            vprint(f"Models {result['selected_indices']}", level=1)

                            excluded_model = set(available_indices) - set(result['selected_indices'])
                            if excluded_model:
                                vprint(f"Excluded trial model(s) {excluded_model}", level=2)
                                excluded_temp.update(excluded_model)
                            vprint(f"Previous accuracy: {best_accuracy:.4f}, Previous Weighted Accuracy: {best_weighted_accuracy:.4f}, Previous loss: {best_loss:.4f}, Previous kappa: {best_kappa:.4f}", level=2)

                            best_accuracy = result['accuracy']
                            best_weighted_accuracy = result['weighted_accuracy']
                            best_loss = result['loss']
                            best_predictions = result['predictions']
                            best_combination = result['selected_indices']
                            best_kappa = result['kappa']

                            vprint(f"New best accuracy: {best_accuracy:.4f}, New best Weighted Accuracy: {best_weighted_accuracy:.4f}, New best loss: {best_loss:.4f}, New best kappa: {best_kappa:.4f}", level=1)
                            
                            write_save_best_combo_results(best_predictions, best_accuracy, best_weighted_accuracy, best_loss, valid_labels, best_combination, excluded_temp, result_dir, run_number)
                            save_progress(run_number, best_metrics={'accuracy': best_accuracy, 'weighted_accuracy': best_weighted_accuracy, 'loss': best_loss, 'kappa': best_kappa}, best_combination=best_combination,best_predictions=best_predictions)
                            best_model.save(os.path.join(models_path, f'best_gating_model_run{run_number}.h5'))
            if excluded_temp:
                excluded_models = excluded_temp.copy()
            vprint(f"Excluding model(s) {excluded_models} from future combinations", level=2)
            vprint(f"{max_tries_min} combinations with {num_models} models completed", level=2)
            # num_models -= 1
            if len(steps) == 1:
                break
            elif steps and len(steps) > 1:
                steps = steps[:-1]
                num_models = steps[-1]
            else:
                break
    # Final Results (level=0 to show at all verbosity levels including progress bar)
    vprint("\nFinal Results:", level=0)
    vprint("==============", level=0)
    vprint(f"Best Cohen's Kappa: {best_kappa:.4f}", level=0)
    vprint(f"Best Accuracy: {best_accuracy:.4f}", level=0)
    vprint(f"Best Weighted Accuracy: {best_weighted_accuracy:.4f}", level=0)
    vprint(f"Best Loss: {best_loss:.4f}", level=0)
    # if best_combination:
    #     print(f"Best Model Combination: {best_combination}")
    #     if excluded_models:
    #         print(f"Excluded Models: {excluded_models}")
    #         write_save_best_combo_results(best_predictions, best_accuracy, best_loss, val_labels, best_combination, excluded_models, result_dir, run_number)

    return best_predictions, valid_labels

def load_progress(run_number):
    """Load progress from JSON file with error handling"""
    import time
    progress_file = os.path.join(ck_path, f'training_progress_run{run_number}.json')
    predictions_file = os.path.join(ck_path, f'best_predictions_run{run_number}.npy')
    
    # Initialize default progress - note: no predictions here
    default_progress = {
        'completed_combinations': [],
        'best_metrics': {
            'accuracy': 0,
            'weighted_accuracy': 0,
            'loss': float('inf'),
            'kappa': 0
        },
        'best_combination': None
    }
    
    predictions = None
    max_retries = PROGRESS_MAX_RETRIES
    retry_delay = PROGRESS_RETRY_DELAY
    
    # Load progress dict if exists
    for attempt in range(max_retries):
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    loaded_progress = json.load(f)
                    default_progress.update(loaded_progress)
                break
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Error loading progress file: {e}")
            print("Using default progress values")
    
    # Try to load predictions file
    if os.path.exists(predictions_file):
        try:
            predictions = np.load(predictions_file)
        except Exception as e:
            print(f"Error loading predictions file: {e}")
    
    return default_progress, predictions

def save_progress(run_number, completed_combination=None, best_metrics=None, best_combination=None, best_predictions=None):
    """Save progress to JSON file with proper separation of numpy arrays"""
    import time
    progress_file = os.path.join(ck_path, f'training_progress_run{run_number}.json')
    predictions_file = os.path.join(ck_path, f'best_predictions_run{run_number}.npy')
    max_retries = PROGRESS_MAX_RETRIES
    retry_delay = PROGRESS_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            # Load existing progress (but not predictions)
            progress, _ = load_progress(run_number)
            
            if completed_combination is not None:
                # Convert numpy types or tuples to list of integers
                if isinstance(completed_combination, (np.ndarray, tuple)):
                    completed_combination = [int(x) for x in completed_combination]
                    
                # Ensure the combination isn't already saved
                if completed_combination not in [tuple(x) for x in progress['completed_combinations']]:
                    progress['completed_combinations'].append(completed_combination)
            
            if best_metrics:
                # Ensure all metric values are Python native types
                progress['best_metrics'] = {
                    'accuracy': float(best_metrics['accuracy']),
                    'weighted_accuracy': float(best_metrics['weighted_accuracy']),
                    'loss': float(best_metrics['loss']),
                    'kappa': float(best_metrics['kappa'])
                }
            
            if best_combination is not None:
                # Convert numpy array or other sequence types to list of integers
                if isinstance(best_combination, (np.ndarray, tuple)):
                    best_combination = [int(x) for x in best_combination]
                elif isinstance(best_combination, list):
                    best_combination = [int(x) for x in best_combination]
                progress['best_combination'] = best_combination
            
            # Save progress dict (without predictions)
            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
            except TypeError as e:
                print(f"Error during JSON serialization: {e}")
                print("Progress content:", progress)
                raise
                
            # Save predictions separately only if provided
            if best_predictions is not None:
                try:
                    np.save(predictions_file, best_predictions)
                except Exception as e:
                    print(f"Error saving predictions: {e}")
            break  
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Error saving progress: {e}")
            
def write_save_best_combo_results(best_predictions, best_accuracy, best_weighted_accuracy, best_loss, true_labels, selected_models, excluded_models, result_dir, run_number): 
    # Calculate final metrics 
    y_true = true_labels 
    y_pred = np.argmax(best_predictions, axis=1) 

    # Calculate metrics
    f1_class_0 = f1_score(y_true, y_pred, average=None, zero_division=0)[0] 
    f1_class_1 = f1_score(y_true, y_pred, average=None, zero_division=0)[1] 
    f1_class_2 = f1_score(y_true, y_pred, average=None, zero_division=0)[2] 
    f1_macro = f1_score(y_true, y_pred, average='macro') 
    f1_weighted = f1_score(y_true, y_pred, average='weighted') 
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic') 

    # Save results to file
    results_file = os.path.join(csv_path, f'model_combination_results_{run_number}.txt') 
    with open(results_file, 'w') as f: 
        f.write("Model Training Results\n") 
        f.write("====================\n\n") 
         
        # Overall Metrics 
        f.write("Overall Metrics:\n") 
        f.write("--------------\n") 
        f.write(f"Final Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Final Weighted Accuracy: {best_weighted_accuracy:.4f}\n")
        f.write(f"Final Loss: {best_loss:.4f}\n") 
        f.write(f"Cohen's Kappa: {kappa:.4f}\n\n") 
         
        # F1 Scores 
        f.write("F1 Scores:\n") 
        f.write("----------\n") 
        f.write(f"Class 0 (Inflammatory): {f1_class_0:.4f}\n") 
        f.write(f"Class 1 (Proliferative): {f1_class_1:.4f}\n") 
        f.write(f"Class 2 (Remodeling): {f1_class_2:.4f}\n") 
        f.write(f"Macro Average F1: {f1_macro:.4f}\n") 
        f.write(f"Weighted Average F1: {f1_weighted:.4f}\n\n") 
         
        # Model Information 
        f.write("Model Information:\n") 
        f.write("-----------------\n") 
        f.write(f"Selected Models: {selected_models}\n\n")
        if excluded_models:
            f.write(f"Excluded Models: {excluded_models}\n\n")
         
        # Additional Details 
        f.write("\nDetailed Metrics:\n") 
        f.write("---------------\n") 
        f.write("Confusion Matrix:\n") 
        f.write(str(confusion_matrix(y_true, y_pred))) 
        f.write("\n\nClassification Report:\n") 
        f.write(str(classification_report(y_true, y_pred))) 

    print(f"\nDetailed results saved to {results_file}")
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn1 = Dense(ff_dim, activation="gelu")  # First Dense layer
        self.ffn2 = Dense(embed_dim)  # Second Dense layer
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn1(out1)  # Apply the first Dense layer
        ffn_output = self.ffn2(ffn_output)  # Apply the second Dense layer
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_hierarchical_gating_network(num_models, num_classes, embedding_dim=HIERARCHICAL_EMBEDDING_DIM):
    """
    Creates a hierarchical attention network for combining model predictions

    Args:
        num_models: Number of base models
        num_classes: Number of classification classes (3 for I,P,R)
        embedding_dim: Dimension of the embedding space (from production_config)
    """
    # Input shape: (batch_size, num_models, num_classes)
    inputs = Input(shape=(num_models, num_classes))
    
    # Phase-specific attention branch
    phase_specific_outputs = []
    
    for phase in range(num_classes):
        # Extract phase-specific probabilities
        phase_inputs = tf.keras.layers.Lambda(lambda x: x[:, :, phase])(inputs)
        phase_inputs = Reshape((num_models, 1))(phase_inputs)
        
        # Learn phase-specific attention
        attention = Dense(embedding_dim, activation='relu')(phase_inputs)
        attention = LayerNormalization()(attention)
        attention = Dense(1, activation='softmax')(attention)
        
        # Weight model predictions for this phase
        weighted_phase = Multiply()([phase_inputs, attention])
        weighted_phase = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(weighted_phase)
        
        phase_specific_outputs.append(weighted_phase)
    
    # Combine phase-specific outputs
    phase_concatenated = Concatenate(axis=-1)(phase_specific_outputs)
    
    # Add positional encoding for transformer
    pos_encoding = tf.cast(tf.range(num_classes), tf.float32)[tf.newaxis, :]
    pos_encoding = Dense(embedding_dim)(pos_encoding[:, :, tf.newaxis])
    
    # Project to embedding space
    phase_embedded = Dense(embedding_dim)(phase_concatenated)
    phase_embedded = Add()([phase_embedded, pos_encoding])
    
    # Transformer for inter-phase relationships - from production_config
    transformer_output = TransformerBlock(
        embed_dim=embedding_dim,
        num_heads=HIERARCHICAL_NUM_HEADS,
        ff_dim=embedding_dim * HIERARCHICAL_FF_DIM_MULTIPLIER
    )(phase_embedded)
    
    # Final classification layer with residual connection
    transformer_pooled = GlobalAveragePooling1D()(transformer_output)
    residual = GlobalAveragePooling1D()(phase_embedded)
    
    combined = Concatenate()([transformer_pooled, residual])
    combined = Dense(embedding_dim, activation='relu')(combined)
    combined = LayerNormalization()(combined)
    combined = Dropout(0.1)(combined)
    
    # Output probabilities
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined) ##TRY THIS TO AVOID OVERFITTING
    # outputs = Dense(num_classes, activation='softmax')(combined)
    
    return Model(inputs=inputs, outputs=outputs)
def focal_loss_gating_network(gamma=HIERARCHICAL_FOCAL_GAMMA, alpha=HIERARCHICAL_FOCAL_ALPHA):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        if alpha is None:
            alpha_factor = 1
        else:
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            
        focusing_factor = y_true * K.pow(1. - y_pred, gamma) + \
                         (1. - y_true) * K.pow(y_pred, gamma)
                         
        return K.mean(alpha_factor * focusing_factor * K.binary_crossentropy(y_true, y_pred))
    return focal_loss_fixed
def train_hierarchical_gating_network(predictions_list, true_labels, n_splits=CV_N_SPLITS, patience=HIERARCHICAL_PATIENCE):
    """
    Train the hierarchical gating network using cross-validation

    Args:
        predictions_list: List of model predictions arrays
        true_labels: True class labels (one-hot encoded)
        n_splits: Number of cross-validation splits (from production_config)
        patience: Early stopping patience (from production_config)
    """
    print("\nTraining hierarchical gating network with cross-validation...")
    
    # Basic validation checks
    min_length = min(len(preds) for preds in predictions_list)
    if (true_labels.shape[1] != predictions_list[0].shape[1] or min_length < 6):
        print("Warning: Class mismatch or dataset too small. Using simple averaging.")
        combined_predictions = np.mean([preds[:min_length] for preds in predictions_list], axis=0)
        attention_weights = np.ones(len(predictions_list)) / len(predictions_list)
        print(f"Using equal weights for all models: {attention_weights}")
        return combined_predictions, attention_weights

    # Prepare data
    predictions_list = [preds[:min_length] for preds in predictions_list]
    true_labels = true_labels[:min_length]
    

    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
    
    # Initialize arrays for predictions and metrics
    combined_predictions = np.zeros_like(true_labels)
    fold_attention_weights = []
    fold_metrics = []
    
    # Convert predictions to numpy arrays and stack
    stacked_predictions = np.stack(predictions_list, axis=1)
    
    # Cross-validation loop
    # for fold, (train_idx, val_idx) in enumerate(kf.split(stacked_predictions)):
    for fold, (train_idx, val_idx) in enumerate(skf.split(stacked_predictions, np.argmax(true_labels, axis=1))):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        X_train = stacked_predictions[train_idx]
        y_train = true_labels[train_idx]
        X_val = stacked_predictions[val_idx]
        y_val = true_labels[val_idx]
        
        # Create and compile model - using production_config values
        model = create_hierarchical_gating_network(
            num_models=len(predictions_list),
            num_classes=true_labels.shape[1],
            embedding_dim=HIERARCHICAL_EMBEDDING_DIM
        )
        class_weights = {
            0: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 0)),  # I
            1: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 1)),  # P
            2: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 2))   # R
        }
        # Use focal loss with config parameters
        loss = focal_loss_gating_network(gamma=HIERARCHICAL_FOCAL_GAMMA, alpha=HIERARCHICAL_FOCAL_ALPHA)
        model.compile(
            optimizer=Adam(learning_rate=HIERARCHICAL_LEARNING_RATE),
            loss=loss,
            metrics=['accuracy'],
        )

        # Train with early stopping - using production_config values
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            # class_weight=class_weights,
            epochs=HIERARCHICAL_EPOCHS,
            batch_size=HIERARCHICAL_BATCH_SIZE,
            callbacks=[
                EarlyStopping(
                    monitor='loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss',
                    factor=HIERARCHICAL_REDUCE_LR_FACTOR,
                    patience=HIERARCHICAL_REDUCE_LR_PATIENCE,
                    min_lr=HIERARCHICAL_REDUCE_LR_MIN_LR
                )
            ],
            verbose=HIERARCHICAL_VERBOSE
        )
        
        # Get predictions for validation fold
        combined_predictions[val_idx] = model.predict(X_val, verbose=0)
        
        # Store metrics
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        fold_metrics.append({'val_loss': val_loss, 'val_accuracy': val_acc})
        
        # Extract attention weights for analysis
        attention_model = Model(
            inputs=model.input,
            outputs=model.layers[1].output  # First attention layer
        )
        attention_weights = attention_model.predict(X_val, verbose=0)
        fold_attention_weights.append(np.mean(attention_weights, axis=0))
        
        vprint(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}", level=1)

    # Calculate and print overall metrics
    vprint("\nOverall cross-validation metrics:", level=1)
    mean_val_acc = np.mean([m['val_accuracy'] for m in fold_metrics])
    std_val_acc = np.std([m['val_accuracy'] for m in fold_metrics])
    vprint(f"Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}", level=1)

    # Calculate average attention weights across folds
    overall_attention = np.mean(fold_attention_weights, axis=0)
    vprint("\nOverall model importance weights:", level=1)
    if get_verbosity() == 1:
        for i, weight in enumerate(overall_attention):
            print(f"Model {i + 1}: {np.mean(weight):.3f}")
    
    return combined_predictions, overall_attention

def calculate_and_save_metrics(true_labels, predictions, result_dir, configs):
    """Calculate metrics and save results."""
    if len(true_labels) != len(predictions):
        vprint(f"Warning: Length mismatch - true_labels: {len(true_labels)}, predictions: {len(predictions)}", level=1)
        # Truncate to shorter length
        min_len = min(len(true_labels), len(predictions))
        true_labels = true_labels[:min_len]
        predictions = predictions[:min_len]
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0),
        'f1_classes': f1_score(true_labels, predictions, average=None, labels=[0, 1, 2], zero_division=0),
        'kappa': cohen_kappa_score(true_labels, predictions, weights='quadratic')
    }
    
    # Print results (level=0 to show at all verbosity levels including progress bar)
    vprint("\nFinal Results:", level=0)
    if get_verbosity() <= 1 or get_verbosity() == 3:
        print(classification_report(true_labels, predictions,
                                  target_names=CLASS_LABELS,
                                  labels=[0, 1, 2]))
    vprint(f"\nCohen's Kappa: {metrics['kappa']:.4f}", level=0)
    
    # Save confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix with Specialized Modalities')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(result_dir, 'specialized_confusion_matrix.png'))
    plt.close()
    
    # Save to CSV
    results = {
        # 'Modality Configuration': f"Specialized (I:{'+'.join(configs['I']['modalities'])}, P:{'+'.join(configs['P']['modalities'])}, R:{'+'.join(configs['R']['modalities'])}, I2:{'+'.join(configs['I2']['modalities'])}, P2:{'+'.join(configs['P2']['modalities'])}, R2:{'+'.join(configs['R2']['modalities'])}, I3:{'+'.join(configs['I3']['modalities'])}, P3:{'+'.join(configs['P3']['modalities'])}, R3:{'+'.join(configs['R3']['modalities'])}, I4:{'+'.join(configs['I4']['modalities'])}, P4:{'+'.join(configs['P4']['modalities'])}, R4:{'+'.join(configs['R4']['modalities'])})",
        'Modality Configuration': "Specialized",
        'Accuracy': metrics['accuracy'],
        'Macro F1-score': metrics['f1_macro'],
        'Weighted F1-score': metrics['f1_weighted'],
        'I F1-score': metrics['f1_classes'][0],
        'P F1-score': metrics['f1_classes'][1],
        'R F1-score': metrics['f1_classes'][2],
        "Cohen's Kappa": metrics['kappa'],
        # 'I gamma': configs['I']['loss_params']['gamma'],
        # 'P gamma': configs['P']['loss_params']['gamma'],
        # 'R gamma': configs['R']['loss_params']['gamma'],
        # 'I ordinal weight': configs['I']['loss_params']['ordinal_weight'],
        # 'P ordinal weight': configs['P']['loss_params']['ordinal_weight'],
        # 'R ordinal weight': configs['R']['loss_params']['ordinal_weight']
    }
    
    csv_filename = os.path.join(csv_path, 'specialized_results_V66_augment_RGBGenAug.csv')
    fieldnames = list(results.keys())
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)
    
    return metrics
def create_focal_ordinal_loss2(ordinal_weight, gamma, alpha):
    """Create a focal ordinal loss function with fixed parameters"""
    vprint("\nLoss Function Configuration for depth_rgb:", level=2)
    vprint(f"Ordinal weight: {ordinal_weight}", level=2)
    vprint(f"Gamma: {gamma}", level=2)
    vprint(f"Alpha: {alpha}", level=2)
    def loss_fn(y_true, y_pred):
        if not hasattr(loss_fn, 'params_verified'):
            if get_verbosity() == 2:
                print("\nVerifying loss parameters during first call:")
                print(f"Current gamma: {gamma}")
                print(f"Current alpha: {alpha}")
            loss_fn.params_verified = True
        # Clip prediction values to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Convert alpha to tensor with proper shape
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        
        # Focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.expand_dims(alpha_tensor, 0) * tf.math.pow(1 - y_pred, gamma)
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
    
    return loss_fn

def create_current_get_focal_ordinal_loss(ordinal_weight, gamma, alpha):
    """Create a wrapper function that matches the expected signature"""
    def current_get_focal_ordinal_loss(**kwargs):
        return create_focal_ordinal_loss2(ordinal_weight, gamma, alpha)
    return current_get_focal_ordinal_loss

def perform_grid_search(data_percentage=100, train_patient_percentage=0.8, cv_folds=3):
    """
    Perform grid search over loss function parameters.
    """
    # Define parameter grids
    param_grid = {
        'ordinal_weight': [1.0],
        'gamma': [2.0, 3.0],
        'alpha': [
            [0.598, 0.315, 1.597],  # Your current weights
            [1, 0.5, 2],  # Alternative weights
            [1.5, 0.3, 1.5]  # Another alternative
        ]
    }
    
    # Create results CSV
    results_file = os.path.join(result_dir, 'loss_parameter_search_results.csv')
    fieldnames = ['ordinal_weight', 'gamma', 'alpha', 
                  'accuracy', 'f1_macro', 'f1_weighted',
                  'f1_I', 'f1_P', 'f1_R', 'kappa']
    
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    best_score = 0
    best_params = None
    
    # Store original function
    original_get_focal_ordinal_loss = globals()['get_focal_ordinal_loss']
    
    # Iterate over parameter combinations
    for ordinal_weight in param_grid['ordinal_weight']:
        for gamma in param_grid['gamma']:
            for alpha in param_grid['alpha']:
                vprint(f"\nTesting parameters:", level=2)
                vprint(f"Ordinal Weight: {ordinal_weight}", level=2)
                vprint(f"Gamma: {gamma}", level=2)
                vprint(f"Alpha: {alpha}", level=2)
                
                try:
                    # Create and set the current loss function
                    current_get_focal_ordinal_loss = create_current_get_focal_ordinal_loss(
                        ordinal_weight, gamma, alpha
                    )
                    globals()['get_focal_ordinal_loss'] = current_get_focal_ordinal_loss
                    
                    # Run your existing main function
                    metrics = main_with_specialized_evaluation(
                        data_percentage=data_percentage,
                        train_patient_percentage=train_patient_percentage,
                        cv_folds=cv_folds
                    )
                    
                    # Prepare results
                    result = {
                        'ordinal_weight': ordinal_weight,
                        'gamma': gamma,
                        'alpha': str(alpha),
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_macro': metrics.get('f1_macro', 0),
                        'f1_weighted': metrics.get('f1_weighted', 0),
                        'f1_I': metrics.get('f1_classes', [0, 0, 0])[0],
                        'f1_P': metrics.get('f1_classes', [0, 0, 0])[1],
                        'f1_R': metrics.get('f1_classes', [0, 0, 0])[2],
                        'kappa': metrics.get('kappa', 0)
                    }
                    
                    # Append results
                    with open(results_file, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result)
                    
                    # Update best parameters
                    current_score = result['f1_weighted']
                    if current_score > best_score:
                        best_score = current_score
                        best_params = {
                            'ordinal_weight': ordinal_weight,
                            'gamma': gamma,
                            'alpha': alpha
                        }
                    
                    vprint(f"\nCurrent Results:", level=1)
                    vprint(f"F1 Weighted: {result['f1_weighted']:.4f}", level=1)
                    vprint(f"Kappa: {result['kappa']:.4f}", level=1)

                except Exception as e:
                    vprint(f"Error during parameter combination: {str(e)}", level=1)
                    import traceback
                    traceback.print_exc()
                    
                finally:
                    # Restore original function
                    globals()['get_focal_ordinal_loss'] = original_get_focal_ordinal_loss
                    
                    # Clear memory
                    tf.keras.backend.clear_session()
                    gc.collect()
    
    vprint("\nGrid Search Complete!", level=1)
    if best_params:
        vprint("\nBest Parameters:", level=1)
        vprint(f"Ordinal Weight: {best_params['ordinal_weight']}", level=1)
        vprint(f"Gamma: {best_params['gamma']}", level=1)
        vprint(f"Alpha: {best_params['alpha']}", level=1)
        vprint(f"Best F1 Weighted: {best_score:.4f}", level=1)
    
    return best_params, results_file
def main_search(data_percentage, train_patient_percentage=0.8, cv_folds=3):
    """
    Test all modality combinations and save results to CSV.

    Args:
        data_percentage: Percentage of data to use (1-100)
        train_patient_percentage: Percentage of patients for training (ignored if cv_folds > 1)
        cv_folds: Number of k-fold CV folds (default: 3). Set to 0 or 1 for single split.

    Configuration is read from production_config.py:
    - MODALITY_SEARCH_MODE: 'all' for all 31 combinations, 'custom' for INCLUDED_COMBINATIONS
    - EXCLUDED_COMBINATIONS: Combinations to exclude
    - INCLUDED_COMBINATIONS: Combinations to include (when mode='custom')
    - RESULTS_CSV_FILENAME: Output CSV filename
    """
    # Determine number of iterations
    if cv_folds <= 1:
        num_iterations = 1
        cv_mode_name = "single split"
    else:
        num_iterations = cv_folds
        cv_mode_name = f"{cv_folds}-fold CV"
    # Use config for CSV filename (saved in organized csv subfolder)
    csv_filename = os.path.join(csv_path, RESULTS_CSV_FILENAME)
    fieldnames = ['Modalities', 'Accuracy (Mean)', 'Accuracy (Std)',
                  'Macro Avg F1-score (Mean)', 'Macro Avg F1-score (Std)',
                  'Weighted Avg F1-score (Mean)', 'Weighted Avg F1-score (Std)',
                  'I F1-score (Mean)', 'P F1-score (Mean)', 'R F1-score (Mean)',
                  "Cohen's Kappa (Mean)", "Cohen's Kappa (Std)"]

    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Use modality configuration from production_config
    modalities = ALL_MODALITIES
    excluded_combinations = EXCLUDED_COMBINATIONS
    included_combinations = INCLUDED_COMBINATIONS
    search_mode = MODALITY_SEARCH_MODE

    # Generate all possible combinations (31 total)
    all_combinations = []
    for r in range(1, len(modalities) + 1):
        all_combinations.extend(itertools.combinations(modalities, r))

    # Filter combinations based on search mode
    if search_mode == 'all':
        # Test all combinations, excluding any in EXCLUDED_COMBINATIONS
        combinations_to_process = [comb for comb in all_combinations if comb not in excluded_combinations]
        vprint(f"\n{'='*80}", level=1)
        vprint(f"MODALITY SEARCH MODE: ALL COMBINATIONS ({len(combinations_to_process)} combinations)", level=1)
        vprint(f"{'='*80}", level=1)
        if excluded_combinations:
            vprint(f"Excluded: {len(excluded_combinations)} combinations", level=1)
    elif search_mode == 'custom':
        # Test only combinations in INCLUDED_COMBINATIONS
        combinations_to_process = [comb for comb in all_combinations if comb in included_combinations]
        vprint(f"\n{'='*80}", level=1)
        vprint(f"MODALITY SEARCH MODE: CUSTOM ({len(combinations_to_process)} combinations)", level=1)
        vprint(f"{'='*80}", level=1)
        vprint(f"Testing only specified combinations from production_config.py", level=1)
    else:
        raise ValueError(f"Invalid MODALITY_SEARCH_MODE: {search_mode}. Must be 'all' or 'custom'")

    vprint(f"Total combinations to test: {len(combinations_to_process)}")
    vprint(f"Cross-validation mode: {cv_mode_name}")
    vprint(f"Iterations per combination: {num_iterations}")
    vprint(f"Total training sessions: {len(combinations_to_process) * num_iterations}")
    vprint(f"Results will be saved to: {csv_filename}")
    vprint(f"{'='*80}\n")

    # Initialize progress bar for progress bar mode
    total_steps = len(combinations_to_process)
    init_progress_bar(total_steps, desc="Testing modality combinations")

    for idx, combination in enumerate(combinations_to_process, 1):
    # for combination in [('metadata','depth_rgb','depth_map','thermal_map')]: #Cedar2
    # for combination in [1]:
        selected_modalities = list(combination)
        # selected_modalities = ['metadata', 'depth_rgb', 'thermal_map']
        vprint(f"\nTesting modalities: {', '.join(selected_modalities)}")
        # Load and prepare the dataset
        data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities)
        # Apply misclassification filtering if configured (from production_config)
        if USE_CORE_DATA or any([THRESHOLD_I, THRESHOLD_P, THRESHOLD_R]):
            thresholds = {'I': THRESHOLD_I, 'P': THRESHOLD_P, 'R': THRESHOLD_R}
            from src.evaluation.metrics import filter_frequent_misclassifications
            data = filter_frequent_misclassifications(data, result_dir, thresholds=thresholds)
        if data_percentage < 100:
            data = data.sample(frac=data_percentage / 100, random_state=789).reset_index(drop=True)  # Phase 7a: seed 789
        vprint(f"Using {data_percentage}% of the data: {len(data)} samples")
        # Perform cross-validation with manual patient split
        run_data = data.copy(deep=True)
        cv_results, confusion_matrices, histories = cross_validation_manual_split(
            run_data, selected_modalities, train_patient_percentage, cv_folds=cv_folds, track_misclass=TRACK_MISCLASS
        )

        # Save predictions per combination for later gating network ensemble
        config_name = '+'.join(selected_modalities)
        for run in range(num_iterations):
            # Load the aggregated predictions that were saved during cross_validation
            train_preds, train_labels = load_aggregated_predictions(run + 1, ck_path, dataset_type='train')
            valid_preds, valid_labels = load_aggregated_predictions(run + 1, ck_path, dataset_type='valid')
            if train_preds is not None and valid_preds is not None and len(train_preds) > 0 and len(valid_preds) > 0:
                # Save as combination-specific file (use first prediction if multiple configs)
                save_combination_predictions(run + 1, config_name, train_preds[0], train_labels, ck_path, dataset_type='train')
                save_combination_predictions(run + 1, config_name, valid_preds[0], valid_labels, ck_path, dataset_type='valid')

        # Calculate average metrics and their standard deviations with error handling
        avg_accuracy = np.mean([m['accuracy'] for m in cv_results])
        std_accuracy = np.std([m['accuracy'] for m in cv_results])
        avg_f1_macro = np.mean([m['f1_macro'] for m in cv_results])
        std_f1_macro = np.std([m['f1_macro'] for m in cv_results])
        avg_f1_weighted = np.mean([m['f1_weighted'] for m in cv_results])
        std_f1_weighted = np.std([m['f1_weighted'] for m in cv_results])
        avg_kappa = np.mean([m['kappa'] for m in cv_results])
        std_kappa = np.std([m['kappa'] for m in cv_results])

        # Calculate average F1-scores for each class
        f1_classes_list = [m['f1_classes'] for m in cv_results]
        if len(f1_classes_list) > 0:
            avg_f1_classes = np.mean(f1_classes_list, axis=0)
            # Ensure avg_f1_classes is always an array
            if np.isscalar(avg_f1_classes) or avg_f1_classes.ndim == 0:
                avg_f1_classes = np.array([avg_f1_classes, 0.0, 0.0])
            elif len(avg_f1_classes) < 3:
                avg_f1_classes = np.pad(avg_f1_classes, (0, 3 - len(avg_f1_classes)), constant_values=0.0)
        else:
            avg_f1_classes = np.array([0.0, 0.0, 0.0])

        # Prepare results for CSV
        result = {
            'Modalities': '+'.join(selected_modalities),
            'Accuracy (Mean)': avg_accuracy,
            'Accuracy (Std)': std_accuracy,
            'Macro Avg F1-score (Mean)': avg_f1_macro,
            'Macro Avg F1-score (Std)': std_f1_macro,
            'Weighted Avg F1-score (Mean)': avg_f1_weighted,
            'Weighted Avg F1-score (Std)': std_f1_weighted,
            'I F1-score (Mean)': float(avg_f1_classes[0]),
            'P F1-score (Mean)': float(avg_f1_classes[1]),
            'R F1-score (Mean)': float(avg_f1_classes[2]),
            "Cohen's Kappa (Mean)": avg_kappa,
            "Cohen's Kappa (Std)": std_kappa
        }

        # Append the result to the CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

        vprint(f"Results for {', '.join(selected_modalities)} appended to {csv_filename}")

        # Update progress bar with current combination info
        status = f"{'+'.join(selected_modalities)} | Acc: {avg_accuracy:.3f} | F1: {avg_f1_macro:.3f}"
        update_progress(n=1, status=status)

        # Clean up after each modality combination
        try:
            tf.keras.backend.clear_session()
            gc.collect()
            clear_gpu_memory()
            reset_keras()
        except Exception as e:
            vprint(f"Error clearing memory stats: {str(e)}")
        try:
            del data, run_data, cv_results, confusion_matrices, histories, result
        except Exception as e:
            vprint(f"Error deleting variables: {str(e)}", level=2)

    # Close progress bar
    close_progress()

    vprint(f"\nAll results saved to {csv_filename}", level=0)

    # After all combinations are trained, run gating network ensemble across modality combinations
    if len(combinations_to_process) >= 2:
        vprint(f"\n{'='*80}")
        vprint(f"GATING NETWORK ENSEMBLE ACROSS MODALITY COMBINATIONS")
        vprint(f"{'='*80}")
        vprint(f"Ensembling predictions from {len(combinations_to_process)} modality combinations")

        for run in range(num_iterations):
            if cv_folds > 1:
                iter_name = f"Fold {run + 1}/{num_iterations}"
            else:
                iter_name = f"Iteration {run + 1}/{num_iterations}"
            vprint(f"\n{iter_name}", level=1)

            # Collect predictions from all modality combinations for this run
            all_train_predictions = []
            all_valid_predictions = []
            train_labels = None
            valid_labels = None
            combination_names = []

            for combination in combinations_to_process:
                config_name = '+'.join(combination)

                # Load predictions for this specific combination
                train_preds, t_labels = load_combination_predictions(run + 1, config_name, ck_path, dataset_type='train')
                valid_preds, v_labels = load_combination_predictions(run + 1, config_name, ck_path, dataset_type='valid')

                if train_preds is not None and valid_preds is not None:
                    all_train_predictions.append(train_preds)
                    all_valid_predictions.append(valid_preds)
                    combination_names.append(config_name)

                    if train_labels is None:
                        train_labels = t_labels
                        valid_labels = v_labels

                    vprint(f"  Loaded {config_name}: train shape {train_preds.shape}, valid shape {valid_preds.shape}", level=2)
                else:
                    vprint(f"  Warning: Could not load predictions for {config_name}", level=1)

            if len(all_train_predictions) >= 2:
                vprint(f"\nTraining gating network with {len(all_train_predictions)} modality combinations:", level=1)
                if get_verbosity() == 2:
                    for name in combination_names:
                        print(f"  - {name}")

                # Convert labels to class indices if needed
                if train_labels is not None and len(train_labels.shape) > 1:
                    gating_train_labels = np.argmax(train_labels, axis=1)
                else:
                    gating_train_labels = train_labels

                if valid_labels is not None and len(valid_labels.shape) > 1:
                    gating_valid_labels = np.argmax(valid_labels, axis=1)
                else:
                    gating_valid_labels = valid_labels

                try:
                    combined_predictions, final_labels = train_gating_network(
                        all_train_predictions,
                        all_valid_predictions,
                        gating_train_labels,
                        gating_valid_labels,
                        run + 1,
                        find_optimal=False,  # Disabled: gating network search was hanging
                        min_models=2,
                        max_tries=100
                    )

                    # Calculate ensemble metrics
                    if combined_predictions is not None:
                        final_preds = np.argmax(combined_predictions, axis=1)
                        ensemble_accuracy = accuracy_score(gating_valid_labels, final_preds)
                        ensemble_f1 = f1_score(gating_valid_labels, final_preds, average='macro')
                        ensemble_kappa = cohen_kappa_score(gating_valid_labels, final_preds)

                        vprint(f"\nGating Network Ensemble Results for Run {run + 1}:", level=0)
                        vprint(f"  Accuracy: {ensemble_accuracy:.4f}", level=0)
                        vprint(f"  F1 Macro: {ensemble_f1:.4f}", level=0)
                        vprint(f"  Kappa: {ensemble_kappa:.4f}", level=0)
                except Exception as e:
                    vprint(f"Error in gating network ensemble for run {run + 1}: {str(e)}", level=1)
            else:
                vprint(f"  Not enough predictions loaded ({len(all_train_predictions)}) for gating network ensemble", level=1)

    # Final summary - read CSV and show best results (shown at all verbosity levels)
    vprint(f"\n{'='*80}", level=0)
    vprint("FINAL SUMMARY - BEST MODALITY COMBINATIONS", level=0)
    vprint(f"{'='*80}", level=0)
    kappa_col = "Cohen's Kappa (Mean)"
    kappa_std_col = "Cohen's Kappa (Std)"
    try:
        results_df = pd.read_csv(csv_filename)
        if len(results_df) > 0:
            # Best by accuracy
            best_acc_idx = results_df['Accuracy (Mean)'].idxmax()
            best_acc_row = results_df.loc[best_acc_idx]
            vprint(f"\nBest by Accuracy:", level=0)
            vprint(f"  Modalities: {best_acc_row['Modalities']}", level=0)
            vprint(f"  Accuracy: {best_acc_row['Accuracy (Mean)']:.4f} ± {best_acc_row['Accuracy (Std)']:.4f}", level=0)
            vprint(f"  F1 Macro: {best_acc_row['Macro Avg F1-score (Mean)']:.4f}", level=0)
            vprint(f"  Kappa: {best_acc_row[kappa_col]:.4f}", level=0)

            # Best by F1 macro
            best_f1_idx = results_df['Macro Avg F1-score (Mean)'].idxmax()
            if best_f1_idx != best_acc_idx:
                best_f1_row = results_df.loc[best_f1_idx]
                vprint(f"\nBest by F1 Macro:", level=0)
                vprint(f"  Modalities: {best_f1_row['Modalities']}", level=0)
                vprint(f"  Accuracy: {best_f1_row['Accuracy (Mean)']:.4f}", level=0)
                vprint(f"  F1 Macro: {best_f1_row['Macro Avg F1-score (Mean)']:.4f} ± {best_f1_row['Macro Avg F1-score (Std)']:.4f}", level=0)
                vprint(f"  Kappa: {best_f1_row[kappa_col]:.4f}", level=0)

            # Best by Kappa
            best_kappa_idx = results_df[kappa_col].idxmax()
            if best_kappa_idx != best_acc_idx and best_kappa_idx != best_f1_idx:
                best_kappa_row = results_df.loc[best_kappa_idx]
                vprint(f"\nBest by Kappa:", level=0)
                vprint(f"  Modalities: {best_kappa_row['Modalities']}", level=0)
                vprint(f"  Accuracy: {best_kappa_row['Accuracy (Mean)']:.4f}", level=0)
                vprint(f"  F1 Macro: {best_kappa_row['Macro Avg F1-score (Mean)']:.4f}", level=0)
                vprint(f"  Kappa: {best_kappa_row[kappa_col]:.4f} ± {best_kappa_row[kappa_std_col]:.4f}", level=0)

            vprint(f"\nTotal combinations tested: {len(results_df)}", level=0)
    except Exception as e:
        vprint(f"Could not generate summary: {str(e)}", level=1)

    vprint(f"{'='*80}\n", level=0)

def main(mode='search', data_percentage=100, train_patient_percentage=0.8, cv_folds=3):
    """
    Combined main function that can run either modality search or specialized evaluation.

    Args:
        mode (str): Either 'search' or 'specialized' to determine which analysis to run
        data_percentage (float): Percentage of data to use
        train_patient_percentage (float): Percentage of patients to use for training (ignored if cv_folds > 1)
        cv_folds (int): Number of k-fold CV folds (default: 3). Set to 0 or 1 for single split.

    Configuration (from production_config.py):
        OUTLIER_REMOVAL: Enable outlier detection and removal
        OUTLIER_CONTAMINATION: Expected proportion of outliers
        TRACK_MISCLASS: Misclassification tracking mode
        USE_CORE_DATA: Use Bayesian-optimized core dataset
        THRESHOLD_I/P/R: Misclassification filtering thresholds
    """
    # Outlier detection and removal (combination-specific, if enabled)
    if OUTLIER_REMOVAL:
        # Determine which modalities will be tested
        modalities_to_test = []
        if mode.lower() == 'search':
            if MODALITY_SEARCH_MODE == 'all':
                # Generate all 31 combinations
                from itertools import chain, combinations
                all_modalities = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
                modalities_to_test = list(chain.from_iterable(
                    combinations(all_modalities, r) for r in range(1, len(all_modalities) + 1)
                ))
            else:  # custom
                modalities_to_test = INCLUDED_COMBINATIONS
        elif mode.lower() == 'specialized':
            # Specialized mode typically tests specific combinations
            modalities_to_test = INCLUDED_COMBINATIONS if INCLUDED_COMBINATIONS else []

        if modalities_to_test:
            vprint("\n" + "="*80, level=1)
            vprint("OUTLIER DETECTION AND REMOVAL (COMBINATION-SPECIFIC)", level=1)
            vprint("="*80, level=1)
            vprint(f"Contamination rate: {OUTLIER_CONTAMINATION*100:.0f}%", level=1)
            vprint(f"Processing {len(modalities_to_test)} modality combination(s)", level=1)
            vprint(f"Image size: {IMAGE_SIZE} (must match cache if using cache)", level=2)
            vprint("Hybrid mode: Uses cache if available, else extracts on-the-fly with ImageNet weights", level=2)
            vprint("", level=1)

            # Process each combination
            for i, combination in enumerate(modalities_to_test, 1):
                combo_name = '_'.join(sorted(combination))
                vprint(f"[{i}/{len(modalities_to_test)}] {combo_name}", level=1)

                try:
                    # Detect outliers for this combination (hybrid mode: cache or on-the-fly)
                    cleaned_df, outlier_df, output_file = detect_outliers_combination(
                        combination=combination,
                        contamination=OUTLIER_CONTAMINATION,
                        random_state=RANDOM_SEED,
                        force_recompute=False,
                        use_cache=True,  # Enable hybrid mode
                        image_size=IMAGE_SIZE,
                        batch_size=OUTLIER_BATCH_SIZE
                    )

                    if cleaned_df is not None:
                        # Apply cleaned dataset
                        success = apply_cleaned_dataset_combination(
                            combination=combination,
                            contamination=OUTLIER_CONTAMINATION,
                            backup=True
                        )

                        if success:
                            vprint(f"  ✓ Applied for {combo_name}: {len(cleaned_df)} samples", level=1)
                            if outlier_df is not None:
                                vprint(f"    Removed {len(outlier_df)} outliers", level=2)
                        else:
                            vprint(f"  ⚠ Could not apply for {combo_name}, using original", level=1)
                    else:
                        vprint(f"  ⚠ Detection failed for {combo_name}, using original", level=1)

                except Exception as e:
                    vprint(f"  ⚠ Error for {combo_name}: {str(e)}", level=1)
                    vprint(f"    Continuing with original dataset", level=2)

            vprint("", level=1)
            vprint("="*80 + "\n", level=1)
        else:
            vprint("Outlier removal skipped (no modality combinations specified)", level=2)
    else:
        vprint("Outlier removal disabled", level=2)
    # Clear any existing cache files to ensure fresh tf_records for each run
    import glob
    cache_patterns = [
        os.path.join(result_dir, 'tf_cache_train*'),
        os.path.join(result_dir, 'tf_cache_valid*'),
        'tf_cache_train*',
        'tf_cache_valid*'
    ]

    for pattern in cache_patterns:
        try:
            cache_files = glob.glob(pattern)
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                except Exception as e:
                    vprint(f"Warning: Could not remove cache file {cache_file}: {str(e)}", level=2)
        except Exception as e:
            vprint(f"Warning: Error while processing pattern {pattern}: {str(e)}", level=2)

    if mode.lower() == 'search':
        main_search(data_percentage, train_patient_percentage, cv_folds=cv_folds)
    elif mode.lower() == 'specialized':
        # Note: Specialized mode not yet implemented, falling back to search mode
        vprint("Warning: Specialized mode not implemented, running search mode instead", level=0)
        main_search(data_percentage, train_patient_percentage, cv_folds=cv_folds)
    else:
        raise ValueError("Mode must be either 'search' or 'specialized'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate multimodal models for Diabetic Foot Ulcer healing phase classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all 31 modality combinations with default settings (multi-GPU)
  python src/main.py --mode search

  # Use CPU only (no GPUs)
  python src/main.py --mode search --device-mode cpu

  # Use single GPU only (auto-selects best GPU)
  python src/main.py --mode search --device-mode single

  # Use specific GPUs (custom)
  python src/main.py --mode search --device-mode custom --custom-gpus 0 1

  # Quick test with 10% of data
  python src/main.py --mode search --data_percentage 10 --cv_folds 1

  # Run with 3-fold cross-validation
  python src/main.py --mode search --cv_folds 3

  # Filter out low-memory GPUs (require 12GB+)
  python src/main.py --mode search --min-gpu-memory 12.0

Configuration:
  GPU selection:
  - --device-mode: 'cpu', 'single', 'multi' (default), or 'custom'
  - Automatic filtering: >=8GB memory, non-display GPUs
  - Single mode: auto-selects best GPU
  - Multi mode: uses TensorFlow MirroredStrategy

  Modality combinations are configured in src/utils/production_config.py:
  - MODALITY_SEARCH_MODE: 'all' (test all 31) or 'custom' (test specific ones)
  - EXCLUDED_COMBINATIONS: List of combinations to skip
  - INCLUDED_COMBINATIONS: List to test when mode='custom'
  - All other hyperparameters (epochs, batch size, etc.)
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['search', 'specialized', 'grid_search'],
        default='search',
        help="""Mode of operation:
        'search': Test modality combinations and save results to CSV
        'specialized': Run specialized evaluation with gating networks
        'grid_search': Perform grid search for hyperparameter tuning
        (default: search)"""
    )

    parser.add_argument(
        "--data_percentage",
        type=float,
        default=100.0,
        help="""Percentage of total data to use (1-100).
        Useful for quick testing with subset of data.
        Examples: 10 (quick test), 50 (half data), 100 (full data)
        (default: 100.0)"""
    )

    parser.add_argument(
        "--train_patient_percentage",
        type=float,
        default=0.67,
        help="""Percentage of patients to use for training (0.0-1.0).
        The rest will be used for validation.
        Patient-level split ensures no data leakage.
        Examples: 0.67 (67%% train), 0.70 (70%% train), 0.80 (80%% train)
        (default: 0.67)"""
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=3,
        help="""Number of cross-validation folds with patient-level splitting.
        Each fold uses a different subset as validation.
        All data is validated exactly once across all folds.
        Set to 0 or 1 for single train/val split (no cross-validation).
        Examples: 0 (single split), 3 (3-fold CV), 5 (5-fold CV)
        (default: 3)"""
    )

    parser.add_argument(
        "--resume_mode",
        type=str,
        choices=['fresh', 'auto', 'from_data', 'from_models', 'from_predictions'],
        default='auto',
        help="""Control checkpoint resumption and cleanup behavior:
        'fresh': Delete ALL checkpoints (models, predictions, cache, CSV) - start completely fresh
        'auto': Keep all checkpoints, auto-resume from latest state (default)
        'from_data': Keep processed data only, delete models/predictions - retrain from scratch
        'from_models': Keep model weights, delete predictions - regenerate predictions only
        'from_predictions': Keep Block A predictions, delete Block B gating models - retrain ensemble only

        Use Cases:
        - fresh: New experiment, clean slate
        - auto: Continue interrupted training
        - from_data: Retrain with different hyperparameters
        - from_models: Regenerate predictions after model changes
        - from_predictions: Fine-tune gating network only
        (default: auto)"""
    )

    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="""Output verbosity level:
        0 = MINIMAL: Only essential info (errors, final results)
        1 = NORMAL: Standard output (default - current behavior)
        2 = DETAILED: Include debug info, intermediate metrics
        3 = PROGRESS_BAR: Print settings at start, then only progress bar with time estimates
        (default: 1)"""
    )

    parser.add_argument(
        "--device-mode",
        type=str,
        choices=['cpu', 'single', 'multi', 'custom'],
        default='multi',
        help="""Device mode for training:
        'cpu': Use CPU only (disables all GPUs)
        'single': Use single best GPU (auto-selected, >=8GB, non-display)
        'multi': Use all available GPUs (>=8GB, non-display)
        'custom': Use specific GPUs (requires --custom-gpus)

        GPU selection criteria:
        - Minimum 8GB memory (configurable with --min-gpu-memory)
        - Excludes display GPUs by default (unless --include-display-gpus)
        - Single mode: selects GPU with most memory
        - Multi mode: uses TensorFlow MirroredStrategy for data parallelism
        (default: multi)"""
    )

    parser.add_argument(
        "--custom-gpus",
        type=int,
        nargs='+',
        default=None,
        help="""GPU IDs to use in 'custom' device mode.
        Must be valid GPU indices from nvidia-smi.
        GPU filtering (memory, display) still applies.
        Examples: --custom-gpus 0 (single GPU 0)
                  --custom-gpus 0 1 (GPUs 0 and 1)
                  --custom-gpus 1 2 3 (GPUs 1, 2, and 3)
        (default: None - required for custom mode)"""
    )

    parser.add_argument(
        "--min-gpu-memory",
        type=float,
        default=8.0,
        help="""Minimum GPU memory in GB for GPU selection.
        GPUs with less memory will be excluded.
        Useful to filter out small/old GPUs.
        Examples: 8.0 (exclude GPUs < 8GB)
                  12.0 (exclude GPUs < 12GB)
                  24.0 (only high-end GPUs)
        (default: 8.0)"""
    )

    parser.add_argument(
        "--include-display-gpus",
        action='store_true',
        help="""Include GPUs with active displays in selection.
        By default, display GPUs are excluded to avoid interference.
        Use this flag if you want to train on your display GPU.
        WARNING: Training on display GPU may cause UI lag.
        (default: False - display GPUs excluded)"""
    )

    args = parser.parse_args()

    # Set up GPU/device strategy BEFORE importing TensorFlow models
    # This must happen early to properly configure CUDA_VISIBLE_DEVICES
    try:
        strategy, selected_gpus = setup_device_strategy(
            mode=args.device_mode,
            custom_gpus=args.custom_gpus,
            min_memory_gb=args.min_gpu_memory,
            exclude_display=not args.include_display_gpus,
            verbose=True  # Always show GPU config
        )
    except ValueError as e:
        print(f"\n❌ GPU configuration error: {e}")
        print("Falling back to CPU mode...")
        strategy, selected_gpus = setup_device_strategy(mode='cpu', verbose=True)

    # Store strategy in global scope for use in training functions
    # This allows training_utils.py to access the strategy
    globals()['DISTRIBUTION_STRATEGY'] = strategy
    globals()['SELECTED_GPUS'] = selected_gpus

    # Now that CUDA_VISIBLE_DEVICES is set, configure TensorFlow
    tf.random.set_seed(RANDOM_SEED)
    apply_environment_config()  # Apply threading and determinism settings
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    # Calculate effective batch size now that strategy is set up
    globals()['strategy'] = strategy  # Update global strategy variable
    batch_size = global_batch_size // strategy.num_replicas_in_sync
    globals()['batch_size'] = batch_size
    vprint(f"\nBatch size per replica: {batch_size} (global batch size: {global_batch_size}, replicas: {strategy.num_replicas_in_sync})", level=0)

    # Set verbosity level
    set_verbosity(args.verbosity)

    # Suppress library warnings when in progress bar mode (verbosity 3)
    if args.verbosity == 3:
        import warnings
        import numpy as np
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress NumPy runtime warnings
        warnings.filterwarnings('ignore', message='.*OpenSSL.*')
        warnings.filterwarnings('ignore', message='.*CUDA.*')
        warnings.filterwarnings('ignore', message='.*cuDNN.*')
        np.seterr(all='ignore')  # Suppress all NumPy warnings
        # Note: TF_CPP_MIN_LOG_LEVEL must be set before TensorFlow import (done at top of file)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('absl').setLevel(logging.ERROR)

    # Print configuration (always shown, even in progress bar mode)
    vprint("\n" + "="*80, level=0)
    vprint("DFU MULTIMODAL CLASSIFICATION - PRODUCTION PIPELINE", level=0)
    vprint("="*80, level=0)
    vprint(f"Mode: {args.mode}", level=0)
    vprint(f"Resume mode: {args.resume_mode}", level=0)
    vprint(f"Data percentage: {args.data_percentage}%", level=0)
    vprint(f"Verbosity: {args.verbosity} ({'MINIMAL' if args.verbosity == 0 else 'NORMAL' if args.verbosity == 1 else 'DETAILED' if args.verbosity == 2 else 'PROGRESS_BAR'})", level=0)

    # Device information
    if selected_gpus:
        if len(selected_gpus) == 1:
            vprint(f"Device: GPU {selected_gpus[0]} (single GPU mode)", level=0)
        else:
            vprint(f"Device: GPUs {selected_gpus} (multi-GPU mode, MirroredStrategy)", level=0)
            vprint(f"  Replicas: {strategy.num_replicas_in_sync}× batch size distribution", level=0)
    else:
        vprint(f"Device: CPU only", level=0)

    if args.cv_folds > 1:
        vprint(f"Cross-validation: {args.cv_folds}-fold CV (patient-level)", level=0)
    else:
        vprint(f"Train/validation split: {args.train_patient_percentage*100:.0f}% train / {(1-args.train_patient_percentage)*100:.0f}% val", level=0)
    vprint(f"\nConfiguration loaded from: src/utils/production_config.py", level=0)
    vprint(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}", level=0)
    vprint(f"Batch size: {GLOBAL_BATCH_SIZE}", level=0)
    if len(selected_gpus) > 1:
        per_gpu_batch = get_effective_batch_size(GLOBAL_BATCH_SIZE, strategy)
        vprint(f"  Per-GPU batch: {per_gpu_batch} ({GLOBAL_BATCH_SIZE} / {len(selected_gpus)} GPUs)", level=0)
    vprint(f"Max epochs: {N_EPOCHS} (with early stopping)", level=0)
    if args.mode == 'search':
        vprint(f"Modality search mode: {MODALITY_SEARCH_MODE}", level=0)
        if MODALITY_SEARCH_MODE == 'all':
            vprint(f"Will test all 31 modality combinations", level=0)
            if EXCLUDED_COMBINATIONS:
                vprint(f"Excluding {len(EXCLUDED_COMBINATIONS)} combinations", level=0)
        else:
            vprint(f"Will test {len(INCLUDED_COMBINATIONS)} custom combinations", level=0)
    vprint("="*80 + "\n", level=0)

    # Handle resume mode cleanup
    from src.utils.config import cleanup_for_resume_mode
    cleanup_for_resume_mode(resume_mode=args.resume_mode)

    # Clear memory before starting
    clear_gpu_memory()
    reset_keras()
    clear_cuda_memory()

    # Build thresholds dict if any threshold is specified
    thresholds = None

    # Load optimized thresholds from auto_polish_dataset_v2.py if --core-data is set
    # Data cleaning configuration summary
    vprint("\nData Cleaning Configuration (from production_config.py):", level=0)
    vprint(f"  Outlier removal: {OUTLIER_REMOVAL}", level=0)
    if OUTLIER_REMOVAL:
        vprint(f"  Outlier contamination: {OUTLIER_CONTAMINATION*100:.0f}%", level=0)
        vprint(f"  Outlier batch size: {OUTLIER_BATCH_SIZE}", level=0)
    vprint(f"  Misclassification tracking: {TRACK_MISCLASS}", level=0)
    if USE_CORE_DATA or any([THRESHOLD_I, THRESHOLD_P, THRESHOLD_R]):
        vprint(f"  Core data mode: {USE_CORE_DATA}", level=0)
        if any([THRESHOLD_I, THRESHOLD_P, THRESHOLD_R]):
            vprint(f"  Thresholds: I={THRESHOLD_I}, P={THRESHOLD_P}, R={THRESHOLD_R}", level=0)

    # Run the selected mode
    main(args.mode, args.data_percentage, args.train_patient_percentage, args.cv_folds)

    # Clear memory after completion
    clear_gpu_memory()
    reset_keras()
    clear_cuda_memory()
