"""
Training utilities including cross-validation, checkpointing, and evaluation.
Functions for managing training runs, saving/loading predictions, and result aggregation.
"""

import os
import pandas as pd
import numpy as np
import pickle
import csv
import random
import gc
import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import get_project_paths, get_output_paths, CLASS_LABELS
from src.utils.debug import clear_gpu_memory
from src.utils.verbosity import vprint, get_verbosity
from src.utils.production_config import (
    GLOBAL_BATCH_SIZE, N_EPOCHS, IMAGE_SIZE,
    SEARCH_MULTIPLE_CONFIGS, SEARCH_CONFIG_VARIANTS,
    GRID_SEARCH_GAMMAS, GRID_SEARCH_ALPHAS, FOCAL_ORDINAL_WEIGHT
)
from src.data.dataset_utils import prepare_cached_datasets, BatchVisualizationCallback, TrainingHistoryCallback
from src.data.generative_augmentation_v2 import AugmentationConfig, GenerativeAugmentationManager, GenerativeAugmentationCallback
from src.models.builders import create_multimodal_model, MetadataConfidenceCallback
from src.models.losses import get_focal_ordinal_loss, weighted_f1_score
from src.evaluation.metrics import track_misclassifications

# Get paths
directory, result_dir, root = get_project_paths()
output_paths = get_output_paths(result_dir)
ck_path = output_paths['checkpoints']
models_path = output_paths['models']
csv_path = output_paths['csv']
misclass_path = output_paths['misclassifications']
vis_path = output_paths['visualizations']
logs_path = output_paths['logs']

class EpochMemoryCallback(tf.keras.callbacks.Callback):
    """Memory management callback that's compatible with distribution strategy"""
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        # tf.keras.backend.clear_session()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, _ in enumerate(gpus):
                tf.config.experimental.reset_memory_stats(f'GPU:{i}')
class NaNMonitorCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor for NaN values in validation metrics
    and trigger training restart if detected.
    """
    def __init__(self):
        super(NaNMonitorCallback, self).__init__()
        self.nan_detected = False
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'val_weighted_f1_score' in logs:
            if np.isnan(logs['val_weighted_f1_score']):
                vprint("\nNaN detected in validation weighted F1 score. Triggering training restart...", level=1)
                self.nan_detected = True
                self.model.stop_training = True
def clean_up_training_resources():
    """Helper function to clean up resources before restarting training"""
    # Clean GPU memory
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Try to clear GPU memory stats
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                tf.config.experimental.reset_memory_stats(f'GPU:{i}')
    except Exception as e:
        vprint(f"Error clearing memory stats: {str(e)}", level=2)
    
    # Remove cache files
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

def create_checkpoint_filename(selected_modalities, run=1, config_name=0):
    modality_str = '_'.join(sorted(selected_modalities))
    checkpoint_name = f'{modality_str}_{run}_{config_name}.h5'
    return os.path.join(models_path, checkpoint_name)
class EscapingReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=5, 
                 escape_factor=5.0, min_lr=1e-6, escape_patience=2):
        """
        Modified ReduceLROnPlateau that temporarily increases LR to escape plateaus.
        
        Args:
            monitor: Quantity to monitor
            factor: Factor by which to reduce LR when plateauing
            patience: Number of epochs with no improvement before reducing LR
            escape_factor: Factor by which to increase LR when trying to escape
            min_lr: Minimum learning rate
            escape_patience: Number of plateau detections before trying to escape
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.escape_factor = escape_factor
        self.min_lr = min_lr
        self.escape_patience = escape_patience
        
        # Internal state
        self.best = float('inf')
        self.wait = 0
        self.plateau_count = 0
        self.is_escaping = False
        self.escape_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        # Get and store current learning rate at the start of epoch
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        # Make sure LR is included in logs
        logs['lr'] = current_lr
        
        # If we're in escape mode, check if it helped
        if self.is_escaping:
            if current < self.best:
                # Escape was successful, reset state
                vprint(f'\nEscape successful! New best {self.monitor}: {current:.6f}', level=2)
                self.best = current
                self.wait = 0
                self.plateau_count = 0
                self.is_escaping = False
            elif epoch - self.escape_epoch >= 2:  # Give escape 2 epochs to work
                # Escape didn't work, revert and reduce LR
                # print(f'\nEscape unsuccessful, reducing LR from {current_lr:.2e} to {current_lr * self.factor:.2e}')
                new_lr = max(current_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.is_escaping = False
                self.wait = 0
            return
            
        # Check if improved
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
        # Check if we should try escaping
        if self.wait >= self.patience:
            self.plateau_count += 1
            
            # If we've hit enough plateaus, try escaping
            if self.plateau_count >= self.escape_patience:
                new_lr = current_lr * self.escape_factor
                vprint(f'\nAttempting to escape plateau by increasing LR from {current_lr:.2e} to {new_lr:.2e}', level=2)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.is_escaping = True
                self.escape_epoch = epoch
                self.plateau_count = 0
            else:
                # Otherwise, reduce LR as normal
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr != current_lr:
                    # print(f'\nReducing LR from {current_lr:.2e} to {new_lr:.2e}')
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.wait = 0
class ProcessedDataManager:
    def __init__(self, data, directory):
        """
        Initialize the ProcessedDataManager.
        
        Args:
            data: The input data DataFrame
            directory: Base directory for the project
        """
        self.directory = directory
        self.data = data.copy(deep=True)
        self.all_modality_shapes = {}
        self.cached_datasets = {}
        
    def process_all_modalities(self):
        """Process all modalities and store their shapes."""
        # Get metadata shape after preprocessing
        vprint("Processing metadata shape...", level=2)
        temp_data = self.data.copy()
        temp_train, _, _, _, _, _ = prepare_cached_datasets(
            temp_data,
            ['metadata'],
            train_patient_percentage=0.8,
            batch_size=1,
            run=0,
            for_shape_inference=True
        )
        for batch in temp_train.take(1):
            self.all_modality_shapes['metadata'] = batch[0]['metadata_input'].shape[1:]
            break

        # Set image shapes
        vprint("Setting image shapes...", level=2)
        for modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
            self.all_modality_shapes[modality] = (IMAGE_SIZE, IMAGE_SIZE, 3)
        
        del temp_train, temp_data
        gc.collect()
        clear_gpu_memory()
        clear_cache_files()
        
        
    def get_shapes_for_modalities(self, selected_modalities):
        """Get input shapes for selected modalities."""
        return {mod: self.all_modality_shapes[mod] for mod in selected_modalities if mod in self.all_modality_shapes}
class CohenKappa(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name='cohen_kappa', **kwargs):
        super(CohenKappa, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Initialize confusion matrix
        self.confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true values to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Update confusion matrix
        confusion = tf.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                mask = tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, j))
                count = tf.reduce_sum(tf.cast(mask, tf.float32))
                confusion = tf.tensor_scatter_nd_add(
                    confusion,
                    [[i, j]],
                    [count]
                )
        
        self.confusion_matrix.assign_add(confusion)
        
    def result(self):
        # Calculate observed agreement
        n = tf.reduce_sum(self.confusion_matrix)
        observed = tf.reduce_sum(tf.linalg.diag_part(self.confusion_matrix)) / n
        
        # Calculate expected agreement
        row_sums = tf.reduce_sum(self.confusion_matrix, axis=1)
        col_sums = tf.reduce_sum(self.confusion_matrix, axis=0)
        expected = tf.reduce_sum((row_sums * col_sums) / (n * n))
        
        # Calculate kappa
        kappa = (observed - expected) / (1.0 - expected + tf.keras.backend.epsilon())
        return kappa
        
    def reset_state(self):
        # Reset confusion matrix to zeros
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))
class WeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, alpha_values, name='weighted_accuracy', **kwargs):
        super(WeightedAccuracy, self).__init__(name=name, **kwargs)
        self.alpha_values = tf.constant(alpha_values, dtype=tf.float32)
        self.weighted_true_positives = self.add_weight(name='weighted_tp', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Create a mask for correct predictions
        correct_predictions = tf.cast(tf.equal(y_pred, y_true), tf.float32)
        
        # Apply class weights based on true labels
        weights = tf.gather(self.alpha_values, y_true)
        weighted_correct = correct_predictions * weights
        
        # Update metrics
        self.weighted_true_positives.assign_add(tf.reduce_sum(weighted_correct))
        self.total_samples.assign_add(tf.reduce_sum(weights))

    def result(self):
        return self.weighted_true_positives / self.total_samples

    def reset_state(self):
        self.weighted_true_positives.assign(0.0)
        self.total_samples.assign(0.0)

class MacroF1Score(tf.keras.metrics.Metric):
    """Macro-averaged F1 score metric - robust to class imbalance."""
    def __init__(self, num_classes=3, name='macro_f1', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Per-class true positives, false positives, false negatives
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)

        # Calculate per-class metrics using vectorized operations
        # Create one-hot encodings for efficient computation
        y_true_one_hot = tf.one_hot(y_true, self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, self.num_classes)

        # True positives: both predicted and true
        tp_per_class = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)

        # False positives: predicted but not true
        fp_per_class = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)

        # False negatives: true but not predicted
        fn_per_class = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        # Update weights - use assign_add on the entire tensor, not individual elements
        self.tp.assign_add(tp_per_class)
        self.fp.assign_add(fp_per_class)
        self.fn.assign_add(fn_per_class)

    def result(self):
        # Calculate F1 per class
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1_per_class = 2 * precision * recall / (precision + recall + 1e-7)

        # Macro average (equal weight to each class)
        macro_f1 = tf.reduce_mean(f1_per_class)
        return macro_f1

    def reset_state(self):
        self.tp.assign(tf.zeros((self.num_classes,)))
        self.fp.assign(tf.zeros((self.num_classes,)))
        self.fn.assign(tf.zeros((self.num_classes,)))

def analyze_modality_contributions(attention_outputs, modality_names):
    """Normalize attention values from each modality to [0, 1] range"""
    normalized_outputs = []
    
    for attention in attention_outputs:
        # Flatten for easier processing
        flat_values = attention.reshape(-1)
        
        # Get min and max
        min_val = np.min(flat_values)
        max_val = np.max(flat_values)
        
        # Normalize to [0, 1]
        normalized = (flat_values - min_val) / (max_val - min_val + 1e-7)
        
        # Reshape back to original shape
        normalized = normalized.reshape(attention.shape)
        normalized_outputs.append(normalized)
    
    return normalized_outputs
class ModalityContributionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, save_dir='modality_analysis', monitor='val_weighted_accuracy', mode='max', run_number=None):
        super().__init__()
        self.save_dir = save_dir
        self.val_data = val_data
        os.makedirs(save_dir, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.run_number = run_number
        self.best_attention_values = None
        self.best_modality_names = None
        
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
            vprint(f"\nGenerating modality contribution analysis for epoch {epoch + 1}", level=2)
            
            # # Find layers with attention outputs
            # modality_outputs = {}
            # for layer in self.model.layers:
            #     if 'modular_attention' in layer.name or 'metadata_attention' in layer.name:
            #         modality_name = layer.name.split('_')[0] + "_" + layer.name.split('_')[1]
            #         if "metadata" in modality_name:
            #             modality_name = "metadata"
            #         modality_outputs[modality_name] = layer
            
            modality_outputs = {}
            outputs = []
            for layer in self.model.layers:
                # Look for any layer with final outputs from each modality branch
                # if any(mod in layer.name for mod in ['metadata_BN', 'depth_rgb_projection3', 'depth_map_projection3', 'thermal_map_projection3']):
                if any(mod in layer.name for mod in ['metadata_BN', 'depth_rgb_BN_proj3', 'depth_map_BN_proj3', 'thermal_map_BN_proj3']):
                # if any(mod in layer.name for mod in ['metadata_attention', 'depth_rgb_modular_attention', 'depth_map_modular_attention', 'thermal_map_modular_attention']):
                    modality_name = layer.name.split('_')[0] + "_" + layer.name.split('_')[1]
                    if "metadata" in modality_name:
                        modality_name = "metadata"
                    modality_outputs[modality_name] = layer
                    outputs.append(layer.output)
            
            if modality_outputs:
                # Create visualization model
                vis_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[self.model.output] + outputs
                )
                
                # Get predictions and attention values
                predictions = vis_model.predict(self.val_data, verbose=0)
                overall_pred = predictions[0]  # Model predictions
                attention_outputs = predictions[1:]  # Attention outputs
                
                # Normalize attention outputs
                attention_outputs = analyze_modality_contributions(
                    attention_outputs, 
                    list(modality_outputs.keys())
                )
                
                # Store best attention values for this run
                self.best_attention_values = attention_outputs
                self.best_modality_names = list(modality_outputs.keys())
                
                # Create multiple visualizations
                fig = plt.figure(figsize=(15, 10))
                gs = plt.GridSpec(2, 2)
                
                # 1. Average attention magnitude per modality
                ax1 = fig.add_subplot(gs[0, 0])
                avg_magnitudes = [np.mean(att) for att in attention_outputs]
                std_magnitudes = [np.std(att) for att in attention_outputs]
                bars = ax1.bar(modality_outputs.keys(), avg_magnitudes, yerr=std_magnitudes)
                ax1.set_title('Average Attention Magnitude per Modality')
                ax1.set_ylabel('Magnitude')
                
                # 2. Attention distribution violin plot
                ax2 = fig.add_subplot(gs[0, 1])
                violin_data = [att.flatten() for att in attention_outputs]
                ax2.violinplot(violin_data)
                ax2.set_xticks(range(1, len(modality_outputs) + 1))
                ax2.set_xticklabels(modality_outputs.keys())
                ax2.set_title('Distribution of Attention Values')
                ax2.set_ylabel('Attention Value')
                ax2.grid(True, axis='y', linestyle='-', linewidth=0.5, color='gray')
                ax2.tick_params(axis='both', which='major', labelsize=18)
                
                # # 3. Feature importance heatmap
                # ax3 = fig.add_subplot(gs[1, :])
                # feature_importance = np.array([np.mean(att, axis=0) for att in attention_outputs[1]])
                # sns.heatmap(feature_importance, 
                #           xticklabels=range(feature_importance.shape[1]),
                #           yticklabels=list(modality_outputs.keys())[1],
                #           cmap='viridis',
                #           ax=ax3)
                # ax3.set_title('Feature Importance per Modality')
                # ax3.set_xlabel('Feature Dimension')
                # ax3.set_ylabel('Modality')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f'modality_analysis_epoch_{epoch+1}.png'))
                plt.close()
                
                # Save numerical results
                with open(os.path.join(self.save_dir, f'modality_stats_epoch_{epoch+1}.txt'), 'w') as f:
                    f.write(f"Epoch {epoch + 1} Modality Analysis\n")
                    f.write("=" * 50 + "\n\n")
                    for mod_name, attention in zip(modality_outputs.keys(), attention_outputs):
                        f.write(f"\n{mod_name.upper()} MODALITY:\n")
                        f.write(f"Mean attention: {np.mean(attention):.4f}\n")
                        f.write(f"Std attention: {np.std(attention):.4f}\n")
                        f.write(f"Max attention: {np.max(attention):.4f}\n")
                        f.write(f"Min attention: {np.min(attention):.4f}\n")
                        
                # Save the raw attention values for later averaging
                np.save(os.path.join(self.save_dir, f'attention_values_run{self.run_number}.npy'), 
                       {'attention_outputs': attention_outputs, 
                        'modality_names': list(modality_outputs.keys())})
def average_attention_values(result_dir, num_runs):
    """Calculate and visualize average attention values across all runs using violin plots."""
    all_attention_outputs = []
    modality_names = None
    name_mapping = {
        'metadata': 'Metadata',
        'depth_rgb': 'RGB',
        'depth_map': 'Depth',
        'thermal_map': 'Thermal'
    }
    # Load attention values from each run
    for run in range(1, num_runs + 1):
        file_path = os.path.join(result_dir, 'modality_analysis', f'attention_values_run{run}.npy')
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item()
            # Store both means and raw values
            all_attention_outputs.append({
                'means': [np.mean(att) for att in data['attention_outputs']],
                'raw_values': data['attention_outputs']
            })
            if modality_names is None:
                modality_names = data['modality_names']
    
    if not all_attention_outputs:
        vprint("No attention values found!", level=1)
        return

    # Prepare data for violin plots
    num_modalities = len(modality_names)
    attention_data = [[] for _ in range(num_modalities)]
    run_means_per_modality = [[] for _ in range(num_modalities)]
    
    # Collect all values and means
    for run_data in all_attention_outputs:
        for i in range(num_modalities):
            mean_value = run_data['means'][i]
            run_means_per_modality[i].append(mean_value)
            
            # Only take the attention scores, not the full raw values
            # Assuming attention scores are what we want to visualize
            attention_scores = run_data['raw_values'][i]
            if isinstance(attention_scores, np.ndarray):
                # Take the mean across appropriate dimensions if needed
                if attention_scores.ndim > 1:
                    attention_scores = np.mean(attention_scores, axis=tuple(range(attention_scores.ndim-1)))
                attention_data[i].extend(attention_scores.flatten())

    # Create visualization with violin plots
    plt.figure(figsize=(15, 8))
    
    # Create violin plot with quartile lines
    parts = plt.violinplot(attention_data, positions=range(1, num_modalities + 1),
                          showmeans=True, showextrema=True, showmedians=True)
    
    # Customize violin plot appearance
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    
    # Add individual points for run means
    for i in range(num_modalities):
        plt.scatter([i + 1] * len(run_means_per_modality[i]), 
                   run_means_per_modality[i],
                   color='white', edgecolor='black', 
                   s=100, zorder=3, alpha=0.6)
    
    # Add min, max, and quartile annotations
    for i, data in enumerate(attention_data):
        x_pos = i + 1
        min_val = np.min(data)
        max_val = np.max(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        plt.text(x_pos, min_val, f'{min_val:.3f}', ha='center', va='bottom')
        plt.text(x_pos, max_val, f'{max_val:.3f}', ha='center', va='top')
        plt.text(x_pos + 0.2, q1, f'Q1: {q1:.3f}', ha='left', va='center')
        plt.text(x_pos + 0.2, q3, f'Q3: {q3:.3f}', ha='left', va='center')
    
    modality_namest = [name_mapping.get(name, name) for name in modality_names]
    modality_names = modality_namest
    # Customize plot
    plt.title('Distribution of Attention Values per Modality', fontsize=16, pad=20)
    plt.ylabel('Attention Value', fontsize=16)
    plt.xlabel('Modality', fontsize=16)
    plt.xticks(range(1, num_modalities + 1), modality_names, rotation=45, fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='Distribution'),
        plt.Line2D([0], [0], color='red', label='Mean', linewidth=2),
        plt.Line2D([0], [0], color='black', label='Median', linewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='w', markeredgecolor='black',
                  markersize=10, label='Run Means')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'modality_analysis', 'attention_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open(os.path.join(result_dir, 'modality_analysis', 'attention_distribution_stats.txt'), 'w') as f:
        f.write("Attention Distribution Analysis Across All Runs\n")
        f.write("=" * 50 + "\n\n")
        for i, mod_name in enumerate(modality_names):
            f.write(f"\n{mod_name.upper()} MODALITY:\n")
            data = attention_data[i]
            f.write(f"Overall mean: {np.mean(data):.4f}\n")
            f.write(f"Overall median: {np.median(data):.4f}\n")
            f.write(f"Overall std: {np.std(data):.4f}\n")
            f.write(f"Q1 (25th percentile): {np.percentile(data, 25):.4f}\n")
            f.write(f"Q3 (75th percentile): {np.percentile(data, 75):.4f}\n")
            f.write(f"Min value: {np.min(data):.4f}\n")
            f.write(f"Max value: {np.max(data):.4f}\n")
            
            f.write("\nRun means:\n")
            for run_idx, run_mean in enumerate(run_means_per_modality[i]):
                f.write(f"Run {run_idx + 1}: {run_mean:.4f}\n")
            f.write("\n")
def cross_validation_manual_split(data, configs, train_patient_percentage=0.8, n_runs=None, cv_folds=3):
    """
    Perform cross-validation using cached dataset pipeline.

    Args:
        data: Input DataFrame
        configs: Dictionary of configurations for different modality combinations, or a list of modalities
        train_patient_percentage: Percentage of data to use for training (ignored if cv_folds > 1)
        n_runs: DEPRECATED - Number of random holdout runs (backwards compatibility)
        cv_folds: Number of k-fold CV folds (default: 3). Set to 0 or 1 for single train/val split.

    Returns:
        Tuple of (all_metrics, all_confusion_matrices, all_histories)
    """
    # Handle backwards compatibility: if n_runs is specified, use it
    if n_runs is not None:
        vprint(f"Warning: n_runs parameter is deprecated. Please use cv_folds instead.", level=1)
        cv_folds = 0  # Force single split mode when using legacy n_runs
        use_legacy_mode = True
        num_iterations = n_runs
    else:
        use_legacy_mode = False
        if cv_folds <= 1:
            num_iterations = 1  # Single split
        else:
            num_iterations = cv_folds  # k-fold CV
    # Handle configs being passed as a list instead of dict
    if isinstance(configs, list):
        modality_list = configs  # Save original list
        modality_name = '+'.join(modality_list)

        if SEARCH_MULTIPLE_CONFIGS and SEARCH_CONFIG_VARIANTS > 1:
            # Create multiple configs with different loss parameters for gating network
            configs = {}
            num_variants = min(SEARCH_CONFIG_VARIANTS, len(GRID_SEARCH_GAMMAS) * len(GRID_SEARCH_ALPHAS))

            variant_idx = 0
            for gamma in GRID_SEARCH_GAMMAS[:SEARCH_CONFIG_VARIANTS]:
                for alpha_set in GRID_SEARCH_ALPHAS[:max(1, SEARCH_CONFIG_VARIANTS // len(GRID_SEARCH_GAMMAS))]:
                    if variant_idx >= num_variants:
                        break

                    config_name = f"{modality_name}_v{variant_idx + 1}"
                    configs[config_name] = {
                        'modalities': modality_list,
                        'batch_size': GLOBAL_BATCH_SIZE,
                        'max_epochs': N_EPOCHS,
                        'image_size': IMAGE_SIZE,
                        'gamma': gamma,
                        'alpha': alpha_set,
                        'ordinal_weight': FOCAL_ORDINAL_WEIGHT
                    }
                    variant_idx += 1

                if variant_idx >= num_variants:
                    break

            vprint(f"Created {len(configs)} config variants for {modality_name} with different loss parameters", level=2)
        else:
            # Original behavior: single config
            configs = {
                modality_name: {
                    'modalities': modality_list,
                    'batch_size': GLOBAL_BATCH_SIZE,
                    'max_epochs': N_EPOCHS,
                    'image_size': IMAGE_SIZE
                }
            }

    # Extract common parameters from configs (all configs should have same values)
    first_config = next(iter(configs.values()))
    batch_size = first_config['batch_size']
    max_epochs = first_config['max_epochs']
    image_size = first_config['image_size']

    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')

    # Setup distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    all_metrics = []
    all_confusion_matrices = []
    all_histories = []
    all_gating_results = []
    all_runs_metrics = []

    # Generate patient folds for k-fold CV (if cv_folds > 1)
    if not use_legacy_mode and cv_folds > 1:
        vprint(f"\n{'='*80}", level=1)
        vprint(f"GENERATING {cv_folds}-FOLD CROSS-VALIDATION SPLITS (PATIENT-LEVEL)", level=1)
        vprint(f"{'='*80}", level=1)
        from src.data.dataset_utils import create_patient_folds
        patient_fold_splits = create_patient_folds(data, n_folds=cv_folds, random_state=42, max_imbalance=0.3)
        vprint(f"Generated {len(patient_fold_splits)} folds", level=1)
        vprint(f"All data will be validated exactly once across all folds", level=1)
        vprint(f"{'='*80}\n", level=1)
    else:
        patient_fold_splits = None

    for iteration_idx in range(num_iterations):
        # Use appropriate naming: "Fold" for k-fold CV, "Run" for legacy mode
        if not use_legacy_mode and cv_folds > 1:
            iteration_name = f"Fold {iteration_idx + 1}/{cv_folds}"
        else:
            iteration_name = f"Run {iteration_idx + 1}/{num_iterations}"

        # For backwards compatibility, maintain "run" variable for file naming
        run = iteration_idx
        # Clean up after each modality combination
        try:
            clear_gpu_memory()
            clear_cache_files()
        except Exception as e:
            vprint(f"Error clearing memory stats: {str(e)}", level=2)
        # Reset random seeds for next iteration
        random.seed(42 + run * (run + 3))
        tf.random.set_seed(42 + run * (run + 3))
        np.random.seed(42 + run * (run + 3))
        os.environ['PYTHONHASHSEED'] = str(42 + run * (run + 3))

        vprint(f"\n{iteration_name}", level=1)

        # Get patient splits for this iteration (k-fold CV or random split)
        if patient_fold_splits is not None:
            # K-fold CV: use pre-computed fold splits
            fold_train_patients, fold_valid_patients = patient_fold_splits[iteration_idx]
            vprint(f"Using pre-computed fold {iteration_idx + 1} patient split", level=1)
        else:
            # Legacy mode or single split: let prepare_cached_datasets handle it
            fold_train_patients, fold_valid_patients = None, None

        # Check if this iteration is already complete
        if is_run_complete(run + 1, ck_path):
            vprint(f"\n{iteration_name} is already complete. Moving to next...", level=1)
            continue

        # Try to load aggregated predictions first (only useful when training multiple configs for same modalities)
        # In search mode (single config per combination), skip this to force fresh training
        run_predictions_list_t, run_true_labels_t = None, None
        run_predictions_list_v, run_true_labels_v = None, None

        if len(configs) > 1:
            # Only check for existing predictions if we have multiple configs (specialized mode)
            run_predictions_list_t, run_true_labels_t = load_aggregated_predictions(run + 1, ck_path, dataset_type='train')
            run_predictions_list_v, run_true_labels_v = load_aggregated_predictions(run + 1, ck_path, dataset_type='valid')

        if run_predictions_list_t is not None and run_predictions_list_v is not None and run_true_labels_t is not None and run_true_labels_v is not None:
            vprint(f"\nLoaded aggregated predictions for run {run + 1}", level=1)
            vprint(f"Number of models: {len(run_predictions_list_t)}", level=2)
            vprint(f"Shape of predictions from first model: {run_predictions_list_t[0].shape}", level=2)
            vprint(f"Labels shape: {run_true_labels_t.shape}", level=2)

            # Proceed directly to gating network training with loaded predictions
            if len(run_predictions_list_t) == len(configs) and len(run_predictions_list_v) == len(configs):
                vprint(f"\nTraining gating network for run {run + 1}...", level=1)
                try:
                    # Import here to avoid circular dependency
                    from src.main import train_gating_network

                    # Convert labels back to class indices for gating network
                    gating_labels_t = np.argmax(run_true_labels_t, axis=1) if len(run_true_labels_t.shape) > 1 else run_true_labels_t
                    gating_labels_v = np.argmax(run_true_labels_v, axis=1) if len(run_true_labels_v.shape) > 1 else run_true_labels_v
                    combined_predictions, gating_labels = train_gating_network(
                        run_predictions_list_t,
                        run_predictions_list_v,
                        gating_labels_t,
                        gating_labels_v,
                        run + 1,
                        find_optimal=False,
                        min_models=2,
                        max_tries=200
                    )

                    # Calculate and store gating network results
                    final_predictions = np.argmax(combined_predictions, axis=1)
                    gating_metrics = {
                        'run': run + 1,
                        'accuracy': accuracy_score(gating_labels, final_predictions),
                        'f1_macro': f1_score(gating_labels, final_predictions, average='macro'),
                        'f1_weighted': f1_score(gating_labels, final_predictions, average='weighted'),
                        'kappa': cohen_kappa_score(gating_labels, final_predictions, weights='quadratic'),
                        'confusion_matrix': confusion_matrix(gating_labels, final_predictions)
                    }
                    
                    vprint(f"\nGating Network Results for Run {run + 1}:", level=1)
                    vprint(f"Accuracy: {gating_metrics['accuracy']:.4f}", level=1)
                    vprint(f"F1 Macro: {gating_metrics['f1_macro']:.4f}", level=1)
                    vprint(f"Kappa: {gating_metrics['kappa']:.4f}", level=1)

                    all_gating_results.append(gating_metrics)
                    save_run_results(gating_metrics, run + 1, result_dir)
                    continue  # Move to next run

                except Exception as e:
                    vprint(f"Error in gating network training: {str(e)}", level=1)
                    # Fall through to regenerate predictions
                    run_predictions_list_t = []
                    run_predictions_list_v = []
                    run_true_labels_t = None
                    run_true_labels_v = None
            else:
                vprint(f"Found incomplete set of predictions ({len(run_predictions_list_t)} of {len(configs)})", level=1)
                run_predictions_list_t = []
                run_predictions_list_v = []
                run_true_labels_t = None
                run_true_labels_v = None
        else:
            run_predictions_list_t = []
            run_predictions_list_v = []
            run_true_labels_t = None
            run_true_labels_v = None
            
        # Get list of completed configs for this run
        completed_configs = get_completed_configs_for_run(run + 1, configs.keys(), ck_path, dataset_type='valid')
        if completed_configs:
            vprint(f"\nFound completed configs for run {run + 1}: {completed_configs}", level=1)
        
        # Initialize data manager for this run
        data_manager = ProcessedDataManager(data.copy(), directory)
        data_manager.process_all_modalities()
        
        # Setup augmentation once per run
        aug_config = AugmentationConfig()
        aug_config.generative_settings['output_size']['width'] = IMAGE_SIZE
        aug_config.generative_settings['output_size']['height'] = IMAGE_SIZE
        
        gen_manager = GenerativeAugmentationManager(
            base_dir=os.path.join(directory, 'Codes/MultimodalClassification/ImageGeneration/models_5_7'),
            config=aug_config
        )
        
        # Get all unique modalities from all configs
        all_modalities = set()
        for config in configs.values():
            all_modalities.update(config['modalities'])
        all_modalities = list(all_modalities)
        
        vprint(f"\nPreparing datasets for {iteration_name} with all modalities: {all_modalities}", level=1)
        # Create cached datasets once for all modalities
        master_train_dataset, pre_aug_dataset, master_valid_dataset, master_steps_per_epoch, master_validation_steps, master_alpha_value = prepare_cached_datasets(
            data_manager.data,
            all_modalities,  # Use all modalities
            train_patient_percentage=train_patient_percentage,
            batch_size=batch_size,
            gen_manager=gen_manager,
            aug_config=aug_config,
            run=run,
            image_size=image_size,
            train_patients=fold_train_patients,  # Pass pre-computed fold splits for k-fold CV
            valid_patients=fold_valid_patients
        )
        
        run_metrics = []
        
        # For each modality combination
        for config_name, config in configs.items():
            # Before starting new config
            try:
                gc.collect()  # Keep basic garbage collection
                if gpus:
                    for i in range(len(gpus)):
                        try:
                            tf.config.experimental.reset_memory_stats(f'GPU:{i}')  # Reset GPU stats without clearing session
                        except:
                            pass
            except Exception as e:
                vprint(f"Error in cleanup between configs: {str(e)}", level=2)
            # First check if this config is in completed_configs
            if config_name in completed_configs:
                predictions_t, labels_t = load_run_predictions(run + 1, config_name, ck_path, dataset_type='train')
                predictions_v, labels_v = load_run_predictions(run + 1, config_name, ck_path, dataset_type='valid')
                vprint(f"\nLoading existing predictions for {config_name}", level=1)
                run_predictions_list_t.append(predictions_t)
                if run_true_labels_t is None:
                    run_true_labels_t = labels_t
                run_predictions_list_v.append(predictions_v)
                if run_true_labels_v is None:
                    run_true_labels_v = labels_v
                continue
            elif os.path.exists(create_checkpoint_filename(config['modalities'], run+1, config_name)):
                # If not in completed_configs but weights exist, we need to regenerate predictions
                vprint(f"\nFound weights but no predictions for {config_name}, regenerating predictions", level=1)
            else:
                vprint(f"\nNo existing data found for {config_name}, starting fresh", level=1)

            selected_modalities = config['modalities']
            # Display proper iteration context
            if not use_legacy_mode and cv_folds > 1:
                vprint(f"\nTraining {config_name} with modalities: {selected_modalities}, fold {run + 1}/{cv_folds}", level=1)
            else:
                vprint(f"\nTraining {config_name} with modalities: {selected_modalities}, run {run + 1}/{num_iterations}", level=1)
            
            training_successful = False
            max_retries = 3
            retry_count = 0
            
            while not training_successful and retry_count < max_retries:
                try:
                    # Filter the master datasets for the selected modalities
                    train_dataset = filter_dataset_modalities(master_train_dataset, selected_modalities)
                    pre_aug_train_dataset = filter_dataset_modalities(pre_aug_dataset, selected_modalities)
                    valid_dataset = filter_dataset_modalities(master_valid_dataset, selected_modalities)
                    # Get a single epoch's worth of data by taking the specified number of steps
                    all_labels = []
                    for batch in pre_aug_train_dataset.take(master_steps_per_epoch):
                        _, labels = batch
                        all_labels.extend(np.argmax(labels.numpy(), axis=1))

                    # Calculate class weights
                    master_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
                    master_class_weights_dict = {i: weight for i, weight in enumerate(master_class_weights)}

                    # # Print dataset information for inspection
                    # print("\nInspecting train_dataset:")
                    # for batch in train_dataset.take(1):
                    #     features, labels = batch
                    #     print("Features:")
                    #     for key, value in features.items():
                    #         print(f"  {key}: shape={value.shape}")
                    #     print(f"Labels: shape={labels.shape}")
                    
                    # print("\nInspecting valid_dataset:")
                    # for batch in valid_dataset.take(1):
                    #     features, labels = batch
                    #     print("Features:")
                    #     for key, value in features.items():
                    #         print(f"  {key}: shape={value.shape}")
                    #     print(f"Labels: shape={labels.shape}")
                        
                    steps_per_epoch = master_steps_per_epoch
                    validation_steps = master_validation_steps
                    # Default values (used when config_name doesn't end with 1, 2, or 3)
                    alpha_value = master_alpha_value  # Proportional class weights
                    class_weights_dict = {i: 1 for i in range(3)}
                    class_weights = [1, 1, 1]
                    if config_name.endswith('1'):
                        alpha_value = master_alpha_value # Proportional class weights (When no mixed_sampling is used)
                        class_weights_dict = {i: 1 for i in range(3)}
                        class_weights = [1, 1, 1]
                    elif config_name.endswith('2'):
                        alpha_value = [1, 1, 1]
                        class_weights_dict = master_class_weights_dict
                        class_weights = master_class_weights
                    elif config_name.endswith('3'):
                        alpha_value = [4, 1, 4]
                        class_weights_dict = {0: 4, 1: 1, 2: 4}
                        class_weights = [4, 1, 4]
                    # alpha_value = [4, 1, 4]  # Equal class weights
                    vprint(f"Alpha values (ordered) [I, P, R]: {[round(a, 3) for a in alpha_value]}", level=2)
                    vprint(f"Class weights: {class_weights_dict} or {class_weights}", level=2)
                    
                    # Create and train model
                    with strategy.scope():
                        weighted_acc = WeightedAccuracy(alpha_values=class_weights)
                        input_shapes = data_manager.get_shapes_for_modalities(selected_modalities)
                        model = create_multimodal_model(input_shapes, selected_modalities, None)

                        # Use loss parameters from config if available, otherwise use defaults
                        ordinal_weight = config.get('ordinal_weight', 0.05)
                        gamma = config.get('gamma', 2.0)
                        alpha = config.get('alpha', class_weights)
                        loss = get_focal_ordinal_loss(num_classes=3, ordinal_weight=ordinal_weight, gamma=gamma, alpha=alpha)
                        macro_f1 = MacroF1Score(num_classes=3)
                        model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss=loss,
                            metrics=['accuracy', weighted_f1_score, weighted_acc, macro_f1, CohenKappa(num_classes=3)]
                        )
                        # Create distributed datasets
                        train_dataset_dis = strategy.experimental_distribute_dataset(train_dataset)
                        valid_dataset_dis = strategy.experimental_distribute_dataset(valid_dataset)
                        callbacks = [
                            EarlyStopping(
                                patience=20,
                                restore_best_weights=True,
                                monitor='val_loss', #'loss',
                                min_delta=0.01,
                                mode='min',
                                verbose=1
                            ),
                            ReduceLROnPlateau(
                                factor=0.50,
                                patience=5,
                                monitor='val_loss', #'loss',
                                min_delta=0.01,
                                min_lr=1e-10,
                                mode='min',
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                create_checkpoint_filename(selected_modalities, run+1, config_name),
                                monitor='val_macro_f1',
                                save_best_only=True,
                                mode='max',
                                save_weights_only=True
                            ),
                            EpochMemoryCallback(strategy), #ACTIVATE THIS WHEN GENERATIVE NOT USED
                            GenerativeAugmentationCallback(gen_manager),
                            NaNMonitorCallback()
                        ]

                        # visualize_dataset(
                        #     train_dataset=train_dataset,
                        #     selected_modalities=selected_modalities,
                        #     save_dir=os.path.join(result_dir, 'train_visualizations_check'),
                        #     max_samples_per_page=20,  # Number of samples per page
                        #     dataset_portion=100,      # Visualize 10% of the dataset
                        #     dpi=75,                 # Image quality
                        #     total_samples=2500
                        # )
                        # visualize_dataset(
                        #     train_dataset=valid_dataset,
                        #     selected_modalities=selected_modalities,
                        #     save_dir=os.path.join(result_dir, 'valid_visualizations_check'),
                        #     max_samples_per_page=20,  # Number of samples per page
                        #     dataset_portion=100,      # Visualize 10% of the dataset
                        #     dpi=75,                 # Image quality
                        #     total_samples=700
                        # )
                        # Add visualization callbacks for first run
                        if run == 0:
                            callbacks.insert(3, BatchVisualizationCallback(
                                dataset=train_dataset,
                                modalities=selected_modalities,
                                freq=1000,
                                max_samples=20,
                                run=run + 1,
                                save_dir=os.path.join(result_dir, 'batch_visualizations_generative')
                            ))
                            callbacks.insert(4, TrainingHistoryCallback(
                                save_dir=os.path.join(result_dir, 'training_plots_generative'),
                                update_freq=5000
                            ))
                            if 'metadata' in selected_modalities:
                                callbacks.insert(5, MetadataConfidenceCallback(
                                    selected_modalities=selected_modalities,
                                    log_dir=os.path.join(result_dir, 'metadata_confidence_logs')
                                ))
                        VISUALIZE_MODALITIES = False
                        if VISUALIZE_MODALITIES:
                                callbacks.append(ModalityContributionCallback(
                                    val_data=valid_dataset,
                                    save_dir=os.path.join(result_dir, 'modality_analysis'),
                                    monitor='val_weighted_accuracy',
                                    mode='max',
                                    run_number=run + 1
                                ))

                        # Train model
                        if os.path.exists(create_checkpoint_filename(selected_modalities, run+1, config_name)):
                            model.load_weights(create_checkpoint_filename(selected_modalities, run+1, config_name))
                            vprint("Loaded existing weights", level=1)
                        else:
                            vprint("No existing pretrained weights found", level=1)
                            vprint(f"Total model trainable weights: {len(model.trainable_weights)}", level=2)
                            history = model.fit(
                                train_dataset_dis,
                                epochs=max_epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=valid_dataset_dis,
                                validation_steps=validation_steps,
                                callbacks=callbacks,
                                verbose=0
                            )
                        
                        model.load_weights(create_checkpoint_filename(selected_modalities, run+1, config_name)) # Load best Validation weights
                        # Evaluate training data
                        y_true_t = []
                        y_pred_t = []
                        probabilities_t = []
                        all_sample_ids_t = []

                        with strategy.scope():
                            for batch in pre_aug_train_dataset.take(steps_per_epoch):
                                batch_inputs, batch_labels = batch
                                with strategy.scope():
                                    batch_pred = model.predict(batch_inputs, verbose=0)
                                y_true_t.extend(np.argmax(batch_labels, axis=1))
                                y_pred_t.extend(np.argmax(batch_pred, axis=1))
                                probabilities_t.extend(batch_pred)
                                all_sample_ids_t.extend(batch_inputs['sample_id'].numpy())
                                
                                del batch_inputs, batch_labels, batch_pred
                                gc.collect()

                        save_run_predictions(run + 1, config_name, np.array(probabilities_t), np.array(y_true_t), ck_path, dataset_type='train')
                        # Store probabilities for gating network
                        run_predictions_list_t.append(np.array(probabilities_t))
                        if run_true_labels_t is None:
                            run_true_labels_t = np.array(y_true_t)

                        # Track misclassifications
                        sample_ids_t = np.array(all_sample_ids_t)
                        track_misclassifications(np.array(y_true_t), np.array(y_pred_t), sample_ids_t, selected_modalities, misclass_path)
                        
                        # Evaluate model
                        y_true_v = []
                        y_pred_v = []
                        probabilities_v = []
                        all_sample_ids_v = []

                        with strategy.scope():
                            for batch in valid_dataset.take(validation_steps):
                                batch_inputs, batch_labels = batch
                                with strategy.scope():
                                    batch_pred = model.predict(batch_inputs, verbose=0)
                                y_true_v.extend(np.argmax(batch_labels, axis=1))
                                y_pred_v.extend(np.argmax(batch_pred, axis=1))
                                probabilities_v.extend(batch_pred)
                                all_sample_ids_v.extend(batch_inputs['sample_id'].numpy())
                                
                                del batch_inputs, batch_labels, batch_pred
                                gc.collect()

                        save_run_predictions(run + 1, config_name, np.array(probabilities_v), np.array(y_true_v), ck_path, dataset_type='valid')
                        # Store probabilities for gating network
                        run_predictions_list_v.append(np.array(probabilities_v))
                        if run_true_labels_v is None:
                            run_true_labels_v = np.array(y_true_v)
                        
                        # Track misclassifications
                        sample_ids_v = np.array(all_sample_ids_v)
                        track_misclassifications(np.array(y_true_v), np.array(y_pred_v), sample_ids_v, selected_modalities, misclass_path)

                        # Calculate metrics
                        accuracy = accuracy_score(y_true_v, y_pred_v)
                        f1_macro = f1_score(y_true_v, y_pred_v, average='macro', zero_division=0)
                        f1_weighted = f1_score(y_true_v, y_pred_v, average='weighted', zero_division=0)
                        f1_classes = f1_score(y_true_v, y_pred_v, average=None, zero_division=0, labels=[0, 1, 2])
                        kappa = cohen_kappa_score(y_true_v, y_pred_v, weights='quadratic')
                        
                        # Store results
                        cm = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1, 2])
                        all_confusion_matrices.append(cm)
                        
                        metrics_dict = {
                            'run': run + 1,
                            'config': config_name,
                            'modalities': selected_modalities,
                            'accuracy': accuracy,
                            'f1_macro': f1_macro,
                            'f1_weighted': f1_weighted,
                            'f1_classes': f1_classes,
                            'kappa': kappa,
                            'y_true': y_true_v,
                            'y_pred': y_pred_v,
                            'probabilities': probabilities_v
                        }
                        save_run_metrics(metrics_dict, run + 1, result_dir)
                        run_metrics.append(metrics_dict)
                        
                        # Print results (level=0 for final metrics to show at all verbosity levels)
                        vprint(f"\nRun {run + 1} Results for {config_name}:", level=0)
                        if get_verbosity() <= 1 or get_verbosity() == 3:
                            print(classification_report(y_true_v, y_pred_v,
                                                    target_names=CLASS_LABELS,
                                                    labels=[0, 1, 2],
                                                    zero_division=0))
                        vprint(f"Cohen's Kappa: {kappa:.4f}", level=0)

                        training_successful = True

                except Exception as e:
                    vprint(f"Error during training (attempt {retry_count + 1}/{max_retries}): {str(e)}", level=0)
                    import traceback
                    vprint(f"Traceback: {traceback.format_exc()}", level=2)
                    clean_up_training_resources()
                    retry_count += 1
                    continue
                
                finally:
                    # Clean up
                    gen_manager.cleanup()
                    gc.collect()

            # Check if training succeeded
            if not training_successful:
                vprint(f"ERROR: Training failed for {config_name} after {max_retries} attempts. Skipping this configuration.", level=0)
                continue

        
        save_run_metrics(run_metrics, run + 1, result_dir)
        save_aggregated_predictions(run + 1, run_predictions_list_t, run_true_labels_t, ck_path, dataset_type='train')
        save_aggregated_predictions(run + 1, run_predictions_list_v, run_true_labels_v, ck_path, dataset_type='valid')
        
        # Train gating network if we have all predictions
        if len(run_predictions_list_v) == len(configs) and len(run_predictions_list_t) == len(configs):
            vprint(f"\nTraining gating network for run {run + 1}...", level=1)
            try:
                # # First validate and correct predictions
                # truncated_predictions_t, run_true_labels_t = correct_and_validate_predictions(
                #     run_predictions_list_t, run_true_labels_t, "train")

                # # Process validation data
                # truncated_predictions_v, run_true_labels_v = correct_and_validate_predictions(
                #     run_predictions_list_v, run_true_labels_v, "valid")

                vprint(f"\nNumber of models: {len(run_predictions_list_t)}", level=2)
            except Exception as e:
                vprint(f"Error in prediction validation: {str(e)}", level=1)
            try:
                # Import here to avoid circular dependency
                from src.main import train_gating_network

                # Convert labels back to class indices for gating network
                gating_labels_t = np.argmax(run_true_labels_t, axis=1) if len(run_true_labels_t.shape) > 1 else run_true_labels_t
                gating_labels_v = np.argmax(run_true_labels_v, axis=1) if len(run_true_labels_v.shape) > 1 else run_true_labels_v
                combined_predictions, gating_labels = train_gating_network(
                    run_predictions_list_t,
                    run_predictions_list_v,
                    gating_labels_t,
                    gating_labels_v,
                    run + 1,
                    find_optimal=False,
                    min_models=2,
                    max_tries=200
                )
                
                # Calculate and store gating network results for this run
                final_predictions = np.argmax(combined_predictions, axis=1)
                gating_metrics = {
                    'run': run + 1,
                    'accuracy': accuracy_score(gating_labels, final_predictions),
                    'f1_macro': f1_score(gating_labels, final_predictions, average='macro'),
                    'f1_weighted': f1_score(gating_labels, final_predictions, average='weighted'),
                    'kappa': cohen_kappa_score(gating_labels, final_predictions, weights='quadratic'),
                    'confusion_matrix': confusion_matrix(gating_labels, final_predictions)
                }
                
                vprint(f"\nGating Network Results for Run {run + 1}:", level=1)
                vprint(f"Accuracy: {gating_metrics['accuracy']:.4f}", level=1)
                vprint(f"F1 Macro: {gating_metrics['f1_macro']:.4f}", level=1)
                vprint(f"Kappa: {gating_metrics['kappa']:.4f}", level=1)

                all_gating_results.append(gating_metrics)
                # Save individual run results
                save_run_results(gating_metrics, run + 1, result_dir)

            except Exception as e:
                vprint(f"Error in gating network training for run {run + 1}: {str(e)}", level=1)
                gating_metrics = None
            
            all_runs_metrics.extend(run_metrics)
            # Clean up after the run
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                clear_gpu_memory()
            except Exception as e:
                vprint(f"Error clearing memory stats: {str(e)}", level=2)

    # Save aggregated results
    save_aggregated_results(all_runs_metrics, configs, result_dir)
    save_gating_results(all_gating_results, result_dir)
    
    return all_runs_metrics, all_confusion_matrices, all_histories
def correct_and_validate_predictions(predictions_list, true_labels, dataset_type="train"):
    """
    Correct and validate model predictions.
    
    Args:
        predictions_list: List of model predictions
        true_labels: True labels
        dataset_type: String indicating "train" or "valid" for logging
    
    Returns:
        tuple: (truncated_predictions, truncated_labels)
    """
    corrected_predictions = []
    
    # Validate and correct predictions
    for i, preds in enumerate(predictions_list):
        if not isinstance(preds, np.ndarray) or preds is None:
            vprint(f"Warning: Invalid {dataset_type} predictions from model {i}. Skipping...", level=1)
            continue

        preds = np.array(preds, dtype=np.float32)
        if len(preds.shape) != 2:
            vprint(f"Warning: Invalid {dataset_type} shape {preds.shape} from model {i}. Skipping...", level=1)
            continue

        # Check and correct predictions
        if preds.shape[1] != 3:
            vprint(f"Warning: Model {i} {dataset_type} predictions have {preds.shape[1]} classes instead of 3", level=1)
            corrected = np.zeros((preds.shape[0], 3), dtype=np.float32)
            for c in range(min(preds.shape[1], 3)):
                corrected[:, c] = preds[:, c]
            for c in range(preds.shape[1], 3):
                corrected[:, c] = 1e-7
            row_sums = corrected.sum(axis=1, keepdims=True)
            corrected = corrected / row_sums
            corrected_predictions.append(corrected)
        else:
            corrected_predictions.append(preds)
    
    if not corrected_predictions:
        raise ValueError(f"No valid {dataset_type} predictions found after correction")
    
    # Find minimum length
    min_length = min(p.shape[0] for p in corrected_predictions)
    # print(f"Minimum length of {dataset_type} predictions: {min_length}")
    
    # Truncate predictions and labels
    truncated_predictions = [p[:min_length] for p in corrected_predictions]
    truncated_labels = true_labels[:min_length]
    
    # Convert labels to one-hot if needed
    if len(truncated_labels.shape) == 1:
        truncated_labels = tf.keras.utils.to_categorical(truncated_labels, num_classes=3)
    
    # print(f"Shape of {dataset_type} predictions from first model: {truncated_predictions[0].shape}")
    # print(f"Shape of {dataset_type} labels: {truncated_labels.shape}")
    
    # Verify all predictions have correct shape
    for i, preds in enumerate(truncated_predictions):
        if preds.shape != (min_length, 3):
            raise ValueError(f"Model {i} {dataset_type} predictions have incorrect shape: {preds.shape}")
    
    return truncated_predictions, truncated_labels
def save_run_results(metrics, run_number, result_dir):
    """Save gating network results for an individual run."""
    csv_filename = os.path.join(csv_path, f'gating_network_run_{run_number}_results.csv')
    
    # Format metrics for CSV
    results = [{
        'run': int(metrics['run']),
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'kappa': float(metrics['kappa'])
    }]
    
    # Save confusion matrix separately
    cm_filename = os.path.join(csv_path, f'gating_network_run_{run_number}_confusion_matrix.csv')
    np.savetxt(cm_filename, metrics['confusion_matrix'], delimiter=',', fmt='%d')
    
    # Save metrics to CSV
    fieldnames = list(results[0].keys())
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
def save_run_metrics(run_metrics, run_number, result_dir):
    # Handle single metrics dict vs list of metrics
    is_single_metric = not isinstance(run_metrics, list)
    
    # Format the metrics
    if is_single_metric:
        csv_filename = os.path.join(csv_path, f'modality_results_run_{run_number}.csv')
        formatted_result = {
            'config': run_metrics['config'],
            'modalities': '+'.join(run_metrics['modalities']),
            'accuracy': run_metrics['accuracy'],
            'f1_macro': run_metrics['f1_macro'],
            'f1_weighted': run_metrics['f1_weighted'],
            'I_f1': run_metrics['f1_classes'][0],
            'P_f1': run_metrics['f1_classes'][1],
            'R_f1': run_metrics['f1_classes'][2],
            'kappa': run_metrics['kappa']
        }
        fieldnames = list(formatted_result.keys())
        
        # Save to CSV with append mode
        write_mode = 'w' if not os.path.exists(csv_filename) else 'a'
        with open(csv_filename, write_mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_mode == 'w':
                writer.writeheader()
            writer.writerow(formatted_result)
            
    else:
        csv_filename = os.path.join(csv_path, f'modality_results_run_{run_number}_list.csv')
        # Format list of metrics
        formatted_results = []
        for m in run_metrics:
            formatted_results.append({
                'config': m['config'],
                'modalities': '+'.join(m['modalities']),
                'accuracy': m['accuracy'],
                'f1_macro': m['f1_macro'],
                'f1_weighted': m['f1_weighted'],
                'I_f1': m['f1_classes'][0],
                'P_f1': m['f1_classes'][1],
                'R_f1': m['f1_classes'][2],
                'kappa': m['kappa']
            })
        
        # Save list to CSV (overwrite mode)
        if formatted_results:
            fieldnames = list(formatted_results[0].keys())
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(formatted_results)
def save_gating_results(all_gating_results, result_dir):
    """Save aggregated gating network results to CSV."""
    if not all_gating_results:
        return
        
    # Calculate average metrics
    results = [{
        'accuracy_mean': np.mean([r['accuracy'] for r in all_gating_results]),
        'accuracy_std': np.std([r['accuracy'] for r in all_gating_results]),
        'f1_macro_mean': np.mean([r['f1_macro'] for r in all_gating_results]),
        'f1_macro_std': np.std([r['f1_macro'] for r in all_gating_results]),
        'f1_weighted_mean': np.mean([r['f1_weighted'] for r in all_gating_results]),
        'f1_weighted_std': np.std([r['f1_weighted'] for r in all_gating_results]),
        'kappa_mean': np.mean([r['kappa'] for r in all_gating_results]),
        'kappa_std': np.std([r['kappa'] for r in all_gating_results])
    }]
    
    # Save to file
    csv_filename = os.path.join(csv_path, 'gating_network_averaged_results.csv')
    fieldnames = list(results[0].keys())
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_aggregated_results(all_metrics, configs, result_dir):
    """Save aggregated results to CSV file."""
    results = []
    
    # Group metrics by configuration
    for config_name in configs.keys():
        config_metrics = [m for m in all_metrics if m['config'] == config_name]
        
        if not config_metrics:
            continue
            
        avg_accuracy = np.mean([m['accuracy'] for m in config_metrics])
        std_accuracy = np.std([m['accuracy'] for m in config_metrics])
        avg_f1_macro = np.mean([m['f1_macro'] for m in config_metrics])
        std_f1_macro = np.std([m['f1_macro'] for m in config_metrics])
        avg_f1_weighted = np.mean([m['f1_weighted'] for m in config_metrics])
        std_f1_weighted = np.std([m['f1_weighted'] for m in config_metrics])
        avg_kappa = np.mean([m['kappa'] for m in config_metrics])
        std_kappa = np.std([m['kappa'] for m in config_metrics])
        
        # Calculate average F1-scores for each class
        f1_classes_list = [m['f1_classes'] for m in config_metrics]
        avg_f1_classes = np.mean(f1_classes_list, axis=0)
        
        results.append({
            'Modalities': '+'.join(configs[config_name]['modalities']),
            'Accuracy (Mean)': avg_accuracy,
            'Accuracy (Std)': std_accuracy,
            'Macro Avg F1-score (Mean)': avg_f1_macro,
            'Macro Avg F1-score (Std)': std_f1_macro,
            'Weighted Avg F1-score (Mean)': avg_f1_weighted,
            'Weighted Avg F1-score (Std)': std_f1_weighted,
            'I F1-score (Mean)': avg_f1_classes[0],
            'P F1-score (Mean)': avg_f1_classes[1],
            'R F1-score (Mean)': avg_f1_classes[2],
            "Cohen's Kappa (Mean)": avg_kappa,
            "Cohen's Kappa (Std)": std_kappa
        })
    
    # Save to CSV
    csv_filename = os.path.join(csv_path, 'modality_results_averaged.csv')
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    vprint(f"Results saved to {csv_filename}", level=1)
def save_run_predictions(run_number, config_name, predictions, true_labels, ck_path, dataset_type='valid'):
    """Save predictions and true labels for a specific run and config."""
    pred_file = os.path.join(ck_path, f'pred_run{run_number}_{config_name}_{dataset_type}.npy')
    labels_file = os.path.join(ck_path, f'true_label_run{run_number}_{config_name}_{dataset_type}.npy')
    np.save(pred_file, predictions)
    np.save(labels_file, true_labels)

def load_run_predictions(run_number, config_name, ck_path, dataset_type='valid'):
    """Load predictions and true labels for a specific run and config."""
    pred_file = os.path.join(ck_path, f'pred_run{run_number}_{config_name}_{dataset_type}.npy')
    labels_file = os.path.join(ck_path, f'true_label_run{run_number}_{config_name}_{dataset_type}.npy')
    
    if os.path.exists(pred_file) and os.path.exists(labels_file):
        return np.load(pred_file), np.load(labels_file)
    return None, None

def get_completed_configs_for_run(run_number, config_names, ck_path, dataset_type='valid'):
    """Get list of configs that have completed predictions for a specific run."""
    completed = []
    for config_name in config_names:
        pred_file = os.path.join(ck_path, f'pred_run{run_number}_{config_name}_{dataset_type}.npy')
        labels_file = os.path.join(ck_path, f'true_label_run{run_number}_{config_name}_{dataset_type}.npy')
        if os.path.exists(pred_file) and os.path.exists(labels_file):
            completed.append(config_name)
    return completed

def load_aggregated_predictions(run_number, ck_path, dataset_type='valid'):
    """Load all predictions for a run if they exist."""
    aggregated_preds_file = os.path.join(ck_path, f'sum_pred_run{run_number}_{dataset_type}.npy')
    aggregated_labels_file = os.path.join(ck_path, f'sum_label_run{run_number}_{dataset_type}.npy')

    if os.path.exists(aggregated_preds_file) and os.path.exists(aggregated_labels_file):
        return np.load(aggregated_preds_file), np.load(aggregated_labels_file)
    return None, None


def save_combination_predictions(run_number, combination_name, predictions, labels, ck_path, dataset_type='valid'):
    """Save predictions for a specific modality combination."""
    # Sanitize combination name for filename
    safe_name = combination_name.replace('+', '_').replace(' ', '_')
    preds_file = os.path.join(ck_path, f'combo_pred_{safe_name}_run{run_number}_{dataset_type}.npy')
    labels_file = os.path.join(ck_path, f'combo_label_{safe_name}_run{run_number}_{dataset_type}.npy')

    np.save(preds_file, predictions)
    np.save(labels_file, labels)
    vprint(f"Saved {combination_name} predictions to {preds_file}", level=1)


def load_combination_predictions(run_number, combination_name, ck_path, dataset_type='valid'):
    """Load predictions for a specific modality combination."""
    # Sanitize combination name for filename
    safe_name = combination_name.replace('+', '_').replace(' ', '_')
    preds_file = os.path.join(ck_path, f'combo_pred_{safe_name}_run{run_number}_{dataset_type}.npy')
    labels_file = os.path.join(ck_path, f'combo_label_{safe_name}_run{run_number}_{dataset_type}.npy')

    if os.path.exists(preds_file) and os.path.exists(labels_file):
        return np.load(preds_file), np.load(labels_file)
    return None, None

def save_aggregated_predictions(run_number, predictions_list, true_labels, ck_path, dataset_type='valid'):
    """Save aggregated predictions with validation and correction."""
    aggregated_preds_file = os.path.join(ck_path, f'sum_pred_run{run_number}_{dataset_type}.npy')
    aggregated_labels_file = os.path.join(ck_path, f'sum_label_run{run_number}_{dataset_type}.npy')
    
    # Validate and correct predictions
    corrected_predictions = []
    for i, preds in enumerate(predictions_list):
        if preds is None:
            vprint(f"Warning: Predictions from model {i} are None. Skipping...", level=1)
            continue

        preds = np.array(preds)
        if len(preds.shape) != 2:
            vprint(f"Warning: Invalid shape {preds.shape} from model {i}. Skipping...", level=1)
            continue

        # Check if we have predictions for all three classes
        if preds.shape[1] != 3:
            vprint(f"Warning: Model {i} predictions have {preds.shape[1]} classes instead of 3", level=1)
            # Create corrected array with proper shape
            corrected_preds = np.zeros((preds.shape[0], 3))

            # Copy existing predictions
            for c in range(min(preds.shape[1], 3)):
                corrected_preds[:, c] = preds[:, c]

            # For missing classes, set very low confidence
            for c in range(preds.shape[1], 3):
                # Use small non-zero value to avoid numerical issues
                corrected_preds[:, c] = 1e-7

            # Renormalize probabilities to sum to 1
            row_sums = corrected_preds.sum(axis=1, keepdims=True)
            corrected_preds = corrected_preds / row_sums

            vprint(f"Corrected predictions shape: {corrected_preds.shape}", level=2)
            corrected_predictions.append(corrected_preds)
        else:
            corrected_predictions.append(preds)

    # Final validation
    if not corrected_predictions:
        vprint(f"Warning: No predictions to save for run {run_number} ({dataset_type}). Skipping...", level=0)
        return  # Skip saving instead of raising error

    shapes = [p.shape for p in corrected_predictions]
    if len(set(shapes)) > 1:
        vprint(f"Warning: Inconsistent shapes after correction: {shapes}", level=1)
        # Find minimum number of samples across all predictions
        min_samples = min(s[0] for s in shapes)
        # Truncate all predictions to minimum length
        corrected_predictions = [p[:min_samples] for p in corrected_predictions]
        if true_labels is not None:
            true_labels = true_labels[:min_samples]
            
    try:
        np.save(aggregated_preds_file, corrected_predictions)
        if true_labels is not None:
            np.save(aggregated_labels_file, true_labels)
    except Exception as e:
        vprint(f"Error saving predictions: {str(e)}", level=1)
        raise

def is_run_complete(run_number, ck_path):
    """Check if a run has completed gating metrics."""
    gating_metrics_file = os.path.join(ck_path, f'gating_network_run_{run_number}_results.csv')
    return os.path.exists(gating_metrics_file)
def main_with_specialized_evaluation(data_percentage=100, train_patient_percentage=0.8, n_runs=3):
    """
    Run specialized evaluation with the new cross-validation structure.
    """

    configs = {
        'a32': {#0
            'modalities': ['metadata'],
        },
        'b32': {#1
            'modalities': ['depth_rgb'],
        },
        'c32': {#2
            'modalities': ['depth_map'],
        },
        'd32': {#3
            'modalities': ['thermal_map'],
        },
        # 'e32': {#4*
        #     'modalities': ['metadata','depth_rgb'],
        # },
        # 'f32': {#5*
        #     'modalities': ['metadata','depth_map'],
        # },
        # 'g32': {#6*
        #     'modalities': ['metadata','thermal_map'],
        # },
        # 'h32': {#7
        #     'modalities': ['depth_rgb','depth_map'],
        # },
        # 'i32': {#8
        #     'modalities': ['depth_rgb','thermal_map'],
        # },
        # # 'i32': {#8 TEMPT
        # #     'modalities': ['depth_rgb','thermal_rgb'],
        # # },
        'j32': {#9*
            'modalities': ['depth_map','thermal_map'],
        },
        'k32': {#10
            'modalities': ['metadata','depth_rgb','depth_map'],
        },
        'z32': {#11
            'modalities': ['metadata','depth_rgb','thermal_map'],
        },
        'm32': {#12*
            'modalities': ['metadata','depth_map','thermal_map'],
        },
        'n32': {#13
            'modalities': ['depth_rgb','depth_map','thermal_map'],
        },
        # 'p3222': {#14
        #     'modalities': ['metadata','depth_rgb','depth_map','thermal_map'],
        # },
    }
    
    # Clear any existing cache files
    clear_cache_files()
    
    # Prepare initial dataset
    vprint("Preparing initial dataset...", level=1)
    data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file,
                         list(set([mod for config in configs.values()
                                 for mod in config['modalities']])))

    # Filter frequent misclassifications
    vprint("Filtering frequent misclassifications...", level=1)
    data = filter_frequent_misclassifications(
        data, result_dir,
        thresholds={'I': 3, 'P': 2, 'R': 3}
    )

    if data_percentage < 100:
        data = data.sample(frac=data_percentage / 100, random_state=42).reset_index(drop=True)

    # Run cross-validation
    vprint("\nStarting cross-validation...", level=1)
    metrics, confusion_matrices, histories = cross_validation_manual_split(
        data, configs, train_patient_percentage, n_runs
    )
    # After all runs are complete
    # average_attention_values(result_dir, num_runs=n_runs)
    
    # # Train gating network if needed
    # if os.path.exists(os.path.join(result_dir, 'predictions_list.npy')):
    #     print("\nTraining gating network...")
    #     predictions_list = np.load(os.path.join(result_dir, 'predictions_list.npy'), allow_pickle=True)
    #     true_labels_onehot = np.load(os.path.join(result_dir, 'true_labels_onehot.npy'), allow_pickle=True)
    #     true_labels = np.argmax(true_labels_onehot, axis=1)
        
    #     combined_predictions = train_gating_network(
    #         predictions_list, true_labels, run_number=1,
    #         find_optimal=True, min_models=2, max_tries=150
    #     )
        
    #     final_predictions = np.argmax(combined_predictions, axis=1)
    #     final_metrics = calculate_and_save_metrics(true_labels, final_predictions, result_dir, configs)
    
    return metrics

def filter_dataset_modalities(dataset, selected_modalities):
    """
    Filter a dataset to only include the selected modalities.
    
    Args:
        dataset: A tf.data.Dataset that contains all modalities
        selected_modalities: List of modalities to keep
    
    Returns:
        A new dataset containing only the selected modalities
    """
    def filter_features(features, labels):
        filtered_features = {}
        # Always include sample_id
        filtered_features['sample_id'] = features['sample_id']
        
        # Include only selected modalities
        for modality in selected_modalities:
            modality_key = f'{modality}_input'
            if modality_key in features:
                filtered_features[modality_key] = features[modality_key]
            else:
                raise KeyError(f"Modality {modality} not found in dataset")
        
        return filtered_features, labels
    
    # return dataset.map(filter_features, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.map(filter_features, num_parallel_calls=2)

def clear_cache_files():
    """Clear any existing cache files."""
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
                    vprint(f"Removed cache file: {cache_file}", level=2)
                except Exception as e:
                    vprint(f"Warning: Could not remove cache file {cache_file}: {str(e)}", level=2)
        except Exception as e:
            vprint(f"Warning: Error while processing pattern {pattern}: {str(e)}", level=2)
