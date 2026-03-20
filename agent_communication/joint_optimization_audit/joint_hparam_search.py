#!/usr/bin/env python3
"""
Joint Hyperparameter Optimization — Bayesian Search (23 dimensions)

Comprehensive Bayesian optimization that jointly searches ALL parameters
(image backbones, heads, RF, fusion, training dynamics) for the
metadata + depth_rgb + thermal_map fusion pipeline.

Uses scikit-optimize gp_minimize over a 23-dimensional search space:
  - Image backbone architectures (depth_rgb + thermal_map)
  - Image head architectures and dropout
  - Image pre-training learning rates and augmentation (mixup)
  - Image size
  - RF hyperparameters (n_estimators, max_depth, feature_selection_k, min_samples_leaf)
  - Fusion strategy (feature_concat / prob_concat)
  - Fusion head architecture and projection dimension
  - Fusion training dynamics (stage1_lr, epochs, label_smoothing, focal_gamma)
  - Stage 2 fine-tuning (epochs, unfreeze_pct)

Key features:
  - Pre-training weight cache keyed by full image config fingerprint
  - Dataset cache per RF config fingerprint
  - Resume support via checkpoint JSON + warm-start gp_minimize (x0/y0)
  - Every trial tracks meta_kappa (RF baseline), fusion_kappa, kappa_delta
  - After Bayesian trials: Top-3 5-fold validation with paired t-test
  - Output: CSV results, best config JSON, log file

Prerequisites:
  - scikit-optimize: pip install scikit-optimize
  - All src/ modules on PYTHONPATH (handled automatically)

Usage:
  # Fresh start (backs up old results):
  python agent_communication/joint_optimization_audit/joint_hparam_search.py --fresh

  # Resume from checkpoint:
  python agent_communication/joint_optimization_audit/joint_hparam_search.py

  # Dry run (print search space and exit):
  python agent_communication/joint_optimization_audit/joint_hparam_search.py --dry-run

  # Custom trial count:
  python agent_communication/joint_optimization_audit/joint_hparam_search.py --n-trials 50
"""

import os
import sys
import gc
import time
import json
import math
import random
import csv
import hashlib
import shutil
import datetime
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

AUDIT_DIR = os.path.join(PROJECT_ROOT, 'agent_communication', 'joint_optimization_audit')
WEIGHTS_DIR = os.path.join(AUDIT_DIR, 'weights')
LOGS_DIR = os.path.join(AUDIT_DIR, 'logs')
RESULTS_CSV = os.path.join(AUDIT_DIR, 'joint_search_results.csv')
CHECKPOINT_PATH = os.path.join(AUDIT_DIR, 'joint_search_checkpoint.json')
BEST_CONFIG_PATH = os.path.join(AUDIT_DIR, 'joint_best_config.json')

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Set environment before TF import
os.environ["OMP_NUM_THREADS"] = "2"
os.environ['TF_NUM_INTEROP_THREADS'] = "2"
os.environ['TF_NUM_INTRAOP_THREADS'] = "4"
os.environ['TF_DETERMINISTIC_OPS'] = "1"
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix

# Multi-GPU setup — must happen before any model/dataset creation
from src.utils.gpu_config import setup_device_strategy
_strategy, _selected_gpus = setup_device_strategy(mode='multi')

from src.utils.config import RANDOM_SEED
from src.models.losses import get_focal_ordinal_loss

# In-memory pre-training weight caches
# key: fingerprint string -> {layer_name: weights}
_pretrain_weight_cache = {}

# In-memory dataset cache fingerprints (to detect when RF config changes)
_dataset_cache_fingerprints = {}


# ─────────────────────────────────────────────────────────────────────
# Weight save/load helpers
# ─────────────────────────────────────────────────────────────────────

def save_model_weights(model, path_without_ext):
    """Save model weights, trying .weights.h5 first then .h5 for Keras compat."""
    os.makedirs(os.path.dirname(path_without_ext), exist_ok=True)
    try:
        fpath = path_without_ext + '.weights.h5'
        model.save_weights(fpath)
        print(f"  [WEIGHTS] Saved: {fpath}")
        return fpath
    except Exception:
        fpath = path_without_ext + '.h5'
        model.save_weights(fpath)
        print(f"  [WEIGHTS] Saved: {fpath}")
        return fpath


def load_model_weights(model, path_without_ext):
    """Load model weights, trying .weights.h5 first then .h5."""
    for ext in ['.weights.h5', '.h5']:
        fpath = path_without_ext + ext
        if os.path.exists(fpath):
            model.load_weights(fpath)
            print(f"  [WEIGHTS] Loaded: {fpath}")
            return fpath
    return None


# ─────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class JointSearchConfig:
    """Single joint optimization trial configuration."""
    name: str = "trial_0"

    # Image modality backbones
    depth_rgb_backbone: str = 'EfficientNetB0'
    thermal_map_backbone: str = 'EfficientNetB2'

    # Image modality heads
    depth_rgb_head_units: list = field(default_factory=lambda: [128])
    thermal_map_head_units: list = field(default_factory=lambda: [128, 32])

    # Head dropout (shared across both image modalities)
    head_dropout: float = 0.3

    # Image pre-training LRs
    depth_rgb_pretrain_lr: float = 0.001
    thermal_map_pretrain_lr: float = 0.003

    # Mixup per modality
    depth_rgb_use_mixup: bool = False
    thermal_map_use_mixup: bool = True

    # Image size (shared)
    image_size: int = 256

    # RF hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 8
    rf_feature_selection_k: int = 40
    rf_min_samples_leaf: int = 5

    # Fusion architecture
    fusion_strategy: str = 'feature_concat'
    image_projection_dim: int = 0
    fusion_head_units: list = field(default_factory=list)

    # Fusion training — Stage 1
    stage1_lr: float = 5e-4
    stage1_epochs: int = 300
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0

    # Fusion training — Stage 2
    stage2_epochs: int = 0
    stage2_unfreeze_pct: float = 0.2

    # Fixed parameters (not searched)
    batch_size: int = 64
    early_stop_patience: int = 20
    reduce_lr_patience: int = 10
    alpha_sum: float = 3.0
    head_use_bn: bool = True
    head_l2: float = 0.0
    n_folds: int = 5
    fold: int = 0


# ─────────────────────────────────────────────────────────────────────
# Fingerprint helpers
# ─────────────────────────────────────────────────────────────────────

def image_pretrain_fingerprint(modality, cfg):
    """Compute fingerprint for a single image modality's pre-training config.

    This fingerprint uniquely identifies the pre-trained weights produced by
    a given combination of backbone, head, dropout, LR, mixup, image_size,
    and fold. Two trials with the same fingerprint can share cached weights.
    """
    if modality == 'depth_rgb':
        key_parts = [
            modality,
            cfg.depth_rgb_backbone,
            str(cfg.depth_rgb_head_units),
            f"{cfg.head_dropout:.3f}",
            f"{cfg.depth_rgb_pretrain_lr:.6f}",
            str(int(cfg.depth_rgb_use_mixup)),
            str(cfg.image_size),
            str(cfg.fold),
        ]
    elif modality == 'thermal_map':
        key_parts = [
            modality,
            cfg.thermal_map_backbone,
            str(cfg.thermal_map_head_units),
            f"{cfg.head_dropout:.3f}",
            f"{cfg.thermal_map_pretrain_lr:.6f}",
            str(int(cfg.thermal_map_use_mixup)),
            str(cfg.image_size),
            str(cfg.fold),
        ]
    else:
        raise ValueError(f"Unknown modality: {modality}")
    raw = '|'.join(key_parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def rf_config_fingerprint(cfg):
    """Compute fingerprint for RF-related config that affects cached datasets."""
    key_parts = [
        str(cfg.rf_n_estimators),
        str(cfg.rf_max_depth),
        str(cfg.rf_feature_selection_k),
        str(cfg.rf_min_samples_leaf),
        str(cfg.image_size),
    ]
    raw = '|'.join(key_parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────
# Metrics (Keras)
# ─────────────────────────────────────────────────────────────────────

class CohenKappaMetric(tf.keras.metrics.Metric):
    """Quadratic-weighted Cohen's Kappa as a Keras metric."""
    def __init__(self, num_classes=3, name='cohen_kappa', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_matrix = self.add_weight(
            name='conf_matrix', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.argmax(y_pred, axis=-1)
        y_true_cls = tf.argmax(y_true, axis=-1)
        confusion = tf.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                mask = tf.logical_and(tf.equal(y_true_cls, i), tf.equal(y_pred_cls, j))
                count = tf.reduce_sum(tf.cast(mask, tf.float32))
                confusion = tf.tensor_scatter_nd_add(confusion, [[i, j]], [count])
        self.conf_matrix.assign_add(confusion)

    def result(self):
        n = tf.reduce_sum(self.conf_matrix)
        k = self.num_classes
        indices = tf.cast(tf.range(k), tf.float32)
        w = tf.square(tf.expand_dims(indices, 1) - tf.expand_dims(indices, 0))
        w = w / tf.cast(tf.square(k - 1), tf.float32)
        row_sums = tf.reduce_sum(self.conf_matrix, axis=1)
        col_sums = tf.reduce_sum(self.conf_matrix, axis=0)
        expected = tf.einsum('i,j->ij', row_sums, col_sums) / n
        obs = tf.reduce_sum(w * self.conf_matrix) / n
        exp = tf.reduce_sum(w * expected) / n
        return 1.0 - obs / (exp + tf.keras.backend.epsilon())

    def reset_state(self):
        self.conf_matrix.assign(tf.zeros_like(self.conf_matrix))


# ─────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────

def make_focal_loss(gamma=2.0, alpha=None, label_smoothing=0.0):
    """Focal loss with per-class alpha weights and optional label smoothing."""
    if alpha is None:
        alpha = [1.0, 1.0, 1.0]
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / 3.0
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha_tensor * tf.math.pow(1.0 - y_pred, gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss


# ─────────────────────────────────────────────────────────────────────
# LR schedules
# ─────────────────────────────────────────────────────────────────────

class CosineAnnealingSchedule(tf.keras.callbacks.Callback):
    """Cosine annealing LR with optional warmup."""
    def __init__(self, initial_lr, total_epochs, warmup_epochs=0, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


def apply_mixup(features, labels, alpha=0.2):
    """Mixup augmentation — blends pairs of samples."""
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


# ─────────────────────────────────────────────────────────────────────
# Model building — standalone pre-training
# ─────────────────────────────────────────────────────────────────────

def build_pretrained_backbone(input_shape, backbone_name, modality_name):
    """Build pretrained backbone — supports all 6 architectures in the search space.

    Uses single-model instantiation (weights='imagenet' directly) for EfficientNet.
    For DenseNet/ResNet/MobileNet, applies preprocessing Lambda inside the model.
    """
    from tensorflow.keras.layers import Lambda as LambdaLayer

    inp = Input(shape=input_shape, name=f'{modality_name}_input')

    backbone_map = {
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'EfficientNetB2': tf.keras.applications.EfficientNetB2,
        'EfficientNetB3': tf.keras.applications.EfficientNetB3,
        'DenseNet121': tf.keras.applications.DenseNet121,
        'ResNet50V2': tf.keras.applications.ResNet50V2,
        'MobileNetV3Large': tf.keras.applications.MobileNetV3Large,
    }

    preprocess_map = {
        'EfficientNetB0': None,
        'EfficientNetB2': None,
        'EfficientNetB3': None,
        'DenseNet121': tf.keras.applications.densenet.preprocess_input,
        'ResNet50V2': tf.keras.applications.resnet_v2.preprocess_input,
        'MobileNetV3Large': tf.keras.applications.mobilenet_v3.preprocess_input,
    }

    BackboneClass = backbone_map[backbone_name]
    preprocess_fn = preprocess_map[backbone_name]

    # Single-model instantiation — identical to standalone audit scripts
    base_model = BackboneClass(weights='imagenet', include_top=False, pooling='avg')

    if preprocess_fn is not None:
        x = LambdaLayer(lambda img: preprocess_fn(img),
                        name=f'{modality_name}_preprocess')(inp)
        x = base_model(x)
    else:
        x = base_model(inp)

    return inp, x, base_model


def build_pretrain_model(modality_name, backbone_name, head_units, head_dropout,
                         head_use_bn=True, head_l2=0.0, image_size=256):
    """Build standalone pre-training model: Input -> backbone -> head -> softmax(3).

    Matches the exact architecture used in standalone pipeline audits so that
    pre-training kappa values are reproducible.
    """
    input_shape = (image_size, image_size, 3)
    inp, features, base_model = build_pretrained_backbone(
        input_shape, backbone_name, modality_name)

    x = features
    regularizer = tf.keras.regularizers.l2(head_l2) if head_l2 > 0 else None
    for i, units in enumerate(head_units):
        x = Dense(units, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizer,
                  name=f'head_dense_{i}')(x)
        if head_use_bn:
            x = BatchNormalization(name=f'head_bn_{i}')(x)
        x = Dropout(head_dropout, name=f'head_drop_{i}')(x)

    output = Dense(3, activation='softmax', name='output', dtype='float32')(x)
    model = Model(inputs=inp, outputs=output)

    return model, base_model


# ─────────────────────────────────────────────────────────────────────
# Data pipeline — reuses project's existing data loading
# ─────────────────────────────────────────────────────────────────────

def load_data(image_modalities=None):
    """Load the dataset for fusion (metadata + depth_rgb + thermal_map)."""
    if image_modalities is None:
        image_modalities = ["depth_rgb", "thermal_map"]

    from src.utils.config import get_project_paths, get_data_paths
    from src.data.image_processing import prepare_dataset

    directory, result_dir, root = get_project_paths()
    data_paths = get_data_paths(root)

    csv_file = data_paths['csv_file']
    depth_bb_file = data_paths['bb_depth_csv']
    thermal_bb_file = data_paths['bb_thermal_csv']

    selected_modalities = ['metadata'] + image_modalities
    data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities)

    modalities_str = ' + '.join(selected_modalities)
    print(f"Loaded {len(data)} samples for fusion: {modalities_str}")
    return data, directory, result_dir


def prepare_fusion_datasets(data, cfg, fold_idx=0):
    """Prepare train/valid datasets for a single fold of fusion training.

    RF hyperparameters from cfg are patched into production_config before
    calling prepare_cached_datasets (which trains the RF and bakes its
    predictions into the cached dataset).
    """
    from src.data.dataset_utils import create_patient_folds, prepare_cached_datasets
    from src.data.generative_augmentation_v3 import AugmentationConfig

    data = data.copy()
    patient_folds = create_patient_folds(data, n_folds=cfg.n_folds, random_state=42)
    train_patients, valid_patients = patient_folds[fold_idx]

    aug_config = AugmentationConfig()
    aug_config.generative_settings['output_size']['width'] = cfg.image_size
    aug_config.generative_settings['output_size']['height'] = cfg.image_size

    selected_modalities = ['metadata', 'depth_rgb', 'thermal_map']

    # Override RF hyperparameters in production_config
    import src.utils.production_config as _pcfg
    orig_rf_n_estimators = _pcfg.RF_N_ESTIMATORS
    orig_rf_max_depth = _pcfg.RF_MAX_DEPTH
    orig_rf_min_samples_leaf = _pcfg.RF_MIN_SAMPLES_LEAF
    orig_rf_feature_selection_k = _pcfg.RF_FEATURE_SELECTION_K

    _pcfg.RF_N_ESTIMATORS = cfg.rf_n_estimators
    _pcfg.RF_MAX_DEPTH = cfg.rf_max_depth
    _pcfg.RF_MIN_SAMPLES_LEAF = cfg.rf_min_samples_leaf
    _pcfg.RF_FEATURE_SELECTION_K = cfg.rf_feature_selection_k

    # Cache dir keyed by RF config fingerprint
    from src.utils.config import get_project_paths
    _, search_result_dir, _ = get_project_paths()
    rf_fp = rf_config_fingerprint(cfg)
    cache_base = os.path.join(search_result_dir, 'search_cache')
    cache_prefix = 'joint_fusion_meta_depth_rgb_thermal_map_'
    search_cache_dir = os.path.join(cache_base, cache_prefix + rf_fp)

    # Auto-clean old caches with different RF fingerprints
    import glob as _glob
    for old_cache in _glob.glob(os.path.join(cache_base, cache_prefix + '*')):
        if old_cache != search_cache_dir and os.path.isdir(old_cache):
            print(f"  Cleaning old fusion cache: {os.path.basename(old_cache)}")
            shutil.rmtree(old_cache)

    train_ds, pre_aug_ds, valid_ds, steps_per_epoch, val_steps, alpha_values = prepare_cached_datasets(
        data,
        selected_modalities,
        batch_size=cfg.batch_size,
        gen_manager=None,
        aug_config=aug_config,
        run=fold_idx,
        image_size=cfg.image_size,
        train_patients=train_patients,
        valid_patients=valid_patients,
        cache_dir=search_cache_dir,
    )

    # Restore overridden globals
    _pcfg.RF_N_ESTIMATORS = orig_rf_n_estimators
    _pcfg.RF_MAX_DEPTH = orig_rf_max_depth
    _pcfg.RF_MIN_SAMPLES_LEAF = orig_rf_min_samples_leaf
    _pcfg.RF_FEATURE_SELECTION_K = orig_rf_feature_selection_k

    def remove_sample_id(features, labels):
        return {k: v for k, v in features.items() if k != 'sample_id'}, labels

    train_ds_clean = train_ds.map(remove_sample_id, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds_clean = valid_ds.map(remove_sample_id, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds_clean, valid_ds_clean, steps_per_epoch, val_steps, alpha_values


def prepare_pretrain_dataset(data, modality, cfg, fold_idx=0):
    """Prepare train/valid datasets for a single image modality's pre-training.

    Uses standalone-style single-modality dataset (no metadata, no RF).
    """
    from src.data.dataset_utils import create_patient_folds, prepare_cached_datasets
    from src.data.generative_augmentation_v3 import AugmentationConfig
    from src.utils.config import get_project_paths
    import src.utils.production_config as _pcfg
    import src.data.dataset_utils as _ds_utils

    data = data.copy()
    patient_folds = create_patient_folds(data, n_folds=cfg.n_folds, random_state=42)
    train_patients, valid_patients = patient_folds[fold_idx]

    aug_config = AugmentationConfig()
    aug_config.generative_settings['output_size']['width'] = cfg.image_size
    aug_config.generative_settings['output_size']['height'] = cfg.image_size

    # Override backbone globals for data pipeline normalization
    # Standalone scripts force 'EfficientNetB0' to keep [0,255] pipeline for
    # ALL pretrained backbones (EfficientNet has built-in Rescaling, others get
    # preprocess_input inside the model). We do the same here.
    _orig_rgb_bb = _pcfg.RGB_BACKBONE
    _orig_map_bb = _pcfg.MAP_BACKBONE
    _orig_ds_rgb_bb = _ds_utils.RGB_BACKBONE
    _orig_ds_map_bb = _ds_utils.MAP_BACKBONE

    is_rgb = (modality == 'depth_rgb')
    if is_rgb:
        _pcfg.RGB_BACKBONE = 'EfficientNetB0'
        _ds_utils.RGB_BACKBONE = 'EfficientNetB0'
    else:
        _pcfg.MAP_BACKBONE = 'EfficientNetB0'
        _ds_utils.MAP_BACKBONE = 'EfficientNetB0'

    _, pt_result_dir, _ = get_project_paths()
    pt_cache_dir = os.path.join(pt_result_dir, 'search_cache',
                                f'joint_{modality}_{cfg.image_size}_pretrain')

    pt_train_ds, _, pt_valid_ds, pt_steps, pt_val_steps, pt_alpha = prepare_cached_datasets(
        data,
        [modality],
        batch_size=cfg.batch_size,
        gen_manager=None,
        aug_config=aug_config,
        run=fold_idx,
        image_size=cfg.image_size,
        train_patients=train_patients,
        valid_patients=valid_patients,
        cache_dir=pt_cache_dir,
    )

    # Restore overridden globals
    _pcfg.RGB_BACKBONE = _orig_rgb_bb
    _pcfg.MAP_BACKBONE = _orig_map_bb
    _ds_utils.RGB_BACKBONE = _orig_ds_rgb_bb
    _ds_utils.MAP_BACKBONE = _orig_ds_map_bb

    # Extract just the image tensor (remove sample_id, metadata, etc.)
    _pt_input_key = f'{modality}_input'
    def _extract_image(features, labels):
        return features[_pt_input_key], labels
    pt_train_ds = pt_train_ds.map(_extract_image, num_parallel_calls=tf.data.AUTOTUNE)
    pt_valid_ds = pt_valid_ds.map(_extract_image, num_parallel_calls=tf.data.AUTOTUNE)

    return pt_train_ds, pt_valid_ds, pt_steps, pt_val_steps, pt_alpha


# ─────────────────────────────────────────────────────────────────────
# Pre-training logic
# ─────────────────────────────────────────────────────────────────────

def pretrain_image_modality(modality, cfg, data, fold_idx=0):
    """Pre-train a single image modality and return backbone weights + kappa.

    Checks in-memory cache and on-disk weight cache before training.
    Returns: (backbone_weights_dict, pretrain_kappa)
    """
    fp = image_pretrain_fingerprint(modality, cfg)

    # Check in-memory cache
    if fp in _pretrain_weight_cache:
        print(f"  PRE-TRAINING: {modality} — using in-memory cached weights (fp={fp[:8]})")
        return _pretrain_weight_cache[fp], _pretrain_weight_cache.get(fp + '_kappa', 0.0)

    # Check on-disk cache
    mod_weights_dir = os.path.join(WEIGHTS_DIR, modality)
    os.makedirs(mod_weights_dir, exist_ok=True)
    disk_path = os.path.join(mod_weights_dir, fp)
    disk_kappa_path = disk_path + '_kappa.json'

    if modality == 'depth_rgb':
        backbone_name = cfg.depth_rgb_backbone
        head_units = cfg.depth_rgb_head_units
        pretrain_lr = cfg.depth_rgb_pretrain_lr
        use_mixup = cfg.depth_rgb_use_mixup
    else:
        backbone_name = cfg.thermal_map_backbone
        head_units = cfg.thermal_map_head_units
        pretrain_lr = cfg.thermal_map_pretrain_lr
        use_mixup = cfg.thermal_map_use_mixup

    # Try loading from disk
    for ext in ['.weights.h5', '.h5']:
        if os.path.exists(disk_path + ext):
            print(f"  PRE-TRAINING: {modality} — loading from disk cache (fp={fp[:8]})")
            with _strategy.scope():
                pt_model, pt_base = build_pretrain_model(
                    modality, backbone_name, head_units,
                    cfg.head_dropout, cfg.head_use_bn, cfg.head_l2, cfg.image_size)
            load_model_weights(pt_model, disk_path)
            backbone_weights = {}
            for layer in pt_model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 10:
                    backbone_weights['__backbone__'] = layer.get_weights()
                    break
            _pretrain_weight_cache[fp] = backbone_weights
            pt_kappa = 0.0
            if os.path.exists(disk_kappa_path):
                with open(disk_kappa_path, 'r') as f:
                    pt_kappa = json.load(f).get('kappa', 0.0)
            _pretrain_weight_cache[fp + '_kappa'] = pt_kappa
            del pt_model, pt_base
            gc.collect()
            tf.keras.backend.clear_session()
            return backbone_weights, pt_kappa

    # Need to train from scratch
    print(f"\n  PRE-TRAINING: {modality} (backbone={backbone_name}, head={head_units}, "
          f"lr={pretrain_lr}, mixup={use_mixup}, img_size={cfg.image_size})")

    pt_train_ds, pt_valid_ds, pt_steps, pt_val_steps, pt_alpha = \
        prepare_pretrain_dataset(data, modality, cfg, fold_idx)

    # Apply mixup if configured
    if use_mixup:
        def _pt_mixup(img, labels):
            batch_size = tf.shape(img)[0]
            u1 = tf.random.uniform([], 0.0, 1.0)
            u2 = tf.random.uniform([], 0.0, 1.0)
            lam = tf.maximum(u1, u2)
            indices = tf.random.shuffle(tf.range(batch_size))
            mixed_img = lam * img + (1.0 - lam) * tf.gather(img, indices)
            mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
            return mixed_img, mixed_labels
        pt_train_ds = pt_train_ds.map(_pt_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    # Compute alpha values
    pt_alpha_arr = np.array(pt_alpha)
    pt_alpha_arr = pt_alpha_arr / pt_alpha_arr.sum() * cfg.alpha_sum
    pt_alpha_list = pt_alpha_arr.tolist()

    # Set deterministic seeds
    _pt_seed = 42 + fold_idx * (fold_idx + 3)
    tf.random.set_seed(_pt_seed)
    np.random.seed(_pt_seed)
    random.seed(_pt_seed)

    strategy = _strategy

    with strategy.scope():
        pt_model, pt_base = build_pretrain_model(
            modality, backbone_name, head_units,
            cfg.head_dropout, cfg.head_use_bn, cfg.head_l2, cfg.image_size)

        # Freeze backbone — only train head + classifier
        pt_base.trainable = False

        pt_trainable = len(pt_model.trainable_weights)
        pt_total = len(pt_model.weights)
        print(f"    Trainable weights: {pt_trainable}/{pt_total} (backbone frozen)")

        pt_loss = make_focal_loss(gamma=2.0, alpha=pt_alpha_list, label_smoothing=0.0)
        pt_opt = Adam(learning_rate=pretrain_lr, clipnorm=1.0)
        pt_model.compile(
            optimizer=pt_opt,
            loss=pt_loss,
            metrics=['accuracy', CohenKappaMetric(num_classes=3)]
        )

    pt_train_dis = strategy.experimental_distribute_dataset(pt_train_ds)
    pt_valid_dis = strategy.experimental_distribute_dataset(pt_valid_ds)

    # Stage 1: Head-only training (backbone frozen)
    pt_callbacks = [
        EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_cohen_kappa',
            min_delta=0.001,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.50,
            patience=7,
            monitor='val_cohen_kappa',
            min_delta=0.001,
            min_lr=1e-7,
            mode='max',
        ),
    ]

    pt_history = pt_model.fit(
        pt_train_dis,
        epochs=50,
        steps_per_epoch=pt_steps,
        validation_data=pt_valid_dis,
        validation_steps=pt_val_steps,
        callbacks=pt_callbacks,
        verbose=0
    )

    pt_kappas = pt_history.history.get('val_cohen_kappa', [])
    pt_best = max(pt_kappas) if pt_kappas else 0.0
    pt_best_ep = int(np.argmax(pt_kappas)) + 1 if pt_kappas else 0
    print(f"    Stage 1 best: val_kappa={pt_best:.4f} at epoch {pt_best_ep}/{len(pt_kappas)}")

    # Save S1 weights before optional Stage 2
    s1_weights = pt_model.get_weights()

    # Stage 2: Fine-tuning (partial backbone unfreeze) — only if modality traditionally uses it
    # depth_rgb: 30 finetune epochs, unfreeze 20%
    # thermal_map: 50 finetune epochs, unfreeze 50%, freeze BN
    if modality == 'depth_rgb':
        ft_epochs = 30
        ft_lr = 1e-5
        unfreeze_pct = 0.2
        freeze_bn = False
    else:
        ft_epochs = 50
        ft_lr = 1e-5
        unfreeze_pct = 0.5
        freeze_bn = True

    if ft_epochs > 0:
        pt_base.trainable = True
        n_layers = len(pt_base.layers)
        freeze_until = int(n_layers * (1.0 - unfreeze_pct))
        for sub_layer in pt_base.layers[:freeze_until]:
            sub_layer.trainable = False
        if freeze_bn:
            for sub_layer in pt_base.layers:
                if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                    sub_layer.trainable = False
        unfrozen_count = n_layers - freeze_until
        bn_note = ", BN frozen" if freeze_bn else ""
        print(f"    Stage 2: unfreezing top {unfreeze_pct*100:.0f}% "
              f"({unfrozen_count}/{n_layers} layers{bn_note})")

        with strategy.scope():
            s2_opt = Adam(learning_rate=ft_lr, clipnorm=1.0)
            pt_model.compile(
                optimizer=s2_opt,
                loss=pt_loss,
                metrics=['accuracy', CohenKappaMetric(num_classes=3)]
            )

        s2_callbacks = [
            EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_cohen_kappa',
                min_delta=0.001,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=0.50,
                patience=5,
                monitor='val_cohen_kappa',
                min_delta=0.001,
                min_lr=1e-8,
                mode='max',
            ),
        ]

        s2_history = pt_model.fit(
            pt_train_dis,
            epochs=ft_epochs,
            steps_per_epoch=pt_steps,
            validation_data=pt_valid_dis,
            validation_steps=pt_val_steps,
            callbacks=s2_callbacks,
            verbose=0
        )

        s2_kappas = s2_history.history.get('val_cohen_kappa', [])
        s2_best = max(s2_kappas) if s2_kappas else 0.0
        s2_best_ep = int(np.argmax(s2_kappas)) + 1 if s2_kappas else 0
        print(f"    Stage 2 best: val_kappa={s2_best:.4f} at epoch {s2_best_ep}/{len(s2_kappas)}")

        if s2_best < pt_best:
            print(f"    Stage 2 did NOT improve (s2={s2_best:.4f} < s1={pt_best:.4f}). Restoring S1 weights.")
            pt_model.set_weights(s1_weights)

    # Post-eval: sklearn kappa
    pt_y_true = []
    pt_y_pred = []
    for batch in pt_valid_ds:
        batch_inputs, batch_labels = batch
        batch_pred = pt_model.predict(batch_inputs, verbose=0)
        pt_y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        pt_y_pred.extend(np.argmax(batch_pred, axis=1))
    pt_y_true = np.array(pt_y_true)
    pt_y_pred = np.array(pt_y_pred)
    pt_post_kappa = cohen_kappa_score(pt_y_true, pt_y_pred, weights='quadratic')
    print(f"    Pre-train POST-EVAL: kappa={pt_post_kappa:.4f} (n={len(pt_y_true)})")

    # Extract backbone weights for transfer
    backbone_weights = {}
    for layer in pt_model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone_weights['__backbone__'] = layer.get_weights()
            break

    # Save to disk cache
    save_model_weights(pt_model, disk_path)
    with open(disk_kappa_path, 'w') as f:
        json.dump({'kappa': float(pt_post_kappa), 'fingerprint': fp}, f)

    # Save to in-memory cache
    _pretrain_weight_cache[fp] = backbone_weights
    _pretrain_weight_cache[fp + '_kappa'] = pt_post_kappa

    del pt_model, pt_base, pt_train_ds, pt_valid_ds, pt_train_dis, pt_valid_dis
    gc.collect()
    tf.keras.backend.clear_session()

    return backbone_weights, pt_post_kappa


# ─────────────────────────────────────────────────────────────────────
# Fusion model building
# ─────────────────────────────────────────────────────────────────────

def build_fusion_model(cfg, image_input_shapes, metadata_input_shape):
    """Build a fusion model using project's standard builders.

    Patches production_config.MODALITY_CONFIGS and related globals
    to match the current trial's config, builds the model via the
    project's create_image_branch / create_metadata_branch / fusion logic,
    then restores all patched globals.
    """
    import src.utils.production_config as _pcfg
    import src.models.builders as _builders
    from tensorflow.keras.layers import concatenate

    # Save originals for restoration
    orig_modality_configs = deepcopy(_pcfg.MODALITY_CONFIGS)
    orig_get_modality_config = _pcfg.get_modality_config
    orig_builders_get_modality_config = _builders.get_modality_config
    orig_fusion_strategy = _pcfg.FUSION_STRATEGY
    orig_fusion_proj_dim = _pcfg.FUSION_IMAGE_PROJECTION_DIM
    orig_rgb_backbone = _pcfg.RGB_BACKBONE
    orig_map_backbone = _pcfg.MAP_BACKBONE
    import src.data.dataset_utils as _ds_utils
    orig_ds_rgb_bb = _ds_utils.RGB_BACKBONE
    orig_ds_map_bb = _ds_utils.MAP_BACKBONE

    try:
        # Patch MODALITY_CONFIGS for this trial
        _pcfg.MODALITY_CONFIGS['depth_rgb']['backbone'] = cfg.depth_rgb_backbone
        _pcfg.MODALITY_CONFIGS['depth_rgb']['head_units'] = cfg.depth_rgb_head_units
        _pcfg.MODALITY_CONFIGS['depth_rgb']['head_l2'] = cfg.head_l2

        _pcfg.MODALITY_CONFIGS['thermal_map']['backbone'] = cfg.thermal_map_backbone
        _pcfg.MODALITY_CONFIGS['thermal_map']['head_units'] = cfg.thermal_map_head_units
        _pcfg.MODALITY_CONFIGS['thermal_map']['head_l2'] = cfg.head_l2

        # Patch backbone globals for data pipeline
        if cfg.depth_rgb_backbone.startswith('EfficientNet'):
            _pcfg.RGB_BACKBONE = cfg.depth_rgb_backbone
            _ds_utils.RGB_BACKBONE = cfg.depth_rgb_backbone
        else:
            # Non-EfficientNet: keep EfficientNetB0 for pipeline normalization
            # (preprocessing is applied inside the model)
            _pcfg.RGB_BACKBONE = 'EfficientNetB0'
            _ds_utils.RGB_BACKBONE = 'EfficientNetB0'

        if cfg.thermal_map_backbone.startswith('EfficientNet'):
            _pcfg.MAP_BACKBONE = cfg.thermal_map_backbone
            _ds_utils.MAP_BACKBONE = cfg.thermal_map_backbone
        else:
            _pcfg.MAP_BACKBONE = 'EfficientNetB0'
            _ds_utils.MAP_BACKBONE = 'EfficientNetB0'

        _pcfg.FUSION_STRATEGY = cfg.fusion_strategy
        _pcfg.FUSION_IMAGE_PROJECTION_DIM = cfg.image_projection_dim

        # Also invalidate the builder's model cache so it re-creates with new config
        if hasattr(_builders.create_efficientnet_branch, '_model_cache'):
            _builders.create_efficientnet_branch._model_cache = {}
        if hasattr(_builders.create_generic_backbone_branch, '_model_cache'):
            _builders.create_generic_backbone_branch._model_cache = {}

        # Build the model inputs manually rather than calling create_multimodal_model
        # to have full control over the fusion head
        inputs = {}

        # Metadata branch
        metadata_input, rf_probs = _builders.create_metadata_branch(metadata_input_shape, 0)
        inputs['metadata_input'] = metadata_input

        # Image branches
        image_feature_branches = {}
        for modality in ['depth_rgb', 'thermal_map']:
            image_input, features = _builders.create_image_branch(
                image_input_shapes[modality], modality)
            inputs[f'{modality}_input'] = image_input
            image_feature_branches[modality] = features

        # Merge image features
        image_features_list = [image_feature_branches[m] for m in ['depth_rgb', 'thermal_map']]
        image_features_concat = concatenate(image_features_list, name='concat_image_features')

        # Optional image feature projection
        proj_dim = cfg.image_projection_dim
        if proj_dim > 0:
            orig_dim = image_features_concat.shape[-1]
            print(f"  Image feature projection: {orig_dim} -> {proj_dim}")
            image_features_projected = Dense(
                proj_dim, activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name='fusion_image_projection')(image_features_concat)
            image_features_projected = BatchNormalization(
                name='fusion_image_proj_bn')(image_features_projected)
            image_features_projected = Dropout(
                0.3, name='fusion_image_proj_dropout')(image_features_projected)
        else:
            image_features_projected = image_features_concat

        # Fusion strategy dispatch
        strategy = cfg.fusion_strategy

        if strategy == 'feature_concat':
            fused = concatenate([rf_probs, image_features_projected], name='fusion_concat')
            x = fused
            for i, units in enumerate(cfg.fusion_head_units):
                x = Dense(units, activation='relu', kernel_initializer='he_normal',
                          name=f'fusion_dense_{i}')(x)
                x = BatchNormalization(name=f'fusion_bn_{i}')(x)
                x = Dropout(cfg.head_dropout, name=f'fusion_drop_{i}')(x)
            output = Dense(3, activation='softmax', name='output', dtype='float32',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

        elif strategy == 'prob_concat':
            image_probs = Dense(3, activation='softmax', name='image_classifier',
                                dtype='float32')(image_features_concat)
            fused = concatenate([rf_probs, image_probs], name='fusion_concat')
            x = fused
            for i, units in enumerate(cfg.fusion_head_units):
                x = Dense(units, activation='relu', kernel_initializer='he_normal',
                          name=f'fusion_dense_{i}')(x)
                x = BatchNormalization(name=f'fusion_bn_{i}')(x)
                x = Dropout(cfg.head_dropout, name=f'fusion_drop_{i}')(x)
            output = Dense(3, activation='softmax', name='output', dtype='float32')(x)

        else:
            raise ValueError(f"Unknown fusion_strategy: {strategy}")

        model = Model(inputs=inputs, outputs=output)

    finally:
        # Restore all patched globals
        _pcfg.MODALITY_CONFIGS = orig_modality_configs
        _pcfg.get_modality_config = orig_get_modality_config
        _builders.get_modality_config = orig_builders_get_modality_config
        _pcfg.FUSION_STRATEGY = orig_fusion_strategy
        _pcfg.FUSION_IMAGE_PROJECTION_DIM = orig_fusion_proj_dim
        _pcfg.RGB_BACKBONE = orig_rgb_backbone
        _pcfg.MAP_BACKBONE = orig_map_backbone
        _ds_utils.RGB_BACKBONE = orig_ds_rgb_bb
        _ds_utils.MAP_BACKBONE = orig_ds_map_bb
        if hasattr(_builders.create_efficientnet_branch, '_model_cache'):
            _builders.create_efficientnet_branch._model_cache = {}
        if hasattr(_builders.create_generic_backbone_branch, '_model_cache'):
            _builders.create_generic_backbone_branch._model_cache = {}

    return model


# ─────────────────────────────────────────────────────────────────────
# Train a single trial end-to-end
# ─────────────────────────────────────────────────────────────────────

def train_single_trial(cfg, data, fold_idx=0):
    """Train a single joint optimization trial and return metrics dict.

    End-to-end pipeline:
    1. Pre-train both image modalities (or load from cache)
    2. Prepare fusion dataset (with RF configured per trial)
    3. Evaluate RF baseline
    4. Build fusion model, transfer backbone weights
    5. Train fusion Stage 1 (frozen backbone)
    6. Optional Stage 2 (partial unfreeze)
    7. Post-eval
    """
    print(f"\n{'='*80}")
    print(f"TRIAL: {cfg.name}")
    print(f"  depth_rgb: backbone={cfg.depth_rgb_backbone}, head={cfg.depth_rgb_head_units}, "
          f"lr={cfg.depth_rgb_pretrain_lr}, mixup={cfg.depth_rgb_use_mixup}")
    print(f"  thermal_map: backbone={cfg.thermal_map_backbone}, head={cfg.thermal_map_head_units}, "
          f"lr={cfg.thermal_map_pretrain_lr}, mixup={cfg.thermal_map_use_mixup}")
    print(f"  image_size={cfg.image_size}, head_dropout={cfg.head_dropout}")
    print(f"  rf: n_est={cfg.rf_n_estimators}, max_depth={cfg.rf_max_depth}, "
          f"feat_k={cfg.rf_feature_selection_k}, leaf={cfg.rf_min_samples_leaf}")
    print(f"  fusion: strategy={cfg.fusion_strategy}, proj_dim={cfg.image_projection_dim}, "
          f"head={cfg.fusion_head_units}")
    print(f"  training: stage1_lr={cfg.stage1_lr}, stage1_epochs={cfg.stage1_epochs}, "
          f"label_smoothing={cfg.label_smoothing}, focal_gamma={cfg.focal_gamma}")
    print(f"  stage2: epochs={cfg.stage2_epochs}, unfreeze_pct={cfg.stage2_unfreeze_pct}")
    print(f"  fold={fold_idx}")
    print(f"{'='*80}")

    start_time = time.time()

    # ── Step 1: Pre-train each image modality ──
    pretrained_weights = {}
    pretrain_kappas = {}
    for modality in ['depth_rgb', 'thermal_map']:
        weights, kappa = pretrain_image_modality(modality, cfg, data, fold_idx)
        pretrained_weights[modality] = weights
        pretrain_kappas[modality] = kappa

    # ── Step 2: Prepare fusion datasets ──
    train_ds, valid_ds, steps_per_epoch, val_steps, alpha_values = \
        prepare_fusion_datasets(data, cfg, fold_idx)

    if cfg.alpha_sum == 0:
        alpha_list = [1.0, 1.0, 1.0]
    else:
        alpha_arr = np.array(alpha_values)
        alpha_arr = alpha_arr / alpha_arr.sum() * cfg.alpha_sum
        alpha_list = alpha_arr.tolist()

    # Infer input shapes from dataset
    for batch in train_ds.take(1):
        batch_inputs, batch_labels = batch
        image_input_shapes = {
            mod: batch_inputs[f'{mod}_input'].shape[1:]
            for mod in ['depth_rgb', 'thermal_map']
        }
        metadata_input_shape = batch_inputs['metadata_input'].shape[1:]
        break

    # ── Step 3: Evaluate metadata (RF) baseline ──
    meta_y_true = []
    meta_y_pred = []
    for batch in valid_ds:
        batch_inputs, batch_labels = batch
        rf_probs = batch_inputs['metadata_input'].numpy()
        meta_y_pred.extend(np.argmax(rf_probs, axis=1))
        meta_y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
    meta_y_true = np.array(meta_y_true)
    meta_y_pred = np.array(meta_y_pred)
    meta_kappa = cohen_kappa_score(meta_y_true, meta_y_pred, weights='quadratic')
    meta_acc = accuracy_score(meta_y_true, meta_y_pred)
    meta_f1 = f1_score(meta_y_true, meta_y_pred, average='macro', zero_division=0)
    n_val_samples = len(meta_y_true)
    print(f"\n  METADATA (RF) PRE-EVAL: kappa={meta_kappa:.4f}, acc={meta_acc:.4f}, "
          f"f1={meta_f1:.4f} (n={n_val_samples})")

    # ── Step 4: Build fusion model, transfer backbone weights ──
    strategy = _strategy

    with strategy.scope():
        model = build_fusion_model(cfg, image_input_shapes, metadata_input_shape)

        # Transfer pre-trained backbone weights
        total_transferred = 0
        fusion_backbones = {}
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                fusion_backbones[layer.name] = layer

        for modality in ['depth_rgb', 'thermal_map']:
            if modality not in pretrained_weights or '__backbone__' not in pretrained_weights[modality]:
                continue
            matched_layer = None
            for name, layer in fusion_backbones.items():
                if name.startswith(modality):
                    matched_layer = layer
                    break
            if matched_layer is None:
                print(f"  Warning: no fusion backbone found for {modality} "
                      f"(available: {list(fusion_backbones.keys())})")
                continue
            try:
                matched_layer.set_weights(pretrained_weights[modality]['__backbone__'])
                total_transferred += 1
                print(f"  Transferred backbone weights for {modality} -> {matched_layer.name}")
            except Exception as e:
                print(f"  Warning: failed to transfer {modality} backbone ({matched_layer.name}): {e}")
        print(f"  Transferred {total_transferred} pre-trained backbones to fusion model")

        # Freeze backbone — only train fusion layers
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                layer.trainable = False

        trainable_count = len(model.trainable_weights)
        total_count = len(model.weights)
        print(f"  Trainable weights: {trainable_count}/{total_count} (backbone frozen)")

        loss_fn = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.0,
                                          gamma=cfg.focal_gamma, alpha=alpha_list,
                                          label_smoothing=cfg.label_smoothing)

        model.compile(
            optimizer=Adam(learning_rate=cfg.stage1_lr, clipnorm=1.0),
            loss=loss_fn,
            metrics=['accuracy', CohenKappaMetric(num_classes=3)],
            jit_compile=True
        )

    train_ds_dis = strategy.experimental_distribute_dataset(train_ds)
    valid_ds_dis = strategy.experimental_distribute_dataset(valid_ds)

    callbacks = [
        EarlyStopping(
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            monitor='val_cohen_kappa',
            min_delta=0.001,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.50,
            patience=cfg.reduce_lr_patience,
            monitor='val_cohen_kappa',
            min_delta=0.001,
            min_lr=1e-7,
            mode='max',
        ),
    ]

    # ── Step 5: Fusion training Stage 1 ──
    history = model.fit(
        train_ds_dis,
        epochs=cfg.stage1_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds_dis,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0
    )

    s1_kappas = history.history.get('val_cohen_kappa', [])
    best_s1_kappa = max(s1_kappas) if s1_kappas else 0.0
    best_s1_epoch = int(np.argmax(s1_kappas)) + 1 if s1_kappas else 0
    print(f"  Fusion S1 best: val_kappa={best_s1_kappa:.4f} at epoch {best_s1_epoch}/{len(s1_kappas)}")

    # ── Step 6: Optional Stage 2 ──
    fusion_s1_weights = model.get_weights()
    best_s2_kappa = -1.0
    if cfg.stage2_epochs > 0:
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                layer.trainable = True
                n_layers = len(layer.layers)
                freeze_until = int(n_layers * (1.0 - cfg.stage2_unfreeze_pct))
                for sub_layer in layer.layers[:freeze_until]:
                    sub_layer.trainable = False
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = False
                unfrozen = n_layers - freeze_until
                print(f"  Stage 2: unfreezing top {cfg.stage2_unfreeze_pct*100:.0f}% "
                      f"({unfrozen}/{n_layers} layers, BN frozen)")

        with strategy.scope():
            s2_lr = cfg.stage1_lr * 0.1  # Stage 2 uses 10x lower LR
            model.compile(
                optimizer=Adam(learning_rate=s2_lr, clipnorm=1.0),
                loss=loss_fn,
                metrics=['accuracy', CohenKappaMetric(num_classes=3)],
                jit_compile=True
            )

        s2_callbacks = [
            EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_cohen_kappa',
                min_delta=0.001,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=0.50,
                patience=5,
                monitor='val_cohen_kappa',
                min_delta=0.001,
                min_lr=1e-8,
                mode='max',
            ),
        ]

        s2_history = model.fit(
            train_ds_dis,
            epochs=cfg.stage2_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_ds_dis,
            validation_steps=val_steps,
            callbacks=s2_callbacks,
            verbose=0
        )

        s2_kappas = s2_history.history.get('val_cohen_kappa', [])
        best_s2_kappa = max(s2_kappas) if s2_kappas else 0.0
        s2_best_ep = int(np.argmax(s2_kappas)) + 1 if s2_kappas else 0
        print(f"  Fusion S2 best: val_kappa={best_s2_kappa:.4f} at epoch {s2_best_ep}/{len(s2_kappas)}")

        if best_s2_kappa < best_s1_kappa:
            print(f"  Fusion S2 did NOT improve ({best_s2_kappa:.4f} < {best_s1_kappa:.4f}). Restoring S1 weights.")
            model.set_weights(fusion_s1_weights)

    # ── Step 7: Post-training evaluation ──
    y_true_v = []
    y_pred_v = []
    for batch in valid_ds:
        batch_inputs, batch_labels = batch
        batch_pred = model.predict(batch_inputs, verbose=0)
        y_true_v.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred_v.extend(np.argmax(batch_pred, axis=1))
    y_true_v = np.array(y_true_v)
    y_pred_v = np.array(y_pred_v)

    kappa = cohen_kappa_score(y_true_v, y_pred_v, weights='quadratic')
    accuracy = accuracy_score(y_true_v, y_pred_v)
    f1_macro = f1_score(y_true_v, y_pred_v, average='macro', zero_division=0)

    elapsed = time.time() - start_time
    kappa_delta = kappa - meta_kappa

    print(f"\n  POST-EVAL: kappa={kappa:.4f}, acc={accuracy:.4f}, f1={f1_macro:.4f}")
    print(f"  kappa_delta (vs RF): {kappa_delta:+.4f}")
    print(f"  Time: {elapsed:.0f}s")

    result = {
        'name': cfg.name,
        'trial_num': int(cfg.name.split('_')[1]) if cfg.name.startswith('trial_') else 0,
        'depth_rgb_backbone': cfg.depth_rgb_backbone,
        'depth_rgb_head_units': str(cfg.depth_rgb_head_units),
        'depth_rgb_head_dropout': cfg.head_dropout,
        'depth_rgb_pretrain_lr': cfg.depth_rgb_pretrain_lr,
        'depth_rgb_use_mixup': cfg.depth_rgb_use_mixup,
        'thermal_map_backbone': cfg.thermal_map_backbone,
        'thermal_map_head_units': str(cfg.thermal_map_head_units),
        'thermal_map_head_dropout': cfg.head_dropout,
        'thermal_map_pretrain_lr': cfg.thermal_map_pretrain_lr,
        'thermal_map_use_mixup': cfg.thermal_map_use_mixup,
        'image_size': cfg.image_size,
        'rf_n_estimators': cfg.rf_n_estimators,
        'rf_max_depth': cfg.rf_max_depth if cfg.rf_max_depth is not None else 'None',
        'rf_feature_selection_k': cfg.rf_feature_selection_k,
        'rf_min_samples_leaf': cfg.rf_min_samples_leaf,
        'fusion_strategy': cfg.fusion_strategy,
        'image_projection_dim': cfg.image_projection_dim,
        'fusion_head_units': str(cfg.fusion_head_units),
        'stage1_lr': cfg.stage1_lr,
        'stage1_epochs': cfg.stage1_epochs,
        'label_smoothing': cfg.label_smoothing,
        'focal_gamma': cfg.focal_gamma,
        'stage2_epochs': cfg.stage2_epochs,
        'stage2_unfreeze_pct': cfg.stage2_unfreeze_pct,
        'fold': fold_idx,
        'depth_rgb_pretrain_kappa': pretrain_kappas.get('depth_rgb', 0.0),
        'thermal_map_pretrain_kappa': pretrain_kappas.get('thermal_map', 0.0),
        'meta_kappa': meta_kappa,
        'meta_acc': meta_acc,
        'meta_f1': meta_f1,
        'best_s1_kappa': best_s1_kappa,
        'best_s1_epoch': best_s1_epoch,
        'best_s2_kappa': best_s2_kappa,
        'post_eval_kappa': kappa,
        'post_eval_acc': accuracy,
        'post_eval_f1': f1_macro,
        'kappa_delta': kappa_delta,
        'trainable_weights': trainable_count,
        'total_weights': total_count,
        'elapsed_seconds': elapsed,
        'n_val_samples': n_val_samples,
    }

    del model, train_ds, valid_ds, train_ds_dis, valid_ds_dis
    gc.collect()
    tf.keras.backend.clear_session()

    return result


# ─────────────────────────────────────────────────────────────────────
# Search space definition
# ─────────────────────────────────────────────────────────────────────

def get_search_space():
    """Define the 23-dimensional Bayesian optimization search space."""
    from skopt.space import Real, Categorical

    search_space = [
        Categorical(['EfficientNetB0', 'EfficientNetB2', 'EfficientNetB3',
                      'DenseNet121', 'ResNet50V2', 'MobileNetV3Large'],
                     name='depth_rgb_backbone'),
        Categorical(['[32]', '[64]', '[128]', '[256]', '[128,32]', '[128,64]'],
                     name='depth_rgb_head'),
        Categorical(['EfficientNetB0', 'EfficientNetB2', 'EfficientNetB3',
                      'DenseNet121', 'ResNet50V2', 'MobileNetV3Large'],
                     name='thermal_map_backbone'),
        Categorical(['[32]', '[64]', '[128]', '[256]', '[128,32]', '[128,64]'],
                     name='thermal_map_head'),
        Categorical([32, 64, 128, 224, 256], name='image_size'),
        Real(1e-4, 5e-3, prior='log-uniform', name='depth_rgb_pretrain_lr'),
        Real(1e-4, 5e-3, prior='log-uniform', name='thermal_map_pretrain_lr'),
        Categorical([0, 1], name='depth_rgb_use_mixup'),
        Categorical([0, 1], name='thermal_map_use_mixup'),
        Real(0.1, 0.6, name='head_dropout'),
        Categorical([10, 50, 100, 200, 300, 500], name='rf_n_estimators'),
        Categorical([3, 5, 8, 10, 15, 20, None], name='rf_max_depth'),
        Categorical([10, 20, 30, 40, 60, 80], name='rf_feature_selection_k'),
        Categorical([1, 3, 5, 10], name='rf_min_samples_leaf'),
        Categorical(['feature_concat', 'prob_concat'], name='fusion_strategy'),
        Categorical([0, 4, 8, 16, 32, 64], name='image_projection_dim'),
        Categorical(['[]', '[16]', '[32]', '[64]', '[128]', '[64,32]'],
                     name='fusion_head'),
        Real(5e-5, 5e-3, prior='log-uniform', name='stage1_lr'),
        Categorical([100, 200, 300, 500], name='stage1_epochs'),
        Real(0.0, 0.2, name='label_smoothing'),
        Categorical([1.0, 2.0, 3.0], name='focal_gamma'),
        Categorical([0, 30, 50, 100], name='stage2_epochs'),
        Real(0.05, 0.5, name='stage2_unfreeze_pct'),
    ]
    return search_space


def parse_head_units(s):
    """Parse string head units like '[128,32]' -> [128, 32], '[]' -> []."""
    s = s.strip()
    if s in ('[]', ''):
        return []
    s = s.strip('[]')
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def decode_params_to_config(params, trial_num=0, fold=0):
    """Convert gp_minimize parameter vector to JointSearchConfig.

    The params list is ordered to match get_search_space().
    """
    (depth_rgb_backbone, depth_rgb_head, thermal_map_backbone, thermal_map_head,
     image_size, depth_rgb_pretrain_lr, thermal_map_pretrain_lr,
     depth_rgb_use_mixup, thermal_map_use_mixup, head_dropout,
     rf_n_estimators, rf_max_depth, rf_feature_selection_k, rf_min_samples_leaf,
     fusion_strategy, image_projection_dim, fusion_head,
     stage1_lr, stage1_epochs, label_smoothing, focal_gamma,
     stage2_epochs, stage2_unfreeze_pct) = params

    cfg = JointSearchConfig(
        name=f"trial_{trial_num}",
        depth_rgb_backbone=depth_rgb_backbone,
        depth_rgb_head_units=parse_head_units(depth_rgb_head),
        thermal_map_backbone=thermal_map_backbone,
        thermal_map_head_units=parse_head_units(thermal_map_head),
        image_size=int(image_size),
        depth_rgb_pretrain_lr=float(depth_rgb_pretrain_lr),
        thermal_map_pretrain_lr=float(thermal_map_pretrain_lr),
        depth_rgb_use_mixup=bool(depth_rgb_use_mixup),
        thermal_map_use_mixup=bool(thermal_map_use_mixup),
        head_dropout=float(head_dropout),
        rf_n_estimators=int(rf_n_estimators),
        rf_max_depth=int(rf_max_depth) if rf_max_depth is not None else None,
        rf_feature_selection_k=int(rf_feature_selection_k),
        rf_min_samples_leaf=int(rf_min_samples_leaf),
        fusion_strategy=fusion_strategy,
        image_projection_dim=int(image_projection_dim),
        fusion_head_units=parse_head_units(fusion_head),
        stage1_lr=float(stage1_lr),
        stage1_epochs=int(stage1_epochs),
        label_smoothing=float(label_smoothing),
        focal_gamma=float(focal_gamma),
        stage2_epochs=int(stage2_epochs),
        stage2_unfreeze_pct=float(stage2_unfreeze_pct),
        fold=fold,
    )
    return cfg


def config_to_params(cfg):
    """Convert JointSearchConfig back to parameter list for warm-start."""
    return [
        cfg.depth_rgb_backbone,
        str(cfg.depth_rgb_head_units).replace(' ', ''),
        cfg.thermal_map_backbone,
        str(cfg.thermal_map_head_units).replace(' ', ''),
        cfg.image_size,
        cfg.depth_rgb_pretrain_lr,
        cfg.thermal_map_pretrain_lr,
        int(cfg.depth_rgb_use_mixup),
        int(cfg.thermal_map_use_mixup),
        cfg.head_dropout,
        cfg.rf_n_estimators,
        cfg.rf_max_depth,
        cfg.rf_feature_selection_k,
        cfg.rf_min_samples_leaf,
        cfg.fusion_strategy,
        cfg.image_projection_dim,
        str(cfg.fusion_head_units).replace(' ', ''),
        cfg.stage1_lr,
        cfg.stage1_epochs,
        cfg.label_smoothing,
        cfg.focal_gamma,
        cfg.stage2_epochs,
        cfg.stage2_unfreeze_pct,
    ]


# ─────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'name', 'trial_num',
    'depth_rgb_backbone', 'depth_rgb_head_units', 'depth_rgb_head_dropout',
    'depth_rgb_pretrain_lr', 'depth_rgb_use_mixup',
    'thermal_map_backbone', 'thermal_map_head_units', 'thermal_map_head_dropout',
    'thermal_map_pretrain_lr', 'thermal_map_use_mixup',
    'image_size',
    'rf_n_estimators', 'rf_max_depth', 'rf_feature_selection_k', 'rf_min_samples_leaf',
    'fusion_strategy', 'image_projection_dim', 'fusion_head_units',
    'stage1_lr', 'stage1_epochs', 'label_smoothing', 'focal_gamma',
    'stage2_epochs', 'stage2_unfreeze_pct',
    'fold',
    'depth_rgb_pretrain_kappa', 'thermal_map_pretrain_kappa',
    'meta_kappa', 'meta_acc', 'meta_f1',
    'best_s1_kappa', 'best_s1_epoch', 'best_s2_kappa',
    'post_eval_kappa', 'post_eval_acc', 'post_eval_f1',
    'kappa_delta',
    'trainable_weights', 'total_weights',
    'elapsed_seconds', 'n_val_samples',
]


def save_result_row(result, filepath):
    """Append a single result row to the CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def load_completed_results(filepath):
    """Load previously completed results from CSV for resume support."""
    if not os.path.exists(filepath):
        return []

    results = []
    float_keys = ['depth_rgb_pretrain_lr', 'thermal_map_pretrain_lr',
                   'depth_rgb_head_dropout', 'thermal_map_head_dropout',
                   'head_dropout', 'stage1_lr', 'label_smoothing',
                   'focal_gamma', 'stage2_unfreeze_pct',
                   'depth_rgb_pretrain_kappa', 'thermal_map_pretrain_kappa',
                   'meta_kappa', 'meta_acc', 'meta_f1',
                   'best_s1_kappa', 'best_s2_kappa',
                   'post_eval_kappa', 'post_eval_acc', 'post_eval_f1',
                   'kappa_delta', 'elapsed_seconds']
    int_keys = ['trial_num', 'image_size', 'rf_n_estimators',
                'rf_feature_selection_k', 'rf_min_samples_leaf',
                'image_projection_dim', 'stage1_epochs', 'stage2_epochs',
                'fold', 'best_s1_epoch', 'trainable_weights', 'total_weights',
                'n_val_samples']
    bool_keys = ['depth_rgb_use_mixup', 'thermal_map_use_mixup']

    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in float_keys:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            for key in int_keys:
                if key in row and row[key]:
                    try:
                        row[key] = int(float(row[key]))
                    except (ValueError, TypeError):
                        pass
            for key in bool_keys:
                if key in row and row[key]:
                    row[key] = row[key] in ('True', 'true', '1')
            # Handle rf_max_depth which can be None
            if 'rf_max_depth' in row:
                if row['rf_max_depth'] in ('None', 'none', ''):
                    row['rf_max_depth'] = None
                else:
                    try:
                        row['rf_max_depth'] = int(float(row['rf_max_depth']))
                    except (ValueError, TypeError):
                        row['rf_max_depth'] = None
            results.append(row)

    return results


def result_to_config(r):
    """Reconstruct a JointSearchConfig from a result dict."""
    depth_rgb_head = r.get('depth_rgb_head_units', '[128]')
    if isinstance(depth_rgb_head, str):
        depth_rgb_head = parse_head_units(depth_rgb_head)
    thermal_map_head = r.get('thermal_map_head_units', '[128,32]')
    if isinstance(thermal_map_head, str):
        thermal_map_head = parse_head_units(thermal_map_head)
    fusion_head = r.get('fusion_head_units', '[]')
    if isinstance(fusion_head, str):
        fusion_head = parse_head_units(fusion_head)

    rf_max_depth = r.get('rf_max_depth', 8)
    if rf_max_depth in ('None', 'none', '', None):
        rf_max_depth = None
    else:
        try:
            rf_max_depth = int(float(rf_max_depth))
        except (ValueError, TypeError):
            rf_max_depth = None

    cfg = JointSearchConfig(
        name=r.get('name', 'restored'),
        depth_rgb_backbone=r.get('depth_rgb_backbone', 'EfficientNetB0'),
        depth_rgb_head_units=depth_rgb_head,
        thermal_map_backbone=r.get('thermal_map_backbone', 'EfficientNetB2'),
        thermal_map_head_units=thermal_map_head,
        head_dropout=float(r.get('depth_rgb_head_dropout', r.get('head_dropout', 0.3))),
        depth_rgb_pretrain_lr=float(r.get('depth_rgb_pretrain_lr', 0.001)),
        thermal_map_pretrain_lr=float(r.get('thermal_map_pretrain_lr', 0.003)),
        depth_rgb_use_mixup=bool(r.get('depth_rgb_use_mixup', False)),
        thermal_map_use_mixup=bool(r.get('thermal_map_use_mixup', True)),
        image_size=int(r.get('image_size', 256)),
        rf_n_estimators=int(r.get('rf_n_estimators', 200)),
        rf_max_depth=rf_max_depth,
        rf_feature_selection_k=int(r.get('rf_feature_selection_k', 40)),
        rf_min_samples_leaf=int(r.get('rf_min_samples_leaf', 5)),
        fusion_strategy=r.get('fusion_strategy', 'feature_concat'),
        image_projection_dim=int(r.get('image_projection_dim', 0)),
        fusion_head_units=fusion_head,
        stage1_lr=float(r.get('stage1_lr', 5e-4)),
        stage1_epochs=int(r.get('stage1_epochs', 300)),
        label_smoothing=float(r.get('label_smoothing', 0.1)),
        focal_gamma=float(r.get('focal_gamma', 2.0)),
        stage2_epochs=int(r.get('stage2_epochs', 0)),
        stage2_unfreeze_pct=float(r.get('stage2_unfreeze_pct', 0.2)),
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────────

def save_checkpoint(trial_results, best_result, best_kappa):
    """Save checkpoint JSON for resume support."""
    checkpoint = {
        'n_completed': len(trial_results),
        'best_kappa': best_kappa,
        'best_result': best_result,
        'trial_history': [],
        'timestamp': datetime.datetime.now().isoformat(),
    }
    for r in trial_results:
        entry = {
            'name': r['name'],
            'post_eval_kappa': r['post_eval_kappa'],
            'params': config_to_params(result_to_config(r)),
        }
        checkpoint['trial_history'].append(entry)

    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)


def load_checkpoint():
    """Load checkpoint JSON if it exists."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Main Bayesian optimization loop
# ─────────────────────────────────────────────────────────────────────

def run_bayesian_search(data, n_trials=100, fresh=False):
    """Run Bayesian optimization over the joint search space.

    Returns list of all trial results and the best config.
    """
    from skopt import gp_minimize
    from skopt.space import Real, Categorical

    search_space = get_search_space()

    print(f"\n{'#'*80}")
    print(f"BAYESIAN OPTIMIZATION: {n_trials} trials over {len(search_space)} dimensions")
    print(f"{'#'*80}\n")

    # Load existing results for resume
    all_results = []
    x0, y0 = None, None
    n_completed = 0

    if not fresh:
        all_results = load_completed_results(RESULTS_CSV)
        checkpoint = load_checkpoint()
        if all_results and checkpoint:
            n_completed = len(all_results)
            print(f"RESUME: Found {n_completed} completed trials")

            # Build x0/y0 for warm-starting gp_minimize
            x0_list = []
            y0_list = []
            for entry in checkpoint.get('trial_history', []):
                params = entry.get('params')
                kappa = entry.get('post_eval_kappa', 0.0)
                if params is not None and kappa != 0.0:
                    x0_list.append(params)
                    y0_list.append(-kappa)  # Negative for minimization

            if x0_list:
                x0 = x0_list
                y0 = y0_list
                print(f"  Warm-starting with {len(x0)} prior evaluations")
            else:
                print("  No valid prior evaluations for warm-start")
        elif all_results:
            # Have CSV but no checkpoint — rebuild warm-start from CSV
            n_completed = len(all_results)
            print(f"RESUME: Found {n_completed} completed trials (no checkpoint, rebuilding)")
            x0_list = []
            y0_list = []
            for r in all_results:
                try:
                    cfg = result_to_config(r)
                    params = config_to_params(cfg)
                    kappa = float(r.get('post_eval_kappa', 0.0))
                    if kappa > 0:
                        x0_list.append(params)
                        y0_list.append(-kappa)
                except Exception:
                    pass
            if x0_list:
                x0 = x0_list
                y0 = y0_list
                print(f"  Warm-starting with {len(x0)} prior evaluations from CSV")
    else:
        if os.path.exists(RESULTS_CSV):
            backup = RESULTS_CSV + '.bak'
            shutil.copy2(RESULTS_CSV, backup)
            os.remove(RESULTS_CSV)
            print(f"FRESH START: Backed up old results to {backup}")
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    remaining = n_trials - n_completed
    if remaining <= 0:
        print(f"All {n_trials} trials already completed")
        return all_results

    print(f"  Will run {remaining} new trials (of {n_trials} total)")

    # Best tracking
    best_kappa = max([r.get('post_eval_kappa', 0.0) for r in all_results], default=0.0)
    best_result = max(all_results, key=lambda r: r.get('post_eval_kappa', 0.0)) if all_results else None

    trial_counter = [n_completed]

    def objective(params):
        """Objective function for gp_minimize. Evaluates a single config on fold 0."""
        trial_num = trial_counter[0]
        trial_counter[0] += 1

        try:
            cfg = decode_params_to_config(params, trial_num=trial_num, fold=0)
            result = train_single_trial(cfg, data, fold_idx=0)

            # Save result
            save_result_row(result, RESULTS_CSV)
            all_results.append(result)

            # Update best
            nonlocal best_kappa, best_result
            kappa = result['post_eval_kappa']
            is_new_best = kappa > best_kappa
            if is_new_best:
                best_kappa = kappa
                best_result = result
                print(f"\n  *** NEW BEST: trial_{trial_num} kappa={kappa:.4f} ***\n")

            # Save checkpoint
            save_checkpoint(all_results, best_result, best_kappa)

            # Memory cleanup
            gc.collect()
            tf.keras.backend.clear_session()

            # Return negative kappa for minimization
            return -kappa

        except Exception as e:
            print(f"\n  !!! Trial {trial_num} FAILED: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            tf.keras.backend.clear_session()
            return 0.0  # Neutral score for failed trials

    # Run Bayesian optimization
    gp_kwargs = {
        'func': objective,
        'dimensions': search_space,
        'n_calls': remaining,
        'random_state': 42,
        'verbose': True,
        'n_initial_points': min(10, remaining),
    }
    if x0 and y0:
        gp_kwargs['x0'] = x0
        gp_kwargs['y0'] = y0
        gp_kwargs['n_initial_points'] = 0  # Skip random phase — prior points seed the GP

    gp_result = gp_minimize(**gp_kwargs)

    print(f"\n{'='*80}")
    print(f"BAYESIAN OPTIMIZATION COMPLETE")
    print(f"  Total trials: {len(all_results)}")
    print(f"  Best kappa: {best_kappa:.4f}")
    if best_result:
        print(f"  Best config: {best_result['name']}")
    print(f"{'='*80}\n")

    return all_results


# ─────────────────────────────────────────────────────────────────────
# Top-N 5-fold validation
# ─────────────────────────────────────────────────────────────────────

def run_topn_validation(data, all_results, top_n=3, n_folds=5):
    """Run 5-fold validation on the top-N configs from Bayesian search.

    Returns dict mapping config name -> {cfg, fold_results, mean_kappa, std_kappa}.
    """
    from scipy import stats

    print(f"\n{'#'*80}")
    print(f"TOP-{top_n} VALIDATION: {n_folds}-fold CV")
    print(f"{'#'*80}\n")

    # Sort by post_eval_kappa descending, de-duplicate by config fingerprint
    sorted_results = sorted(all_results, key=lambda r: r.get('post_eval_kappa', 0.0), reverse=True)

    seen_fingerprints = set()
    top_results = []
    for r in sorted_results:
        fp = (
            r.get('depth_rgb_backbone', ''),
            r.get('depth_rgb_head_units', ''),
            r.get('thermal_map_backbone', ''),
            r.get('thermal_map_head_units', ''),
            str(r.get('image_size', 256)),
            str(r.get('rf_n_estimators', 200)),
            str(r.get('rf_max_depth', 8)),
            r.get('fusion_strategy', ''),
            str(r.get('image_projection_dim', 0)),
            r.get('fusion_head_units', ''),
            str(r.get('stage1_epochs', 300)),
            str(r.get('stage2_epochs', 0)),
        )
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            top_results.append(r)
            if len(top_results) == top_n:
                break

    print(f"Selected Top-{top_n} configs for {n_folds}-fold validation:")
    for i, r in enumerate(top_results):
        print(f"  {i+1}. {r['name']} -> kappa={r['post_eval_kappa']:.4f}")

    # Load completed results to skip already-done folds
    completed = load_completed_results(RESULTS_CSV)
    completed_names = {r['name'] for r in completed}

    topn_fold_results = {}

    for rank, r in enumerate(top_results):
        base_cfg = result_to_config(r)
        base_cfg.n_folds = n_folds
        tag = f"TOP{top_n}_{rank+1}"
        fold_results = []

        for fold_idx in range(n_folds):
            fold_name = f"{tag}_fold{fold_idx+1}"
            if fold_name in completed_names:
                cached = [cr for cr in completed if cr['name'] == fold_name][0]
                print(f"  SKIP (cached): {fold_name} -> kappa={cached['post_eval_kappa']:.4f}")
                fold_results.append(cached)
            else:
                cfg = deepcopy(base_cfg)
                cfg.name = fold_name
                cfg.fold = fold_idx
                result = train_single_trial(cfg, data, fold_idx=fold_idx)
                save_result_row(result, RESULTS_CSV)
                fold_results.append(result)
                gc.collect()
                tf.keras.backend.clear_session()

        fold_kappas = [float(fr['post_eval_kappa']) for fr in fold_results]
        topn_fold_results[tag] = {
            'cfg': base_cfg,
            'orig_name': r['name'],
            'fold0_kappa': float(r['post_eval_kappa']),
            'fold_results': fold_results,
            'mean_kappa': np.mean(fold_kappas),
            'std_kappa': np.std(fold_kappas),
        }
        print(f"\n  {tag} ({r['name']}): mean_kappa={np.mean(fold_kappas):.4f} "
              f"+/- {np.std(fold_kappas):.4f}")

    # Run baseline 5-fold
    print(f"\n{'='*60}")
    print(f"BASELINE: {n_folds}-fold CV with default parameters")
    print(f"{'='*60}")

    baseline_cfg = JointSearchConfig(name="BASELINE")
    baseline_cfg.n_folds = n_folds
    baseline_fold_results = []

    for fold_idx in range(n_folds):
        fold_name = f"BASELINE_fold{fold_idx+1}"
        if fold_name in completed_names:
            cached = [cr for cr in completed if cr['name'] == fold_name][0]
            print(f"  SKIP (cached): {fold_name} -> kappa={cached['post_eval_kappa']:.4f}")
            baseline_fold_results.append(cached)
        else:
            cfg = deepcopy(baseline_cfg)
            cfg.name = fold_name
            cfg.fold = fold_idx
            result = train_single_trial(cfg, data, fold_idx=fold_idx)
            save_result_row(result, RESULTS_CSV)
            baseline_fold_results.append(result)
            gc.collect()
            tf.keras.backend.clear_session()

    baseline_kappas = [float(fr['post_eval_kappa']) for fr in baseline_fold_results]
    print(f"\n  BASELINE: mean_kappa={np.mean(baseline_kappas):.4f} "
          f"+/- {np.std(baseline_kappas):.4f}")

    # Summary with paired t-tests
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Rank':<6} {'Config':<30} {'Mean':>8} {'Std':>8} {'p-value':>10}")
    print(f"  {'-'*65}")

    # Baseline row
    print(f"  {'BL':<6} {'BASELINE':<30} {np.mean(baseline_kappas):>8.4f} "
          f"{np.std(baseline_kappas):>8.4f} {'---':>10}")

    sorted_topn = sorted(topn_fold_results.items(),
                         key=lambda x: x[1]['mean_kappa'], reverse=True)

    for rank, (tag, info) in enumerate(sorted_topn, start=1):
        fk = [float(r['post_eval_kappa']) for r in info['fold_results']]
        if len(fk) == len(baseline_kappas) and len(fk) >= 2:
            t_stat, p_value = stats.ttest_rel(fk, baseline_kappas)
            p_str = f"{p_value:.4f}"
        else:
            p_str = "N/A"

        print(f"  {rank:<6} {info['orig_name']:<30} {info['mean_kappa']:>8.4f} "
              f"{info['std_kappa']:>8.4f} {p_str:>10}")

    # Print detailed fold results
    print(f"\n{'─'*60}")
    print("DETAILED FOLD RESULTS")
    print(f"{'─'*60}")

    def print_fold_table(label, fold_results):
        kappas = [float(r['post_eval_kappa']) for r in fold_results]
        accs = [float(r['post_eval_acc']) for r in fold_results]
        f1s = [float(r['post_eval_f1']) for r in fold_results]
        print(f"\n  {label}:")
        print(f"  {'Fold':<8} {'Kappa':>8} {'Accuracy':>10} {'F1':>10}")
        print(f"  {'-'*38}")
        for i, r in enumerate(fold_results):
            print(f"  Fold {i+1:<3} {float(r['post_eval_kappa']):>8.4f} "
                  f"{float(r['post_eval_acc']):>10.4f} {float(r['post_eval_f1']):>10.4f}")
        print(f"  {'-'*38}")
        print(f"  {'Mean':<8} {np.mean(kappas):>8.4f} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f}")
        print(f"  {'Std':<8} {np.std(kappas):>8.4f} {np.std(accs):>10.4f} {np.std(f1s):>10.4f}")
        return kappas

    print_fold_table("BASELINE", baseline_fold_results)
    for tag, info in sorted_topn:
        print_fold_table(f"{tag}: {info['orig_name']}", info['fold_results'])

    # Determine winner
    winner_tag, winner_info = sorted_topn[0]
    winner_kappas = [float(r['post_eval_kappa']) for r in winner_info['fold_results']]

    print(f"\n{'─'*60}")
    print(f"STATISTICAL COMPARISON: {winner_info['orig_name']} vs BASELINE")
    print(f"{'─'*60}")

    kappa_diff = np.mean(winner_kappas) - np.mean(baseline_kappas)
    print(f"  Mean Kappa diff: {kappa_diff:+.4f} "
          f"({np.mean(baseline_kappas):.4f} -> {np.mean(winner_kappas):.4f})")

    if len(winner_kappas) == len(baseline_kappas) and len(winner_kappas) >= 2:
        t_stat, p_value = stats.ttest_rel(winner_kappas, baseline_kappas)
        print(f"\n  Paired t-test (n={len(winner_kappas)} folds):")
        print(f"    t-statistic = {t_stat:.4f}")
        print(f"    p-value     = {p_value:.4f}")
        if p_value < 0.05:
            label = "WINNER" if kappa_diff > 0 else "BASELINE"
            print(f"    -> Statistically significant (p < 0.05): {label} is better")
        else:
            print(f"    -> NOT statistically significant (p >= 0.05)")

    # Save best config
    bl_mean = np.mean(baseline_kappas)
    winner_mean = np.mean(winner_kappas)

    if winner_mean > bl_mean:
        final_cfg = winner_info['cfg']
        final_label = winner_info['orig_name']
        final_kappas = winner_kappas
        print(f"\n  WINNER: {final_label} (kappa {winner_mean:.4f} > baseline {bl_mean:.4f})")
    else:
        final_cfg = JointSearchConfig(name="BASELINE")
        final_label = "BASELINE"
        final_kappas = baseline_kappas
        print(f"\n  BASELINE wins (kappa {bl_mean:.4f} >= winner {winner_mean:.4f})")

    config_dict = asdict(final_cfg)
    config_dict['selection_method'] = 'bayesian_optimization_joint_search'
    config_dict['five_fold_results'] = {
        'mean_kappa': float(np.mean(final_kappas)),
        'std_kappa': float(np.std(final_kappas)),
        'fold_kappas': [float(k) for k in final_kappas],
    }
    with open(BEST_CONFIG_PATH, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"\nBest config saved to: {BEST_CONFIG_PATH}")

    return topn_fold_results, baseline_fold_results


# ─────────────────────────────────────────────────────────────────────
# Dry run — print search space and exit
# ─────────────────────────────────────────────────────────────────────

def dry_run():
    """Print search space details and exit."""
    search_space = get_search_space()

    print(f"\n{'='*80}")
    print(f"JOINT HYPERPARAMETER SEARCH — DRY RUN")
    print(f"{'='*80}\n")

    print(f"Search space: {len(search_space)} dimensions\n")
    total_combos = 1
    for dim in search_space:
        if hasattr(dim, 'categories'):
            n = len(dim.categories)
            total_combos *= n
            cats_str = ', '.join(str(c) for c in dim.categories)
            print(f"  {dim.name:<30} Categorical({n}): [{cats_str}]")
        else:
            print(f"  {dim.name:<30} Real({dim.low}, {dim.high}, prior='{dim.prior}')")
            total_combos *= 100  # Approximate for continuous

    print(f"\nApproximate search space size: {total_combos:.2e}")
    print(f"\nCSV columns ({len(CSV_COLUMNS)}):")
    for col in CSV_COLUMNS:
        print(f"  {col}")

    print(f"\nOutput files:")
    print(f"  Results CSV:  {RESULTS_CSV}")
    print(f"  Checkpoint:   {CHECKPOINT_PATH}")
    print(f"  Best config:  {BEST_CONFIG_PATH}")
    print(f"  Weights dir:  {WEIGHTS_DIR}")
    print(f"  Logs dir:     {LOGS_DIR}")

    # Print default config
    default_cfg = JointSearchConfig()
    print(f"\nDefault config (baseline):")
    for k, v in asdict(default_cfg).items():
        print(f"  {k}: {v}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(fresh=False, n_trials=100, n_folds=5, top_n=3, dry_run_mode=False):
    """Main entry point for joint hyperparameter optimization."""

    if dry_run_mode:
        dry_run()
        return

    print("=" * 80)
    print("JOINT HYPERPARAMETER OPTIMIZATION")
    print(f"  Modalities: metadata + depth_rgb + thermal_map")
    print(f"  Strategy: Bayesian optimization ({n_trials} trials, 23 dimensions)")
    print(f"  Validation: Top-{top_n} configs, {n_folds}-fold CV with paired t-test")
    print("=" * 80)

    data, directory, result_dir = load_data()

    # Phase 1: Bayesian optimization
    all_results = run_bayesian_search(data, n_trials=n_trials, fresh=fresh)

    if not all_results:
        print("No results produced. Exiting.")
        return

    # Phase 2: Top-N validation
    topn_fold_results, baseline_fold_results = run_topn_validation(
        data, all_results, top_n=top_n, n_folds=n_folds)

    print(f"\nResults saved to: {RESULTS_CSV}")
    print(f"Best config saved to: {BEST_CONFIG_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint Hyperparameter Optimization (metadata + depth_rgb + thermal_map)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start clean (backup old results)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of Bayesian trials (default: 100)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Folds for Top-N validation (default: 5)')
    parser.add_argument('--top-n', type=int, default=3,
                        help='Top configs for validation (default: 3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print search space and exit')
    args = parser.parse_args()

    # Set up dual logging (stdout + log file)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOGS_DIR, f'joint_hparam_search_{timestamp}.log')

    class TeeStream:
        """Write to both a file and the original stream."""
        def __init__(self, stream, log_file):
            self.stream = stream
            self.log_file = log_file
        def write(self, data):
            self.stream.write(data)
            self.log_file.write(data)
            self.log_file.flush()
        def flush(self):
            self.stream.flush()
            self.log_file.flush()

    log_file = open(log_path, 'w')
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    print(f"Logging to: {log_path}")

    try:
        main(fresh=args.fresh, n_trials=args.n_trials, n_folds=args.n_folds,
             top_n=args.top_n, dry_run_mode=args.dry_run)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to: {log_path}")
