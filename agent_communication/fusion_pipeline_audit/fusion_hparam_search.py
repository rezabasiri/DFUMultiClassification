#!/usr/bin/env python3
"""
Fusion Pipeline Hyperparameter Search — Standalone Script

Systematically tests fusion-specific training configurations for
metadata + depth_rgb + thermal_map to maximize val_cohen_kappa.

Uses Top-3 R1 carry-forward strategy (same as standalone audits):
  Round 1: Fusion learning rate (STAGE1_LR)          (6 configs)
           -> Top-3 R1 winners each get their own independent R2-R7 branch
  Round 2: Fusion training epochs                    (4 configs per branch)
  Round 3: Metadata confidence scaling               (6 configs per branch)
  Round 4: Cross-modal attention regularization      (4 configs per branch)
  Round 5: Metadata-image attention asymmetry        (4 configs per branch)
  Round 6: RF hyperparameters                        (6 configs per branch)
  Round 7: Stage 2 fine-tuning in fusion context     (4 configs per branch)
  Top 3:   Best configs across ALL branches, 5-fold CV (15 folds)
  Baseline: Default parameters, 5-fold CV            (5 folds)
  Summary compares top configs against baseline with paired t-test.

Fusion-specific vs standalone searches:
  - Standalone searches optimize backbone, head, loss, augmentation per modality
  - This search optimizes HOW modalities are COMBINED — parameters that only
    activate when 3 modalities are present
  - All standalone modality configs (backbone, head, etc.) are loaded from
    production_config.MODALITY_CONFIGS and kept fixed during fusion search
  - Fuses metadata + depth_rgb + thermal_map (3-modality fusion)

Prerequisites:
  - Standalone modality configs (backbone, head) should be finalized first
  - Each experiment pre-trains image branches (frozen backbone, head-only) on the
    same fold split, then freezes the backbone for fusion training — matching the
    approach in src/main.py to avoid overfitting with ~2K images

Usage:
  # Fresh start (backs up old results):
  python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --fresh

  # Resume from where it left off (default — loads standalone weights if available):
  python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py

  # Force re-train image branches even if standalone weights exist:
  python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --force-retrain

Results written to: agent_communication/fusion_pipeline_audit/fusion_search_results.csv
"""

import os
import sys
import gc
import time
import json
import math
import random
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

AUDIT_DIR = os.path.join(PROJECT_ROOT, 'agent_communication', 'fusion_pipeline_audit')

# Standalone audit weight directories — standalone scripts save per-fold weights here
_STANDALONE_WEIGHTS_DIRS = {
    'depth_rgb': os.path.join(PROJECT_ROOT, 'agent_communication', 'depth_rgb_pipeline_audit', 'weights'),
    'thermal_map': os.path.join(PROJECT_ROOT, 'agent_communication', 'thermal_map_pipeline_audit', 'weights'),
}


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


def find_standalone_weights(modality, config_name, fold_idx):
    """Find saved standalone weights for a given modality and fold.

    Looks for weights saved by standalone scripts in their audit directories.
    The standalone scripts save weights as {config_name}_fold{fold_idx+1}.{ext}
    where config_name is e.g. 'BASELINE' or 'TOP3_1'.

    Returns path_without_ext if found, None otherwise.
    """
    weights_dir = _STANDALONE_WEIGHTS_DIRS.get(modality)
    if weights_dir is None or not os.path.isdir(weights_dir):
        return None

    # Standalone saves as {cfg.name} which is "{config_name}_fold{fold_idx+1}"
    weight_name = f"{config_name}_fold{fold_idx + 1}"
    path_without_ext = os.path.join(weights_dir, weight_name)

    for ext in ['.weights.h5', '.h5']:
        if os.path.exists(path_without_ext + ext):
            return path_without_ext
    return None


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

# Standalone best configs for pre-training — ensures each image arm in fusion
# is pre-trained with the exact same hyperparameters that were optimized by the
# standalone pipeline audits.  Keys: modality name -> dict of training params.
#
# depth_rgb: BASELINE — EfficientNetB0 frozen, head=[128]
#   Mean kappa=0.2943 +/- 0.0581 over 5 folds
#   DenseNet121 winner was not significantly better (+0.008), using simpler baseline
# thermal_map: B3_R6_ft_top50_50ep — EfficientNetB2 frozen, head=[128,32]
#   Best standalone config from thermal_map_best_config.json
_STANDALONE_CONFIGS = {
    'depth_rgb': {
        'backbone': 'EfficientNetB0',
        'freeze': 'frozen',
        'head_units': [128],
        'head_dropout': 0.3,
        'head_use_bn': True,
        'head_l2': 0.0,
        'learning_rate': 0.001,
        'stage1_epochs': 50,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
        'batch_size': 64,
        'lr_schedule': 'plateau',
        'warmup_epochs': 0,
        'loss_type': 'focal',
        'focal_gamma': 2.0,
        'alpha_sum': 3.0,
        'label_smoothing': 0.0,
        'use_augmentation': True,
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'image_size': 256,
        'optimizer': 'adam',
        'weight_decay': 0.0001,
        'finetune_lr': 1e-5,
        'finetune_epochs': 30,
        'unfreeze_pct': 0.2,
        'freeze_bn_in_stage2': False,  # depth_rgb standalone does NOT freeze BN in stage 2
    },
    'thermal_map': {
        'backbone': 'EfficientNetB2',
        'freeze': 'frozen',
        'head_units': [128, 32],
        'head_dropout': 0.3,
        'head_use_bn': True,
        'head_l2': 0.0,
        'learning_rate': 0.003,
        'stage1_epochs': 50,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
        'batch_size': 64,
        'lr_schedule': 'plateau',
        'warmup_epochs': 0,
        'loss_type': 'focal',
        'focal_gamma': 2.0,
        'alpha_sum': 3.0,
        'label_smoothing': 0.0,
        'use_augmentation': True,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'image_size': 256,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'finetune_lr': 1e-5,  # R6 screening used 5e-6, but pick_best() lost it; 5-fold CV ran with 1e-5
        'finetune_epochs': 50,
        'unfreeze_pct': 0.5,
        'freeze_bn_in_stage2': True,  # thermal_map standalone freezes BN in stage 2
    },
}
for _mod, _cfg in _STANDALONE_CONFIGS.items():
    print(f"  Standalone config for {_mod}: backbone={_cfg['backbone']}, "
          f"head={_cfg['head_units']}, epochs={_cfg['stage1_epochs']}+{_cfg['finetune_epochs']}ft")

# Cache pre-trained image weights per (modality, fold) to avoid redundant training.
# All configs in the same fold use the same data split and the same pre-training
# hyperparams, so the pre-trained weights are identical.
_pretrain_cache = {}  # key: (modality, fold_idx) -> {layer_name: weights}

# ─────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FusionSearchConfig:
    """Single fusion experiment configuration.

    Standalone modality params (backbone, head, loss, etc.) are loaded from
    _STANDALONE_CONFIGS and NOT searched here — they were already optimized
    by standalone pipeline audits.
    """
    name: str = "baseline"

    # Modalities
    image_modalities: list = field(default_factory=lambda: ["depth_rgb", "thermal_map"])

    # Fusion architecture strategy
    # - prob_concat: RF probs (3) + image probs (3) → Dense(3). Current baseline. 21 params.
    # - feature_concat: RF probs (3) + image features (128+32) → fusion head → Dense(3).
    # - feature_concat_attn: feature_concat + cross-modal attention between branches.
    # - gated: learned per-class gate: gate * rf_probs + (1-gate) * image_probs.
    # - hybrid: RF probs (3) + image features (160) + image probs (3) → fusion head → Dense(3).
    fusion_strategy: str = "prob_concat"

    # Fusion head (layers between concat and final Dense(3) output)
    # Empty list = direct Dense(3) on concat (current behavior for prob_concat).
    fusion_head_units: list = field(default_factory=list)
    fusion_head_dropout: float = 0.3
    fusion_head_bn: bool = True

    # Temperature scaling for image softmax (applied before fusion in prob-level strategies)
    # 1.0 = standard softmax, <1.0 = sharper, >1.0 = smoother
    image_temperature: float = 1.0

    # Gate network hidden dim (only used when fusion_strategy="gated")
    gate_hidden_dim: int = 32

    # Fusion training — Stage 1 (frozen backbone after pre-training, fusion layers only)
    stage1_lr: float = 1e-4           # STAGE1_LR in production_config
    stage1_epochs: int = 200          # N_EPOCHS (main.py uses full budget + early stopping)
    early_stop_patience: int = 20     # EARLY_STOP_PATIENCE in production_config
    reduce_lr_patience: int = 10      # REDUCE_LR_PATIENCE in production_config
    batch_size: int = 64
    lr_schedule: str = "plateau"      # plateau, cosine
    warmup_epochs: int = 0

    # Fusion training — Stage 2 (partial backbone unfreeze + lower LR fine-tuning)
    stage2_lr: float = 1e-5           # STAGE2_LR
    stage2_epochs: int = 0            # 0 = skip Stage 2 (default for fusion in main.py)
    stage2_unfreeze_pct: float = 0.2  # STAGE2_UNFREEZE_PCT

    # Metadata confidence scaling (ConfidenceBasedMetadataAttention)
    meta_min_scale: float = 1.5       # min_scale in ConfidenceBasedMetadataAttention
    meta_max_scale: float = 3.0       # max_scale

    # Cross-modal attention (only effective when fusion_strategy="feature_concat_attn")
    fusion_query_dim: int = 64        # Dense units in fusion_query_i
    fusion_query_l2: float = 0.001    # L2 on fusion_query_i
    meta_query_scale: float = 0.8     # score scaling when metadata queries images
    image_query_scale: float = 1.5    # score scaling when images query metadata

    # Loss — uses same focal_ordinal_loss as main.py (from src/models/losses.py)
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    alpha_sum: float = 3.0
    label_smoothing: float = 0.1      # Matches main.py: _resolve_training_params → max(modality label_smoothing)

    # RF hyperparameters (metadata pipeline)
    rf_n_estimators: int = 200
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 5
    rf_min_samples_split: int = 10
    rf_feature_selection_k: int = 40

    # Augmentation (image modality — use standalone settings)
    use_augmentation: bool = True
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    image_size: int = 256

    # Cross-validation (5-fold to match standalone audits — 3-fold gives different splits!)
    n_folds: int = 5
    fold: int = 0


# ─────────────────────────────────────────────────────────────────────
# Metrics
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
# Model building — standalone pre-training (matches standalone audit scripts)
# ─────────────────────────────────────────────────────────────────────

def build_pretrained_backbone(input_shape, backbone_name, modality_name):
    """Build pretrained backbone — matches standalone audit scripts EXACTLY.

    Uses single-model instantiation (weights='imagenet' directly) just like
    standalone scripts. Input name uses modality_name for dataset compatibility.
    """
    from tensorflow.keras.layers import Lambda

    inp = Input(shape=input_shape, name=f'{modality_name}_input')

    backbone_map = {
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'EfficientNetB2': tf.keras.applications.EfficientNetB2,
        'DenseNet121': tf.keras.applications.DenseNet121,
        'ResNet50V2': tf.keras.applications.ResNet50V2,
    }

    preprocess_map = {
        'EfficientNetB0': None,
        'EfficientNetB2': None,
        'DenseNet121': tf.keras.applications.densenet.preprocess_input,
        'ResNet50V2': tf.keras.applications.resnet_v2.preprocess_input,
    }

    BackboneClass = backbone_map[backbone_name]
    preprocess_fn = preprocess_map[backbone_name]

    # Single-model instantiation — identical to standalone audit scripts.
    # No temp+copy approach; Keras loads ImageNet weights directly.
    base_model = BackboneClass(weights='imagenet', include_top=False, pooling='avg')

    if preprocess_fn is not None:
        x = Lambda(lambda img: preprocess_fn(img), name=f'{modality_name}_preprocess')(inp)
        x = base_model(x)
    else:
        x = base_model(inp)

    return inp, x, base_model


def build_pretrain_model(sa_cfg, modality_name):
    """Build standalone pre-training model — identical to standalone audit build_model().

    This creates: Input → backbone → head_dense → BN → Dropout → ... → softmax(3)
    Matches the exact architecture used in standalone pipeline audits so that
    pre-training kappa values are reproducible.
    """
    input_shape = (sa_cfg['image_size'], sa_cfg['image_size'], 3)
    inp, features, base_model = build_pretrained_backbone(
        input_shape, sa_cfg['backbone'], modality_name)

    x = features
    regularizer = tf.keras.regularizers.l2(sa_cfg['head_l2']) if sa_cfg['head_l2'] > 0 else None
    for i, units in enumerate(sa_cfg['head_units']):
        x = Dense(units, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizer,
                  name=f'head_dense_{i}')(x)
        if sa_cfg['head_use_bn']:
            x = BatchNormalization(name=f'head_bn_{i}')(x)
        x = Dropout(sa_cfg['head_dropout'], name=f'head_drop_{i}')(x)

    output = Dense(3, activation='softmax', name='output', dtype='float32')(x)
    model = Model(inputs=inp, outputs=output)

    return model, base_model


# ─────────────────────────────────────────────────────────────────────
# Model building — fusion-specific
# ─────────────────────────────────────────────────────────────────────

def _build_fusion_head(x, cfg: FusionSearchConfig, name_prefix='fusion'):
    """Build fusion head layers between concatenated features and final output.

    Args:
        x: Tensor after concatenation.
        cfg: Config with fusion_head_units, fusion_head_dropout, fusion_head_bn.
        name_prefix: Prefix for layer names.
    Returns:
        Tensor after fusion head (before final Dense(3)).
    """
    for i, units in enumerate(cfg.fusion_head_units):
        x = Dense(units, activation='relu', kernel_initializer='he_normal',
                  name=f'{name_prefix}_dense_{i}')(x)
        if cfg.fusion_head_bn:
            x = BatchNormalization(name=f'{name_prefix}_bn_{i}')(x)
        x = Dropout(cfg.fusion_head_dropout, name=f'{name_prefix}_drop_{i}')(x)
    return x


def _build_cross_modal_attention(branches, branch_names, cfg: FusionSearchConfig):
    """Apply cross-modal attention between branches (for feature_concat_attn strategy).

    Each branch queries every other branch via learned projections. Skip connections
    preserve original representations. Metadata-image asymmetry is applied via
    configurable scaling factors.

    Uses only Keras layers (no raw tf ops) for compatibility with Keras 3+.

    Args:
        branches: List of feature tensors (one per modality).
        branch_names: List of modality names (e.g. ['metadata', 'depth_rgb', 'thermal_map']).
        cfg: Config with fusion_query_dim, fusion_query_l2, meta_query_scale, image_query_scale.
    Returns:
        List of attended feature tensors (same length as branches).
    """
    from tensorflow.keras.layers import Multiply, Add, Lambda

    metadata_idx = None
    for i, name in enumerate(branch_names):
        if name == 'metadata':
            metadata_idx = i
            break

    attention_outputs = []
    for i, branch in enumerate(branches):
        original_branch = branch
        query = Dense(cfg.fusion_query_dim, activation='tanh', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(cfg.fusion_query_l2),
                      name=f'attn_query_{i}')(branch)

        attended_branches = []
        for j, other_branch in enumerate(branches):
            if i == j:
                continue
            # Project other_branch to same dim as query for element-wise interaction
            key = Dense(cfg.fusion_query_dim, activation='tanh', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(cfg.fusion_query_l2),
                        name=f'attn_key_{i}_{j}')(other_branch)
            interaction = Multiply(name=f'attn_interact_{i}_{j}')([query, key])
            # Produce raw logit score, then apply scaling before sigmoid
            logit = Dense(1, activation=None, name=f'attn_logit_{i}_{j}')(interaction)

            # Apply asymmetric scaling based on metadata vs image query direction
            if metadata_idx is not None and i == metadata_idx:
                scale = cfg.meta_query_scale
            elif metadata_idx is not None and j == metadata_idx:
                scale = cfg.image_query_scale
            else:
                scale = 1.0

            weight = Lambda(lambda x, s=scale: tf.nn.sigmoid(x * s),
                            name=f'attn_weight_{i}_{j}')(logit)

            # Weight the other branch, then project to query branch's dim for skip connection
            attended = Multiply(name=f'attn_attended_{i}_{j}')([other_branch, weight])
            attended_proj = Dense(original_branch.shape[-1], activation=None,
                                  name=f'attn_proj_{i}_{j}')(attended)
            attended_branches.append(attended_proj)

        if attended_branches:
            if len(attended_branches) == 1:
                attended_sum = attended_branches[0]
            else:
                attended_sum = Add(name=f'attn_sum_{i}')(attended_branches)
            attention_outputs.append(Add(name=f'attn_skip_{i}')([original_branch, attended_sum]))
        else:
            attention_outputs.append(branch)

    return attention_outputs


def build_fusion_model(cfg: FusionSearchConfig, image_input_shapes: dict, metadata_input_shape):
    """Build a fusion model using the specified fusion_strategy.

    Builds image branches and metadata branch using project's create_image_branch
    and create_metadata_branch (with patched modality configs from _STANDALONE_CONFIGS),
    then wires them together according to cfg.fusion_strategy.

    Strategies:
        prob_concat: image_features → Dense(3, softmax) → concat with RF → Dense(3).
        feature_concat: concat [RF probs + image features] → fusion head → Dense(3).
        feature_concat_attn: cross-modal attention on features, then concat → head → Dense(3).
        gated: learned per-class gate between RF probs and image probs.
        hybrid: concat [RF probs + image features + image probs] → fusion head → Dense(3).

    Args:
        cfg: Fusion search configuration.
        image_input_shapes: Dict mapping image modality name to shape.
        metadata_input_shape: Shape of metadata input, e.g. (3,).
    """
    import src.utils.production_config as _pcfg
    import src.models.builders as _builders
    from tensorflow.keras.layers import concatenate

    # Patch get_modality_config so image branches use standalone-audited configs
    orig_get_modality_config = _pcfg.get_modality_config
    orig_builders_get_modality_config = _builders.get_modality_config

    def patched_get_modality_config(modality):
        if modality in _STANDALONE_CONFIGS:
            sa = _STANDALONE_CONFIGS[modality]
            return {
                'backbone': sa['backbone'],
                'head_units': sa['head_units'],
                'head_l2': sa.get('head_l2', 0.0),
                'label_smoothing': sa.get('label_smoothing', 0.0),
                'finetune_epochs': sa.get('finetune_epochs', 30),
            }
        return orig_get_modality_config(modality)

    _pcfg.get_modality_config = patched_get_modality_config
    _builders.get_modality_config = patched_get_modality_config

    try:
        # Build branches using project's standard builders
        inputs = {}
        image_feature_branches = {}  # modality -> feature tensor (before classifier)

        # Metadata branch: Input(3) → cast float32 → rf_probs (3)
        metadata_input, rf_probs = _builders.create_metadata_branch(metadata_input_shape, 0)
        inputs['metadata_input'] = metadata_input

        # Image branches: Input → backbone → projection head → features
        for modality in cfg.image_modalities:
            image_input, features = _builders.create_image_branch(
                image_input_shapes[modality], modality)
            inputs[f'{modality}_input'] = image_input
            image_feature_branches[modality] = features

        # Merge image features (order matches cfg.image_modalities)
        image_features_list = [image_feature_branches[m] for m in cfg.image_modalities]
        if len(image_features_list) > 1:
            image_features_concat = concatenate(image_features_list, name='concat_image_features')
        else:
            image_features_concat = image_features_list[0]

        # Build image classifier (Dense(3)) — used by prob-level strategies
        def _make_image_probs(image_feats, temperature):
            if temperature == 1.0:
                return Dense(3, activation='softmax', name='image_classifier',
                             dtype='float32')(image_feats)
            else:
                logits = Dense(3, activation=None, name='image_logits',
                               dtype='float32')(image_feats)
                scaled = Lambda(lambda x: x / temperature,
                                name='temperature_scale')(logits)
                return tf.keras.layers.Softmax(name='image_classifier', dtype='float32')(scaled)

        # === Strategy dispatch ===
        strategy = cfg.fusion_strategy

        if strategy == 'prob_concat':
            # Current baseline: image features → Dense(3, softmax) → concat RF → Dense(3)
            image_probs = _make_image_probs(image_features_concat, cfg.image_temperature)
            fused = concatenate([rf_probs, image_probs], name='fusion_concat')
            x = _build_fusion_head(fused, cfg)
            output = Dense(3, activation='softmax', name='output', dtype='float32')(x)

        elif strategy == 'feature_concat':
            # Feature-level: concat [RF probs (3) + image features (128+32)] → head → Dense(3)
            fused = concatenate([rf_probs, image_features_concat], name='fusion_concat')
            x = _build_fusion_head(fused, cfg)
            output = Dense(3, activation='softmax', name='output', dtype='float32')(x)

        elif strategy == 'feature_concat_attn':
            # Feature-level + cross-modal attention
            branch_names = ['metadata'] + cfg.image_modalities
            branch_tensors = [rf_probs] + image_features_list
            attended = _build_cross_modal_attention(branch_tensors, branch_names, cfg)
            fused = concatenate(attended, name='fusion_concat')
            x = _build_fusion_head(fused, cfg)
            output = Dense(3, activation='softmax', name='output', dtype='float32')(x)

        elif strategy == 'gated':
            # Gated fusion: learn per-class gate between RF and image probs
            image_probs = _make_image_probs(image_features_concat, cfg.image_temperature)
            gate_input = concatenate([rf_probs, image_features_concat], name='gate_input')
            gate = Dense(cfg.gate_hidden_dim, activation='relu',
                         name='gate_hidden')(gate_input)
            gate = Dense(3, activation='sigmoid', name='gate_weights',
                         dtype='float32')(gate)
            # gate * rf_probs + (1-gate) * image_probs
            from tensorflow.keras.layers import Multiply, Add
            inv_gate = Lambda(lambda g: 1.0 - g, name='inv_gate')(gate)
            gated_rf = Multiply(name='gated_rf')([gate, rf_probs])
            gated_img = Multiply(name='gated_img')([inv_gate, image_probs])
            output = Add(name='output')([gated_rf, gated_img])

        elif strategy == 'hybrid':
            # Hybrid: concat [RF probs + image features + image probs] → head → Dense(3)
            image_probs = _make_image_probs(image_features_concat, cfg.image_temperature)
            fused = concatenate([rf_probs, image_features_concat, image_probs],
                                name='fusion_concat')
            x = _build_fusion_head(fused, cfg)
            output = Dense(3, activation='softmax', name='output', dtype='float32')(x)

        else:
            raise ValueError(f"Unknown fusion_strategy: {strategy}")

        model = Model(inputs=inputs, outputs=output)

    finally:
        # Restore originals (always, even on error)
        _pcfg.get_modality_config = orig_get_modality_config
        _builders.get_modality_config = orig_builders_get_modality_config

    return model


# ─────────────────────────────────────────────────────────────────────
# Data pipeline — reuses project's existing data loading
# ─────────────────────────────────────────────────────────────────────

def load_data(image_modalities=None):
    """Load the dataset for fusion (metadata + depth_rgb + thermal_map).

    NOTE: Core data filtering (USE_CORE_DATA) is intentionally DISABLED here.
    The fusion search should use ALL data so that results are not biased by
    thresholds optimized for standalone paths. Filtering can be re-evaluated
    after the best fusion config is found.
    """
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


def prepare_datasets(data, cfg: FusionSearchConfig, fold_idx=0):
    """Prepare train/valid datasets for a single fold of fusion training."""
    from src.data.dataset_utils import create_patient_folds, prepare_cached_datasets
    from src.data.generative_augmentation_v3 import AugmentationConfig

    data = data.copy()

    patient_folds = create_patient_folds(data, n_folds=cfg.n_folds, random_state=42)
    train_patients, valid_patients = patient_folds[fold_idx]

    aug_config = AugmentationConfig()
    aug_config.generative_settings['output_size']['width'] = cfg.image_size
    aug_config.generative_settings['output_size']['height'] = cfg.image_size

    selected_modalities = ['metadata'] + cfg.image_modalities

    # Override RF hyperparameters in production_config for this experiment
    import src.utils.production_config as _pcfg
    orig_rf_n_estimators = _pcfg.RF_N_ESTIMATORS
    orig_rf_max_depth = _pcfg.RF_MAX_DEPTH
    orig_rf_min_samples_leaf = _pcfg.RF_MIN_SAMPLES_LEAF
    orig_rf_min_samples_split = _pcfg.RF_MIN_SAMPLES_SPLIT
    orig_rf_feature_selection_k = _pcfg.RF_FEATURE_SELECTION_K

    _pcfg.RF_N_ESTIMATORS = cfg.rf_n_estimators
    _pcfg.RF_MAX_DEPTH = cfg.rf_max_depth
    _pcfg.RF_MIN_SAMPLES_LEAF = cfg.rf_min_samples_leaf
    _pcfg.RF_MIN_SAMPLES_SPLIT = cfg.rf_min_samples_split
    _pcfg.RF_FEATURE_SELECTION_K = cfg.rf_feature_selection_k

    # Temporarily override augmentation settings
    import src.data.dataset_utils as _ds_utils
    original_aug = _pcfg.USE_GENERAL_AUGMENTATION
    if not cfg.use_augmentation:
        _pcfg.USE_GENERAL_AUGMENTATION = False
        _ds_utils.USE_GENERAL_AUGMENTATION = False

    # Cache dir under results/search_cache/
    # RF params are part of the cache key because RF probabilities are baked into
    # the cached dataset. To avoid multi-GB cache proliferation, we auto-clean
    # old fusion caches when RF params change (only one RF variant cached at a time).
    from src.utils.config import get_project_paths
    _, search_result_dir, _ = get_project_paths()
    aug_suffix = 'aug' if cfg.use_augmentation else 'noaug'
    rf_key = f"rf{cfg.rf_n_estimators}_d{cfg.rf_max_depth}_l{cfg.rf_min_samples_leaf}_k{cfg.rf_feature_selection_k}"
    img_key = '_'.join(cfg.image_modalities)
    cache_base = os.path.join(search_result_dir, 'search_cache')
    cache_prefix = f'fusion_meta_{img_key}_{cfg.image_size}_{aug_suffix}_'
    search_cache_dir = os.path.join(cache_base, cache_prefix + rf_key)

    # Auto-clean old fusion caches with different RF params to save disk space
    import glob as _glob
    for old_cache in _glob.glob(os.path.join(cache_base, cache_prefix + 'rf*')):
        if old_cache != search_cache_dir and os.path.isdir(old_cache):
            import shutil
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
    _pcfg.USE_GENERAL_AUGMENTATION = original_aug
    _ds_utils.USE_GENERAL_AUGMENTATION = original_aug
    _pcfg.RF_N_ESTIMATORS = orig_rf_n_estimators
    _pcfg.RF_MAX_DEPTH = orig_rf_max_depth
    _pcfg.RF_MIN_SAMPLES_LEAF = orig_rf_min_samples_leaf
    _pcfg.RF_MIN_SAMPLES_SPLIT = orig_rf_min_samples_split
    _pcfg.RF_FEATURE_SELECTION_K = orig_rf_feature_selection_k

    def remove_sample_id(features, labels):
        return {k: v for k, v in features.items() if k != 'sample_id'}, labels

    train_ds_clean = train_ds.map(remove_sample_id, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds_clean = valid_ds.map(remove_sample_id, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds_clean, valid_ds_clean, pre_aug_ds, steps_per_epoch, val_steps, alpha_values



# ─────────────────────────────────────────────────────────────────────
# Training logic
# ─────────────────────────────────────────────────────────────────────

def train_single_config(cfg: FusionSearchConfig, data, fold_idx=0,
                        force_retrain=False):
    """Train a single fusion configuration and return metrics.

    Args:
        force_retrain: If True, always re-train image branches even if standalone
            weights are available. If False (default), load standalone weights when found.
    """
    modalities_str = '+'.join(['metadata'] + cfg.image_modalities)
    print(f"\n{'='*80}")
    print(f"CONFIG [FUSION {modalities_str}]: {cfg.name}")
    print(f"  fusion_strategy={cfg.fusion_strategy}, head={cfg.fusion_head_units}, "
          f"head_drop={cfg.fusion_head_dropout}, head_bn={cfg.fusion_head_bn}")
    print(f"  image_temperature={cfg.image_temperature}, gate_hidden_dim={cfg.gate_hidden_dim}")
    print(f"  stage1_lr={cfg.stage1_lr}, stage1_epochs={cfg.stage1_epochs}")
    print(f"  stage2_lr={cfg.stage2_lr}, stage2_epochs={cfg.stage2_epochs}, unfreeze_pct={cfg.stage2_unfreeze_pct}")
    print(f"  meta_scale=[{cfg.meta_min_scale}, {cfg.meta_max_scale}]")
    print(f"  fusion_query_dim={cfg.fusion_query_dim}, fusion_query_l2={cfg.fusion_query_l2}")
    print(f"  meta_query_scale={cfg.meta_query_scale}, image_query_scale={cfg.image_query_scale}")
    print(f"  rf: n_est={cfg.rf_n_estimators}, max_depth={cfg.rf_max_depth}, "
          f"leaf={cfg.rf_min_samples_leaf}, split={cfg.rf_min_samples_split}, feat_k={cfg.rf_feature_selection_k}")
    print(f"  loss={cfg.loss_type}, gamma={cfg.focal_gamma}, alpha_sum={cfg.alpha_sum}")
    print(f"  batch={cfg.batch_size}, lr_schedule={cfg.lr_schedule}, warmup={cfg.warmup_epochs}")
    print(f"  aug={cfg.use_augmentation}, mixup={cfg.use_mixup}({cfg.mixup_alpha}), img_size={cfg.image_size}")
    print(f"  fold={fold_idx}")
    print(f"{'='*80}")

    start_time = time.time()

    train_ds, valid_ds, pre_aug_ds, steps_per_epoch, val_steps, alpha_values = \
        prepare_datasets(data, cfg, fold_idx=fold_idx)

    if cfg.alpha_sum == 0:
        alpha_list = [1.0, 1.0, 1.0]
    else:
        alpha_arr = np.array(alpha_values)
        alpha_arr = alpha_arr / alpha_arr.sum() * cfg.alpha_sum
        alpha_list = alpha_arr.tolist()
    print(f"  Alpha values (sum={cfg.alpha_sum}): {[round(a, 3) for a in alpha_list]}")

    # Infer input shapes from dataset
    for batch in train_ds.take(1):
        batch_inputs, batch_labels = batch
        image_input_shapes = {
            mod: batch_inputs[f'{mod}_input'].shape[1:]
            for mod in cfg.image_modalities
        }
        metadata_input_shape = batch_inputs['metadata_input'].shape[1:]
        break

    strategy = _strategy

    # ── Step 0: Evaluate metadata (RF) branch before fusion ──
    # RF probabilities are pre-computed during data preparation. Evaluate by
    # taking argmax of the RF probs and comparing to true labels.
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
    print(f"\n  METADATA (RF) PRE-EVAL: kappa={meta_kappa:.4f}, acc={meta_acc:.4f}, "
          f"f1={meta_f1:.4f} (n={len(meta_y_true)})")

    # ── Step 1: Pre-train each image modality using standalone-matched params ──
    # Each image arm is pre-trained with the EXACT hyperparameters from its
    # standalone pipeline audit (hardcoded in _STANDALONE_CONFIGS).  This ensures
    # the pre-training kappa matches the standalone result (e.g., depth_rgb
    # fold 0 ≈ 0.1975, thermal_map fold 0 ≈ 0.2833).
    pretrained_weights = {}  # modality -> {layer_name: weights}

    for image_modality in cfg.image_modalities:
        cache_key = (image_modality, fold_idx)

        if cache_key in _pretrain_cache:
            # Reuse cached pre-trained weights (same fold = same data split)
            pretrained_weights[image_modality] = _pretrain_cache[cache_key]
            print(f"\n  PRE-TRAINING: {image_modality} — using cached weights (fold {fold_idx})")
            continue

        # Resolve pre-training hyperparameters from standalone config
        sa_cfg = _STANDALONE_CONFIGS[image_modality]

        # Check for saved standalone weights (from standalone audit scripts)
        # The standalone saves BASELINE weights per fold as "BASELINE_fold{n}"
        # We look for BASELINE weights first since that's the default standalone config.
        if not force_retrain:
            sa_weight_path = find_standalone_weights(image_modality, 'BASELINE', fold_idx)
            if sa_weight_path is not None:
                print(f"\n  PRE-TRAINING: {image_modality} — loading standalone weights")
                print(f"    Source: {sa_weight_path}")
                with strategy.scope():
                    sa_model, sa_base = build_pretrain_model(sa_cfg, image_modality)
                loaded = load_model_weights(sa_model, sa_weight_path)
                if loaded:
                    pretrained_weights[image_modality] = {}
                    for layer in sa_model.layers:
                        if hasattr(layer, 'layers'):  # Sub-model (backbone)
                            pretrained_weights[image_modality]['__backbone__'] = layer.get_weights()
                            break
                    _pretrain_cache[cache_key] = pretrained_weights[image_modality]
                    del sa_model, sa_base
                    gc.collect()
                    tf.keras.backend.clear_session()
                    continue
                else:
                    print(f"    Failed to load, will re-train")
                    del sa_model, sa_base
                    gc.collect()
                    tf.keras.backend.clear_session()
        pt_lr            = sa_cfg['learning_rate']
        pt_epochs        = sa_cfg['stage1_epochs']
        pt_es_patience   = sa_cfg['early_stop_patience']
        pt_rlr_patience  = sa_cfg['reduce_lr_patience']
        pt_gamma         = sa_cfg['focal_gamma']
        pt_label_smooth  = sa_cfg['label_smoothing']
        pt_alpha_sum     = sa_cfg['alpha_sum']
        pt_ft_epochs     = sa_cfg['finetune_epochs']
        pt_ft_lr         = sa_cfg['finetune_lr']
        pt_unfreeze_pct  = sa_cfg['unfreeze_pct']
        pt_optimizer     = sa_cfg['optimizer']
        pt_weight_decay  = sa_cfg['weight_decay']
        pt_use_aug       = sa_cfg['use_augmentation']
        pt_use_mixup     = sa_cfg['use_mixup']
        pt_mixup_alpha   = sa_cfg['mixup_alpha']
        pt_image_size    = sa_cfg['image_size']
        pt_backbone      = sa_cfg['backbone']
        pt_lr_schedule   = sa_cfg.get('lr_schedule', 'plateau')
        pt_warmup_epochs = sa_cfg.get('warmup_epochs', 0)

        print(f"\n  PRE-TRAINING: {image_modality} (standalone config)")
        print(f"    backbone={pt_backbone}, head={sa_cfg['head_units']}, "
              f"lr={pt_lr}, epochs={pt_epochs}+{pt_ft_epochs}ft")
        print(f"    aug={pt_use_aug}, mixup={pt_use_mixup}({pt_mixup_alpha}), "
              f"img_size={pt_image_size}, schedule={pt_lr_schedule}, warmup={pt_warmup_epochs}")
        if pt_ft_epochs > 0:
            print(f"    Stage 2: finetune_epochs={pt_ft_epochs}, finetune_lr={pt_ft_lr}, "
                  f"unfreeze_pct={pt_unfreeze_pct}")

        # Build standalone dataset for just this image modality (same patient split)
        from src.data.dataset_utils import create_patient_folds as _cpf, prepare_cached_datasets as _pcd
        from src.data.generative_augmentation_v3 import AugmentationConfig as _AugCfg
        from src.utils.config import get_project_paths as _gpp
        import src.utils.production_config as _pcfg
        import src.data.dataset_utils as _ds_utils

        pt_patient_folds = _cpf(data, n_folds=cfg.n_folds, random_state=42)
        pt_train_patients, pt_valid_patients = pt_patient_folds[fold_idx]

        pt_aug_config = _AugCfg()
        pt_aug_config.generative_settings['output_size']['width'] = pt_image_size
        pt_aug_config.generative_settings['output_size']['height'] = pt_image_size

        # Override data pipeline globals to match standalone audit exactly
        _orig_aug = _pcfg.USE_GENERAL_AUGMENTATION
        _orig_rgb_bb = _pcfg.RGB_BACKBONE
        _orig_map_bb = _pcfg.MAP_BACKBONE
        _orig_ds_aug = _ds_utils.USE_GENERAL_AUGMENTATION
        _orig_ds_rgb_bb = _ds_utils.RGB_BACKBONE
        _orig_ds_map_bb = _ds_utils.MAP_BACKBONE

        if not pt_use_aug:
            _pcfg.USE_GENERAL_AUGMENTATION = False
            _ds_utils.USE_GENERAL_AUGMENTATION = False

        # Set backbone for data pipeline normalization — match standalone exactly.
        # Standalone scripts force 'EfficientNetB0' to keep [0,255] pipeline for
        # ALL pretrained backbones (EfficientNet has built-in Rescaling, others get
        # preprocess_input inside the model). We do the same here.
        is_rgb = (image_modality == 'depth_rgb')
        if is_rgb:
            _pcfg.RGB_BACKBONE = 'EfficientNetB0'
            _ds_utils.RGB_BACKBONE = 'EfficientNetB0'
        else:
            _pcfg.MAP_BACKBONE = 'EfficientNetB0'
            _ds_utils.MAP_BACKBONE = 'EfficientNetB0'

        _, pt_result_dir, _ = _gpp()
        pt_aug_suffix = 'aug' if pt_use_aug else 'noaug'
        # Use the same cache path as standalone scripts — ensures identical cached data.
        # Standalone uses: {modality}_{image_size}_pretrained_{aug_suffix}
        pt_cache_dir = os.path.join(pt_result_dir, 'search_cache',
            f'{image_modality}_{pt_image_size}_pretrained_{pt_aug_suffix}')

        pt_train_ds, _, pt_valid_ds, pt_steps, pt_val_steps, pt_alpha = _pcd(
            data,
            [image_modality],  # Single modality — matches standalone audit approach
            batch_size=sa_cfg['batch_size'],
            gen_manager=None,
            aug_config=pt_aug_config,
            run=fold_idx,
            image_size=pt_image_size,
            train_patients=pt_train_patients,
            valid_patients=pt_valid_patients,
            cache_dir=pt_cache_dir,
        )

        # Restore overridden globals
        _pcfg.USE_GENERAL_AUGMENTATION = _orig_aug
        _pcfg.RGB_BACKBONE = _orig_rgb_bb
        _pcfg.MAP_BACKBONE = _orig_map_bb
        _ds_utils.USE_GENERAL_AUGMENTATION = _orig_ds_aug
        _ds_utils.RGB_BACKBONE = _orig_ds_rgb_bb
        _ds_utils.MAP_BACKBONE = _orig_ds_map_bb

        # Compute alpha values for this pre-training (same as standalone)
        if pt_alpha_sum == 0:
            pt_alpha_list = [1.0, 1.0, 1.0]
        else:
            pt_alpha_arr = np.array(pt_alpha)
            pt_alpha_arr = pt_alpha_arr / pt_alpha_arr.sum() * pt_alpha_sum
            pt_alpha_list = pt_alpha_arr.tolist()
        print(f"    Alpha values (sum={pt_alpha_sum}): {[round(a, 3) for a in pt_alpha_list]}")

        # Remove sample_id and extract the single image tensor for pre-training.
        # The model Input is named '{modality}_input' and the dataset produces
        # {'sample_id': ..., '{modality}_input': tensor}. Extracting the tensor
        # directly avoids Keras warnings about input structure mismatch.
        _pt_input_key = f'{image_modality}_input'
        def _extract_image(features, labels):
            return features[_pt_input_key], labels
        pt_train_ds = pt_train_ds.map(_extract_image, num_parallel_calls=tf.data.AUTOTUNE)
        pt_valid_ds = pt_valid_ds.map(_extract_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply mixup if standalone config uses it
        if pt_use_mixup:
            _mixup_a = pt_mixup_alpha
            def _pt_mixup(img, labels):
                batch_size = tf.shape(img)[0]
                u1 = tf.random.uniform([], 0.0, 1.0)
                u2 = tf.random.uniform([], 0.0, 1.0)
                lam = tf.maximum(u1, u2) if _mixup_a <= 0.3 else (u1 + u2) / 2.0
                indices = tf.random.shuffle(tf.range(batch_size))
                mixed_img = lam * img + (1.0 - lam) * tf.gather(img, indices)
                mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
                return mixed_img, mixed_labels
            pt_train_ds = pt_train_ds.map(
                _pt_mixup, num_parallel_calls=tf.data.AUTOTUNE
            )

        # Reset TF random state to match standalone's per-config fresh start.
        # Standalone runs clear_session between configs; here we do the same
        # before each modality pre-training to get identical initialization.
        import random as _random
        _pt_seed = 42 + fold_idx * (fold_idx + 3)
        tf.random.set_seed(_pt_seed)
        np.random.seed(_pt_seed)
        _random.seed(_pt_seed)

        with strategy.scope():
            pretrain_model, pt_base_model = build_pretrain_model(sa_cfg, image_modality)

            # Freeze backbone — only train head + classifier (matches standalone)
            pt_base_model.trainable = False

            pt_trainable = len(pretrain_model.trainable_weights)
            pt_total = len(pretrain_model.weights)
            print(f"    Trainable weights: {pt_trainable}/{pt_total} (backbone frozen)")

            pt_loss = make_focal_loss(gamma=pt_gamma, alpha=pt_alpha_list,
                                      label_smoothing=pt_label_smooth)
            if pt_optimizer == 'adamw':
                try:
                    pt_opt = tf.keras.optimizers.AdamW(
                        learning_rate=pt_lr, weight_decay=pt_weight_decay, clipnorm=1.0)
                except (AttributeError, TypeError):
                    pt_opt = Adam(learning_rate=pt_lr, clipnorm=1.0)
            else:
                pt_opt = Adam(learning_rate=pt_lr, clipnorm=1.0)
            pretrain_model.compile(
                optimizer=pt_opt,
                loss=pt_loss,
                metrics=['accuracy', CohenKappaMetric(num_classes=3)]
            )

        pt_train_dis = strategy.experimental_distribute_dataset(pt_train_ds)
        pt_valid_dis = strategy.experimental_distribute_dataset(pt_valid_ds)

        # Stage 1: Head-only training (backbone frozen)
        pt_callbacks = [
            EarlyStopping(
                patience=pt_es_patience,
                restore_best_weights=True,
                monitor='val_cohen_kappa',
                min_delta=0.001,
                mode='max',
                verbose=1
            ),
        ]
        if pt_lr_schedule == 'cosine':
            pt_callbacks.append(CosineAnnealingSchedule(
                initial_lr=pt_lr,
                total_epochs=pt_epochs,
                warmup_epochs=pt_warmup_epochs,
                min_lr=1e-7
            ))
        else:
            pt_callbacks.append(ReduceLROnPlateau(
                factor=0.50,
                patience=pt_rlr_patience,
                monitor='val_cohen_kappa',
                min_delta=0.001,
                min_lr=1e-7,
                mode='max',
            ))

        pt_history = pretrain_model.fit(
            pt_train_dis,
            epochs=pt_epochs,
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

        # Save S1 weights before Stage 2 — restore if S2 doesn't improve
        s1_weights = pretrain_model.get_weights()

        # Stage 2: Fine-tuning (partial backbone unfreeze + lower LR)
        # Matches standalone exactly: base_model.trainable=True, freeze bottom layers,
        # and optionally freeze all BN layers (thermal_map standalone does this).
        if pt_ft_epochs > 0 and pt_base_model is not None:
            pt_base_model.trainable = True
            n_layers = len(pt_base_model.layers)
            freeze_until = int(n_layers * (1.0 - pt_unfreeze_pct))
            for sub_layer in pt_base_model.layers[:freeze_until]:
                sub_layer.trainable = False
            # Freeze BN layers if standalone config does so (e.g. thermal_map)
            if sa_cfg.get('freeze_bn_in_stage2', False):
                for sub_layer in pt_base_model.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = False
            unfrozen_count = n_layers - freeze_until
            bn_note = ", BN frozen" if sa_cfg.get('freeze_bn_in_stage2', False) else ""
            print(f"    Stage 2: unfreezing top {pt_unfreeze_pct*100:.0f}% "
                  f"({unfrozen_count}/{n_layers} layers{bn_note})")

            with strategy.scope():
                if pt_optimizer == 'adamw':
                    try:
                        s2_opt = tf.keras.optimizers.AdamW(
                            learning_rate=pt_ft_lr, weight_decay=pt_weight_decay, clipnorm=1.0)
                    except (AttributeError, TypeError):
                        s2_opt = Adam(learning_rate=pt_ft_lr, clipnorm=1.0)
                else:
                    s2_opt = Adam(learning_rate=pt_ft_lr, clipnorm=1.0)
                pretrain_model.compile(
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

            s2_history = pretrain_model.fit(
                pt_train_dis,
                epochs=pt_ft_epochs,
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

            # If Stage 2 didn't improve, restore Stage 1 weights
            if s2_best < pt_best:
                print(f"    Stage 2 did NOT improve (s2={s2_best:.4f} < s1={pt_best:.4f}). Restoring S1 weights.")
                pretrain_model.set_weights(s1_weights)

        # Post-eval: compute sklearn kappa on validation set for direct comparison
        pt_y_true = []
        pt_y_pred = []
        for batch in pt_valid_ds:
            batch_inputs, batch_labels = batch
            batch_pred = pretrain_model.predict(batch_inputs, verbose=0)
            pt_y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
            pt_y_pred.extend(np.argmax(batch_pred, axis=1))
        pt_y_true = np.array(pt_y_true)
        pt_y_pred = np.array(pt_y_pred)
        pt_post_kappa = cohen_kappa_score(pt_y_true, pt_y_pred, weights='quadratic')
        pt_post_acc = accuracy_score(pt_y_true, pt_y_pred)
        pt_post_f1 = f1_score(pt_y_true, pt_y_pred, average='macro', zero_division=0)
        print(f"    Pre-train POST-EVAL: kappa={pt_post_kappa:.4f}, acc={pt_post_acc:.4f}, "
              f"f1={pt_post_f1:.4f} (n={len(pt_y_true)})")

        # Save backbone weights for transfer to fusion model.
        # The fusion model (create_multimodal_model) uses different head layer names
        # than the standalone model, so we only transfer the backbone (EfficientNet)
        # weights which are the most important (fine-tuned in Stage 2).
        pretrained_weights[image_modality] = {}
        for layer in pretrain_model.layers:
            if hasattr(layer, 'layers'):  # Sub-model (EfficientNet backbone)
                pretrained_weights[image_modality]['__backbone__'] = layer.get_weights()
                break

        # Cache for other configs using the same fold
        _pretrain_cache[cache_key] = pretrained_weights[image_modality]

        del pretrain_model, pt_train_ds, pt_valid_ds, pt_train_dis, pt_valid_dis
        gc.collect()
        tf.keras.backend.clear_session()

    # ── Step 2: Build fusion model, transfer pretrained weights, freeze backbone ──
    with strategy.scope():
        model = build_fusion_model(cfg, image_input_shapes, metadata_input_shape)

        # Transfer pre-trained backbone weights into fusion model.
        # Match by modality name prefix in layer name (e.g. 'depth_rgb_efficientnetb0').
        total_transferred = 0
        fusion_backbones = {}
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:  # Sub-model (backbone)
                fusion_backbones[layer.name] = layer

        for image_modality in cfg.image_modalities:
            if image_modality not in pretrained_weights or '__backbone__' not in pretrained_weights[image_modality]:
                continue
            # Find the fusion backbone whose name starts with this modality
            matched_layer = None
            for name, layer in fusion_backbones.items():
                if name.startswith(image_modality):
                    matched_layer = layer
                    break
            if matched_layer is None:
                print(f"  Warning: no fusion backbone found for {image_modality} "
                      f"(available: {list(fusion_backbones.keys())})")
                continue
            try:
                matched_layer.set_weights(pretrained_weights[image_modality]['__backbone__'])
                total_transferred += 1
                print(f"  Transferred backbone weights for {image_modality} -> {matched_layer.name}")
            except Exception as e:
                print(f"  Warning: failed to transfer {image_modality} backbone ({matched_layer.name}): {e}")
        print(f"  Transferred {total_transferred} pre-trained backbones to fusion model")

        # Freeze backbone — only train fusion layers (matches main.py)
        # "Unfreezing causes BatchNorm stat disruption + overfitting with ~2K images"
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Sub-model (EfficientNet)
                layer.trainable = False

        trainable_count = len(model.trainable_weights)
        total_count = len(model.weights)
        print(f"  Trainable weights: {trainable_count}/{total_count} (backbone frozen, fusion layers trainable)")

        loss_fn = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.0,
                                          gamma=cfg.focal_gamma, alpha=alpha_list,
                                          label_smoothing=cfg.label_smoothing)

        model.compile(
            optimizer=Adam(learning_rate=cfg.stage1_lr, clipnorm=1.0),
            loss=loss_fn,
            metrics=['accuracy', CohenKappaMetric(num_classes=3)],
            jit_compile=True
        )

    if cfg.use_mixup:
        train_ds = train_ds.map(
            lambda f, l: apply_mixup(f, l, alpha=cfg.mixup_alpha),
            num_parallel_calls=tf.data.AUTOTUNE
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
    ]

    if cfg.lr_schedule == 'cosine':
        callbacks.append(CosineAnnealingSchedule(
            initial_lr=cfg.stage1_lr,
            total_epochs=cfg.stage1_epochs,
            warmup_epochs=cfg.warmup_epochs,
            min_lr=1e-7
        ))
    else:
        callbacks.append(ReduceLROnPlateau(
            factor=0.50,
            patience=cfg.reduce_lr_patience,
            monitor='val_cohen_kappa',
            min_delta=0.001,
            min_lr=1e-7,
            mode='max',
        ))

    # ── Step 3: Fusion training (frozen backbone) ──
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
    print(f"  Fusion best: val_kappa={best_s1_kappa:.4f} at epoch {best_s1_epoch}/{len(s1_kappas)}")

    # ── Step 4: Optional Stage 2 — partial backbone unfreeze + lower LR ──
    # Save S1 weights before Stage 2 — restore if S2 doesn't improve
    fusion_s1_weights = model.get_weights()
    best_s2_kappa = 0.0
    ran_stage2 = False
    if cfg.stage2_epochs > 0:
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Sub-model (backbone)
                layer.trainable = True
                n_layers = len(layer.layers)
                freeze_until = int(n_layers * (1.0 - cfg.stage2_unfreeze_pct))
                for sub_layer in layer.layers[:freeze_until]:
                    sub_layer.trainable = False
                # Keep BatchNorm frozen
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = False
                unfrozen = n_layers - freeze_until
                print(f"  Stage 2: unfreezing top {cfg.stage2_unfreeze_pct*100:.0f}% "
                      f"({unfrozen}/{n_layers} layers, BN frozen)")

        with strategy.scope():
            model.compile(
                optimizer=Adam(learning_rate=cfg.stage2_lr, clipnorm=1.0),
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
        best_s2_epoch = int(np.argmax(s2_kappas)) + 1 if s2_kappas else 0
        print(f"  Stage 2 best: val_kappa={best_s2_kappa:.4f} at epoch {best_s2_epoch}/{len(s2_kappas)}")
        ran_stage2 = True

        # If Stage 2 didn't improve, restore Stage 1 weights
        if best_s2_kappa < best_s1_kappa:
            print(f"  Fusion Stage 2 did NOT improve (s2={best_s2_kappa:.4f} < s1={best_s1_kappa:.4f}). Restoring S1 weights.")
            model.set_weights(fusion_s1_weights)

    # Post-training evaluation
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
    cm = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1, 2])

    elapsed = time.time() - start_time
    print(f"\n  POST-EVAL: kappa={kappa:.4f}, acc={accuracy:.4f}, f1={f1_macro:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  Time: {elapsed:.0f}s")

    result = {
        'name': cfg.name,
        'image_modalities': '+'.join(cfg.image_modalities),
        'fusion_strategy': cfg.fusion_strategy,
        'fusion_head_units': str(cfg.fusion_head_units),
        'fusion_head_dropout': cfg.fusion_head_dropout,
        'fusion_head_bn': cfg.fusion_head_bn,
        'image_temperature': cfg.image_temperature,
        'gate_hidden_dim': cfg.gate_hidden_dim,
        'stage1_lr': cfg.stage1_lr,
        'stage1_epochs': cfg.stage1_epochs,
        'stage2_lr': cfg.stage2_lr,
        'stage2_epochs': cfg.stage2_epochs if ran_stage2 else 0,
        'stage2_unfreeze_pct': cfg.stage2_unfreeze_pct,
        'meta_min_scale': cfg.meta_min_scale,
        'meta_max_scale': cfg.meta_max_scale,
        'fusion_query_dim': cfg.fusion_query_dim,
        'fusion_query_l2': cfg.fusion_query_l2,
        'meta_query_scale': cfg.meta_query_scale,
        'image_query_scale': cfg.image_query_scale,
        'loss_type': cfg.loss_type,
        'focal_gamma': cfg.focal_gamma,
        'alpha_sum': cfg.alpha_sum,
        'label_smoothing': cfg.label_smoothing,
        'rf_n_estimators': cfg.rf_n_estimators,
        'rf_max_depth': cfg.rf_max_depth,
        'rf_min_samples_leaf': cfg.rf_min_samples_leaf,
        'rf_min_samples_split': cfg.rf_min_samples_split,
        'rf_feature_selection_k': cfg.rf_feature_selection_k,
        'batch_size': cfg.batch_size,
        'lr_schedule': cfg.lr_schedule,
        'warmup_epochs': cfg.warmup_epochs,
        'use_augmentation': cfg.use_augmentation,
        'use_mixup': cfg.use_mixup,
        'mixup_alpha': cfg.mixup_alpha,
        'image_size': cfg.image_size,
        'fold': fold_idx,
        'pretrained_loaded': True,
        'trainable_weights': trainable_count,
        'total_weights': total_count,
        'best_s1_kappa': best_s1_kappa,
        'best_s1_epoch': best_s1_epoch,
        'best_s2_kappa': best_s2_kappa if ran_stage2 else -1,
        'post_eval_kappa': kappa,
        'post_eval_acc': accuracy,
        'post_eval_f1': f1_macro,
        'elapsed_seconds': elapsed,
        'n_val_samples': len(y_true_v),
    }

    del model, train_ds, valid_ds, pre_aug_ds
    gc.collect()
    tf.keras.backend.clear_session()

    return result


# ─────────────────────────────────────────────────────────────────────
# Search rounds — fusion-specific parameters
# ─────────────────────────────────────────────────────────────────────

def round1_fusion_lr():
    """Round 1: Fusion learning rate (STAGE1_LR).

    LR for fusion-layer training (backbone frozen, pre-trained).
    Image branches pre-trained at PRETRAIN_LR=1e-3; fusion needs lower LR
    to learn cross-modal combination without destroying image features.
    """
    configs = []
    for lr in [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3]:
        name = f"R1_lr{lr:.0e}".replace('+', '')
        cfg = FusionSearchConfig(name=name, stage1_lr=lr)
        configs.append(cfg)
    return configs


def round2_fusion_epochs(best_r1: FusionSearchConfig):
    """Round 2: Fusion training epochs and early stopping patience.

    Main.py uses N_EPOCHS=200 with EARLY_STOP_PATIENCE=20.
    Test whether more/fewer epochs or different patience helps fusion.
    """
    configs = []
    for epochs, patience in [(100, 15), (200, 20), (200, 30), (300, 25)]:
        cfg = deepcopy(best_r1)
        cfg.name = f"R2_ep{epochs}_pat{patience}"
        cfg.stage1_epochs = epochs
        cfg.early_stop_patience = patience
        configs.append(cfg)
    return configs


def round3_fusion_strategy(best_r2: FusionSearchConfig):
    """Round 3: Fusion architecture strategy.

    Tests fundamentally different ways to combine modalities:
    - prob_concat: current baseline — fuse at probability level (21 params)
    - feature_concat: fuse at feature level (163 → Dense(3))
    - feature_concat_attn: feature-level + cross-modal attention
    - gated: learned per-class gate between RF and image probs
    - hybrid: all signals (RF probs + image features + image probs) combined
    """
    configs = []
    strategies = ['prob_concat', 'feature_concat', 'feature_concat_attn',
                  'gated', 'hybrid']
    for strat in strategies:
        cfg = deepcopy(best_r2)
        cfg.name = f"R3_{strat}"
        cfg.fusion_strategy = strat
        # Start with no hidden fusion head — let strategy speak for itself
        cfg.fusion_head_units = []
        configs.append(cfg)
    return configs


def round4_fusion_head(best_r3: FusionSearchConfig):
    """Round 4: Fusion head architecture.

    Tests different head depths/widths between the fusion concat and final Dense(3).
    Also tests dropout rate and BatchNorm within the head.
    """
    configs = []

    # Part A: Head size variants
    head_variants = [
        ('head_none',    []),
        ('head_32',      [32]),
        ('head_64',      [64]),
        ('head_64_32',   [64, 32]),
        ('head_128_64',  [128, 64]),
    ]
    for name, units in head_variants:
        cfg = deepcopy(best_r3)
        cfg.name = f"R4_{name}"
        cfg.fusion_head_units = units
        configs.append(cfg)

    # Part B: Dropout and BN variants (using a mid-size head)
    for drop, bn, label in [(0.1, True, 'drop01_bn'), (0.3, True, 'drop03_bn'),
                             (0.5, True, 'drop05_bn'), (0.3, False, 'drop03_nobn')]:
        cfg = deepcopy(best_r3)
        cfg.name = f"R4_{label}"
        cfg.fusion_head_units = [64]
        cfg.fusion_head_dropout = drop
        cfg.fusion_head_bn = bn
        configs.append(cfg)

    return configs


def round5_strategy_params(best_r4: FusionSearchConfig):
    """Round 5: Strategy-specific parameters.

    Searches parameters that are specific to the winning fusion strategy:
    - feature_concat_attn: attention query dim, L2, asymmetry
    - gated: gate hidden dim
    - prob_concat/hybrid/feature_concat: temperature scaling + metadata scaling
    """
    configs = []
    strat = best_r4.fusion_strategy

    if strat == 'feature_concat_attn':
        # Attention parameters
        attn_variants = [
            ('dim32_l2_0',       32,  0.0,   1.0, 1.0),   # Small, no reg, symmetric
            ('dim64_l2_1e3',     64,  0.001, 0.8, 1.5),   # Default
            ('dim64_l2_1e2',     64,  0.01,  0.8, 1.5),   # Stronger reg
            ('dim128_l2_1e3',    128, 0.001, 0.8, 1.5),   # Larger query
            ('dim64_strong_img', 64,  0.001, 0.5, 2.0),   # Images heavily guided by meta
            ('dim64_symmetric',  64,  0.001, 1.0, 1.0),   # No asymmetry
        ]
        for name, dim, l2, mqs, iqs in attn_variants:
            cfg = deepcopy(best_r4)
            cfg.name = f"R5_{name}"
            cfg.fusion_query_dim = dim
            cfg.fusion_query_l2 = l2
            cfg.meta_query_scale = mqs
            cfg.image_query_scale = iqs
            configs.append(cfg)

    elif strat == 'gated':
        # Gate hidden dim
        for hidden in [16, 32, 64, 128]:
            cfg = deepcopy(best_r4)
            cfg.name = f"R5_gate{hidden}"
            cfg.gate_hidden_dim = hidden
            configs.append(cfg)
        # Also test temperature for the image probs inside the gate
        for temp in [0.5, 1.0, 2.0]:
            cfg = deepcopy(best_r4)
            cfg.name = f"R5_temp{temp}"
            cfg.image_temperature = temp
            configs.append(cfg)

    else:
        # prob_concat, feature_concat, hybrid: temperature + metadata scaling
        # Temperature scaling
        for temp in [0.5, 0.75, 1.0, 1.5, 2.0]:
            cfg = deepcopy(best_r4)
            cfg.name = f"R5_temp{temp}"
            cfg.image_temperature = temp
            configs.append(cfg)
        # Metadata confidence scaling
        for name, min_s, max_s in [('no_scale', 1.0, 1.0), ('mild', 1.0, 2.0),
                                    ('default', 1.5, 3.0), ('strong', 2.0, 4.0)]:
            cfg = deepcopy(best_r4)
            cfg.name = f"R5_meta_{name}"
            cfg.meta_min_scale = min_s
            cfg.meta_max_scale = max_s
            configs.append(cfg)

    return configs


def round6_rf_params(best_r5: FusionSearchConfig):
    """Round 6: RF hyperparameters — these directly affect metadata quality."""
    configs = []
    variants = [
        ('rf100_d6',     100,  6,  5, 10, 40),
        ('rf200_d8',     200,  8,  5, 10, 40),   # Current default
        ('rf300_d10',    300,  10, 5, 10, 40),
        ('rf200_d12',    200,  12, 3,  5, 40),   # Deeper, less regularized
        ('rf200_d8_k20', 200,  8,  5, 10, 20),   # Fewer selected features
        ('rf200_d8_k60', 200,  8,  5, 10, 60),   # More selected features
    ]
    for name, n_est, depth, leaf, split, k in variants:
        cfg = deepcopy(best_r5)
        cfg.name = f"R6_{name}"
        cfg.rf_n_estimators = n_est
        cfg.rf_max_depth = depth
        cfg.rf_min_samples_leaf = leaf
        cfg.rf_min_samples_split = split
        cfg.rf_feature_selection_k = k
        configs.append(cfg)
    return configs


def round7_stage2_finetune(best_r6: FusionSearchConfig):
    """Round 7: Stage 2 — partial backbone unfreeze + lower LR fine-tuning.

    After fusion training with frozen backbone, tests whether partially
    unfreezing the backbone and fine-tuning with a lower LR helps.
    Risk: BatchNorm stat disruption + overfitting on ~2K images.
    """
    configs = []
    variants = [
        ('no_ft',            0,   0.0, 0.0),      # No Stage 2 (default for fusion in main.py)
        ('ft_top10_50ep',    50,  0.1, 1e-5),     # Conservative: 10% unfreeze, STAGE2_FINETUNE_EPOCHS
        ('ft_top20_50ep',    50,  0.2, 1e-5),     # Standard: 20% unfreeze (STAGE2_UNFREEZE_PCT)
        ('ft_top20_50ep_lr', 50,  0.2, 5e-6),     # Lower LR variant
    ]
    for name, epochs, pct, lr in variants:
        cfg = deepcopy(best_r6)
        cfg.name = f"R7_{name}"
        cfg.stage2_epochs = epochs
        cfg.stage2_unfreeze_pct = pct
        cfg.stage2_lr = lr
        configs.append(cfg)
    return configs


# ─────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────

def save_results(results: List[Dict], filepath: str):
    """Append results to CSV file."""
    if not results:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)


def pick_best(results: List[Dict], metric='post_eval_kappa') -> Tuple[Dict, FusionSearchConfig]:
    """Pick the best result and reconstruct its config."""
    best = max(results, key=lambda r: r[metric])
    print(f"\n  BEST from this round: {best['name']} "
          f"(kappa={best['post_eval_kappa']:.4f}, s1_kappa={best['best_s1_kappa']:.4f})")

    # Reconstruct fusion_head_units from string representation if needed
    raw_head_units = best.get('fusion_head_units', '[]')
    if isinstance(raw_head_units, str):
        import ast
        fusion_head_units = ast.literal_eval(raw_head_units)
    else:
        fusion_head_units = raw_head_units if raw_head_units else []

    cfg = FusionSearchConfig(
        name=best['name'],
        fusion_strategy=best.get('fusion_strategy', 'prob_concat'),
        fusion_head_units=fusion_head_units,
        fusion_head_dropout=float(best.get('fusion_head_dropout', 0.3)),
        fusion_head_bn=best.get('fusion_head_bn', True),
        image_temperature=float(best.get('image_temperature', 1.0)),
        gate_hidden_dim=int(best.get('gate_hidden_dim', 32)),
        stage1_lr=float(best['stage1_lr']),
        stage1_epochs=int(best['stage1_epochs']),
        stage2_lr=float(best.get('stage2_lr', 1e-5)),
        stage2_epochs=int(best.get('stage2_epochs', 0)),
        stage2_unfreeze_pct=float(best.get('stage2_unfreeze_pct', 0.2)),
        meta_min_scale=float(best.get('meta_min_scale', 1.5)),
        meta_max_scale=float(best.get('meta_max_scale', 3.0)),
        fusion_query_dim=int(best.get('fusion_query_dim', 64)),
        fusion_query_l2=float(best.get('fusion_query_l2', 0.001)),
        meta_query_scale=float(best.get('meta_query_scale', 0.8)),
        image_query_scale=float(best.get('image_query_scale', 1.5)),
        loss_type=best.get('loss_type', 'focal'),
        focal_gamma=float(best.get('focal_gamma', 2.0)),
        alpha_sum=float(best.get('alpha_sum', 3.0)),
        label_smoothing=float(best.get('label_smoothing', 0.0)),
        rf_n_estimators=int(best.get('rf_n_estimators', 200)),
        rf_max_depth=int(best.get('rf_max_depth', 8)),
        rf_min_samples_leaf=int(best.get('rf_min_samples_leaf', 5)),
        rf_min_samples_split=int(best.get('rf_min_samples_split', 10)),
        rf_feature_selection_k=int(best.get('rf_feature_selection_k', 40)),
        batch_size=int(best.get('batch_size', 64)),
        lr_schedule=best.get('lr_schedule', 'plateau'),
        warmup_epochs=int(best.get('warmup_epochs', 0)),
        use_augmentation=best.get('use_augmentation', True),
        use_mixup=best.get('use_mixup', False),
        mixup_alpha=float(best.get('mixup_alpha', 0.2)),
        image_size=int(best.get('image_size', 256)),
    )

    return best, cfg


def pick_topk(results: List[Dict], k: int = 3,
              metric='post_eval_kappa') -> List[Tuple[Dict, FusionSearchConfig]]:
    """Pick the top-K results by metric, de-duplicating by stage1_lr.

    Returns a list of (result_dict, FusionSearchConfig) tuples, sorted by metric descending.
    """
    sorted_results = sorted(results, key=lambda r: r[metric], reverse=True)

    seen = set()
    topk = []
    for r in sorted_results:
        key = float(r['stage1_lr'])
        if key not in seen:
            seen.add(key)
            _, cfg = pick_best([r])
            topk.append((r, cfg))
            if len(topk) == k:
                break

    print(f"\n  TOP-{k} R1 BRANCHES selected:")
    for i, (r, cfg) in enumerate(topk):
        print(f"    Branch {i+1}: {r['name']} -> kappa={r[metric]:.4f} "
              f"(stage1_lr={cfg.stage1_lr})")

    return topk


def load_completed_results(filepath: str) -> List[Dict]:
    """Load previously completed results from CSV for resume support."""
    if not os.path.exists(filepath):
        return []

    results = []
    float_keys = ['stage1_lr', 'stage2_lr', 'stage2_unfreeze_pct',
                   'meta_min_scale', 'meta_max_scale', 'fusion_query_l2',
                   'meta_query_scale', 'image_query_scale',
                   'focal_gamma', 'alpha_sum', 'label_smoothing', 'mixup_alpha',
                   'fusion_head_dropout', 'image_temperature',
                   'best_s1_kappa', 'best_s2_kappa', 'post_eval_kappa',
                   'post_eval_acc', 'post_eval_f1', 'elapsed_seconds']
    int_keys = ['stage1_epochs', 'stage2_epochs', 'fusion_query_dim',
                'gate_hidden_dim',
                'rf_n_estimators', 'rf_max_depth', 'rf_min_samples_leaf',
                'rf_min_samples_split', 'rf_feature_selection_k',
                'batch_size', 'warmup_epochs', 'image_size',
                'fold', 'trainable_weights', 'total_weights',
                'best_s1_epoch', 'n_val_samples']
    bool_keys = ['use_augmentation', 'use_mixup', 'pretrained_loaded', 'fusion_head_bn']
    # Note: fusion_strategy and fusion_head_units are strings — no conversion needed

    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in float_keys:
                if key in row and row[key]:
                    row[key] = float(row[key])
            for key in int_keys:
                if key in row and row[key]:
                    row[key] = int(row[key])
            for key in bool_keys:
                if key in row and row[key]:
                    row[key] = row[key] in ('True', 'true', '1')
            results.append(row)

    return results


def run_round(round_name: str, configs: List[FusionSearchConfig], data,
              results_csv: str, completed: List[Dict],
              force_retrain=False) -> List[Dict]:
    """Run a round of configs, skipping any already completed."""
    completed_names = {r['name'] for r in completed}
    round_results = []

    for cfg in configs:
        if cfg.name in completed_names:
            cached = [r for r in completed if r['name'] == cfg.name][0]
            print(f"\n  SKIP (cached): {cfg.name} -> kappa={cached['post_eval_kappa']:.4f}")
            round_results.append(cached)
        else:
            result = train_single_config(cfg, data, fold_idx=cfg.fold,
                                          force_retrain=force_retrain)
            round_results.append(result)
            save_results([result], results_csv)

    return round_results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(fresh: bool = False, force_retrain: bool = False):
    image_modalities = ["depth_rgb", "thermal_map"]
    modalities_str = ' + '.join(['metadata'] + image_modalities)
    print("=" * 80)
    print(f"FUSION HYPERPARAMETER SEARCH ({modalities_str})")
    print(f"  Strategy: Top-3 R1 carry-forward with independent R2-R7 branches")
    print("=" * 80)

    data, directory, result_dir = load_data(image_modalities)
    results_csv = os.path.join(AUDIT_DIR, 'fusion_search_results.csv')

    completed = []
    if fresh:
        if os.path.exists(results_csv):
            os.rename(results_csv, results_csv + '.bak')
            print(f"FRESH START — backed up old results to {results_csv}.bak")
    else:
        completed = load_completed_results(results_csv)
        if completed:
            completed_names = [r['name'] for r in completed]
            print(f"RESUME MODE — found {len(completed)} completed configs: {completed_names}")
        else:
            print("RESUME MODE — no previous results found, starting from scratch")

    all_results = []
    R1_TOP_K = 3  # Number of R1 branches to carry forward

    # ─── Round 1: Fusion Learning Rate ───
    print(f"\n{'#'*80}")
    print("ROUND 1: FUSION LEARNING RATE (STAGE1_LR)")
    print(f"{'#'*80}")

    r1_configs = round1_fusion_lr()
    r1_results = run_round("R1", r1_configs, data, results_csv, completed,
                               force_retrain=force_retrain)
    all_results.extend(r1_results)

    # Pick top-K R1 winners (de-duplicated by stage1_lr)
    r1_topk = pick_topk(r1_results, k=R1_TOP_K)

    # ─── Branches: Run R2-R7 independently for each R1 winner ───
    branch_results = {}  # branch_tag -> {r1_cfg, label, r1_kappa, all_results, round_results}
    for branch_idx, (r1_result, r1_cfg) in enumerate(r1_topk):
        branch_tag = f"B{branch_idx+1}"
        branch_label = f"lr{r1_cfg.stage1_lr:.0e}"

        print(f"\n{'#'*80}")
        print(f"BRANCH {branch_tag}: {branch_label} (R1 kappa={r1_result['post_eval_kappa']:.4f})")
        print(f"{'#'*80}")

        branch_all = []
        branch_rounds = {}

        # ─── R2: Fusion Training Epochs ───
        print(f"\n{'='*60}")
        print(f"  {branch_tag} — ROUND 2: FUSION TRAINING EPOCHS")
        print(f"{'='*60}")
        r2_configs = round2_fusion_epochs(r1_cfg)
        for c in r2_configs:
            c.name = f"{branch_tag}_{c.name}"
        r2_results = run_round(f"{branch_tag}_R2", r2_configs, data, results_csv, completed,
                                   force_retrain=force_retrain)
        branch_all.extend(r2_results)
        all_results.extend(r2_results)
        _, best_r2_cfg = pick_best(r2_results)
        branch_rounds['R2'] = r2_results

        # ─── R3: Fusion Strategy ───
        print(f"\n{'='*60}")
        print(f"  {branch_tag} — ROUND 3: FUSION STRATEGY")
        print(f"{'='*60}")
        r3_configs = round3_fusion_strategy(best_r2_cfg)
        for c in r3_configs:
            c.name = f"{branch_tag}_{c.name}"
        r3_results = run_round(f"{branch_tag}_R3", r3_configs, data, results_csv, completed,
                                   force_retrain=force_retrain)
        branch_all.extend(r3_results)
        all_results.extend(r3_results)
        _, best_r3_cfg = pick_best(r3_results)
        branch_rounds['R3'] = r3_results

        # ─── R4: Fusion Head Architecture ───
        print(f"\n{'='*60}")
        print(f"  {branch_tag} — ROUND 4: FUSION HEAD ARCHITECTURE")
        print(f"{'='*60}")
        r4_configs = round4_fusion_head(best_r3_cfg)
        for c in r4_configs:
            c.name = f"{branch_tag}_{c.name}"
        r4_results = run_round(f"{branch_tag}_R4", r4_configs, data, results_csv, completed,
                                   force_retrain=force_retrain)
        branch_all.extend(r4_results)
        all_results.extend(r4_results)
        _, best_r4_cfg = pick_best(r4_results)
        branch_rounds['R4'] = r4_results

        # ─── R5: Strategy-Specific Parameters ───
        print(f"\n{'='*60}")
        print(f"  {branch_tag} — ROUND 5: STRATEGY-SPECIFIC PARAMETERS")
        print(f"{'='*60}")
        r5_configs = round5_strategy_params(best_r4_cfg)
        for c in r5_configs:
            c.name = f"{branch_tag}_{c.name}"
        r5_results = run_round(f"{branch_tag}_R5", r5_configs, data, results_csv, completed,
                                   force_retrain=force_retrain)
        branch_all.extend(r5_results)
        all_results.extend(r5_results)
        _, best_r5_cfg = pick_best(r5_results)
        branch_rounds['R5'] = r5_results

        # ─── R6: RF Hyperparameters ───
        print(f"\n{'='*60}")
        print(f"  {branch_tag} — ROUND 6: RF HYPERPARAMETERS")
        print(f"{'='*60}")
        r6_configs = round6_rf_params(best_r5_cfg)
        for c in r6_configs:
            c.name = f"{branch_tag}_{c.name}"
        r6_results = run_round(f"{branch_tag}_R6", r6_configs, data, results_csv, completed,
                                   force_retrain=force_retrain)
        branch_all.extend(r6_results)
        all_results.extend(r6_results)
        _, best_r6_cfg = pick_best(r6_results)
        branch_rounds['R6'] = r6_results

        # ─── R7: Stage 2 Fine-tuning in Fusion ───
        print(f"\n{'='*60}")
        print(f"  {branch_tag} — ROUND 7: STAGE 2 FINE-TUNING IN FUSION")
        print(f"{'='*60}")
        r7_configs = round7_stage2_finetune(best_r6_cfg)
        for c in r7_configs:
            c.name = f"{branch_tag}_{c.name}"
        r7_results = run_round(f"{branch_tag}_R7", r7_configs, data, results_csv, completed,
                                   force_retrain=force_retrain)
        branch_all.extend(r7_results)
        all_results.extend(r7_results)
        _, best_r7_cfg = pick_best(r7_results)
        branch_rounds['R7'] = r7_results

        branch_results[branch_tag] = {
            'r1_cfg': r1_cfg,
            'label': branch_label,
            'r1_kappa': r1_result['post_eval_kappa'],
            'all_results': branch_all,
            'round_results': branch_rounds,
        }

    # ─── Top 3 Selection: Pick top 3 configs across ALL branches ───
    print(f"\n{'#'*80}")
    print("TOP 3 SELECTION: 5-FOLD VALIDATION OF BEST CONFIGS (CROSS-BRANCH)")
    print(f"{'#'*80}")

    # Pool all results from R1 + all branches
    search_results = sorted(all_results, key=lambda r: r['post_eval_kappa'], reverse=True)

    # De-duplicate by config fingerprint
    seen_configs = set()
    top3_results = []
    for r in search_results:
        fingerprint = (
            r.get('fusion_strategy', 'prob_concat'),
            str(r.get('fusion_head_units', '[]')),
            float(r.get('fusion_head_dropout', 0.3)),
            r.get('fusion_head_bn', True),
            float(r.get('image_temperature', 1.0)),
            int(r.get('gate_hidden_dim', 32)),
            float(r['stage1_lr']), int(r['stage1_epochs']),
            float(r.get('meta_min_scale', 1.5)), float(r.get('meta_max_scale', 3.0)),
            int(r.get('fusion_query_dim', 64)), float(r.get('fusion_query_l2', 0.001)),
            float(r.get('meta_query_scale', 0.8)), float(r.get('image_query_scale', 1.5)),
            int(r.get('rf_n_estimators', 200)), int(r.get('rf_max_depth', 8)),
            int(r.get('rf_min_samples_leaf', 5)), int(r.get('rf_min_samples_split', 10)),
            int(r.get('rf_feature_selection_k', 40)),
            int(r.get('stage2_epochs', 0)), float(r.get('stage2_unfreeze_pct', 0.2)),
        )
        if fingerprint not in seen_configs:
            seen_configs.add(fingerprint)
            top3_results.append(r)
            print(f"  Selected TOP3 #{len(top3_results)}: {r['name']} -> kappa={r['post_eval_kappa']:.4f}")
        if len(top3_results) == 3:
            break

    print(f"\nTop 3 configs by fold-0 kappa (cross-branch):")
    for i, r in enumerate(top3_results):
        print(f"  {i+1}. {r['name']} -> kappa={r['post_eval_kappa']:.4f}")

    # Run each top-3 config on all 5 folds
    N_FINAL_FOLDS = 5
    top3_fold_results = {}
    for rank, r in enumerate(top3_results):
        _, base_cfg = pick_best([r])
        base_cfg.n_folds = N_FINAL_FOLDS
        tag = f"TOP3_{rank+1}"
        fold_configs = []
        for fold_idx in range(N_FINAL_FOLDS):
            cfg = deepcopy(base_cfg)
            cfg.name = f"{tag}_fold{fold_idx+1}"
            cfg.fold = fold_idx
            fold_configs.append(cfg)
        fold_results = run_round(tag, fold_configs, data, results_csv, completed,
                                  force_retrain=force_retrain)
        all_results.extend(fold_results)
        top3_fold_results[tag] = {
            'cfg': base_cfg,
            'orig_name': r['name'],
            'fold0_kappa': r['post_eval_kappa'],
            'fold_results': fold_results,
            'mean_kappa': np.mean([fr['post_eval_kappa'] for fr in fold_results]),
        }

    # ─── Baseline: 5-fold CV with default parameters ───
    print(f"\n{'#'*80}")
    print("BASELINE: 5-FOLD CV WITH DEFAULT PARAMETERS")
    print(f"{'#'*80}")

    baseline_base = FusionSearchConfig(name="BASELINE")
    baseline_base.n_folds = N_FINAL_FOLDS
    baseline_configs = []
    for fold_idx in range(N_FINAL_FOLDS):
        cfg = deepcopy(baseline_base)
        cfg.name = f"BASELINE_fold{fold_idx+1}"
        cfg.fold = fold_idx
        baseline_configs.append(cfg)

    baseline_results = run_round("BASELINE", baseline_configs, data, results_csv, completed,
                                     force_retrain=force_retrain)
    all_results.extend(baseline_results)

    # ─── Summary ───
    print(f"\n{'='*80}")
    print(f"FUSION SEARCH COMPLETE — SUMMARY")
    print(f"{'='*80}")

    # R1 results
    best_r1 = max(r1_results, key=lambda r: r['post_eval_kappa'])
    print(f"  R1: Fusion LR: {best_r1['name']} -> kappa={best_r1['post_eval_kappa']:.4f}")

    # Per-branch best from each round
    for btag, binfo in branch_results.items():
        print(f"\n  {btag} ({binfo['label']}, R1 kappa={binfo['r1_kappa']:.4f}):")
        for rname, rresults in binfo['round_results'].items():
            best = max(rresults, key=lambda r: r['post_eval_kappa'])
            print(f"    {rname}: {best['name']} -> kappa={best['post_eval_kappa']:.4f}")

    def print_config_summary(label, cfg, fold_results):
        print(f"\n{label}:")
        print(f"  fusion_strategy={cfg.fusion_strategy}, head={cfg.fusion_head_units}, "
              f"head_drop={cfg.fusion_head_dropout}, head_bn={cfg.fusion_head_bn}")
        print(f"  image_temperature={cfg.image_temperature}, gate_hidden_dim={cfg.gate_hidden_dim}")
        print(f"  stage1_lr={cfg.stage1_lr}, epochs={cfg.stage1_epochs}")
        print(f"  meta_scale=[{cfg.meta_min_scale}, {cfg.meta_max_scale}]")
        print(f"  fusion_query_dim={cfg.fusion_query_dim}, l2={cfg.fusion_query_l2}")
        print(f"  meta_q={cfg.meta_query_scale}, img_q={cfg.image_query_scale}")
        print(f"  rf: n={cfg.rf_n_estimators}, depth={cfg.rf_max_depth}, "
              f"leaf={cfg.rf_min_samples_leaf}, k={cfg.rf_feature_selection_k}")
        print(f"  stage2: epochs={cfg.stage2_epochs}, unfreeze={cfg.stage2_unfreeze_pct}")
        print(f"  loss={cfg.loss_type}, gamma={cfg.focal_gamma}, alpha_sum={cfg.alpha_sum}, "
              f"label_smoothing={cfg.label_smoothing}")

        fold_kappas = [r['post_eval_kappa'] for r in fold_results]
        fold_accs = [r['post_eval_acc'] for r in fold_results]
        fold_f1s = [r['post_eval_f1'] for r in fold_results]

        print(f"\n  {'Fold':<8} {'Kappa':>8} {'Accuracy':>10} {'F1 (macro)':>12}")
        print(f"  {'-'*40}")
        for i, r in enumerate(fold_results):
            print(f"  Fold {i+1:<3} {r['post_eval_kappa']:>8.4f} {r['post_eval_acc']:>10.4f} {r['post_eval_f1']:>12.4f}")
        print(f"  {'-'*40}")
        print(f"  {'Mean':<8} {np.mean(fold_kappas):>8.4f} {np.mean(fold_accs):>10.4f} {np.mean(fold_f1s):>12.4f}")
        print(f"  {'Std':<8} {np.std(fold_kappas):>8.4f} {np.std(fold_accs):>10.4f} {np.std(fold_f1s):>12.4f}")

        return fold_kappas, fold_accs, fold_f1s

    print(f"\n{'─'*60}")
    print("TOP 3 CONFIGS — 5-FOLD RESULTS")
    print(f"{'─'*60}")

    sorted_top3 = sorted(top3_fold_results.items(),
                         key=lambda x: x[1]['mean_kappa'], reverse=True)

    print(f"\n  {'Rank':<6} {'Config':<30} {'Fold0':>7} {'Mean+/-Std':>14}")
    print(f"  {'-'*60}")
    for rank, (tag, info) in enumerate(sorted_top3):
        fk = [r['post_eval_kappa'] for r in info['fold_results']]
        print(f"  {rank+1:<6} {info['orig_name']:<30} {info['fold0_kappa']:>7.4f} "
              f"{np.mean(fk):>6.4f}+/-{np.std(fk):.4f}")

    winner_tag, winner_info = sorted_top3[0]
    winner_cfg = winner_info['cfg']
    winner_fold_results = winner_info['fold_results']

    print(f"\n{'─'*60}")
    print("DETAILED RESULTS")
    print(f"{'─'*60}")

    for rank, (tag, info) in enumerate(sorted_top3):
        print_config_summary(
            f"#{rank+1}: {info['orig_name']} ({tag})",
            info['cfg'], info['fold_results'])

    print(f"\n{'─'*60}")
    baseline_cfg = FusionSearchConfig(name="BASELINE")
    baseline_cfg.n_folds = N_FINAL_FOLDS
    bl_kappas, bl_accs, bl_f1s = print_config_summary(
        "BASELINE (defaults)", baseline_cfg, baseline_results)

    print(f"\n{'─'*60}")
    print(f"STATISTICAL COMPARISON: #{1} {winner_info['orig_name']} vs BASELINE")
    print(f"{'─'*60}")

    best_kappas = [r['post_eval_kappa'] for r in winner_fold_results]
    best_accs = [r['post_eval_acc'] for r in winner_fold_results]
    best_f1s = [r['post_eval_f1'] for r in winner_fold_results]

    kappa_diff = np.mean(best_kappas) - np.mean(bl_kappas)
    acc_diff = np.mean(best_accs) - np.mean(bl_accs)
    f1_diff = np.mean(best_f1s) - np.mean(bl_f1s)
    print(f"  Mean Kappa diff:    {kappa_diff:+.4f}  ({np.mean(bl_kappas):.4f} -> {np.mean(best_kappas):.4f})")
    print(f"  Mean Accuracy diff: {acc_diff:+.4f}  ({np.mean(bl_accs):.4f} -> {np.mean(best_accs):.4f})")
    print(f"  Mean F1 diff:       {f1_diff:+.4f}  ({np.mean(bl_f1s):.4f} -> {np.mean(best_f1s):.4f})")

    from scipy import stats
    if len(best_kappas) == len(bl_kappas) and len(best_kappas) >= 2:
        t_stat, p_value = stats.ttest_rel(best_kappas, bl_kappas)
        print(f"\n  Paired t-test on kappa (n={len(best_kappas)} folds):")
        print(f"    t-statistic = {t_stat:.4f}")
        print(f"    p-value     = {p_value:.4f}")
        if p_value < 0.05:
            winner_label = "WINNER" if kappa_diff > 0 else "BASELINE"
            print(f"    -> Statistically significant (p < 0.05): {winner_label} is better")
        else:
            print(f"    -> NOT statistically significant (p >= 0.05)")

    print(f"\nResults saved to: {results_csv}")

    # Decide overall best: winner vs baseline (higher mean kappa wins)
    bl_mean_kappa = np.mean(bl_kappas)
    winner_mean_kappa = np.mean(best_kappas)
    if bl_mean_kappa >= winner_mean_kappa:
        final_cfg = baseline_cfg
        print(f"\n  BASELINE wins (kappa {bl_mean_kappa:.4f} >= {winner_mean_kappa:.4f})")
    else:
        final_cfg = winner_cfg
        print(f"\n  {winner_info['orig_name']} wins (kappa {winner_mean_kappa:.4f} > {bl_mean_kappa:.4f})")

    best_config_path = os.path.join(AUDIT_DIR, 'fusion_best_config.json')
    config_dict = asdict(final_cfg)
    with open(best_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Best config saved to: {best_config_path}")


if __name__ == '__main__':
    import datetime
    import argparse

    parser = argparse.ArgumentParser(description='Fusion Pipeline Hyperparameter Search (metadata + depth_rgb + thermal_map)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (back up old results). Default: resume from existing CSV.')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force re-training image branches even if standalone weights exist. '
                             'Default: load standalone weights when available.')
    args = parser.parse_args()

    log_dir = os.path.join(AUDIT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'fusion_hparam_search_{timestamp}.log')

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
        main(fresh=args.fresh, force_retrain=args.force_retrain)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to: {log_path}")
