#!/usr/bin/env python3
"""
Thermal Map Hyperparameter Search — Standalone Script

Systematically tests architecture and training configurations for thermal_map
standalone to maximize val_cohen_kappa. Uses sequential elimination:
  Round 1: Backbone + freeze strategy  (8 configs)
  Round 2: Head architecture            (5 configs)
  Round 3: Loss + regularization        (12 configs)
  Round 4: Training dynamics            (8 configs)
  Round 5: Augmentation + image size    (4 configs)
  Round 6: Fine-tuning strategy         (4 configs)

Map-specific adjustments vs depth_rgb search:
  - SimpleCNN uses 3 layers (128→64→32) instead of 4 — maps are simpler
  - Overrides MAP_BACKBONE (not RGB_BACKBONE) in data pipeline
  - Image sizes: 128 and 256 (maps have less fine detail than RGB)
  - Head R2 uses smaller two-layer [128, 32] (less feature complexity)
  - Augmentation uses spatial + mild sensor noise (no color augmentation)
  - Thermal bbox uses fixed +30px margin (vs FOV correction for depth_map)

Usage:
  # Fresh start (backs up old results):
  python agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py --fresh

  # Resume from where it left off (default):
  python agent_communication/thermal_map_pipeline_audit/thermal_map_hparam_search.py

Results written to: agent_communication/thermal_map_pipeline_audit/thermal_map_search_results.csv
"""

import os
import sys
import gc
import time
import json
import math
import random
import hashlib
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

MODALITY = 'thermal_map'
MODALITY_LABEL = 'THERMAL MAP'
AUDIT_DIR = os.path.join(PROJECT_ROOT, 'agent_communication', 'thermal_map_pipeline_audit')

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
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, GlobalAveragePooling2D,
    BatchNormalization, Dropout, MaxPooling2D
)
from tensorflow.keras.models import Model
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix

# Multi-GPU setup — must happen before any model/dataset creation
from src.utils.gpu_config import setup_device_strategy
_strategy, _selected_gpus = setup_device_strategy(mode='multi')

# ─────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """Single experiment configuration."""
    name: str = "baseline"

    # Backbone
    backbone: str = "EfficientNetB0"

    # Freeze strategy
    freeze: str = "frozen"  # frozen, partial_unfreeze, full_unfreeze

    # Head architecture
    head_units: List[int] = field(default_factory=lambda: [128])
    head_dropout: float = 0.3
    head_use_bn: bool = True
    head_l2: float = 0.0

    # Training
    learning_rate: float = 1e-3
    stage1_epochs: int = 50
    early_stop_patience: int = 15
    reduce_lr_patience: int = 7
    batch_size: int = 64
    lr_schedule: str = "plateau"  # plateau, cosine
    warmup_epochs: int = 0

    # Loss
    loss_type: str = "focal"  # focal, cce
    focal_gamma: float = 2.0
    alpha_sum: float = 3.0
    label_smoothing: float = 0.0

    # Augmentation
    use_augmentation: bool = True
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # Image
    image_size: int = 256

    # Optimizer
    optimizer: str = "adam"  # adam, adamw
    weight_decay: float = 1e-4

    # Fine-tuning
    finetune_lr: float = 1e-5
    finetune_epochs: int = 30
    unfreeze_pct: float = 0.2

    # Cross-validation
    n_folds: int = 5  # 5-fold throughout — matches fusion pipeline splits
    fold: int = 0


# ─────────────────────────────────────────────────────────────────────
# Model building
# ─────────────────────────────────────────────────────────────────────

def build_simple_cnn(input_shape):
    """3-layer CNN for map data — simpler than RGB (no color features)."""
    inp = Input(shape=input_shape, name=f'{MODALITY}_input')
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = GlobalAveragePooling2D()(x)
    return inp, x


def build_pretrained_backbone(input_shape, backbone_name):
    """Build pretrained backbone with ImageNet weights."""
    from tensorflow.keras.layers import Lambda

    inp = Input(shape=input_shape, name=f'{MODALITY}_input')

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

    kwargs = {'weights': 'imagenet', 'include_top': False, 'pooling': 'avg'}
    base_model = BackboneClass(**kwargs)

    if preprocess_fn is not None:
        x = Lambda(lambda img: preprocess_fn(img), name='backbone_preprocess')(inp)
        x = base_model(x)
    else:
        x = base_model(inp)

    return inp, x, base_model


def build_model(cfg: SearchConfig):
    """Build complete model from config."""
    input_shape = (cfg.image_size, cfg.image_size, 3)

    base_model = None
    if cfg.backbone == 'SimpleCNN':
        inp, features = build_simple_cnn(input_shape)
    else:
        inp, features, base_model = build_pretrained_backbone(input_shape, cfg.backbone)

    x = features
    regularizer = tf.keras.regularizers.l2(cfg.head_l2) if cfg.head_l2 > 0 else None
    for i, units in enumerate(cfg.head_units):
        x = Dense(units, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizer,
                  name=f'head_dense_{i}')(x)
        if cfg.head_use_bn:
            x = BatchNormalization(name=f'head_bn_{i}')(x)
        x = Dropout(cfg.head_dropout, name=f'head_drop_{i}')(x)

    output = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inp, outputs=output)

    return model, base_model


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
            num_classes = 3
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha_tensor * tf.math.pow(1.0 - y_pred, gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss


def make_cce_loss(alpha=None, label_smoothing=0.0):
    """Categorical crossentropy with optional alpha weighting and label smoothing."""
    if alpha is None:
        alpha = [1.0, 1.0, 1.0]
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def cce_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            num_classes = 3
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        loss = -y_true * tf.math.log(y_pred) * alpha_tensor
        return tf.reduce_sum(loss, axis=-1)

    return cce_loss


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
# LR schedules & augmentation callbacks
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
    lam = tf.random.uniform([], 0.0, 1.0)
    if alpha > 0:
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
# Data pipeline — reuses project's existing data loading
# ─────────────────────────────────────────────────────────────────────

def load_data():
    """Load the dataset using the project's standard prepare_dataset pipeline."""
    from src.utils.config import get_project_paths, get_data_paths
    from src.data.image_processing import prepare_dataset
    from src.utils.production_config import (
        USE_CORE_DATA, THRESHOLD_I, THRESHOLD_P, THRESHOLD_R
    )

    directory, result_dir, root = get_project_paths()
    data_paths = get_data_paths(root)

    csv_file = data_paths['csv_file']
    depth_bb_file = data_paths['bb_depth_csv']
    thermal_bb_file = data_paths['bb_thermal_csv']

    selected_modalities = [MODALITY]
    data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities)

    if USE_CORE_DATA and all(t is not None for t in [THRESHOLD_I, THRESHOLD_P, THRESHOLD_R]):
        try:
            from src.evaluation.metrics import filter_frequent_misclassifications
            thresholds = {'I': THRESHOLD_I, 'P': THRESHOLD_P, 'R': THRESHOLD_R}
            data = filter_frequent_misclassifications(data, result_dir, thresholds=thresholds)
        except Exception as e:
            print(f"  Warning: Core data filtering failed: {e}")

    print(f"Loaded {len(data)} samples for {MODALITY}")
    if 'Healing Phase Abs' in data.columns:
        class_dist = data['Healing Phase Abs'].value_counts().sort_index()
        print(f"  Class distribution: {dict(class_dist)}")
        print(f"  Unique patients: {data['Patient#'].nunique()}")
    return data, directory, result_dir


def prepare_datasets(data, cfg: SearchConfig, fold_idx=0):
    """Prepare train/valid datasets for a single fold."""
    from src.data.dataset_utils import create_patient_folds, prepare_cached_datasets
    from src.data.generative_augmentation_v3 import AugmentationConfig

    data = data.copy()

    patient_folds = create_patient_folds(data, n_folds=cfg.n_folds, random_state=42)
    train_patients, valid_patients = patient_folds[fold_idx]

    train_mask = data['Patient#'].isin(train_patients)
    valid_mask = data['Patient#'].isin(valid_patients)
    print(f"  [FOLD] n_folds={cfg.n_folds}, fold_idx={fold_idx}, "
          f"train={train_mask.sum()} samples ({len(train_patients)} patients), "
          f"valid={valid_mask.sum()} samples ({len(valid_patients)} patients)")

    aug_config = AugmentationConfig()
    aug_config.generative_settings['output_size']['width'] = cfg.image_size
    aug_config.generative_settings['output_size']['height'] = cfg.image_size

    selected_modalities = [MODALITY]

    # Temporarily override globals that affect the data pipeline
    import src.utils.production_config as _pcfg
    import src.data.dataset_utils as _ds_utils
    original_aug = _pcfg.USE_GENERAL_AUGMENTATION
    original_map_backbone = _pcfg.MAP_BACKBONE

    if not cfg.use_augmentation:
        _pcfg.USE_GENERAL_AUGMENTATION = False
        _ds_utils.USE_GENERAL_AUGMENTATION = False

    # Override MAP_BACKBONE to control normalization in data pipeline
    if cfg.backbone == 'SimpleCNN':
        _pcfg.MAP_BACKBONE = 'SimpleCNN'
        _ds_utils.MAP_BACKBONE = 'SimpleCNN'
    else:
        _pcfg.MAP_BACKBONE = 'EfficientNetB0'
        _ds_utils.MAP_BACKBONE = 'EfficientNetB0'

    # Cache dir under results/search_cache/ parent
    from src.utils.config import get_project_paths
    _, search_result_dir, _ = get_project_paths()
    aug_suffix = 'aug' if cfg.use_augmentation else 'noaug'
    norm_key = 'simplecnn' if cfg.backbone == 'SimpleCNN' else 'pretrained'
    search_cache_dir = os.path.join(search_result_dir, 'search_cache',
        f'{MODALITY}_{cfg.image_size}_{norm_key}_{aug_suffix}')

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

    print(f"  [CACHE] {search_cache_dir}")
    print(f"  [PIPELINE] steps={steps_per_epoch}, val_steps={val_steps}, "
          f"alpha={[round(a, 4) for a in alpha_values]}")

    # Restore overridden globals
    _pcfg.USE_GENERAL_AUGMENTATION = original_aug
    _pcfg.MAP_BACKBONE = original_map_backbone
    _ds_utils.USE_GENERAL_AUGMENTATION = original_aug
    _ds_utils.MAP_BACKBONE = original_map_backbone

    def remove_sample_id(features, labels):
        return {k: v for k, v in features.items() if k != 'sample_id'}, labels

    train_ds_clean = train_ds.map(remove_sample_id, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds_clean = valid_ds.map(remove_sample_id, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds_clean, valid_ds_clean, pre_aug_ds, steps_per_epoch, val_steps, alpha_values


# ─────────────────────────────────────────────────────────────────────
# Training logic
# ─────────────────────────────────────────────────────────────────────

def apply_freeze_strategy(model, base_model, cfg: SearchConfig):
    """Apply freeze strategy to the backbone."""
    if base_model is None:
        return

    if cfg.freeze == 'frozen':
        base_model.trainable = False
    elif cfg.freeze == 'partial_unfreeze':
        base_model.trainable = True
        n_layers = len(base_model.layers)
        freeze_until = int(n_layers * 0.8)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
    elif cfg.freeze == 'full_unfreeze':
        base_model.trainable = True
    else:
        raise ValueError(f"Unknown freeze strategy: {cfg.freeze}")


def train_single_config(cfg: SearchConfig, data, fold_idx=0):
    """Train a single configuration and return metrics."""
    print(f"\n{'='*80}")
    print(f"CONFIG [{MODALITY}]: {cfg.name}")
    print(f"  backbone={cfg.backbone}, freeze={cfg.freeze}")
    print(f"  head={cfg.head_units}, dropout={cfg.head_dropout}, bn={cfg.head_use_bn}, l2={cfg.head_l2}")
    print(f"  loss={cfg.loss_type}, gamma={cfg.focal_gamma}, alpha_sum={cfg.alpha_sum}")
    print(f"  lr={cfg.learning_rate}, epochs={cfg.stage1_epochs}, batch={cfg.batch_size}")
    print(f"  lr_schedule={cfg.lr_schedule}, warmup={cfg.warmup_epochs}")
    print(f"  optimizer={cfg.optimizer}, weight_decay={cfg.weight_decay}")
    print(f"  augmentation={cfg.use_augmentation}, mixup={cfg.use_mixup}({cfg.mixup_alpha})")
    print(f"  label_smooth={cfg.label_smoothing}, image_size={cfg.image_size}, fold={fold_idx}")
    print(f"  unfreeze_pct={cfg.unfreeze_pct}, finetune_lr={cfg.finetune_lr}, finetune_epochs={cfg.finetune_epochs}")
    print(f"  [CONFIG_DICT] {json.dumps(asdict(cfg), indent=None, default=str)}")
    print(f"{'='*80}")

    start_time = time.time()

    train_ds, valid_ds, pre_aug_ds, steps_per_epoch, val_steps, alpha_values = \
        prepare_datasets(data, cfg, fold_idx=fold_idx)

    if cfg.alpha_sum == 0:
        alpha_list = [1.0, 1.0, 1.0]
        print(f"  Alpha values (UNIFORM — no class weighting): {alpha_list}")
    else:
        alpha_arr = np.array(alpha_values)
        alpha_arr = alpha_arr / alpha_arr.sum() * cfg.alpha_sum
        alpha_list = alpha_arr.tolist()
        print(f"  Alpha values (sum={cfg.alpha_sum}): {[round(a, 3) for a in alpha_list]}")

    strategy = _strategy
    with strategy.scope():
        model, base_model = build_model(cfg)
        apply_freeze_strategy(model, base_model, cfg)

        trainable_count = len(model.trainable_weights)
        total_count = len(model.weights)
        print(f"  Trainable weights: {trainable_count}/{total_count}")

        if cfg.loss_type == 'focal':
            loss_fn = make_focal_loss(gamma=cfg.focal_gamma, alpha=alpha_list,
                                       label_smoothing=cfg.label_smoothing)
        elif cfg.loss_type == 'cce':
            loss_fn = make_cce_loss(alpha=alpha_list, label_smoothing=cfg.label_smoothing)
        else:
            raise ValueError(f"Unknown loss: {cfg.loss_type}")

        lr = cfg.learning_rate
        if cfg.freeze == 'full_unfreeze' and cfg.backbone != 'SimpleCNN':
            lr = cfg.finetune_lr
            print(f"  Using finetune LR={lr} (backbone fully trainable)")
        elif cfg.freeze == 'partial_unfreeze' and cfg.backbone != 'SimpleCNN':
            lr = cfg.finetune_lr * 10
            print(f"  Using partial-unfreeze LR={lr}")

        if cfg.optimizer == 'adamw':
            try:
                opt = tf.keras.optimizers.AdamW(
                    learning_rate=lr, weight_decay=cfg.weight_decay, clipnorm=1.0)
                print(f"  Using AdamW (weight_decay={cfg.weight_decay})")
            except (AttributeError, TypeError):
                print("  Warning: AdamW not available, falling back to Adam")
                opt = Adam(learning_rate=lr, clipnorm=1.0)
        else:
            opt = Adam(learning_rate=lr, clipnorm=1.0)

        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=['accuracy', CohenKappaMetric(num_classes=3)]
        )

    if cfg.use_mixup:
        mixup_alpha = cfg.mixup_alpha
        train_ds = train_ds.map(
            lambda f, l: apply_mixup(f, l, alpha=mixup_alpha),
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
            initial_lr=lr,
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
    print(f"  Stage 1 best: val_kappa={best_s1_kappa:.4f} at epoch {best_s1_epoch}/{len(s1_kappas)}")

    # Stage 2: Fine-tuning
    best_s2_kappa = 0.0
    ran_stage2 = False
    if cfg.freeze == 'frozen' and base_model is not None and cfg.finetune_epochs > 0:
        base_model.trainable = True
        n_layers = len(base_model.layers)
        freeze_until = int(n_layers * (1.0 - cfg.unfreeze_pct))
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        unfrozen_count = n_layers - freeze_until
        print(f"  Stage 2: unfreezing top {cfg.unfreeze_pct*100:.0f}% ({unfrozen_count}/{n_layers} layers, BN frozen)")

        with strategy.scope():
            if cfg.optimizer == 'adamw':
                try:
                    s2_opt = tf.keras.optimizers.AdamW(
                        learning_rate=cfg.finetune_lr, weight_decay=cfg.weight_decay, clipnorm=1.0)
                except (AttributeError, TypeError):
                    s2_opt = Adam(learning_rate=cfg.finetune_lr, clipnorm=1.0)
            else:
                s2_opt = Adam(learning_rate=cfg.finetune_lr, clipnorm=1.0)

            model.compile(
                optimizer=s2_opt,
                loss=loss_fn,
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

        s2_history = model.fit(
            train_ds_dis,
            epochs=cfg.finetune_epochs,
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
    from sklearn.metrics import classification_report
    print(f"  Per-class report:\n{classification_report(y_true_v, y_pred_v, target_names=['I','P','R'], labels=[0,1,2], zero_division=0)}")
    print(f"  Time: {elapsed:.0f}s")

    overall_best_kappa = max(best_s1_kappa, best_s2_kappa) if ran_stage2 else best_s1_kappa

    result = {
        'name': cfg.name,
        'backbone': cfg.backbone,
        'freeze': cfg.freeze,
        'head_units': str(cfg.head_units),
        'head_dropout': cfg.head_dropout,
        'head_use_bn': cfg.head_use_bn,
        'head_l2': cfg.head_l2,
        'loss_type': cfg.loss_type,
        'focal_gamma': cfg.focal_gamma,
        'alpha_sum': cfg.alpha_sum,
        'label_smoothing': cfg.label_smoothing,
        'learning_rate': cfg.learning_rate,
        'lr_schedule': cfg.lr_schedule,
        'warmup_epochs': cfg.warmup_epochs,
        'batch_size': cfg.batch_size,
        'optimizer': cfg.optimizer,
        'weight_decay': cfg.weight_decay,
        'use_augmentation': cfg.use_augmentation,
        'use_mixup': cfg.use_mixup,
        'mixup_alpha': cfg.mixup_alpha,
        'image_size': cfg.image_size,
        'stage1_epochs': cfg.stage1_epochs,
        'finetune_epochs': cfg.finetune_epochs if ran_stage2 else 0,
        'unfreeze_pct': cfg.unfreeze_pct,
        'fold': fold_idx,
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

    del model, base_model, train_ds, valid_ds, pre_aug_ds
    gc.collect()
    tf.keras.backend.clear_session()

    return result


# ─────────────────────────────────────────────────────────────────────
# Search rounds
# ─────────────────────────────────────────────────────────────────────

def round1_backbone_freeze():
    """Round 1: Test backbone x freeze strategy combinations.

    Thermal maps encode temperature as pixel intensity. Pretrained ImageNet
    features transfer for structural/edge detection despite domain gap.
    """
    configs = []
    pretrained_backbones = [
        'EfficientNetB0', 'EfficientNetB2',
        'DenseNet121', 'ResNet50V2',
    ]
    for bb in pretrained_backbones:
        for fr in ['frozen', 'partial_unfreeze']:
            name = f"R1_{bb}_{fr}"
            cfg = SearchConfig(
                name=name,
                backbone=bb,
                freeze=fr,
                head_units=[128],
                head_dropout=0.3,
                learning_rate=1e-3,
                finetune_epochs=30 if fr == 'frozen' else 0,
                finetune_lr=1e-5,
            )
            configs.append(cfg)

    return configs


def round2_head(best_r1: SearchConfig):
    """Round 2: Test head architectures.

    Thermal maps have less feature complexity than RGB (temperature gradient
    only), so smaller heads may be sufficient.
    """
    configs = []
    heads = {
        'tiny':      {'units': [32], 'dropout': 0.3, 'bn': True},
        'small':     {'units': [64], 'dropout': 0.3, 'bn': True},
        'medium':    {'units': [128], 'dropout': 0.3, 'bn': True},
        'large':     {'units': [256], 'dropout': 0.3, 'bn': True},
        'two_layer': {'units': [128, 32], 'dropout': 0.3, 'bn': True},
    }

    for head_name, head_cfg in heads.items():
        cfg = deepcopy(best_r1)
        cfg.name = f"R2_{head_name}"
        cfg.head_units = head_cfg['units']
        cfg.head_dropout = head_cfg['dropout']
        cfg.head_use_bn = head_cfg['bn']
        configs.append(cfg)

    return configs


def round3_loss_regularization(best_r2: SearchConfig):
    """Round 3: Test loss function + regularization + class weighting."""
    configs = []

    variants = [
        ('focal_g2_d03',     'focal', 2.0, 0.3, 0.0, 3.0, 0.0,   False, 0.0),
        ('focal_g3_d03',     'focal', 3.0, 0.3, 0.0, 3.0, 0.0,   False, 0.0),
        ('cce_d03',          'cce',   0.0, 0.3, 0.0, 3.0, 0.0,   False, 0.0),
        ('focal_g2_d05',     'focal', 2.0, 0.5, 0.0, 3.0, 0.0,   False, 0.0),
        ('focal_g2_d02',     'focal', 2.0, 0.2, 0.0, 3.0, 0.0,   False, 0.0),
        ('focal_g2_ls01',    'focal', 2.0, 0.3, 0.1, 3.0, 0.0,   False, 0.0),
        ('focal_g2_l2_1e3',  'focal', 2.0, 0.3, 0.0, 3.0, 1e-3,  False, 0.0),
        ('focal_g2_mixup02', 'focal', 2.0, 0.3, 0.0, 3.0, 0.0,   True,  0.2),
        ('focal_g2_alpha0',  'focal', 2.0, 0.3, 0.0, 0.0, 0.0,   False, 0.0),
        ('focal_g2_alpha1',  'focal', 2.0, 0.3, 0.0, 1.0, 0.0,   False, 0.0),
        ('focal_g2_alpha5',  'focal', 2.0, 0.3, 0.0, 5.0, 0.0,   False, 0.0),
        ('focal_g2_alpha8',  'focal', 2.0, 0.3, 0.0, 8.0, 0.0,   False, 0.0),
    ]

    for name, loss, gamma, drop, ls, alpha_sum, l2, mixup, mixup_a in variants:
        cfg = deepcopy(best_r2)
        cfg.name = f"R3_{name}"
        cfg.loss_type = loss
        cfg.focal_gamma = gamma
        cfg.head_dropout = drop
        cfg.label_smoothing = ls
        cfg.alpha_sum = alpha_sum
        cfg.head_l2 = l2
        cfg.use_mixup = mixup
        cfg.mixup_alpha = mixup_a
        configs.append(cfg)

    return configs


def round4_training_dynamics(best_r3: SearchConfig):
    """Round 4: Test learning rate, batch size, LR schedule, warmup, and optimizer."""
    configs = []

    variants = [
        ('lr5e4_b64_plateau',    5e-4, 64, 50,  'plateau', 0, 'adam',  0.0),
        ('lr1e3_b64_plateau',    1e-3, 64, 50,  'plateau', 0, 'adam',  0.0),
        ('lr3e3_b64_plateau',    3e-3, 64, 50,  'plateau', 0, 'adam',  0.0),
        ('lr1e3_b32_plateau',    1e-3, 32, 50,  'plateau', 0, 'adam',  0.0),
        ('lr1e3_b64_cosine',     1e-3, 64, 60,  'cosine',  5, 'adam',  0.0),
        ('lr1e3_b64_e100',       1e-3, 64, 100, 'plateau', 0, 'adam',  0.0),
        ('adamw_wd1e4',          1e-3, 64, 100, 'plateau', 0, 'adamw', 1e-4),
        ('adamw_wd1e3',          1e-3, 64, 100, 'plateau', 0, 'adamw', 1e-3),
    ]

    for name, lr, bs, epochs, schedule, warmup, opt, wd in variants:
        cfg = deepcopy(best_r3)
        cfg.name = f"R4_{name}"
        cfg.learning_rate = lr
        cfg.batch_size = bs
        cfg.stage1_epochs = epochs
        cfg.lr_schedule = schedule
        cfg.warmup_epochs = warmup
        cfg.optimizer = opt
        cfg.weight_decay = wd
        if epochs > 50:
            cfg.early_stop_patience = 20
            cfg.reduce_lr_patience = 10
        configs.append(cfg)

    return configs


def round5_augmentation_imagesize(best_r4: SearchConfig):
    """Round 5: Test augmentation on/off and image size.

    Thermal maps encode temperature gradients — spatial augmentations +
    mild sensor noise. Test 128 (maps have less fine detail) and 256.
    """
    configs = []

    variants = [
        ('aug_on_128',  True,  128),
        ('aug_off_128', False, 128),
        ('aug_on_256',  True,  256),
        ('aug_off_256', False, 256),
    ]

    for name, aug, img_size in variants:
        cfg = deepcopy(best_r4)
        cfg.name = f"R5_{name}"
        cfg.use_augmentation = aug
        cfg.image_size = img_size
        configs.append(cfg)

    return configs


def round6_finetuning(best_r5: SearchConfig):
    """Round 6: Fine-tuning strategy exploration."""
    configs = []

    variants = [
        ('ft_top20_30ep',     0.2,  30, 1e-5),
        ('ft_top40_30ep',     0.4,  30, 1e-5),
        ('ft_top50_50ep',     0.5,  50, 5e-6),
        ('ft_top20_50ep',     0.2,  50, 5e-6),
    ]

    for name, pct, epochs, lr in variants:
        cfg = deepcopy(best_r5)
        cfg.name = f"R6_{name}"
        cfg.unfreeze_pct = pct
        cfg.finetune_epochs = epochs
        cfg.finetune_lr = lr
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


def pick_best(results: List[Dict], metric='post_eval_kappa') -> Tuple[Dict, SearchConfig]:
    """Pick the best result and reconstruct its config."""
    best = max(results, key=lambda r: r[metric])
    print(f"\n  BEST from this round: {best['name']} "
          f"(kappa={best['post_eval_kappa']:.4f}, s1_kappa={best['best_s1_kappa']:.4f})")
    print(f"  → Config: backbone={best['backbone']}, head={best['head_units']}, "
          f"lr={best['learning_rate']}, loss={best['loss_type']}, "
          f"epochs={best['stage1_epochs']}+{best.get('finetune_epochs',0)}ft, "
          f"img={best['image_size']}, n_val={best['n_val_samples']}")

    head_units = best['head_units']
    if isinstance(head_units, str):
        head_units = eval(head_units)

    cfg = SearchConfig(
        name=best['name'],
        backbone=best['backbone'],
        freeze=best['freeze'],
        head_units=head_units,
        head_dropout=float(best['head_dropout']),
        head_use_bn=best['head_use_bn'],
        head_l2=float(best.get('head_l2', 0.0)),
        loss_type=best['loss_type'],
        focal_gamma=float(best['focal_gamma']),
        alpha_sum=float(best['alpha_sum']),
        label_smoothing=float(best['label_smoothing']),
        learning_rate=float(best['learning_rate']),
        lr_schedule=best.get('lr_schedule', 'plateau'),
        warmup_epochs=int(best.get('warmup_epochs', 0)),
        batch_size=int(best['batch_size']),
        optimizer=best.get('optimizer', 'adam'),
        weight_decay=float(best.get('weight_decay', 1e-4)),
        use_augmentation=best['use_augmentation'],
        use_mixup=best.get('use_mixup', False),
        mixup_alpha=float(best.get('mixup_alpha', 0.0)),
        image_size=int(best['image_size']),
        stage1_epochs=int(best['stage1_epochs']),
        finetune_epochs=int(best.get('finetune_epochs', 0)),
        unfreeze_pct=float(best.get('unfreeze_pct', 0.2)),
    )

    return best, cfg


def load_completed_results(filepath: str) -> List[Dict]:
    """Load previously completed results from CSV for resume support."""
    if not os.path.exists(filepath):
        return []

    results = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ['head_dropout', 'head_l2', 'focal_gamma', 'alpha_sum',
                        'label_smoothing', 'learning_rate', 'mixup_alpha',
                        'weight_decay', 'unfreeze_pct',
                        'best_s1_kappa', 'best_s2_kappa', 'post_eval_kappa',
                        'post_eval_acc', 'post_eval_f1', 'elapsed_seconds']:
                if key in row and row[key]:
                    row[key] = float(row[key])
            for key in ['batch_size', 'image_size', 'stage1_epochs', 'finetune_epochs',
                        'fold', 'trainable_weights', 'total_weights', 'best_s1_epoch',
                        'warmup_epochs', 'n_val_samples']:
                if key in row and row[key]:
                    row[key] = int(row[key])
            for key in ['head_use_bn', 'use_augmentation', 'use_mixup']:
                if key in row and row[key]:
                    row[key] = row[key] in ('True', 'true', '1')
            results.append(row)

    return results


def run_round(round_name: str, configs: List[SearchConfig], data,
              results_csv: str, completed: List[Dict]) -> List[Dict]:
    """Run a round of configs, skipping any already completed."""
    completed_names = {r['name'] for r in completed}
    round_results = []

    for cfg in configs:
        if cfg.name in completed_names:
            cached = [r for r in completed if r['name'] == cfg.name][0]
            print(f"\n  SKIP (cached): {cfg.name} → kappa={cached['post_eval_kappa']:.4f}")
            round_results.append(cached)
        else:
            result = train_single_config(cfg, data, fold_idx=cfg.fold)
            round_results.append(result)
            save_results([result], results_csv)

    return round_results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(fresh: bool = False):
    print("=" * 80)
    print(f"{MODALITY_LABEL} HYPERPARAMETER SEARCH")
    print("=" * 80)

    data, directory, result_dir = load_data()
    results_csv = os.path.join(AUDIT_DIR, f'{MODALITY}_search_results.csv')

    completed = []
    if fresh:
        if os.path.exists(results_csv):
            os.rename(results_csv, results_csv + '.bak')
            print(f"FRESH START — backed up old results to {results_csv}.bak")
        # Delete search cache for this modality to avoid stale data
        import shutil
        from src.utils.config import get_project_paths as _gpp
        _, _search_result_dir, _ = _gpp()
        _cache_base = os.path.join(_search_result_dir, 'search_cache')
        if os.path.exists(_cache_base):
            _deleted = []
            for _item in os.listdir(_cache_base):
                if _item.startswith(f'{MODALITY}_'):
                    shutil.rmtree(os.path.join(_cache_base, _item))
                    _deleted.append(_item)
            if _deleted:
                print(f"FRESH START — deleted {len(_deleted)} cache dirs: {_deleted}")
            else:
                print(f"FRESH START — no {MODALITY} cache dirs found to delete")
        else:
            print(f"FRESH START — no search_cache directory found")
    else:
        completed = load_completed_results(results_csv)
        if completed:
            completed_names = [r['name'] for r in completed]
            print(f"RESUME MODE — found {len(completed)} completed configs: {completed_names}")
        else:
            print("RESUME MODE — no previous results found, starting from scratch")

    all_results = []

    print(f"\n{'#'*80}")
    print("ROUND 1: BACKBONE + FREEZE STRATEGY")
    print(f"{'#'*80}")

    r1_configs = round1_backbone_freeze()
    r1_results = run_round("R1", r1_configs, data, results_csv, completed)
    all_results.extend(r1_results)
    best_r1_result, best_r1_cfg = pick_best(r1_results)

    print(f"\n{'#'*80}")
    print("ROUND 2: HEAD ARCHITECTURE")
    print(f"{'#'*80}")

    r2_configs = round2_head(best_r1_cfg)
    r2_results = run_round("R2", r2_configs, data, results_csv, completed)
    all_results.extend(r2_results)
    best_r2_result, best_r2_cfg = pick_best(r2_results)

    print(f"\n{'#'*80}")
    print("ROUND 3: LOSS + REGULARIZATION")
    print(f"{'#'*80}")

    r3_configs = round3_loss_regularization(best_r2_cfg)
    r3_results = run_round("R3", r3_configs, data, results_csv, completed)
    all_results.extend(r3_results)
    best_r3_result, best_r3_cfg = pick_best(r3_results)

    print(f"\n{'#'*80}")
    print("ROUND 4: TRAINING DYNAMICS")
    print(f"{'#'*80}")

    r4_configs = round4_training_dynamics(best_r3_cfg)
    r4_results = run_round("R4", r4_configs, data, results_csv, completed)
    all_results.extend(r4_results)
    best_r4_result, best_r4_cfg = pick_best(r4_results)

    print(f"\n{'#'*80}")
    print("ROUND 5: AUGMENTATION + IMAGE SIZE")
    print(f"{'#'*80}")

    r5_configs = round5_augmentation_imagesize(best_r4_cfg)
    r5_results = run_round("R5", r5_configs, data, results_csv, completed)
    all_results.extend(r5_results)
    best_r5_result, best_r5_cfg = pick_best(r5_results)

    print(f"\n{'#'*80}")
    print("ROUND 6: FINE-TUNING STRATEGY")
    print(f"{'#'*80}")

    r6_configs = round6_finetuning(best_r5_cfg)
    r6_results = run_round("R6", r6_configs, data, results_csv, completed)
    all_results.extend(r6_results)
    best_r6_result, best_r6_cfg = pick_best(r6_results)

    # ─── Top 3 Selection ───
    print(f"\n{'#'*80}")
    print("TOP 3 SELECTION: 5-FOLD VALIDATION OF BEST CONFIGS")
    print(f"{'#'*80}")

    search_results = sorted(all_results, key=lambda r: r['post_eval_kappa'], reverse=True)

    seen_configs = set()
    top3_results = []
    for r in search_results:
        fingerprint = (
            r['backbone'], r['freeze'], str(r['head_units']), float(r['head_dropout']),
            r['head_use_bn'], float(r.get('head_l2', 0)), r['loss_type'],
            float(r['focal_gamma']), float(r['alpha_sum']), float(r['label_smoothing']),
            float(r['learning_rate']), r.get('lr_schedule', 'plateau'),
            int(r.get('warmup_epochs', 0)), int(r['batch_size']),
            r.get('optimizer', 'adam'), float(r.get('weight_decay', 0)),
            r['use_augmentation'], r.get('use_mixup', False),
            float(r.get('mixup_alpha', 0)), int(r['image_size']),
            int(r['stage1_epochs']), int(r.get('finetune_epochs', 0)),
            float(r.get('unfreeze_pct', 0.2)),
        )
        if fingerprint not in seen_configs:
            seen_configs.add(fingerprint)
            top3_results.append(r)
            print(f"  Selected TOP3 #{len(top3_results)}: {r['name']} → kappa={r['post_eval_kappa']:.4f}")
        if len(top3_results) == 3:
            break

    print(f"\nTop 3 configs by fold-0 kappa:")
    for i, r in enumerate(top3_results):
        print(f"  {i+1}. {r['name']} → kappa={r['post_eval_kappa']:.4f}")

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
        fold_results = run_round(tag, fold_configs, data, results_csv, completed)
        all_results.extend(fold_results)
        top3_fold_results[tag] = {
            'cfg': base_cfg,
            'orig_name': r['name'],
            'fold0_kappa': r['post_eval_kappa'],
            'fold_results': fold_results,
            'mean_kappa': np.mean([fr['post_eval_kappa'] for fr in fold_results]),
        }

    # ─── Baseline ───
    print(f"\n{'#'*80}")
    print(f"BASELINE: EfficientNetB0 FROZEN ON ALL 5 FOLDS")
    print(f"{'#'*80}")

    baseline_cfg = SearchConfig(
        name="BASELINE",
        backbone='EfficientNetB0',
        freeze='frozen',
        finetune_epochs=30,
        finetune_lr=1e-5,
        n_folds=N_FINAL_FOLDS,
    )
    baseline_configs = []
    for fold_idx in range(N_FINAL_FOLDS):
        cfg = deepcopy(baseline_cfg)
        cfg.name = f"BASELINE_fold{fold_idx+1}"
        cfg.fold = fold_idx
        baseline_configs.append(cfg)
    baseline_results = run_round("BASELINE", baseline_configs, data, results_csv, completed)
    all_results.extend(baseline_results)

    # ─── Summary ───
    print(f"\n{'='*80}")
    print(f"{MODALITY_LABEL} SEARCH COMPLETE — SUMMARY")
    print(f"{'='*80}")

    for round_name, round_results in [
        ('R1: Backbone+Freeze', r1_results),
        ('R2: Head', r2_results),
        ('R3: Loss+Reg+Alpha', r3_results),
        ('R4: Training+Optim', r4_results),
        ('R5: Aug+ImgSize', r5_results),
        ('R6: FineTuning', r6_results),
    ]:
        best = max(round_results, key=lambda r: r['post_eval_kappa'])
        print(f"  {round_name}: {best['name']} → kappa={best['post_eval_kappa']:.4f}")

    def print_config_summary(label, cfg, fold_results):
        print(f"\n{label}:")
        print(f"  backbone={cfg.backbone}, freeze={cfg.freeze}")
        print(f"  head={cfg.head_units}, dropout={cfg.head_dropout}, bn={cfg.head_use_bn}, l2={cfg.head_l2}")
        print(f"  loss={cfg.loss_type}, gamma={cfg.focal_gamma}, alpha_sum={cfg.alpha_sum}")
        print(f"  lr={cfg.learning_rate}, schedule={cfg.lr_schedule}, batch={cfg.batch_size}")
        print(f"  optimizer={cfg.optimizer}, weight_decay={cfg.weight_decay}")
        print(f"  epochs={cfg.stage1_epochs}+{cfg.finetune_epochs}ft (unfreeze {cfg.unfreeze_pct*100:.0f}%), warmup={cfg.warmup_epochs}")
        print(f"  aug={cfg.use_augmentation}, mixup={cfg.use_mixup}(alpha={cfg.mixup_alpha}), img_size={cfg.image_size}")
        print(f"  label_smoothing={cfg.label_smoothing}")

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

    print(f"\n  {'Rank':<6} {'Config':<30} {'Fold0':>7} {'Mean±Std':>14}")
    print(f"  {'-'*60}")
    for rank, (tag, info) in enumerate(sorted_top3):
        fk = [r['post_eval_kappa'] for r in info['fold_results']]
        print(f"  {rank+1:<6} {info['orig_name']:<30} {info['fold0_kappa']:>7.4f} "
              f"{np.mean(fk):>6.4f}±{np.std(fk):.4f}")

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
    bl_kappas, bl_accs, bl_f1s = print_config_summary(
        "BASELINE (EfficientNetB0 frozen)", baseline_cfg, baseline_results)

    print(f"\n{'─'*60}")
    print(f"STATISTICAL COMPARISON: #{1} {winner_info['orig_name']} vs BASELINE")
    print(f"{'─'*60}")

    best_kappas = [r['post_eval_kappa'] for r in winner_fold_results]
    best_accs = [r['post_eval_acc'] for r in winner_fold_results]
    best_f1s = [r['post_eval_f1'] for r in winner_fold_results]

    kappa_diff = np.mean(best_kappas) - np.mean(bl_kappas)
    acc_diff = np.mean(best_accs) - np.mean(bl_accs)
    f1_diff = np.mean(best_f1s) - np.mean(bl_f1s)
    print(f"  Mean Kappa diff:    {kappa_diff:+.4f}  ({np.mean(bl_kappas):.4f} → {np.mean(best_kappas):.4f})")
    print(f"  Mean Accuracy diff: {acc_diff:+.4f}  ({np.mean(bl_accs):.4f} → {np.mean(best_accs):.4f})")
    print(f"  Mean F1 diff:       {f1_diff:+.4f}  ({np.mean(bl_f1s):.4f} → {np.mean(best_f1s):.4f})")

    from scipy import stats
    if len(best_kappas) == len(bl_kappas) and len(best_kappas) >= 2:
        t_stat, p_value = stats.ttest_rel(best_kappas, bl_kappas)
        print(f"\n  Paired t-test on kappa (n={len(best_kappas)} folds):")
        print(f"    t-statistic = {t_stat:.4f}")
        print(f"    p-value     = {p_value:.4f}")
        if p_value < 0.05:
            winner_label = "WINNER" if kappa_diff > 0 else "BASELINE"
            print(f"    → Statistically significant (p < 0.05): {winner_label} is better")
        else:
            print(f"    → NOT statistically significant (p >= 0.05)")

    print(f"\nResults saved to: {results_csv}")

    best_config_path = os.path.join(AUDIT_DIR, f'{MODALITY}_best_config.json')
    config_dict = asdict(winner_cfg)
    with open(best_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Best config saved to: {best_config_path}")


if __name__ == '__main__':
    import datetime
    import argparse

    parser = argparse.ArgumentParser(description=f'{MODALITY_LABEL} Hyperparameter Search')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (back up old results). Default: resume from existing CSV.')
    args = parser.parse_args()

    log_dir = os.path.join(AUDIT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{MODALITY}_hparam_search_{timestamp}.log')

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
        main(fresh=args.fresh)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to: {log_path}")
