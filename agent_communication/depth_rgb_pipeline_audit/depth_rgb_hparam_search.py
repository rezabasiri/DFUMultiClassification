#!/usr/bin/env python3
"""
Depth RGB Hyperparameter Search — Standalone Script

Systematically tests architecture and training configurations for depth_rgb
standalone to maximize val_cohen_kappa. Uses sequential elimination:
  Round 1: Backbone + freeze strategy  (9 configs)
  Round 2: Head architecture            (4 configs)
  Round 3: Loss + regularization        (6 configs)
  Round 4: Training dynamics            (4 configs)
  Round 5: Augmentation + image size    (4 configs)

Usage:
  /venv/multimodal/bin/python agent_communication/depth_rgb_pipeline_audit/depth_rgb_hparam_search.py

Results written to: results/depth_rgb_search_results.csv
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

# Set environment before TF import
os.environ["OMP_NUM_THREADS"] = "2"
os.environ['TF_NUM_INTEROP_THREADS'] = "2"
os.environ['TF_NUM_INTRAOP_THREADS'] = "4"
os.environ['TF_DETERMINISTIC_OPS'] = "1"
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF noise

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

# ─────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """Single experiment configuration."""
    name: str = "baseline"

    # Backbone
    backbone: str = "EfficientNetB0"  # SimpleCNN, EfficientNetB0, EfficientNetB2

    # Freeze strategy
    freeze: str = "frozen"  # frozen, partial_unfreeze, full_unfreeze

    # Head architecture
    head_units: List[int] = field(default_factory=lambda: [128])
    head_dropout: float = 0.3
    head_use_bn: bool = True

    # Training
    learning_rate: float = 1e-3
    stage1_epochs: int = 50
    early_stop_patience: int = 15
    reduce_lr_patience: int = 7
    batch_size: int = 64

    # Loss
    loss_type: str = "focal"  # focal, cce
    focal_gamma: float = 2.0
    alpha_sum: float = 3.0  # normalization target for alpha values
    label_smoothing: float = 0.0

    # Augmentation
    use_augmentation: bool = True

    # Image
    image_size: int = 256

    # Fine-tuning LR (only used when freeze != 'frozen')
    finetune_lr: float = 1e-5
    finetune_epochs: int = 30

    # Which fold to run (0-indexed)
    fold: int = 0


# ─────────────────────────────────────────────────────────────────────
# Model building
# ─────────────────────────────────────────────────────────────────────

def build_simple_cnn(input_shape):
    """4-layer CNN from scratch — no pretrained weights."""
    inp = Input(shape=input_shape, name='depth_rgb_input')
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = GlobalAveragePooling2D()(x)
    return inp, x


def build_efficientnet(input_shape, variant='EfficientNetB0'):
    """EfficientNet backbone with ImageNet weights."""
    inp = Input(shape=input_shape, name='depth_rgb_input')

    backbone_map = {
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'EfficientNetB2': tf.keras.applications.EfficientNetB2,
    }
    ENetClass = backbone_map[variant]

    # Load with ImageNet weights
    base_model = ENetClass(weights='imagenet', include_top=False, pooling='avg')
    x = base_model(inp)

    return inp, x, base_model


def build_model(cfg: SearchConfig):
    """Build complete model from config."""
    input_shape = (cfg.image_size, cfg.image_size, 3)

    base_model = None
    if cfg.backbone == 'SimpleCNN':
        inp, features = build_simple_cnn(input_shape)
    else:
        inp, features, base_model = build_efficientnet(input_shape, cfg.backbone)

    # Head
    x = features
    for i, units in enumerate(cfg.head_units):
        x = Dense(units, activation='relu', kernel_initializer='he_normal',
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
# Metrics (reused from main codebase)
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
# Data pipeline — reuses project's existing data loading
# ─────────────────────────────────────────────────────────────────────

def load_data():
    """Load the dataset using the project's standard prepare_dataset pipeline."""
    from src.utils.config import get_project_paths, get_data_paths, get_output_paths
    from src.data.image_processing import prepare_dataset
    from src.utils.production_config import (
        USE_CORE_DATA, THRESHOLD_I, THRESHOLD_P, THRESHOLD_R
    )

    directory, result_dir, root = get_project_paths()
    data_paths = get_data_paths(root)

    csv_file = data_paths['csv_file']
    depth_bb_file = data_paths['bb_depth_csv']
    thermal_bb_file = data_paths['bb_thermal_csv']

    # Use prepare_dataset (same as main.py) — creates best_matching.csv if needed
    selected_modalities = ['depth_rgb']
    data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities)

    # Apply core data filtering (same as main.py)
    if USE_CORE_DATA and all(t is not None for t in [THRESHOLD_I, THRESHOLD_P, THRESHOLD_R]):
        try:
            from src.evaluation.metrics import filter_frequent_misclassifications
            thresholds = {'I': THRESHOLD_I, 'P': THRESHOLD_P, 'R': THRESHOLD_R}
            data = filter_frequent_misclassifications(data, result_dir, thresholds=thresholds)
        except Exception as e:
            print(f"  Warning: Core data filtering failed: {e}")

    print(f"Loaded {len(data)} samples")
    return data, directory, result_dir


def prepare_datasets(data, cfg: SearchConfig, fold_idx=0):
    """Prepare train/valid datasets for a single fold using the project's pipeline."""
    from src.data.dataset_utils import create_patient_folds, prepare_cached_datasets
    from src.data.generative_augmentation_v3 import AugmentationConfig

    import pandas as pd

    # Convert labels
    data = data.copy()
    if data['Healing Phase Abs'].dtype in ['object', 'str'] or pd.api.types.is_string_dtype(data['Healing Phase Abs']):
        data['Healing Phase Abs'] = data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})

    # Note: core data filtering already applied in load_data()

    # Get patient folds
    patient_folds = create_patient_folds(data, n_folds=3, random_state=42)
    train_patients, valid_patients = patient_folds[fold_idx]

    # Augmentation
    aug_config = AugmentationConfig()
    aug_config.generative_settings['output_size']['width'] = cfg.image_size
    aug_config.generative_settings['output_size']['height'] = cfg.image_size

    selected_modalities = ['depth_rgb']

    # Temporarily override USE_GENERAL_AUGMENTATION if this config disables augmentation
    import src.utils.production_config as _pcfg
    import src.data.dataset_utils as _ds_utils
    original_aug = _pcfg.USE_GENERAL_AUGMENTATION
    if not cfg.use_augmentation:
        _pcfg.USE_GENERAL_AUGMENTATION = False
        _ds_utils.USE_GENERAL_AUGMENTATION = False

    # Use a unique cache dir for this search to avoid collisions with main training
    from src.utils.config import get_project_paths, get_output_paths
    _, search_result_dir, _ = get_project_paths()
    aug_suffix = 'aug' if cfg.use_augmentation else 'noaug'
    search_cache_dir = os.path.join(search_result_dir, f'tf_records_search_{cfg.image_size}_{aug_suffix}')

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

    # Restore augmentation setting
    _pcfg.USE_GENERAL_AUGMENTATION = original_aug
    if hasattr(_ds_utils, 'USE_GENERAL_AUGMENTATION'):
        _ds_utils.USE_GENERAL_AUGMENTATION = original_aug

    # Remove sample_id from datasets for model.fit()
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
        return  # SimpleCNN — all trainable

    if cfg.freeze == 'frozen':
        base_model.trainable = False
    elif cfg.freeze == 'partial_unfreeze':
        base_model.trainable = True
        # Freeze bottom 80% of layers
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
    print(f"CONFIG: {cfg.name}")
    print(f"  backbone={cfg.backbone}, freeze={cfg.freeze}")
    print(f"  head={cfg.head_units}, dropout={cfg.head_dropout}, bn={cfg.head_use_bn}")
    print(f"  loss={cfg.loss_type}, gamma={cfg.focal_gamma}, alpha_sum={cfg.alpha_sum}")
    print(f"  lr={cfg.learning_rate}, epochs={cfg.stage1_epochs}, batch={cfg.batch_size}")
    print(f"  augmentation={cfg.use_augmentation}, label_smooth={cfg.label_smoothing}")
    print(f"  image_size={cfg.image_size}, fold={fold_idx}")
    print(f"{'='*80}")

    start_time = time.time()

    # Prepare data
    train_ds, valid_ds, pre_aug_ds, steps_per_epoch, val_steps, alpha_values = \
        prepare_datasets(data, cfg, fold_idx=fold_idx)

    # Normalize alpha to target sum
    alpha_arr = np.array(alpha_values)
    alpha_arr = alpha_arr / alpha_arr.sum() * cfg.alpha_sum
    alpha_list = alpha_arr.tolist()
    print(f"  Alpha values (sum={cfg.alpha_sum}): {[round(a, 3) for a in alpha_list]}")

    # Build model
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model, base_model = build_model(cfg)

        # Apply freeze strategy
        apply_freeze_strategy(model, base_model, cfg)

        trainable_count = len(model.trainable_weights)
        total_count = len(model.weights)
        print(f"  Trainable weights: {trainable_count}/{total_count}")

        # Build loss
        if cfg.loss_type == 'focal':
            loss_fn = make_focal_loss(gamma=cfg.focal_gamma, alpha=alpha_list,
                                       label_smoothing=cfg.label_smoothing)
        elif cfg.loss_type == 'cce':
            loss_fn = make_cce_loss(alpha=alpha_list, label_smoothing=cfg.label_smoothing)
        else:
            raise ValueError(f"Unknown loss: {cfg.loss_type}")

        # Determine learning rate for this stage
        lr = cfg.learning_rate
        if cfg.freeze == 'full_unfreeze' and cfg.backbone != 'SimpleCNN':
            lr = cfg.finetune_lr  # Use lower LR when backbone is trainable
            print(f"  Using finetune LR={lr} (backbone fully trainable)")
        elif cfg.freeze == 'partial_unfreeze' and cfg.backbone != 'SimpleCNN':
            lr = cfg.finetune_lr * 10  # Middle ground LR
            print(f"  Using partial-unfreeze LR={lr}")

        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),
            loss=loss_fn,
            metrics=['accuracy', CohenKappaMetric(num_classes=3)]
        )

    # Distribute datasets
    train_ds_dis = strategy.experimental_distribute_dataset(train_ds)
    valid_ds_dis = strategy.experimental_distribute_dataset(valid_ds)

    # Callbacks
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

    # Train Stage 1 (head only, or full model if SimpleCNN/full_unfreeze)
    history = model.fit(
        train_ds_dis,
        epochs=cfg.stage1_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds_dis,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0  # Quiet — we print summary at end
    )

    # Get best val_kappa from Stage 1
    s1_kappas = history.history.get('val_cohen_kappa', [])
    best_s1_kappa = max(s1_kappas) if s1_kappas else 0.0
    best_s1_epoch = int(np.argmax(s1_kappas)) + 1 if s1_kappas else 0
    print(f"  Stage 1 best: val_kappa={best_s1_kappa:.4f} at epoch {best_s1_epoch}/{len(s1_kappas)}")

    # Stage 2: Fine-tuning (only for frozen backbone configs)
    best_s2_kappa = 0.0
    ran_stage2 = False
    if cfg.freeze == 'frozen' and base_model is not None and cfg.finetune_epochs > 0:
        # Try partial unfreeze for Stage 2
        base_model.trainable = True
        n_layers = len(base_model.layers)
        freeze_until = int(n_layers * 0.8)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        with strategy.scope():
            model.compile(
                optimizer=Adam(learning_rate=cfg.finetune_lr, clipnorm=1.0),
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

    # Post-training evaluation on validation set (using best weights)
    y_true_v = []
    y_pred_v = []

    # Use pre_aug_ds for validation evaluation (has sample_id, finite)
    # Actually, we need the validation split. Let's re-iterate valid_ds.
    # We need to get the unfiltered validation dataset
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

    # Overall best kappa (max of stage1 and stage2)
    overall_best_kappa = max(best_s1_kappa, best_s2_kappa) if ran_stage2 else best_s1_kappa

    result = {
        'name': cfg.name,
        'backbone': cfg.backbone,
        'freeze': cfg.freeze,
        'head_units': str(cfg.head_units),
        'head_dropout': cfg.head_dropout,
        'head_use_bn': cfg.head_use_bn,
        'loss_type': cfg.loss_type,
        'focal_gamma': cfg.focal_gamma,
        'alpha_sum': cfg.alpha_sum,
        'label_smoothing': cfg.label_smoothing,
        'learning_rate': cfg.learning_rate,
        'batch_size': cfg.batch_size,
        'use_augmentation': cfg.use_augmentation,
        'image_size': cfg.image_size,
        'stage1_epochs': cfg.stage1_epochs,
        'finetune_epochs': cfg.finetune_epochs if ran_stage2 else 0,
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

    # Cleanup
    del model, base_model, train_ds, valid_ds, pre_aug_ds
    gc.collect()
    tf.keras.backend.clear_session()

    return result


# ─────────────────────────────────────────────────────────────────────
# Search rounds
# ─────────────────────────────────────────────────────────────────────

def round1_backbone_freeze():
    """Round 1: Test backbone x freeze strategy combinations."""
    configs = []
    backbones = ['SimpleCNN', 'EfficientNetB0', 'EfficientNetB2']
    freezes = ['frozen', 'partial_unfreeze', 'full_unfreeze']

    for bb in backbones:
        for fr in freezes:
            # SimpleCNN has no pretrained weights — freeze is meaningless
            if bb == 'SimpleCNN' and fr != 'full_unfreeze':
                continue

            name = f"R1_{bb}_{fr}"
            cfg = SearchConfig(
                name=name,
                backbone=bb,
                freeze=fr,
                head_units=[128],
                head_dropout=0.3,
                learning_rate=1e-3,
                finetune_epochs=30 if fr == 'frozen' else 0,  # Only frozen gets Stage 2
                finetune_lr=1e-5,
            )
            configs.append(cfg)

    return configs


def round2_head(best_r1: SearchConfig):
    """Round 2: Test head architectures with best backbone/freeze from R1."""
    configs = []
    heads = {
        'small':     {'units': [64], 'dropout': 0.3, 'bn': True},
        'medium':    {'units': [128], 'dropout': 0.3, 'bn': True},
        'large':     {'units': [256], 'dropout': 0.3, 'bn': True},
        'two_layer': {'units': [256, 64], 'dropout': 0.3, 'bn': True},
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
    """Round 3: Test loss function + regularization combos."""
    configs = []

    variants = [
        ('focal_g2_d03', 'focal', 2.0, 0.3, 0.0, 3.0),
        ('focal_g3_d03', 'focal', 3.0, 0.3, 0.0, 3.0),
        ('cce_d03',       'cce',   0.0, 0.3, 0.0, 3.0),
        ('focal_g2_d05', 'focal', 2.0, 0.5, 0.0, 3.0),
        ('focal_g2_d02', 'focal', 2.0, 0.2, 0.0, 3.0),
        ('focal_g2_ls01','focal', 2.0, 0.3, 0.1, 3.0),
    ]

    for name, loss, gamma, drop, ls, alpha_sum in variants:
        cfg = deepcopy(best_r2)
        cfg.name = f"R3_{name}"
        cfg.loss_type = loss
        cfg.focal_gamma = gamma
        cfg.head_dropout = drop
        cfg.label_smoothing = ls
        cfg.alpha_sum = alpha_sum
        configs.append(cfg)

    return configs


def round4_training_dynamics(best_r3: SearchConfig):
    """Round 4: Test learning rate, batch size, training length."""
    configs = []

    variants = [
        ('lr5e4_b64_e50',  5e-4, 64, 50),
        ('lr1e3_b64_e50',  1e-3, 64, 50),   # Current default
        ('lr3e3_b64_e50',  3e-3, 64, 50),
        ('lr1e3_b32_e50',  1e-3, 32, 50),
        ('lr1e3_b64_e100', 1e-3, 64, 100),
    ]

    for name, lr, bs, epochs in variants:
        cfg = deepcopy(best_r3)
        cfg.name = f"R4_{name}"
        cfg.learning_rate = lr
        cfg.batch_size = bs
        cfg.stage1_epochs = epochs
        if epochs > 50:
            cfg.early_stop_patience = 20
            cfg.reduce_lr_patience = 10
        configs.append(cfg)

    return configs


def round5_augmentation_imagesize(best_r4: SearchConfig):
    """Round 5: Test augmentation on/off and image size."""
    configs = []

    variants = [
        ('aug_on_256',  True,  256),
        ('aug_off_256', False, 256),
        ('aug_on_224',  True,  224),
        ('aug_off_224', False, 224),
    ]

    for name, aug, img_size in variants:
        cfg = deepcopy(best_r4)
        cfg.name = f"R5_{name}"
        cfg.use_augmentation = aug
        cfg.image_size = img_size
        configs.append(cfg)

    return configs


# ─────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────

def save_results(results: List[Dict], filepath: str):
    """Append results to CSV file."""
    if not results:
        return

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

    # Reconstruct config from result
    cfg = SearchConfig(
        name=best['name'],
        backbone=best['backbone'],
        freeze=best['freeze'],
        head_units=eval(best['head_units']),  # stored as string repr of list
        head_dropout=best['head_dropout'],
        head_use_bn=best['head_use_bn'],
        loss_type=best['loss_type'],
        focal_gamma=best['focal_gamma'],
        alpha_sum=best['alpha_sum'],
        label_smoothing=best['label_smoothing'],
        learning_rate=best['learning_rate'],
        batch_size=best['batch_size'],
        use_augmentation=best['use_augmentation'],
        image_size=best['image_size'],
        stage1_epochs=best['stage1_epochs'],
        finetune_epochs=best.get('finetune_epochs', 0),
    )

    return best, cfg


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("DEPTH RGB HYPERPARAMETER SEARCH")
    print("=" * 80)

    # Load data once
    data, directory, result_dir = load_data()
    results_csv = os.path.join(result_dir, 'depth_rgb_search_results.csv')

    # Remove old results if starting fresh
    if os.path.exists(results_csv):
        os.rename(results_csv, results_csv + '.bak')
        print(f"Backed up old results to {results_csv}.bak")

    all_results = []

    # ─── Round 1: Backbone + Freeze ───
    print(f"\n{'#'*80}")
    print("ROUND 1: BACKBONE + FREEZE STRATEGY")
    print(f"{'#'*80}")

    r1_configs = round1_backbone_freeze()
    r1_results = []
    for cfg in r1_configs:
        result = train_single_config(cfg, data, fold_idx=0)
        r1_results.append(result)
        save_results([result], results_csv)
    all_results.extend(r1_results)

    best_r1_result, best_r1_cfg = pick_best(r1_results)

    # ─── Round 2: Head Architecture ───
    print(f"\n{'#'*80}")
    print("ROUND 2: HEAD ARCHITECTURE")
    print(f"{'#'*80}")

    r2_configs = round2_head(best_r1_cfg)
    r2_results = []
    for cfg in r2_configs:
        result = train_single_config(cfg, data, fold_idx=0)
        r2_results.append(result)
        save_results([result], results_csv)
    all_results.extend(r2_results)

    best_r2_result, best_r2_cfg = pick_best(r2_results)

    # ─── Round 3: Loss + Regularization ───
    print(f"\n{'#'*80}")
    print("ROUND 3: LOSS + REGULARIZATION")
    print(f"{'#'*80}")

    r3_configs = round3_loss_regularization(best_r2_cfg)
    r3_results = []
    for cfg in r3_configs:
        result = train_single_config(cfg, data, fold_idx=0)
        r3_results.append(result)
        save_results([result], results_csv)
    all_results.extend(r3_results)

    best_r3_result, best_r3_cfg = pick_best(r3_results)

    # ─── Round 4: Training Dynamics ───
    print(f"\n{'#'*80}")
    print("ROUND 4: TRAINING DYNAMICS")
    print(f"{'#'*80}")

    r4_configs = round4_training_dynamics(best_r3_cfg)
    r4_results = []
    for cfg in r4_configs:
        result = train_single_config(cfg, data, fold_idx=0)
        r4_results.append(result)
        save_results([result], results_csv)
    all_results.extend(r4_results)

    best_r4_result, best_r4_cfg = pick_best(r4_results)

    # ─── Round 5: Augmentation + Image Size ───
    print(f"\n{'#'*80}")
    print("ROUND 5: AUGMENTATION + IMAGE SIZE")
    print(f"{'#'*80}")

    r5_configs = round5_augmentation_imagesize(best_r4_cfg)
    r5_results = []
    for cfg in r5_configs:
        result = train_single_config(cfg, data, fold_idx=0)
        r5_results.append(result)
        save_results([result], results_csv)
    all_results.extend(r5_results)

    best_r5_result, best_r5_cfg = pick_best(r5_results)

    # ─── Final: Run best config on all 3 folds ───
    print(f"\n{'#'*80}")
    print("FINAL: BEST CONFIG ON ALL 3 FOLDS")
    print(f"{'#'*80}")

    final_results = []
    for fold_idx in range(3):
        cfg = deepcopy(best_r5_cfg)
        cfg.name = f"FINAL_fold{fold_idx+1}"
        cfg.fold = fold_idx
        result = train_single_config(cfg, data, fold_idx=fold_idx)
        final_results.append(result)
        save_results([result], results_csv)
    all_results.extend(final_results)

    # ─── Summary ───
    print(f"\n{'='*80}")
    print("SEARCH COMPLETE — SUMMARY")
    print(f"{'='*80}")

    # Per-round best
    for round_name, round_results in [
        ('R1: Backbone+Freeze', r1_results),
        ('R2: Head', r2_results),
        ('R3: Loss+Reg', r3_results),
        ('R4: Training', r4_results),
        ('R5: Aug+ImgSize', r5_results),
    ]:
        best = max(round_results, key=lambda r: r['post_eval_kappa'])
        print(f"  {round_name}: {best['name']} → kappa={best['post_eval_kappa']:.4f}")

    # Final 3-fold results
    print(f"\nFINAL BEST CONFIG: {best_r5_cfg.name}")
    print(f"  backbone={best_r5_cfg.backbone}, freeze={best_r5_cfg.freeze}")
    print(f"  head={best_r5_cfg.head_units}, dropout={best_r5_cfg.head_dropout}")
    print(f"  loss={best_r5_cfg.loss_type}, gamma={best_r5_cfg.focal_gamma}")
    print(f"  lr={best_r5_cfg.learning_rate}, batch={best_r5_cfg.batch_size}")
    print(f"  aug={best_r5_cfg.use_augmentation}, img_size={best_r5_cfg.image_size}")

    fold_kappas = [r['post_eval_kappa'] for r in final_results]
    print(f"\n  3-fold kappas: {[round(k, 4) for k in fold_kappas]}")
    print(f"  Mean kappa:    {np.mean(fold_kappas):.4f} +/- {np.std(fold_kappas):.4f}")

    # Compare to baseline
    baseline_r1 = [r for r in r1_results if 'EfficientNetB0_frozen' in r['name']]
    if baseline_r1:
        baseline_kappa = baseline_r1[0]['post_eval_kappa']
        improvement = np.mean(fold_kappas) - baseline_kappa
        print(f"\n  Baseline kappa (B0 frozen, fold 0): {baseline_kappa:.4f}")
        print(f"  Improvement: {improvement:+.4f}")

    print(f"\nResults saved to: {results_csv}")

    # Save best config as JSON
    best_config_path = os.path.join(result_dir, 'depth_rgb_best_config.json')
    config_dict = asdict(best_r5_cfg)
    with open(best_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Best config saved to: {best_config_path}")


if __name__ == '__main__':
    main()
