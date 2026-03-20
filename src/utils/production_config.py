"""
Production Configuration for main.py

This module contains all hyperparameters and configuration values used in the
production training pipeline (src/main.py). All values are centralized here to
facilitate hyperparameter tuning and experimentation.

Categories:
- Training Parameters
- Gating Network Configuration
- Hierarchical Gating Network Configuration
- Learning Rate Scheduler
- Model Combination Search
- Callbacks Configuration
- Attention Mechanisms
- Loss Function Parameters
- Cross-Validation
- Grid Search
- Environment Configuration
- File I/O
"""

import os

# =============================================================================
# Training Parameters
# =============================================================================

# Core training hyperparameters
IMAGE_SIZE = 128  # Image dimensions (128x128 validated by joint optimization audit — TOP10_4)
GLOBAL_BATCH_SIZE = int(os.environ.get('OVERRIDE_BATCH_SIZE', 64))  # Total batch size across all GPU replicas
PHASE2_BATCH_SIZE_ADJUSTMENT = False  # Auto-adjust batch size in Phase 2 based on modality count/weight (can reduce batch size too aggressively)
N_EPOCHS = 200  # Full training epochs

# EPOCH SETTINGS - Understanding the different epoch parameters:
# ----------------------------------------------------------------
# N_EPOCHS: Total training budget for the entire training process
#   - For pre-training (image-only models): Uses N_EPOCHS epochs
#   - For Stage 1 (frozen image branch): Uses STAGE1_EPOCHS epochs
#   - For Stage 2 (fine-tuning): Uses (N_EPOCHS - STAGE1_EPOCHS) epochs
#   - Production: 200 epochs total
#   - Quick test: 50 epochs total (agent_communication/generative_augmentation/test_generative_aug.py)
#
# STAGE1_EPOCHS: Number of epochs for Stage 1 fusion training (frozen image branch)
#   - Only used in two-stage fusion training
#   - Image branch is frozen, only fusion layers train
#   - Typically ~10% of N_EPOCHS (20 out of 200)
#   - Production: 20 epochs
#   - Quick test: 5 epochs
#
# LR_SCHEDULE_EXPLORATION_EPOCHS: Learning rate schedule exploration period
#   - Defines how long to stay at initial LR before entering cosine annealing
#   - Automatically set to 10% of N_EPOCHS (matches STAGE1_EPOCHS ratio)
#   - Production (N_EPOCHS=200): 20 epochs exploration
#   - After exploration, uses cosine annealing with warm restarts
#
# Example production timeline (N_EPOCHS=200, STAGE1_EPOCHS=20):
#   1. Pre-training: 0-200 epochs (trains image-only model)
#   2. Stage 1: 0-20 epochs (frozen image, train fusion)
#   3. Stage 2: 20-200 epochs (fine-tune everything)

# Default backbones (fallback for modalities not in MODALITY_CONFIGS; also used by data pipeline normalization)
RGB_BACKBONE = 'DenseNet121'
MAP_BACKBONE = 'DenseNet121'

# =============================================================================
# Per-Modality Hyperparameters
# =============================================================================
# Each modality can override the global defaults for backbone, head, loss, and
# fine-tuning.  Values here are validated by joint optimization audit:
#   depth_rgb:   agent_communication/joint_optimization_audit/joint_best_config.json (TOP10_4)
#   thermal_map: agent_communication/joint_optimization_audit/joint_best_config.json (TOP10_4)
#   depth_map:   mirrors thermal_map setup (same modality type)
#
# Keys not present in a modality dict fall back to the global defaults above.

MODALITY_CONFIGS = {
    'depth_rgb': {
        # validated: joint optimization audit TOP10_4 (joint_best_config.json)
        'backbone': 'DenseNet121',
        'head_units': [128, 32],         # two-layer head (joint audit best)
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.001,          # Stage 1 (frozen backbone) LR
        'finetune_lr': 1e-5,            # Stage 2 fine-tuning LR
        'finetune_epochs': 30,
        'unfreeze_pct': 0.05,           # top 5% backbone unfrozen in Stage 2 (joint audit best)
        'freeze_bn_stage2': True,        # freeze BN in Stage 2
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
    'depth_map': {
        # mirrors thermal_map setup (same modality type: map images)
        'backbone': 'DenseNet121',
        'head_units': [128],             # single-layer head (matches thermal_map)
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.001,          # same as thermal_map
        'finetune_lr': 1e-5,
        'finetune_epochs': 30,
        'unfreeze_pct': 0.05,           # top 5% backbone unfrozen (matches thermal_map)
        'freeze_bn_stage2': True,        # freeze BN in Stage 2 (matches thermal_map)
        'use_mixup': False,             # matches thermal_map
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
    'thermal_map': {
        # validated: joint optimization audit TOP10_4 (joint_best_config.json)
        'backbone': 'DenseNet121',
        'head_units': [128],             # single-layer head (joint audit best)
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.001,          # Stage 1 (frozen backbone) LR
        'finetune_lr': 1e-5,
        'finetune_epochs': 30,
        'unfreeze_pct': 0.05,           # top 5% backbone unfrozen in Stage 2 (joint audit best)
        'freeze_bn_stage2': True,        # freeze BN in Stage 2
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
    'thermal_rgb': {
        # mirrors depth_rgb setup (same modality type: RGB images)
        'backbone': 'DenseNet121',
        'head_units': [128, 32],
        'head_l2': 0.0,
        'label_smoothing': 0.0,
        'learning_rate': 0.001,
        'finetune_lr': 1e-5,
        'finetune_epochs': 30,
        'unfreeze_pct': 0.05,
        'freeze_bn_stage2': True,
        'use_mixup': False,
        'mixup_alpha': 0.2,
        'early_stop_patience': 15,
        'reduce_lr_patience': 7,
    },
}

def get_modality_config(modality):
    """Get hyperparameters for a specific modality, falling back to globals.

    Resolved at call time so that globals like LABEL_SMOOTHING / STAGE2_FINETUNE_EPOCHS
    are available regardless of import order.
    """
    cfg = MODALITY_CONFIGS.get(modality, {})
    is_rgb = modality in ['depth_rgb', 'thermal_rgb']
    return {
        'backbone': cfg.get('backbone', RGB_BACKBONE if is_rgb else MAP_BACKBONE),
        'head_units': cfg.get('head_units', 256),
        'head_l2': cfg.get('head_l2', 0.0),
        'label_smoothing': cfg.get('label_smoothing', globals().get('LABEL_SMOOTHING', 0.0)),
        'learning_rate': cfg.get('learning_rate', globals().get('PRETRAIN_LR', 1e-3)),
        'finetune_lr': cfg.get('finetune_lr', globals().get('STAGE2_LR', 1e-5)),
        'finetune_epochs': cfg.get('finetune_epochs', globals().get('STAGE2_FINETUNE_EPOCHS', 50)),
        'unfreeze_pct': cfg.get('unfreeze_pct', globals().get('STAGE2_UNFREEZE_PCT', 0.2)),
        'freeze_bn_stage2': cfg.get('freeze_bn_stage2', True),
        'use_mixup': cfg.get('use_mixup', False),
        'mixup_alpha': cfg.get('mixup_alpha', 0.2),
        'early_stop_patience': cfg.get('early_stop_patience', globals().get('EARLY_STOP_PATIENCE', 20)),
        'reduce_lr_patience': cfg.get('reduce_lr_patience', globals().get('REDUCE_LR_PATIENCE', 10)),
    }

# Fusion-specific training parameters
# Validated by joint optimization audit (agent_communication/joint_optimization_audit/joint_best_config.json)
# Winner: TOP10_4 (DenseNet121+DenseNet121, 128px, feature_concat) — mean kappa 0.353 +/- 0.037 over 5 folds
# Beats metadata-only (0.307) with statistical significance (p=0.013)
PRETRAIN_LR = 1e-3  # Default learning rate for Stage 1 head-only training (overridden by per-modality learning_rate)
STAGE1_LR = 1.716e-3  # Learning rate for Stage 1 fusion training (frozen image branch); joint audit TOP10_4: 1.716e-3
STAGE2_LR = 5e-6  # Learning rate for Stage 2 fusion fine-tuning
FUSION_INIT_RF_WEIGHT = 0.70  # Initial RF weight for learnable fusion (0.0-1.0, image weight = 1 - this)
FUSION_STRATEGY = 'feature_concat'  # Fusion strategy for metadata+image combinations:
                                    #   'feature_concat': concat RF probs + image features -> Dense(3). VALIDATED WINNER.
                                    #   'residual': output = softmax(log(RF) + alpha * correction(images)).
                                    #     Tested in fusion audit R9 — underperformed feature_concat on all folds.
FUSION_RESIDUAL_ALPHA_INIT = 0.01  # Initial value for residual gate scalar (only used when FUSION_STRATEGY='residual')
                                   # Small value means model starts nearly at RF-only performance.
                                   # The network learns to increase alpha when images are informative.
FUSION_IMAGE_PROJECTION_DIM = 0  # Project image features to this dim before fusing with metadata (0=disabled)
                                 # Joint audit TOP10_4 confirms: proj_dim=0 is best for feature_concat
STAGE1_EPOCHS = 500  # Stage 1 fusion training epochs (joint audit TOP10_4: 500 with early stopping patience=20)
STAGE2_FINETUNE_EPOCHS = 30  # Stage 2 fine-tuning epochs (joint audit TOP10_4: 30; Stage 2 helps with conservative unfreeze)
STAGE2_UNFREEZE_PCT = 0.05  # Fraction of backbone to unfreeze in Stage 2 (joint audit TOP10_4: 5%)
DATA_PERCENTAGE = 100  # Percentage of data to use (100.0 = all data, 50.0 = half for faster testing)

# Class imbalance handling - PRODUCTION OPTIMIZED (Phase 7 investigation)
# Options: 'none', 'random', 'smote', 'combined', 'combined_smote'
#   'none':   No resampling — use original imbalanced distribution as-is
#   'random': Simple oversampling to MAX class (Kappa ~0.10) - NOT RECOMMENDED
#   'smote': SMOTE synthetic oversampling to MAX class (Kappa ~0.14)
#   'combined': Undersample majority + oversample minority to MIDDLE class (Kappa ~0.17) - RECOMMENDED
#   'combined_smote': Undersample majority + SMOTE minority to MIDDLE class - NOT for fusion (creates synthetic samples without images)
# For production with 25% outlier removal: Expected Kappa 0.31 ± 0.08
SAMPLING_STRATEGY = 'none'  # PRODUCTION: Use 'combined' for best fusion performance

# Additional class weighting based on original class frequencies (applied ON TOP of sampling strategy)
# --- Alpha values: inverse class frequency weights from ORIGINAL (pre-resampling) distribution ---
# Controls: WeightedF1Score metric, focal loss alpha, and optionally RF/training class weights below
USE_FREQUENCY_BASED_WEIGHTS = True   # Compute alpha values from original class distribution
FREQUENCY_WEIGHT_NORMALIZATION = 3.0  # Alpha values normalized to sum to this value (3.0 = standard; 10.0 was too aggressive)

# --- Neural network loss weighting (independent of RF) ---
# Controls: focal loss per-sample weighting via model.fit(class_weight=...)
# Options: 'uniform'    → [1, 1, 1] (no extra weighting)
#          'balanced'   → sklearn compute_class_weight('balanced') from post-resampling data
#          'frequency'  → alpha values (requires USE_FREQUENCY_BASED_WEIGHTS=True)
TRAINING_CLASS_WEIGHT_MODE = 'frequency'

# --- Random Forest hyperparameters (independent of neural network) ---
# Controls: RF model used to generate metadata probabilities fed into the neural network
# Validated by joint optimization audit: TOP10_4 (joint_best_config.json)
RF_N_ESTIMATORS = 300         # Number of trees (joint audit TOP10_4: 300)
RF_CLASS_WEIGHT = 'frequency'  # 'balanced' (sklearn auto), 'frequency' (alpha values), or None
RF_OOF_FOLDS = 5             # Internal OOF folds for training predictions
RF_MAX_DEPTH = 10             # Limit tree depth (joint audit TOP10_4: 10)
RF_MIN_SAMPLES_LEAF = 5       # Min samples per leaf (prevents memorizing rare patterns)
RF_MIN_SAMPLES_SPLIT = 10     # Min samples to split a node

# Metadata feature selection (Mutual Information)
RF_FEATURE_SELECTION = True  # Enable/disable MI-based feature selection before RF training
RF_FEATURE_SELECTION_K = 80  # Number of top features to keep (joint audit TOP10_4: 80)

# Early stopping and learning rate
# Joint optimization audit TOP10_4 uses 20/10 (same as before)
EARLY_STOP_PATIENCE = 20  # Epochs to wait before stopping (fusion training; standalone uses per-modality values)
REDUCE_LR_PATIENCE = 10  # Epochs to wait before reducing LR (fusion training; standalone uses per-modality values)

# =============================================================================
# Data Cleaning and Outlier Detection
# =============================================================================

# Multimodal outlier detection (Isolation Forest on joint feature space)
OUTLIER_REMOVAL = False  # Enable/disable outlier detection and removal
OUTLIER_CONTAMINATION = 0.25  # Expected proportion of outliers (0.0-1.0)
OUTLIER_BATCH_SIZE = 32  # Batch size for on-the-fly feature extraction

# General augmentation (applied during training only, not validation)
# RGB images: brightness ±60%, contrast 0.6-1.4x, saturation 0.6-1.4x, gaussian noise σ=0.15
# Map images: brightness ±40%, contrast 0.6-1.4x, gaussian noise σ=0.1 (no saturation)
# Applied with 60% probability, different settings for RGB vs maps
USE_GENERAL_AUGMENTATION = True  # Enable/disable general (non-generative) augmentation

# Generative augmentation (Stable Diffusion-based synthetic data generation)
# V3: Uses single conditional SDXL model fine-tuned on all phases
# V2 (legacy): Uses separate SD 1.5 models per modality/phase from results/GenerativeAug_Models/models_5_7/
# Only applies to RGB images (depth_rgb, thermal_rgb)
USE_GENERATIVE_AUGMENTATION = False  # Enable/disable generative augmentation
GENERATIVE_AUG_VERSION = 'v3'  # 'v3' = SDXL conditional model, 'v2' = SD 1.5 per-phase models
GENERATIVE_AUG_PROB = 0.06  # Probability of applying generative augmentation (0.0-1.0)
GENERATIVE_AUG_MIX_RATIO = (0.01, 0.05)  # Range for mixing real/synthetic samples (min, max)
GENERATIVE_AUG_INFERENCE_STEPS = 50  # Diffusion inference steps (25=fast/good quality, 50=higher quality but 2x slower)
GENERATIVE_AUG_BATCH_LIMIT = 32  # Max batch size for generative aug (increased - full GPU mode has more memory available)
GENERATIVE_AUG_PHASES = ['I', 'P', 'R']  # Which phases to generate images for

# SDXL-specific settings (V3)
# Single conditional SDXL model fine-tuned on all phases with phase-specific prompts
# Prompts: PHASE_I/P/R, diabetic foot ulcer, [phase] phase wound
GENERATIVE_AUG_SDXL_MODEL_PATH = 'results/GenerativeAug_Models/sdxl_full/checkpoint_epoch_0035.pt'
GENERATIVE_AUG_SDXL_RESOLUTION = 512  # Native SDXL resolution (will be resized to IMAGE_SIZE if different)
GENERATIVE_AUG_SDXL_GUIDANCE_SCALE = 4.0  # CFG scale (lower = more training data influence, higher = more prompt influence)

# SD 1.5 legacy settings (V2) - kept for backward compatibility
GENERATIVE_AUG_MODEL_PATH = 'results/GenerativeAug_Models/models_5_7'  # Path to SD 1.5 models (per-phase)
GENERATIVE_AUG_MAX_MODELS = 3  # Max SD 1.5 models loaded in GPU memory simultaneously (V2 only)

# Misclassification tracking (for iterative data polishing)
# Options: 'none', 'both', 'valid', 'train'
#   'none': Disable tracking (fastest, default for production)
#   'both': Track from both train and validation sets
#   'valid': Track only from validation set
#   'train': Track only from training set (not recommended)
TRACK_MISCLASS = 'none'  # Misclassification tracking mode: 'both', 'valid', 'train', 'none'

# =============================================================================
# Confidence-Based Filtering (scripts/confidence_based_filtering.py)
# =============================================================================
# Identifies "bad" samples by analyzing model prediction confidence.
# Low-confidence samples are often annotation errors or ambiguous cases.
# Faster than iterative misclassification tracking (requires only 1 training run).

# Enable/disable confidence-based filtering during training
USE_CONFIDENCE_FILTERING = False  # Disabled: replaced by RF LOO influence filtering

# Filtering parameters - PER-CLASS percentiles (bottom X% to remove from each class)
# This allows different filtering intensity for each class based on data quality
CONFIDENCE_FILTER_PERCENTILE_I = 17  # Inflammatory class (class 0)
CONFIDENCE_FILTER_PERCENTILE_P = 23  # Proliferative class (class 1) - majority class
CONFIDENCE_FILTER_PERCENTILE_R = 15  # Remodeling class (class 2) - minority class

# Legacy single percentile (used as fallback if per-class not specified)
CONFIDENCE_FILTER_PERCENTILE = 15  # Remove bottom X% lowest confidence samples (default: 15%)

CONFIDENCE_FILTER_MODE = 'per_class'  # 'global' = bottom X% overall, 'per_class' = bottom X% per class
CONFIDENCE_FILTER_MIN_SAMPLES = 50  # Minimum samples to keep per class (safety limit)
CONFIDENCE_FILTER_MAX_CLASS_REMOVAL_PCT = 30  # Never remove more than X% from ANY class (protects minority classes)

# Confidence metric to use
# 'max_prob': Maximum softmax probability (simple, fast)
# 'margin': Difference between top-2 probabilities (measures uncertainty)
# 'entropy': Prediction entropy (information-theoretic uncertainty)
CONFIDENCE_METRIC = 'max_prob'

# Output files (relative to results directory)
CONFIDENCE_FILTER_RESULTS_FILE = 'confidence_filter_results.json'
CONFIDENCE_FILTER_BAD_SAMPLES_FILE = 'confidence_low_samples.csv'

# Misclassification filtering thresholds (for core-data mode)
# Core dataset mode (uses optimized thresholds from auto_polish_dataset_v2.py)
# When True: filters dataset using thresholds above + frequent_misclassifications_saved.csv
# Requires: frequent_misclassifications_saved.csv in results/ or results/misclassifications_saved/
# If CSV missing: prints warning and continues with unfiltered data
# USE_CORE_DATA = os.environ.get('OVERRIDE_USE_CORE_DATA', 'false').lower() == 'true' # CORE DATA OFF
USE_CORE_DATA = os.environ.get('OVERRIDE_USE_CORE_DATA', 'false').lower() == 'true' or True # CORE DATA ON
THRESHOLD_I = int(os.environ['OVERRIDE_THRESHOLD_I']) if 'OVERRIDE_THRESHOLD_I' in os.environ else 53
THRESHOLD_P = int(os.environ['OVERRIDE_THRESHOLD_P']) if 'OVERRIDE_THRESHOLD_P' in os.environ else 84
THRESHOLD_R = int(os.environ['OVERRIDE_THRESHOLD_R']) if 'OVERRIDE_THRESHOLD_R' in os.environ else 70

# =============================================================================
# RF LOO Influence Filtering (replaces confidence-based filtering for RF)
# =============================================================================
# For each unique metadata pattern, measures OOF Kappa change when removed.
# Patterns whose removal improves Kappa are flagged as harmful and excluded
# from BOTH training and validation sets (applied before the split).
USE_RF_LOO_FILTERING = False
RF_LOO_MIN_INFLUENCE = 0.005    # Min influence score to flag pattern as harmful (0.001 is noise)
RF_LOO_MAX_REMOVAL_PCT = 15     # Max % of patterns to remove per class
RF_LOO_MIN_PATTERNS_PER_CLASS = 50  # Min unique patterns to keep per class

# =============================================================================
# Verbosity and Progress Tracking
# =============================================================================

# Verbosity levels:
# 0 = MINIMAL: Only essential info (errors, final results)
# 1 = NORMAL: Standard output (default - current behavior)
# 2 = DETAILED: Include debug info, intermediate metrics
# 3 = FULL: Everything + progress bars with time estimates
VERBOSITY = 3  # Default verbosity level

# Epoch printing settings
EPOCH_PRINT_INTERVAL = 20  # Print training progress every N epochs (0 = print all epochs)

# Progress bar settings (used when VERBOSITY >= 3)
SHOW_PROGRESS_BAR = True  # Enable/disable progress bar
PROGRESS_BAR_UPDATE_INTERVAL = 1  # Seconds between progress bar updates

# =============================================================================
# Gating Network Configuration
# =============================================================================

# Master switch: enable/disable gating network ensemble after all combinations are trained
USE_GATING_NETWORK = True

# Architecture parameters
GATING_NUM_HEADS_MULTIPLIER = 1  # num_heads = max(8, num_models + this)
GATING_KEY_DIM_MULTIPLIER = 2  # key_dim = max(16, multiplier * (num_models + 1))

# Neural network hyperparameters (moved from hardcoded values in main.py)
DROPOUT_RATE = 0.3  # Dropout rate for ResidualBlock
GATING_DROPOUT_RATE = 0.1  # Dropout rate for gating network MultiHeadAttention layers
GATING_L2_REGULARIZATION = 1e-4  # L2 regularization for gating network dense layers
ATTENTION_TEMPERATURE = 0.1  # Initial temperature for dual-level attention
TEMPERATURE_MIN_VALUE = 0.01  # Minimum temperature value for scaling
ATTENTION_CLASS_NUM_HEADS = 8  # Number of attention heads for class-level attention
ATTENTION_CLASS_KEY_DIM = 32  # Key dimension for class-level attention

# Training parameters
GATING_LEARNING_RATE = 1e-3  # Adam optimizer learning rate
GATING_EPOCHS = 30  # Maximum number of epochs (reduced for faster testing)
GATING_BATCH_SIZE = 128  # Batch size for gating network training (increased for faster training)
GATING_VERBOSE = 0  # Training verbosity (0=silent, 1=progress bar, 2=epoch)

# Callbacks - ReduceLROnPlateau
GATING_REDUCE_LR_FACTOR = 0.5  # Factor to reduce learning rate
GATING_REDUCE_LR_PATIENCE = 10  # Epochs to wait before reducing LR
GATING_REDUCE_LR_MIN_LR = 1e-9  # Minimum learning rate
GATING_REDUCE_LR_MIN_DELTA = 2e-3  # Minimum change to qualify as improvement

# Callbacks - EarlyStopping
GATING_EARLY_STOP_PATIENCE = 30  # Epochs to wait before stopping
GATING_EARLY_STOP_MIN_DELTA = 2e-2  # Minimum change to qualify as improvement
GATING_EARLY_STOP_VERBOSE = 2  # Verbosity level

# =============================================================================
# Hierarchical Gating Network Configuration
# =============================================================================

# Architecture parameters
HIERARCHICAL_EMBEDDING_DIM = 32  # Embedding dimension for hierarchical network
HIERARCHICAL_NUM_HEADS = 2  # Number of attention heads in transformer
HIERARCHICAL_FF_DIM_MULTIPLIER = 2  # Feed-forward dim = embedding_dim * this

# Hierarchical neural network hyperparameters (moved from hardcoded values in main.py)
TRANSFORMER_DROPOUT_RATE = 0.1  # Dropout rate for TransformerBlock
HIERARCHICAL_L2_REGULARIZATION = 0.001  # L2 regularization for hierarchical gating output layer
LAYER_NORM_EPSILON = 1e-6  # Epsilon for LayerNormalization layers

# Training parameters
HIERARCHICAL_LEARNING_RATE = 1e-3  # Adam optimizer learning rate
HIERARCHICAL_EPOCHS = 50  # Maximum number of epochs
HIERARCHICAL_BATCH_SIZE = 32  # Batch size for hierarchical training
HIERARCHICAL_PATIENCE = 20  # Early stopping patience
HIERARCHICAL_VERBOSE = 2  # Training verbosity

# Callbacks - ReduceLROnPlateau
HIERARCHICAL_REDUCE_LR_FACTOR = 0.5  # Factor to reduce learning rate
HIERARCHICAL_REDUCE_LR_PATIENCE = 10  # Epochs to wait before reducing LR
HIERARCHICAL_REDUCE_LR_MIN_LR = 1e-12  # Minimum learning rate

# Focal loss for hierarchical gating
HIERARCHICAL_FOCAL_GAMMA = 3.0  # Gamma parameter for focal loss
HIERARCHICAL_FOCAL_ALPHA = None  # Alpha parameter (None = no class weighting)

# =============================================================================
# Learning Rate Scheduler (DynamicLRSchedule)
# NOTE: These LR_SCHEDULE_* params are currently UNUSED. They are only referenced
# by the DynamicLRSchedule class (defined in src/main.py) which is never
# instantiated anywhere. Kept for potential future use.
# =============================================================================

LR_SCHEDULE_INITIAL_LR = 1e-3  # Initial learning rate
LR_SCHEDULE_MIN_LR = 1e-14  # Minimum learning rate
LR_SCHEDULE_EXPLORATION_EPOCHS = int(N_EPOCHS * 0.10)  # Exploration epochs (10% of N_EPOCHS, auto-adjusts when N_EPOCHS changes)
LR_SCHEDULE_CYCLE_LENGTH = 30  # Initial cycle length
LR_SCHEDULE_CYCLE_MULTIPLIER = 2.0  # Factor to multiply cycle length

# Adaptive LR adjustment
LR_SCHEDULE_PATIENCE_THRESHOLD = 5  # Epochs without improvement before adjusting
LR_SCHEDULE_DECAY_FACTOR = 0.8  # Factor to reduce max LR (20% reduction)

# =============================================================================
# Model Combination Search Configuration
# =============================================================================

# Search parameters
SEARCH_MIN_MODELS = 2  # Minimum number of models in combination
SEARCH_MAX_TRIES = 100  # Maximum combinations to try per model count
SEARCH_STEP_SIZE_FRACTION = 0.20  # Fraction of models for step size (20%)
SEARCH_NUM_PROCESSES = 3  # Maximum number of parallel processes

# Manual exclusions (list of model indices to exclude)
SEARCH_EXCLUDED_MODELS = [800000]  # Models to exclude from search

# =============================================================================
# Attention Visualization Parameters
# =============================================================================

# Visualization frequency
ATTENTION_VIS_FREQUENCY = 100  # Visualize every N epochs (1 = every epoch)

# Heatmap scale ranges (fixed scales for consistent visualization)
ATTENTION_MODEL_VMIN = 0.075  # Minimum value for model attention heatmap
ATTENTION_MODEL_VMAX = 0.275  # Maximum value for model attention heatmap
ATTENTION_CLASS_VMIN = 0.20  # Minimum value for class attention heatmap
ATTENTION_CLASS_VMAX = 0.45  # Maximum value for class attention heatmap

# =============================================================================
# Attention Entropy Loss Parameters
# =============================================================================

# Entropy calculation
ENTROPY_EPSILON = 1e-10  # Small value to prevent log(0)

# Entropy weighting (model vs class attention)
ENTROPY_MODEL_WEIGHT = 0.7  # Weight for model-level entropy (70%)
ENTROPY_CLASS_WEIGHT = 0.3  # Weight for class-level entropy (30%)

# Dynamic entropy weight in loss
ENTROPY_LOSS_WEIGHT = 0.2  # Base weight for entropy in total loss

# =============================================================================
# Loss Function Parameters
# =============================================================================

# Focal ordinal loss defaults (when not specified)
FOCAL_ORDINAL_WEIGHT = 0.0  # Ordinal penalty disabled — argmax-based distance is non-differentiable, adds complexity with no benefit. Pure focal loss with alpha is sufficient.
LABEL_SMOOTHING = 0.0  # Label smoothing for focal loss (joint audit TOP10_4: 0.0)
# Note: HIERARCHICAL_FOCAL_GAMMA and HIERARCHICAL_FOCAL_ALPHA are used for hierarchical gating

# =============================================================================
# Cross-Validation Configuration
# =============================================================================
#
# IMPORTANT: Use --cv_folds flag (NOT this config) to control main training!
#
# --cv_folds N (command line, default: 5)
#   - Controls BOTH main training AND confidence filtering folds
#   - Each fold runs in isolated subprocess to prevent memory leaks
#   - Examples:
#       --cv_folds 3              # Run 3 folds
#       --cv_folds 3 --fold 2     # Run only fold 2 (others load from disk)
#
# CV_N_SPLITS (this config)
#   - Used ONLY for hierarchical gating network
#   - Does NOT affect main training or confidence filtering
#
# Quick testing: python src/main.py --cv_folds 2 --data_percentage 40 --resume_mode fresh
#
CV_N_SPLITS = 5  # For hierarchical gating only (use --cv_folds for main training)
CV_RANDOM_STATE = 42  # Random state for reproducibility
CV_SHUFFLE = True  # Whether to shuffle data before splitting

# =============================================================================
# Grid Search Configuration
# =============================================================================

# Parameter grid for loss function tuning
GRID_SEARCH_ORDINAL_WEIGHTS = [1.0]  # Ordinal weight values to try
GRID_SEARCH_GAMMAS = [2.0]  # Gamma values to try (joint audit TOP10_4: 2.0)
GRID_SEARCH_ALPHAS = [  # Alpha value combinations to try
    [0.598, 0.315, 1.597],  # Current weights
    [1, 0.5, 2],  # Alternative 1
    [1.5, 0.3, 1.5]  # Alternative 2
]

# =============================================================================
# File I/O Configuration
# =============================================================================

# Progress saving/loading
PROGRESS_MAX_RETRIES = 6  # Maximum retry attempts for file operations
PROGRESS_RETRY_DELAY = 0.4  # Delay between retries (seconds)

# =============================================================================
# Modality Search Configuration
# =============================================================================

# All available modalities
# ALL_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
ALL_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_map']

# Search mode: 'all' tests all 31 combinations, 'custom' uses INCLUDED_COMBINATIONS
MODALITY_SEARCH_MODE = 'all'  # Options: 'all', 'custom'

# Combinations to exclude (list of tuples)
EXCLUDED_COMBINATIONS = []  # e.g., [('depth_rgb',), ('thermal_rgb',)]

# Combinations to include (only used when MODALITY_SEARCH_MODE = 'custom')
# Can be overridden via OVERRIDE_INCLUDED_COMBO env var (e.g. "metadata+depth_rgb")
_combo_override = os.environ.get('OVERRIDE_INCLUDED_COMBO')
if _combo_override:
    INCLUDED_COMBINATIONS = [tuple(_combo_override.split('+'))]
else:
    INCLUDED_COMBINATIONS = [
        ('metadata', 'depth_rgb', 'thermal_map',),  # Priority 1: full fusion (target combo)
        ('metadata', 'thermal_map',),                # Priority 2: metadata + best image
        ('metadata', 'depth_rgb',),                  # Priority 3: metadata + depth
        ('metadata',),                               # Priority 4: metadata alone (baseline)
    ]

# Results file naming
RESULTS_CSV_FILENAME = 'modality_combination_results.csv'  # Output CSV filename

# Multiple configs per modality combination for gating network
# Set to False in search mode - gating network is for combining DIFFERENT modality combinations, not configs
# Set to True only if you want to test different loss parameters for the same modality combination
SEARCH_MULTIPLE_CONFIGS = False  # Disabled: gating network should ensemble across modality combinations, not configs
SEARCH_CONFIG_VARIANTS = 2  # Number of config variants to create per modality combination (if enabled)

# =============================================================================
# Environment Configuration
# =============================================================================

# TensorFlow threading configuration (for CPU performance tuning)
TF_OMP_NUM_THREADS = "2"  # OpenMP threads
TF_NUM_INTEROP_THREADS = "2"  # TensorFlow inter-op parallelism threads
TF_NUM_INTRAOP_THREADS = "4"  # TensorFlow intra-op parallelism threads

# TensorFlow determinism (for reproducibility)
TF_DETERMINISTIC_OPS = "1"  # Enable deterministic operations
TF_CUDNN_DETERMINISTIC = "1"  # Enable deterministic cuDNN operations

# =============================================================================
# Helper Functions
# =============================================================================

def get_gating_config():
    """Returns gating network configuration as a dictionary."""
    return {
        'learning_rate': GATING_LEARNING_RATE,
        'epochs': GATING_EPOCHS,
        'batch_size': GATING_BATCH_SIZE,
        'verbose': GATING_VERBOSE,
        'reduce_lr_factor': GATING_REDUCE_LR_FACTOR,
        'reduce_lr_patience': GATING_REDUCE_LR_PATIENCE,
        'reduce_lr_min_lr': GATING_REDUCE_LR_MIN_LR,
        'reduce_lr_min_delta': GATING_REDUCE_LR_MIN_DELTA,
        'early_stop_patience': GATING_EARLY_STOP_PATIENCE,
        'early_stop_min_delta': GATING_EARLY_STOP_MIN_DELTA,
        'early_stop_verbose': GATING_EARLY_STOP_VERBOSE,
    }

def get_hierarchical_config():
    """Returns hierarchical gating network configuration as a dictionary."""
    return {
        'embedding_dim': HIERARCHICAL_EMBEDDING_DIM,
        'num_heads': HIERARCHICAL_NUM_HEADS,
        'ff_dim_multiplier': HIERARCHICAL_FF_DIM_MULTIPLIER,
        'learning_rate': HIERARCHICAL_LEARNING_RATE,
        'epochs': HIERARCHICAL_EPOCHS,
        'batch_size': HIERARCHICAL_BATCH_SIZE,
        'patience': HIERARCHICAL_PATIENCE,
        'verbose': HIERARCHICAL_VERBOSE,
        'reduce_lr_factor': HIERARCHICAL_REDUCE_LR_FACTOR,
        'reduce_lr_patience': HIERARCHICAL_REDUCE_LR_PATIENCE,
        'reduce_lr_min_lr': HIERARCHICAL_REDUCE_LR_MIN_LR,
        'focal_gamma': HIERARCHICAL_FOCAL_GAMMA,
        'focal_alpha': HIERARCHICAL_FOCAL_ALPHA,
    }

def get_lr_schedule_config():
    """Returns learning rate scheduler configuration as a dictionary."""
    return {
        'initial_lr': LR_SCHEDULE_INITIAL_LR,
        'min_lr': LR_SCHEDULE_MIN_LR,
        'exploration_epochs': LR_SCHEDULE_EXPLORATION_EPOCHS,
        'cycle_length': LR_SCHEDULE_CYCLE_LENGTH,
        'cycle_multiplier': LR_SCHEDULE_CYCLE_MULTIPLIER,
        'patience_threshold': LR_SCHEDULE_PATIENCE_THRESHOLD,
        'decay_factor': LR_SCHEDULE_DECAY_FACTOR,
    }

def get_search_config():
    """Returns model combination search configuration as a dictionary."""
    return {
        'min_models': SEARCH_MIN_MODELS,
        'max_tries': SEARCH_MAX_TRIES,
        'step_size_fraction': SEARCH_STEP_SIZE_FRACTION,
        'num_processes': SEARCH_NUM_PROCESSES,
        'excluded_models': SEARCH_EXCLUDED_MODELS,
    }

def get_attention_config():
    """Returns attention mechanism configuration as a dictionary."""
    return {
        'vis_frequency': ATTENTION_VIS_FREQUENCY,
        'model_vmin': ATTENTION_MODEL_VMIN,
        'model_vmax': ATTENTION_MODEL_VMAX,
        'class_vmin': ATTENTION_CLASS_VMIN,
        'class_vmax': ATTENTION_CLASS_VMAX,
        'entropy_epsilon': ENTROPY_EPSILON,
        'entropy_model_weight': ENTROPY_MODEL_WEIGHT,
        'entropy_class_weight': ENTROPY_CLASS_WEIGHT,
        'entropy_loss_weight': ENTROPY_LOSS_WEIGHT,
    }

def get_environment_config():
    """Returns environment configuration as a dictionary."""
    return {
        'omp_num_threads': TF_OMP_NUM_THREADS,
        'interop_threads': TF_NUM_INTEROP_THREADS,
        'intraop_threads': TF_NUM_INTRAOP_THREADS,
        'deterministic_ops': TF_DETERMINISTIC_OPS,
        'cudnn_deterministic': TF_CUDNN_DETERMINISTIC,
    }

def apply_environment_config():
    """Apply environment configuration to os.environ."""
    import os
    os.environ["OMP_NUM_THREADS"] = TF_OMP_NUM_THREADS
    os.environ['TF_NUM_INTEROP_THREADS'] = TF_NUM_INTEROP_THREADS
    os.environ['TF_NUM_INTRAOP_THREADS'] = TF_NUM_INTRAOP_THREADS
    os.environ['TF_DETERMINISTIC_OPS'] = TF_DETERMINISTIC_OPS
    os.environ['TF_CUDNN_DETERMINISTIC'] = TF_CUDNN_DETERMINISTIC
