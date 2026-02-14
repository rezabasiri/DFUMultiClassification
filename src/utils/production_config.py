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

# =============================================================================
# Training Parameters
# =============================================================================

# Core training hyperparameters
IMAGE_SIZE = 256  # Image dimensions (256x256 optimal for fusion - see agent_communication/fusion_fix/FUSION_FIX_GUIDE.md)
GLOBAL_BATCH_SIZE = 32  # Total batch size across all GPU replicas (16 per GPU; reduced from 600 to get ~17 steps/epoch with ~516 training samples instead of 1)
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

# Image backbone selection (for backbone comparison experiments)
# Options: 'SimpleCNN', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3'
# Best combination from 20-test comparison: B3+B1 (Kappa=0.3295, 79.7% improvement over baseline)
RGB_BACKBONE = 'EfficientNetB3'  # Backbone for RGB images (depth_rgb, thermal_rgb)
MAP_BACKBONE = 'EfficientNetB1'  # Backbone for map images (depth_map, thermal_map)

# Fusion-specific training parameters
PRETRAIN_LR = 1e-3  # Learning rate for image-only pre-training (trains projection+classifier layers on top of ImageNet features; higher than fusion stages because pre-training starts from random head weights)
STAGE1_LR = 1e-4  # Learning rate for Stage 1 fusion training (frozen image branch)
STAGE2_LR = 1e-5  # Learning rate for Stage 2 fine-tuning (unfrozen image branch, lower to preserve pre-trained features)
FUSION_INIT_RF_WEIGHT = 0.70  # Initial RF weight for learnable fusion (0.0-1.0, image weight = 1 - this)
STAGE1_EPOCHS = 20  # Stage 1 fusion training epochs (frozen image branch)
DATA_PERCENTAGE = 100  # Percentage of data to use (100.0 = all data, 50.0 = half for faster testing)

# Class imbalance handling - PRODUCTION OPTIMIZED (Phase 7 investigation)
# Options: 'random', 'smote', 'combined', 'combined_smote'
#   'random': Simple oversampling to MAX class (Kappa ~0.10) - NOT RECOMMENDED
#   'smote': SMOTE synthetic oversampling to MAX class (Kappa ~0.14)
#   'combined': Undersample majority + oversample minority to MIDDLE class (Kappa ~0.17) - RECOMMENDED
#   'combined_smote': Undersample majority + SMOTE minority to MIDDLE class - NOT for fusion (creates synthetic samples without images)
# For production with 25% outlier removal: Expected Kappa 0.31 ± 0.08
SAMPLING_STRATEGY = 'combined'  # PRODUCTION: Use 'combined' for best fusion performance

# Additional class weighting based on original class frequencies (applied ON TOP of sampling strategy)
# When enabled, computes alpha weights from inverse class frequencies before resampling,
# then applies these weights during training via model.fit(class_weight=...) or focal loss alpha
# This provides extra emphasis on minority classes even after resampling balances the dataset
USE_FREQUENCY_BASED_WEIGHTS = False  # Disabled: after 'combined' resampling classes are balanced, extra weighting causes degenerate predictions
FREQUENCY_WEIGHT_NORMALIZATION = 3.0  # Weights are normalized to sum to this value (default: 3.0 for 3 classes)

# Metadata feature selection (Mutual Information)
RF_FEATURE_SELECTION = True  # Enable/disable MI-based feature selection before RF training
RF_FEATURE_SELECTION_K = 40  # Number of top features to keep (only used if RF_FEATURE_SELECTION=True)

# Early stopping and learning rate
EARLY_STOP_PATIENCE = 20  # Epochs to wait before stopping (increased for longer training)
REDUCE_LR_PATIENCE = 10  # Epochs to wait before reducing LR (increased for longer training)

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
TRACK_MISCLASS = 'none'  # Misclassification tracking mode

# =============================================================================
# Confidence-Based Filtering (scripts/confidence_based_filtering.py)
# =============================================================================
# Identifies "bad" samples by analyzing model prediction confidence.
# Low-confidence samples are often annotation errors or ambiguous cases.
# Faster than iterative misclassification tracking (requires only 1 training run).

# Enable/disable confidence-based filtering during training
USE_CONFIDENCE_FILTERING = True  # Set to True to enable

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
# Lower threshold = exclude more misclassified samples
# Set to None to use auto-optimized values from bayesian_optimization_results.json
THRESHOLD_I = None  # Inflammatory class threshold
THRESHOLD_P = None  # Proliferative class threshold
THRESHOLD_R = None  # Remodeling class threshold (typically higher to protect minority)

# Core dataset mode (uses optimized thresholds from auto_polish_dataset_v2.py)
USE_CORE_DATA = False  # Use Bayesian-optimized core dataset with misclassification filtering

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
FOCAL_ORDINAL_WEIGHT = 0.5  # Default ordinal penalty weight
# Note: HIERARCHICAL_FOCAL_GAMMA and HIERARCHICAL_FOCAL_ALPHA are used for hierarchical gating

# =============================================================================
# Cross-Validation Configuration
# =============================================================================
#
# IMPORTANT: Use --cv_folds flag (NOT this config) to control main training!
#
# --cv_folds N (command line, default: 3)
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
CV_N_SPLITS = 2  # For hierarchical gating only (use --cv_folds for main training)
CV_RANDOM_STATE = 42  # Random state for reproducibility
CV_SHUFFLE = True  # Whether to shuffle data before splitting

# =============================================================================
# Grid Search Configuration
# =============================================================================

# Parameter grid for loss function tuning
GRID_SEARCH_ORDINAL_WEIGHTS = [1.0]  # Ordinal weight values to try
GRID_SEARCH_GAMMAS = [2.0, 3.0]  # Gamma values to try
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
ALL_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']

# Search mode: 'all' tests all 31 combinations, 'custom' uses INCLUDED_COMBINATIONS
MODALITY_SEARCH_MODE = 'custom'  # Options: 'all', 'custom'

# Combinations to exclude (list of tuples)
EXCLUDED_COMBINATIONS = []  # e.g., [('depth_rgb',), ('thermal_rgb',)]

# Combinations to include (only used when MODALITY_SEARCH_MODE = 'custom')
INCLUDED_COMBINATIONS = [
    ('metadata',),
    # ('metadata', 'depth_rgb'),
] # e.g., [('metadata',), ('depth_rgb', 'thermal_rgb',)]

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
