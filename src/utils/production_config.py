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
IMAGE_SIZE = 450  # Image dimensions (128x128 pixels for better detail)
# IMAGE_SIZE = 64  # Image dimensions (64x64 pixels)
GLOBAL_BATCH_SIZE = 80  # Total batch size across all GPU replicas (for 2x RTX 5090 32GB VRAM with 128x128 images)
N_EPOCHS = 150  # Quick speed test (normally 120 for full training)

# Early stopping and learning rate
EARLY_STOP_PATIENCE = 20  # Epochs to wait before stopping (increased for longer training)
REDUCE_LR_PATIENCE = 5  # Epochs to wait before reducing LR (increased for longer training)

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

# Training parameters
GATING_LEARNING_RATE = 1e-3  # Adam optimizer learning rate
GATING_EPOCHS = 30  # Maximum number of epochs (reduced for faster testing)
GATING_BATCH_SIZE = 128  # Batch size for gating network training (increased for faster training)
GATING_VERBOSE = 0  # Training verbosity (0=silent, 1=progress bar, 2=epoch)

# Callbacks - ReduceLROnPlateau
GATING_REDUCE_LR_FACTOR = 0.5  # Factor to reduce learning rate
GATING_REDUCE_LR_PATIENCE = 5  # Epochs to wait before reducing LR
GATING_REDUCE_LR_MIN_LR = 1e-9  # Minimum learning rate
GATING_REDUCE_LR_MIN_DELTA = 2e-3  # Minimum change to qualify as improvement

# Callbacks - EarlyStopping
GATING_EARLY_STOP_PATIENCE = 20  # Epochs to wait before stopping
GATING_EARLY_STOP_MIN_DELTA = 2e-2  # Minimum change to qualify as improvement
GATING_EARLY_STOP_VERBOSE = 2  # Verbosity level

# =============================================================================
# Hierarchical Gating Network Configuration
# =============================================================================

# Architecture parameters
HIERARCHICAL_EMBEDDING_DIM = 32  # Embedding dimension for hierarchical network
HIERARCHICAL_NUM_HEADS = 2  # Number of attention heads in transformer
HIERARCHICAL_FF_DIM_MULTIPLIER = 2  # Feed-forward dim = embedding_dim * this

# Training parameters
HIERARCHICAL_LEARNING_RATE = 1e-3  # Adam optimizer learning rate
HIERARCHICAL_EPOCHS = 50  # Maximum number of epochs
HIERARCHICAL_BATCH_SIZE = 32  # Batch size for hierarchical training
HIERARCHICAL_PATIENCE = 20  # Early stopping patience
HIERARCHICAL_VERBOSE = 2  # Training verbosity

# Callbacks - ReduceLROnPlateau
HIERARCHICAL_REDUCE_LR_FACTOR = 0.5  # Factor to reduce learning rate
HIERARCHICAL_REDUCE_LR_PATIENCE = 7  # Epochs to wait before reducing LR
HIERARCHICAL_REDUCE_LR_MIN_LR = 1e-12  # Minimum learning rate

# Focal loss for hierarchical gating
HIERARCHICAL_FOCAL_GAMMA = 3.0  # Gamma parameter for focal loss
HIERARCHICAL_FOCAL_ALPHA = None  # Alpha parameter (None = no class weighting)

# =============================================================================
# Learning Rate Scheduler (DynamicLRSchedule)
# =============================================================================

LR_SCHEDULE_INITIAL_LR = 1e-3  # Initial learning rate
LR_SCHEDULE_MIN_LR = 1e-14  # Minimum learning rate
LR_SCHEDULE_EXPLORATION_EPOCHS = 10  # Number of exploration epochs
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
FOCAL_GAMMA = 1.0  # Reduced from 2.0 to be less aggressive with easy examples
FOCAL_ALPHA = 0.25  # Default focal loss alpha

# =============================================================================
# Cross-Validation Configuration
# =============================================================================

CV_N_SPLITS = 3  # Number of cross-validation folds
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
    ('metadata', 'depth_rgb', 'depth_map', 'thermal_map',),  # Temporary: Phase 2 evaluation
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
