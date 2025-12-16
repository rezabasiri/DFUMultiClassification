"""
Configuration file for demo and testing scripts.

This file centralizes all hyperparameters and configuration settings
for easy tuning and experimentation. Values can be modified here
without changing the main code.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Training/validation split
TRAIN_PATIENT_PERCENTAGE = 0.67  # Percentage of patients for training (67%)

# Data split validation threshold
MAX_SPLIT_DIFF = 0.3  # Maximum allowed difference in class distributions (30%)
                      # Relaxed for small demo dataset. Use 0.1 for production.

# Image preprocessing
IMAGE_SIZE = 64  # Image dimensions (64x64) - smaller for faster demo testing
                 # Use 128 for production

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training epochs
N_EPOCHS = 3  # Number of training epochs per run/fold
              # Demo uses 3 for quick testing. Production typically uses 100-1000

# Batch size
BATCH_SIZE = 4  # Batch size for training and validation
                # Demo uses 4 for limited memory. Production typically uses 16-32

# Optimizer settings
LEARNING_RATE = 0.0001  # Adam optimizer learning rate
                        # Typical range: 0.0001 - 0.001

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================

# Focal loss parameters
FOCAL_LOSS_GAMMA = 2.0  # Focal loss gamma parameter
                        # Controls focus on hard examples
                        # Typical range: 0.5 - 5.0

# ============================================================================
# CROSS-VALIDATION CONFIGURATION
# ============================================================================

# Cross-validation settings
USE_CV = False  # Enable/disable cross-validation
                # False: Single train/val split
                # True: K-fold cross-validation

N_FOLDS = 3  # Number of cross-validation folds
             # Only used if USE_CV=True
             # Typical values: 3, 5, 10

# ============================================================================
# MODALITY CONFIGURATION
# ============================================================================

# Available modalities
ALL_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']

# Modality input shapes (automatically determined, but documented here)
# - metadata: (3,) - RF probability features
# - depth_rgb: (IMAGE_SIZE, IMAGE_SIZE, 3)
# - depth_map: (IMAGE_SIZE, IMAGE_SIZE, 3)
# - thermal_rgb: (IMAGE_SIZE, IMAGE_SIZE, 3)
# - thermal_map: (IMAGE_SIZE, IMAGE_SIZE, 3)

# ============================================================================
# VERBOSITY CONFIGURATION
# ============================================================================

# Control output verbosity for different operations
VERBOSE_TRAINING = 0  # Keras training verbose level
                      # 0: Silent
                      # 1: Progress bar
                      # 2: One line per epoch

VERBOSE_PREDICTIONS = 0  # Model prediction verbose level
                         # 0: Silent
                         # 1: Progress bar

VERBOSE_TESTING = True  # Enable/disable detailed test progress prints
                        # True: Show detailed progress for each test
                        # False: Minimal output (results only)

# ============================================================================
# PERFORMANCE REPORTING CONFIGURATION
# ============================================================================

# Number of top performers to display
TOP_N_PERFORMERS = 5  # Number of best performing combinations to show

# Metrics precision
METRICS_PRECISION = 4  # Number of decimal places for metrics display

# ============================================================================
# FILE OUTPUT CONFIGURATION
# ============================================================================

# Results directory structure
RESULTS_DEMO_DIR = "results/demo"  # Directory for demo test results

# Results file naming
RESULTS_FILE_PREFIX = "modality_test_results"  # Prefix for results files
RESULTS_FILE_EXTENSION = ".txt"  # File extension for results

# ============================================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================================

# Random seed base (actual seed varies by run/fold)
RANDOM_SEED_BASE = 42  # Base seed for reproducibility
                       # Each fold uses: RANDOM_SEED_BASE + fold * (fold + 3)

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_demo_config():
    """
    Returns a dictionary with all demo configuration parameters.

    Returns:
        dict: Configuration dictionary
    """
    return {
        # Data
        'train_patient_percentage': TRAIN_PATIENT_PERCENTAGE,
        'max_split_diff': MAX_SPLIT_DIFF,
        'image_size': IMAGE_SIZE,

        # Training
        'n_epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,

        # Loss
        'focal_loss_gamma': FOCAL_LOSS_GAMMA,

        # Cross-validation
        'use_cv': USE_CV,
        'n_folds': N_FOLDS,

        # Verbosity
        'verbose_training': VERBOSE_TRAINING,
        'verbose_predictions': VERBOSE_PREDICTIONS,
        'verbose_testing': VERBOSE_TESTING,

        # Performance
        'top_n_performers': TOP_N_PERFORMERS,
        'metrics_precision': METRICS_PRECISION,
    }

def get_modality_config():
    """
    Returns modality-related configuration.

    Returns:
        dict: Modality configuration dictionary
    """
    return {
        'all_modalities': ALL_MODALITIES,
        'image_size': IMAGE_SIZE,
    }
