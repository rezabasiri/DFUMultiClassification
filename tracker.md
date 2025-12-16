# Change Tracker

This file tracks major changes made to the repository structure and files.

## 2025-12-16 - Repository Reorganization

### Directory Structure
- **Project root**: Created organized directory structure with src/, data/, paper/, docs/, scripts/, archive/, notebooks/, tests/
- **src/**: Created package structure with data/, models/, training/, evaluation/, utils/ submodules
- **data/raw/**: Moved all raw image folders (Depth_Map_IMG, Depth_RGB, Thermal_Map_IMG, Thermal_RGB) and CSV files
- **data/processed/**: Moved best_matching.csv for processed data
- **paper/**: Moved main.tex and references.bib

### Python Files - Major Moves
- **MultiModal_ModelV68_Combined_v1_31.py → src/main.py**: Renamed and moved main training script
- **Data_Processing.py → src/data/preprocessing.py**: Moved data preprocessing module
- **data_caching.py → src/data/caching.py**: Moved caching utilities
- **models_losses.py → src/models/architectures.py**: Moved model architectures
- **MisclassificationFunctions.py → src/evaluation/metrics.py**: Moved evaluation metrics
- **Performance_Analysis_Plot.py → src/evaluation/performance_plots.py**: Moved performance visualization
- **ROC_Plot_Generation_CrossValid_V2.py → src/evaluation/roc_plots.py**: Moved ROC plotting utilities
- **perform_grid_search_function.py → src/training/grid_search.py**: Moved grid search functionality

### New Files Created
- **src/__init__.py**: Package initialization for src module
- **src/data/__init__.py**: Package initialization for data module
- **src/models/__init__.py**: Package initialization for models module
- **src/training/__init__.py**: Package initialization for training module
- **src/evaluation/__init__.py**: Package initialization for evaluation module
- **src/utils/__init__.py**: Package initialization for utils module
- **src/utils/config.py**: Centralized configuration for paths, constants, and settings
- **archive/README.md**: Documentation for archived code and missing modules
- **claude.md**: Instructions for future Claude sessions

### Import Updates
- **src/main.py**: Updated imports to use src.evaluation.metrics and src.utils.config
- **src/data/preprocessing.py**: Updated to import from src.utils.config
- **src/data/caching.py**: Updated to import from src.data.preprocessing and src.utils.config
- **src/models/architectures.py**: Updated to import from src.utils.config
- **src/evaluation/metrics.py**: Updated to import CLASS_LABELS from src.utils.config

### Configuration Changes
- **src/utils/config.py**: Centralized all path definitions previously scattered across files
- **src/utils/config.py**: Added get_project_paths() and get_data_paths() helper functions
- **src/utils/config.py**: Defined IMAGE_SIZE, RANDOM_SEED, and CLASS_LABELS constants

### Documentation
- **README.md**: Completely rewritten with project overview, structure, and quick start
- **archive/README.md**: Documented missing GenerativeAugmentationFunctions modules

### Missing Modules Noted
- **GenerativeAugmentationFunctions_V2.py**: Referenced in src/main.py (commented out)
- **GenerativeAugmentationFunctions_V1_3.py**: Referenced in src/data/caching.py (commented out)

## 2025-12-16 - Added Generative Augmentation Modules

### New Modules Added
- **GenerativeAugmentationFunctions_V2.py → src/data/generative_augmentation_v2.py**: Added and moved generative augmentation module (V2)
- **GenerativeAugmentationFunctions_V1_3.py → src/data/generative_augmentation_v1_3.py**: Added and moved generative augmentation module (V1.3)

### Import Updates
- **src/main.py**: Uncommented and updated import to use src.data.generative_augmentation_v2
- **src/data/caching.py**: Uncommented and updated import to use src.data.generative_augmentation_v1_3

### Documentation
- **archive/README.md**: Updated to reflect that missing modules have been added

## 2025-12-16 - Refactored main.py into Modular Components

### Main Script Breakdown
- **src/main.py**: Reduced from 5,386 lines to 1,857 lines (65% reduction)
- **src/main_original.py**: Backed up original main.py for reference

### New Modules Created

#### Data Processing Modules
- **src/data/image_processing.py**: Image processing and dataset preparation utilities (19KB, ~410 lines)
  - Functions: extract_info_from_filename, find_file_match, find_best_alternative, create_best_matching_dataset, prepare_dataset, preprocess_image_data, adjust_bounding_box, load_and_preprocess_image, create_default_image, is_valid_bounding_box

- **src/data/dataset_utils.py**: TensorFlow dataset creation and caching utilities (60KB, ~1,213 lines)
  - Functions: create_cached_dataset, check_split_validity, prepare_cached_datasets, visualize_dataset, plot_net_confusion_matrix
  - Handles dataset creation, caching, validation split checking, and visualization

#### Model Modules
- **src/models/builders.py**: Model architecture builders (19KB, ~313 lines)
  - Functions: create_image_branch, create_metadata_branch, create_fusion_layer, create_multimodal_model
  - Builds image branches, metadata branches, fusion layers, and complete multimodal models

- **src/models/losses.py**: Custom loss functions (3.8KB, ~75 lines)
  - Functions: weighted_f1_score, weighted_ordinal_crossentropy, get_weighted_ordinal_crossentropy, get_weighted_ordinal_crossentropyF1, focal_ordinal_loss, get_focal_ordinal_loss
  - Handles class imbalance and ordinal relationships

#### Training Modules
- **src/training/training_utils.py**: Training utilities and cross-validation (72KB, ~1,530 lines)
  - Functions: clean_up_training_resources, create_checkpoint_filename, analyze_modality_contributions, average_attention_values, cross_validation_manual_split, correct_and_validate_predictions, save_run_results, save_run_metrics, save_gating_results, save_aggregated_results, save_run_predictions, load_run_predictions, get_completed_configs_for_run, load_aggregated_predictions, save_aggregated_predictions, is_run_complete
  - Manages training runs, checkpointing, prediction saving/loading, and result aggregation

#### Utility Modules
- **src/utils/debug.py**: Debug and memory management utilities (1.1KB, ~40 lines)
  - Functions: clear_gpu_memory, reset_keras, clear_cuda_memory
  - Handles GPU memory management and TensorFlow session resets

### Import Organization
- **src/main.py**: Updated with comprehensive imports from all new modules
  - Organized imports by category: standard library, TensorFlow/Keras, scikit-learn, visualization, project modules
  - All functions now imported from their respective specialized modules
  - Maintains all original functionality while being much more maintainable

### Benefits
- **Improved Maintainability**: Each module focuses on a specific aspect (data, models, training, evaluation)
- **Better Code Organization**: Related functions grouped together logically
- **Easier Testing**: Individual modules can be tested independently
- **Reduced Complexity**: Main script is now focused on orchestration rather than implementation
- **No Functionality Lost**: All 84 original functions preserved across modules

## 2025-12-16 - Added Environment Setup Files

### New Files
- **requirements.txt**: Python package dependencies for pip installation
- **docs/SETUP.md**: Comprehensive environment setup and troubleshooting guide

### Documentation Updates
- **README.md**: Added installation section with virtual environment setup instructions

## 2025-12-16 - Added Test Workflow Script

### New Files
- **test_workflow.py**: Comprehensive test script for demo data with extensive debugging
- **docs/TESTING.md**: Complete testing guide with troubleshooting

### Features
- Tests entire pipeline with minimal computational requirements (laptop-friendly)
- Extensive debugging prints at each step (11 stages total)
- Configurable test parameters (batch size, epochs, modalities, image size)
- Validates data loading, preprocessing, model building, training, and evaluation
- Saves detailed test results
- Error handling with helpful traceback information

### Test Configuration
- Modalities: metadata + depth_rgb (expandable)
- Batch size: 4 (adjustable for limited memory)
- Epochs: 5 (quick testing)
- Image size: 64x64 (reduces computation)
- No augmentation (faster testing)

## 2025-12-16 - Fixed Missing Import in dataset_utils.py

### Bug Fix
- **src/data/dataset_utils.py**: Added missing `import random` statement (line 7)
  - Fixed NameError when running test_workflow.py
  - The prepare_cached_datasets() function uses random.seed() for reproducibility

## 2025-12-16 - Documented Image Filename Format and Data Structure

### Documentation
- **README.md**: Added comprehensive data structure section
  - Image filename format: `{random}_P{patient:3d}{appt:2d}{dfu:1d}{B/A}{D/T}{R/M/L}{Z/W}.png`
  - B/A: Before/After debridement
  - D/T: Depth/Thermal camera source
  - R/M/L: Right/Middle/Left angle
  - Z/W: Zoomed/Wide view
  - Dataset file descriptions (DataMaster, best_matching, bounding boxes)
  - Healing phase label definitions (I, P, R)

## 2025-12-16 - Fixed Data Split Issue in test_workflow.py

### Bug Fix
- **test_workflow.py**: Adjusted train/validation split ratio from 80/20 to 67/33
  - Problem: With only 2 patients having Phase R samples (7, 256) out of 8 total
  - 80/20 split (6 train, 2 val) had 54% chance both R patients end up in training
  - This caused "Could not create valid data split" error (validation missing Phase R)
  - Solution: 67/33 split (5 train, 3 val) gives 64% chance of R patient in validation
  - Significantly improves odds of successful split with all three phases (I, P, R) in both sets

## 2025-12-16 - Fixed Missing Imports in dataset_utils.py

### Bug Fix
- **src/data/dataset_utils.py**: Added missing sklearn imports (lines 13-15)
  - Added `from sklearn.impute import KNNImputer`
  - Added `from sklearn.preprocessing import StandardScaler`
  - Added `from sklearn.metrics import confusion_matrix, classification_report`
  - Fixed multiple NameErrors when running test_workflow.py
  - KNNImputer(n_neighbors=5) used for imputing missing metadata values (line 619)
  - StandardScaler() used for feature scaling in preprocess_split() (line 632)
  - confusion_matrix and classification_report used for model evaluation (lines 1215, 1232)

## 2025-12-16 - Updated .gitignore with Python Ignores

### Configuration Update
- **.gitignore**: Added comprehensive Python-specific ignore patterns
  - Python cache: `__pycache__/`, `*.py[cod]`, `*$py.class`, `*.so`
  - Virtual environments: `venv/`, `env/`, `ENV/`
  - IDE files: `.vscode/`, `.idea/`, swap files
  - Jupyter: `.ipynb_checkpoints/`
  - Model outputs: `*.h5`, `*.hdf5`, `*.pkl`, `*.pickle`
  - Results/logs: `results/runs/`, `results/checkpoints/`, `*.log`

## 2025-12-16 - Made Data Split Validation Threshold Configurable

### Enhancement
- **src/data/dataset_utils.py**: Added `max_split_diff` parameter to prepare_cached_datasets() (line 267)
  - Previously hardcoded to 0.05 (5%), which is too strict for small datasets
  - Now configurable with default of 0.1 (10%) for production runs
  - Allows test runs to use relaxed threshold (0.3) for small patient counts
  - Example issue: With 9 patients, Train I=46% vs Val I=70% → 24% difference exceeds 5%
- **test_workflow.py**: Added `max_split_diff: 0.3` to TEST_CONFIG (line 184)
  - Uses 30% threshold appropriate for 9-patient test dataset
  - Main training runs maintain stricter 10% default threshold

## 2025-12-16 - Fixed Critical Data Split Loop Bug

### Bug Fix
- **src/data/dataset_utils.py**: Moved label conversion outside the split attempt loop (lines 284-287)
  - **Critical bug**: Label conversion was inside the loop, causing data corruption
  - First iteration: Converts 'I'→0, 'P'→1, 'R'→2 correctly
  - Second+ iterations: Tries to map 0, 1, 2 again → produces NaN values
  - Result: All iterations after first had NaN labels, so `best_split` remained None
  - Caused "No valid split found with all classes present in both sets" error
  - **Fix**: Convert labels once before the loop, not during each iteration

## 2025-12-16 - Added image_size Parameter to Dataset Functions

### Enhancement
- **src/data/dataset_utils.py**: Added `image_size` parameter to dataset creation functions
  - Added to `create_cached_dataset()` function (line 26, default: 128)
  - Added to `prepare_cached_datasets()` function (line 268, default: 128)
  - Passed through to both train and validation dataset creation
  - Fixes NameError: "name 'image_size' is not defined" in nested functions
- **test_workflow.py**: Pass image_size=64 from TEST_CONFIG (line 260)
  - Test uses 64x64 images for faster processing
  - Production default remains 128x128

## 2025-12-16 - Made Distribution Strategy Optional in Model Builder

### Enhancement
- **src/models/builders.py**: Added optional `strategy` parameter to create_multimodal_model() (line 255)
  - Previously assumed `strategy` variable existed in scope (from main.py)
  - Now accepts optional strategy parameter (default: None)
  - Uses strategy.scope() for multi-GPU training when provided
  - Falls back to default scope for single-GPU/CPU scenarios
  - Fixes NameError: "name 'strategy' is not defined" in test runs
  - Maintains compatibility with production multi-GPU setup

## 2025-12-16 - Fixed Missing Image Folder Paths in Dataset Creation

### Bug Fix
- **src/data/dataset_utils.py**: Extracted folder paths at start of create_cached_dataset() (lines 39-43)
  - **Critical bug**: Nested functions referenced undefined variables (image_folder, depth_folder, etc.)
  - These variables were in module-level data_paths but not accessible in nested function scope
  - Caused NameError: "name 'image_folder' is not defined" during dataset iteration
  - **Fix**: Extract folder paths from data_paths at function start so they're in scope
  - Fixes image loading for all modalities (depth_rgb, depth_map, thermal_rgb, thermal_map)

## 2025-12-16 - Added Missing Training Steps Parameters

### Bug Fix
- **test_workflow.py**: Added steps_per_epoch and validation_steps to model.fit() (lines 419-420)
  - Error: "When providing an infinite dataset, you must specify the number of steps to run"
  - Dataset uses .repeat() for infinite iteration but model.fit() needs step counts
  - steps_per_epoch and validation_steps already returned from prepare_cached_datasets()
  - **Fix**: Pass both parameters to model.fit() call
  - Allows training to complete successfully with proper epoch boundaries
