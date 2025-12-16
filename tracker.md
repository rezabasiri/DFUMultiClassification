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
