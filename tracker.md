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
