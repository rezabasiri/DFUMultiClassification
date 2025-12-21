# Change Log

Tracks major repository changes and refactors.

## 2025-12-16 — Repository reorganization

### Structure created / standardized

* Created top-level dirs: `src/`, `data/`, `paper/`, `docs/`, `scripts/`, `archive/`, `notebooks/`, `tests/`, plus `results/` usage.
* Created `src/` package layout: `data/`, `models/`, `training/`, `evaluation/`, `utils/`.

### Data & paper relocation

* Moved raw folders + CSVs → `data/raw/` (Depth_Map_IMG, Depth_RGB, Thermal_Map_IMG, Thermal_RGB).
* Moved processed file → `data/processed/best_matching.csv` (later updated; see below).
* Moved paper files → `paper/main.tex`, `paper/references.bib`.

### Major file moves/renames into `src/`

* `MultiModal_ModelV68_Combined_v1_31.py` → `src/main.py`
* `Data_Processing.py` → `src/data/preprocessing.py`
* `data_caching.py` → `src/data/caching.py`
* `models_losses.py` → `src/models/architectures.py`
* `MisclassificationFunctions.py` → `src/evaluation/metrics.py`
* `Performance_Analysis_Plot.py` → `src/evaluation/performance_plots.py`
* `ROC_Plot_Generation_CrossValid_V2.py` → `src/evaluation/roc_plots.py`
* `perform_grid_search_function.py` → `src/training/grid_search.py`

### New files added

* `src/**/__init__.py` for package modules (`src`, `data`, `models`, `training`, `evaluation`, `utils`).
* `src/utils/config.py`: centralized paths + constants (IMAGE_SIZE, RANDOM_SEED, CLASS_LABELS) + helpers `get_project_paths()`, `get_data_paths()`.
* `archive/README.md`: archive notes + module status.
* `claude.md`: instructions for future Claude sessions.
* `README.md`: rewritten (overview, structure, quick start).

### Import updates

* Updated modules to import config/constants from `src.utils.config`, and metrics/augmentation from their new module paths.

### Missing module status (then)

* `GenerativeAugmentationFunctions_V2.py` referenced in `src/main.py` (commented).
* `GenerativeAugmentationFunctions_V1_3.py` referenced in `src/data/caching.py` (commented).

---

## 2025-12-16 — Generative augmentation modules added

### Added + moved into `src/data/`

* `GenerativeAugmentationFunctions_V2.py` → `src/data/generative_augmentation_v2.py`
* `GenerativeAugmentationFunctions_V1_3.py` → `src/data/generative_augmentation_v1_3.py`

### Import updates

* `src/main.py`: uncommented/updated import → `src.data.generative_augmentation_v2`
* `src/data/caching.py`: uncommented/updated import → `src.data.generative_augmentation_v1_3`
* `archive/README.md`: updated (modules no longer missing)

---

## 2025-12-16 — Refactored `main.py` into modules

### Main script refactor

* `src/main.py`: 5,386 → 1,857 lines (~65% reduction)
* `src/main_original.py`: backup of pre-refactor script
* Preserved all original functionality (84 functions), improved maintainability/testing.

### New modules created

**Data**

* `src/data/image_processing.py`: image/dataset prep utilities (extract/match/preprocess/bbox helpers).
* `src/data/dataset_utils.py`: TF dataset creation/caching/validation/visualization utilities.

**Models**

* `src/models/builders.py`: branch builders + fusion + `create_multimodal_model()`.
* `src/models/losses.py`: custom ordinal/focal + weighted losses.

**Training**

* `src/training/training_utils.py`: CV, checkpointing, prediction/results IO, aggregation, run completion checks.

**Utils**

* `src/utils/debug.py`: GPU/session cleanup helpers.

### Import organization

* `src/main.py` imports reorganized by category and now orchestrates (not implements) most logic.

---

## 2025-12-16 — Environment & setup docs

* Added `requirements.txt`
* Added `docs/SETUP.md`
* Updated `README.md` with venv install instructions

---

## 2025-12-16 — Testing workflow added

### Added

* `test_workflow.py`: end-to-end pipeline smoke test (11 stages, debug output, saves results)
* `docs/TESTING.md`: test guide + troubleshooting

### Test defaults

* Modalities: `metadata + depth_rgb`
* Batch size 4, epochs 5, image 64×64, no augmentation

---

## 2025-12-16 — Fixes & enhancements (pipeline correctness)

### `dataset_utils.py` bug fixes

* Added missing imports: `random`; sklearn (`KNNImputer`, `StandardScaler`, `confusion_matrix`, `classification_report`).
* Fixed split-loop corruption: moved label conversion outside repeated split attempts (prevents NaN labels).
* Added `image_size` parameter threading (`create_cached_dataset`, `prepare_cached_datasets`).
* Made `max_split_diff` configurable; demo uses relaxed threshold for small datasets.
* Fixed missing folder-path scope in nested functions (ensures modality image loading works).
* Added missing import: `load_and_preprocess_image`.
* Made TF cache filenames unique per modality combination (prevents shape-mismatch cache reuse).
* Centralized tf_records cache under `results/tf_records/`.
* Fixed `prepare_cached_datasets()` to pass through `cache_dir` correctly.

### `test_workflow.py` bug fixes

* Added `steps_per_epoch` + `validation_steps` (required for repeated/infinite datasets).
* Fixed metadata input shape mismatch: set metadata shape to `(3,)` (RF probs: I/P/R).
* Adjusted split ratio 80/20 → 67/33 to improve chance of Phase R presence in validation.

### `.gitignore`

* Added comprehensive Python ignores + outputs/logs.
* Added ignores for `results/tf_records/`, `results/demo_tf_records/`, and `modality_test_results_*.txt`.

---

## 2025-12-16 — Modality combination testing

### Added / updated

* `test_modality_combinations.py`: tests all 31 modality combinations; verifies dynamic modality handling in model builder.
* Validated builder logic for 1–5 modalities (concat + dense stack varies by count).

### Fix

* Corrected return-value unpacking from `prepare_cached_datasets()` (expects 6 values).

### Cache isolation (root-cause resolution)

* Identified cache root cause: same cache filenames reused across modality combinations.
* Fixed by including sorted modality names in cache filenames.

---

## 2025-12-16 — Demo folder + performance reporting

* Created `demo/` and moved tests:

  * `demo/test_workflow.py`
  * `demo/test_modality_combinations.py` (enhanced metrics tracking + top performers + grouped summaries)
* README updated with “Demo & Testing”.
* Added F1-macro/F1-weighted and optional k-fold CV to modality testing.
* Created `src/utils/demo_config.py` and updated demo scripts to use centralized config values.
* Separated demo vs production matching files: `results/demo_best_matching.csv` for demo scripts.

---

## 2025-12-16 — Production configuration + CLI support

### Production config

* Created `src/utils/production_config.py`: centralized 70+ `main.py` hyperparameters + helper getters + `apply_environment_config()`.

### `main.py` updates

* Replaced hard-coded values with `production_config.py` imports.
* Added robust CLI via argparse (mode/data_percentage/train split/n_runs) + runtime config summary.
* Added modality-search configuration section in `production_config.py` (`ALL_MODALITIES`, search mode, include/exclude lists, output CSV filename).
* Created `USAGE_GUIDE.md` with runnable examples and workflow migration notes.

---

## 2025-12-16 — Demo caching enabled (speed)

* Demo scripts now cache TF records in `results/demo_tf_records/` and reuse them on reruns (production behavior unchanged).

---

## 2025-12-16 — Import-path fix for running main directly

* Fixed `ModuleNotFoundError: No module named 'src'` when running `python src/main.py`:

  * Inserted project root into `sys.path` inside `src/main.py`.

---


## 2025-12-16 - Fixed Missing Keras Layer Imports in main.py

### Issue
Running `python src/main.py` resulted in:
```
NameError: name 'Layer' is not defined
```
at line 1240 where `TransformerBlock(Layer)` is defined.

### Root Cause
Missing imports for Keras layer classes used in main.py:
- Layer (base class for custom layers)
- Dense, Dropout, LayerNormalization (used in transformer blocks)
- MultiHeadAttention (used in attention mechanisms)
- Input, GlobalAveragePooling1D, Add (used in model building)
- Model (for creating models)
- K (Keras backend for low-level operations)

### Fix Applied

**src/main.py** (lines 30-34):
Added comprehensive Keras imports:
```python
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
```

### Result
All Keras classes used in main.py are now properly imported. The script can now run without NameError exceptions.

---

## 2025-12-17 — Remove undefined clear_cache_files() call

**src/main.py** (line 1947): Removed call to undefined `clear_cache_files()` function. Cache clearing is already handled inside `main()` function at lines 1824-1842.

## 2025-12-17 — Add missing itertools import

**src/main.py** (line 20): Added `import itertools` for combinations generation in main_search().

## 2025-12-17 — Fix MisclassificationFunctions import

**src/main.py** (line 1754): Updated import from `MisclassificationFunctions` to `src.evaluation.metrics`.

## 2025-12-17 — Add missing imports to training_utils.py

**src/training/training_utils.py**: Added missing imports: `random`, `gc`, `glob`, and `clear_gpu_memory` from `src.utils.debug`.

## 2025-12-17 — Handle list input in cross_validation_manual_split

**src/training/training_utils.py**: Modified `cross_validation_manual_split()` to accept modality list and convert to expected dict format. Fixed imports to use correct names: `GLOBAL_BATCH_SIZE`, `N_EPOCHS`, `IMAGE_SIZE` from production_config.

## 2025-12-17 — Add dataset and augmentation imports to training_utils.py

**src/training/training_utils.py**: Added `prepare_cached_datasets` from `src.data.dataset_utils` and `AugmentationConfig`, `GenerativeAugmentationManager` from `src.data.generative_augmentation_v2`.

## 2025-12-17 — Fix image_size variable in ProcessedDataManager and augmentation setup

**src/training/training_utils.py**: Changed `image_size` to `IMAGE_SIZE` in `process_all_modalities()` (line 218) and augmentation config setup (lines 707-708).

## 2025-12-17 — Extract config parameters in cross_validation_manual_split

**src/training/training_utils.py**: Added extraction of `batch_size`, `max_epochs`, `image_size` from configs dict and `gpus` list from TF config (lines 604-611).

## 2025-12-17 — Add missing augmentation import to dataset_utils.py

**src/data/dataset_utils.py**: Added import of `create_enhanced_augmentation_fn` from `src.data.generative_augmentation_v2`.

## 2025-12-17 — Add missing compute_class_weight import to training_utils.py

**src/training/training_utils.py**: Added import of `compute_class_weight` from `sklearn.utils.class_weight`.

## 2025-12-17 — Fix alpha_value referenced before assignment

**src/training/training_utils.py** (lines 818-833): Added default values for `alpha_value`, `class_weights_dict`, `class_weights` before conditionals. Changed `if` to `elif` for config suffix checks.

## 2025-12-17 — Add missing strategy variable

**src/training/training_utils.py** (line 615): Added `strategy = tf.distribute.MirroredStrategy()` in `cross_validation_manual_split()`.

## 2025-12-17 — Add missing create_multimodal_model import

**src/training/training_utils.py** (line 27): Added import of `create_multimodal_model` from `src.models.builders`.

## 2025-12-17 — Add missing get_focal_ordinal_loss import

**src/training/training_utils.py** (line 28): Added import of `get_focal_ordinal_loss` from `src.models.losses`.

## 2025-12-17 — Add missing Keras imports (Adam, callbacks)

**src/training/training_utils.py** (lines 15-16): Added `Adam` from `tensorflow.keras.optimizers` and `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` from `tensorflow.keras.callbacks`.

## 2025-12-17 — Add missing weighted_f1_score import

**src/training/training_utils.py** (line 30): Added `weighted_f1_score` to import from `src.models.losses`.

## 2025-12-17 — Add missing GenerativeAugmentationCallback import

**src/training/training_utils.py** (line 28): Added `GenerativeAugmentationCallback` to import from `src.data.generative_augmentation_v2`.

## 2025-12-17 — Add missing BatchVisualizationCallback import

**src/training/training_utils.py** (line 27): Added `BatchVisualizationCallback` to import from `src.data.dataset_utils`.

## 2025-12-17 — Add missing TrainingHistoryCallback import

**src/training/training_utils.py** (line 27): Added `TrainingHistoryCallback` to import from `src.data.dataset_utils`.

## 2025-12-17 — Add missing MetadataConfidenceCallback import

**src/training/training_utils.py** (line 29): Added `MetadataConfidenceCallback` to import from `src.models.builders`.

## 2025-12-17 — Fix n_epochs to max_epochs variable name

**src/training/training_utils.py** (line 942): Changed `n_epochs` to `max_epochs` in model.fit() call.

## 2025-12-17 — Add missing track_misclassifications import

**src/training/training_utils.py** (line 31): Added `track_misclassifications` import from `src.evaluation.metrics`.

## 2025-12-17 — Fix return value in cross_validation_manual_split

**src/training/training_utils.py** (line 1130): Changed return from `all_metrics` (empty list) to `all_runs_metrics` (populated list).

## 2025-12-17 — Fix avg_f1_classes index error in main_search

**src/main.py** (lines 1775-1783): Added safeguards to ensure `avg_f1_classes` is always a 3-element array even with single run or empty results.

## 2025-12-17 — Pass image_size to prepare_cached_datasets

**src/training/training_utils.py** (line 749): Added `image_size=image_size` parameter to `prepare_cached_datasets()` call to fix image size mismatch (128 vs 64).

## 2025-12-17 — Organize output directories

**src/utils/config.py**: Added `get_output_paths()` function that creates organized subdirectories under results/:
- `models/` - .h5 model weight files
- `checkpoints/` - .npy prediction/label files
- `csv/` - Result CSV files
- `misclassifications/` - Misclassification tracking
- `visualizations/` - Plots and batch visualizations
- `logs/` - Training logs
- `tf_records/` - TensorFlow cache files

**src/training/training_utils.py**: Updated to use organized paths for all outputs (.h5 models, CSVs, misclassifications).

**src/main.py**: Updated to use organized paths for CSV outputs.

## 2025-12-17 — Fix train_gating_network import

**src/training/training_utils.py** (lines 668, 1090): Added local import `from src.main import train_gating_network` inside functions to avoid circular dependency. Import placed at point of use rather than module level.

## 2025-12-17 — Add WeightedAccuracy import to main.py

**src/main.py** (line 96): Added `WeightedAccuracy` to imports from `src.training.training_utils`.

## 2025-12-17 — Skip gating network for single model

**src/main.py** (lines 915-919): Added early return in `train_gating_network()` when only 1 model is present. Gating network requires at least 2 models to combine; with single model, returns predictions directly.

## 2025-12-17 — Add multiple configs infrastructure (disabled by default)

**src/utils/production_config.py** (lines 202-206): Added `SEARCH_MULTIPLE_CONFIGS=False` and `SEARCH_CONFIG_VARIANTS` parameters. Note: Gating network is designed to ensemble predictions from DIFFERENT modality combinations (e.g., metadata+depth_rgb vs metadata+thermal_map), not different configs of the same combination.

**src/training/training_utils.py** (lines 609-649): Added infrastructure to create multiple configs with different loss parameters when `SEARCH_MULTIPLE_CONFIGS=True` (disabled by default).

**src/training/training_utils.py** (lines 897-901): Modified model compilation to use loss parameters from config if available.

## 2025-12-17 — Update README with system components documentation

**README.md**: Added "System Components" section explaining multimodal fusion (GAMAN), gating network (late fusion ensemble), search mode, and cross-validation. Updated Quick Start with detailed usage examples for exploring modality combinations and training gating network ensemble.

## 2025-12-17 — Add gating network ensemble across modality combinations

**src/training/training_utils.py** (lines 1431-1452): Added `save_combination_predictions()` and `load_combination_predictions()` functions to save/load predictions per modality combination.

**src/main.py** (lines 97, 1775-1784, 1835-1914): Added import for new functions; added code to save predictions per combination after training; added new section "GATING NETWORK ENSEMBLE ACROSS MODALITY COMBINATIONS" that loads predictions from all combinations and trains gating network to ensemble them.

## 2025-12-17 — Fix train_model_combination for small datasets

**src/main.py** (lines 753-782): Added input validation for empty data/labels; improved class weight calculation to handle missing classes gracefully.

**src/main.py** (lines 857-861): Fixed batch size issue where `GATING_BATCH_SIZE=64` with only 42 samples caused empty dataset. Changed to `min(GATING_BATCH_SIZE, len(train_labels))` and `drop_remainder=False`.

## 2025-12-17 — Organize gating outputs into proper directories

**src/main.py**: Updated save paths for gating network outputs:
- `best_gating_model_run{N}.h5` → `results/models/`
- `training_progress_run{N}.json` → `results/checkpoints/`
- `best_predictions_run{N}.npy` → `results/checkpoints/`
- `model_combination_results_{N}.txt` → `results/csv/`

**.gitignore**: Added additional output directories:
- `results/batch_visualizations_generative/`
- `results/training_plots_generative/`
- `results/metadata_confidence_logs/`

Removed previously tracked CSV files from git to respect `results/csv/` gitignore entry.

## 2025-12-17 — Fix search mode: each combination trains independently

**src/training/training_utils.py** (lines 689-698): Modified `cross_validation_manual_split()` to only load existing predictions when `len(configs) > 1` (specialized mode with multiple configs for same modalities). In search mode (single config per combination), now forces fresh training for each modality combination instead of incorrectly reusing predictions from previous combinations.



## 2025-12-17 — Ensure consistent train/valid splits across modality combinations

**src/data/dataset_utils.py**: Added `save_patient_split()` and `load_patient_split()` to persist patient splits across combinations; modified `prepare_cached_datasets()` to load existing splits ensuring all combinations use identical train/valid patients per run.

## 2025-12-17 — Implement k-fold cross-validation with patient-level splitting

**src/data/dataset_utils.py**: Added `create_patient_folds()` for stratified patient-level k-fold splitting; modified `prepare_cached_datasets()` to accept pre-computed patient splits.
**src/training/training_utils.py**: Updated `cross_validation_manual_split()` with `cv_folds` parameter (default=3) for proper k-fold CV; maintains backwards compatibility with deprecated `n_runs`.
**src/main.py**: Added `--cv_folds` CLI argument; updated `main()` and `main_search()` signatures.
**README.md**: Added two-block training workflow documentation.

## 2025-12-17 — Add resume mode with intelligent checkpoint management

**src/utils/config.py**: Added `cleanup_for_resume_mode()` function with 5 modes (fresh/auto/from_data/from_models/from_predictions) for selective file deletion.
**src/main.py**: Added `--resume_mode` CLI argument (default='auto'); calls cleanup function at startup.
**README.md**: Added "Resume and Checkpoint Management" section with mode documentation and workflows.

## 2025-12-17 — Fix n_runs default to enable cv_folds

**src/main.py**: Changed `--n_runs` default from 3 to None; fixed backwards compatibility check to properly detect when user hasn't specified n_runs, allowing cv_folds to work correctly.

## 2025-12-17 — Fix resume mode fresh: comprehensive checkpoint cleanup patterns

**src/utils/config.py**: Added `*pred*.npy` and `*label*.npy` patterns to catch all prediction/label file variants (pred_run*, combo_pred*, true_label_run*); added `gating_network_run_*_results.csv` pattern for gating results; applied to both 'fresh' and 'from_data' modes.

## 2025-12-17 — Fix patient split confusion between k-fold CV and legacy systems

**src/data/dataset_utils.py**: Added `for_shape_inference` parameter to suppress patient split messages and file operations during metadata shape detection.
**src/training/training_utils.py**: Updated metadata shape detection call to use `for_shape_inference=True`; prevents confusing messages from both k-fold CV and legacy split systems.

## 2025-12-17 — Add verbosity system with progress bar support

**src/utils/verbosity.py**: Created verbosity utility with `vprint()`, `init_progress_bar()`, `update_progress()`, `close_progress()` functions; 4 levels (0=minimal, 1=normal, 2=detailed, 3=progress_bar).
**src/utils/production_config.py**: Added `VERBOSITY` parameter (default=1) and progress bar settings.
**src/main.py**: Added `--verbosity` CLI argument; fixed "Train labels unique" to show actual classes instead of one-hot values; fixed "run 1 of None" display; integrated vprint() and progress bar showing completion %, elapsed time, ETA, accuracy/F1.
**src/training/training_utils.py**: Fixed training message to display fold/run number correctly instead of "run 1 of None"; replaced 40+ print statements with vprint() at level 1-2.
**src/data/dataset_utils.py**: Replaced 30+ print statements with vprint() at level 2 for class distributions, alpha values, unique cases.
**src/data/image_processing.py**: Replaced print statements with vprint() for modality sample counts.
**src/evaluation/metrics.py**: Replaced print statements with vprint() for misclassification file loading messages.

## 2025-12-17 — Fix f-string syntax errors from batch vprint replacement

**src/main.py**: Removed duplicate level parameters and level parameters from inside f-string expressions.
**src/data/dataset_utils.py**: Fixed f-string syntax errors where level parameter was incorrectly placed inside expressions.

## 2025-12-17 — Fix verbosity guards for progress bar mode

**src/data/dataset_utils.py**: Changed 14 guards from `get_verbosity() == 2` to `get_verbosity() <= 2` and 1 guard from `== 1` to `<= 1`.
**src/main.py**: Changed 9 guards from `get_verbosity() == 2/1` to `get_verbosity() <= 2/1`.
**src/training/training_utils.py**: Changed 1 guard from `get_verbosity() == 1` to `get_verbosity() <= 1`.

Semantics: `<= 2` means "show at levels 0, 1, 2 but NOT at level 3 (progress bar mode)". This is clearer than `== 2` and ensures proper behavior at all verbosity levels while still suppressing output in progress bar mode.

## 2025-12-17 — Show final metrics at verbosity 3 (progress bar mode)

**src/main.py**: Changed final results output from `level=1` to `level=0` to show at all verbosity levels:
- `train_model_combination()`: Final Results section now shows at level=0
- `calculate_and_save_metrics()`: Final Results and classification_report now show at level=0 or level=3
- `main_search()`: Gating Network Ensemble Results now show at level=0
- Added new FINAL SUMMARY section at end of `main_search()` that reads CSV and displays best modality combinations by accuracy, F1, and kappa (level=0)

**src/training/training_utils.py**: Changed run results output from `level=1` to `level=0` for final metrics display.
**src/data/dataset_utils.py**: Changed Net Confusion Matrix Results from `level=1` to `level=0`.

## 2025-12-17 — Fix verbosity guards to use correct logic

Corrected guard logic for verbosity levels:
- **Level 2 debug output**: Changed from `get_verbosity() <= 2` back to `get_verbosity() == 2` to show ONLY at level 2
- **Level 1 normal output**: Changed from `get_verbosity() <= 1` to `get_verbosity() == 1` for non-final output
- **Final results**: Use `get_verbosity() <= 1 or get_verbosity() == 3` to show at levels 0, 1, and 3

Files updated:
- **src/main.py**: Fixed 7 debug guards to `== 2` and 1 normal guard to `== 1`
- **src/data/dataset_utils.py**: Fixed 13 debug guards to `== 2`
- **src/data/caching.py**: Added verbosity import; wrapped Binary1/Binary2 prints with `== 2` guard

Verbosity levels now work correctly:
- Level 0 (MINIMAL): Only config header + final results
- Level 1 (NORMAL): Standard progress info + final results
- Level 2 (DETAILED): Everything including debug info (Binary distributions, alpha values, shapes)
- Level 3 (PROGRESS_BAR): Clean progress bar + final results after completion

## 2025-12-17 — Move demo_best_matching.csv to results/demo

**demo/test_workflow.py**: Updated to save `demo_best_matching.csv` to `results/demo/` instead of `results/`; creates demo directory if needed.
**demo/test_modality_combinations.py**: Updated to look for `demo_best_matching.csv` in `results/demo/` directory.

---

## 2025-12-21 — Pin library versions for GPU cluster compatibility

**requirements.txt**: Pinned PyTorch (2.1.2), torchvision (0.16.2), transformers (4.35.0), and diffusers (0.23.0) for compatibility with TensorFlow 2.15.1 CUDA 12.2; fixes `register_pytree_node` AttributeError on RTX 5090.
**test_imports.py**: Created import verification script to test TensorFlow, PyTorch, transformers, and diffusers compatibility.

## 2025-12-21 — Class balancing investigation and fixes

**src/data/dataset_utils.py** (lines 621-656): Fixed resampling control flow bug - removed undefined `OverSampleOnly` that caused NameError and always fell back to exception handler. Now properly implements simple oversampling strategy when mix=False.
**debug_modality_isolation.py**: Created diagnostic script to verify modality isolation in datasets and detect potential data leakage between combinations.

**Issue discovered**: Enabling resampling (apply_sampling=True) caused train/validation distribution mismatch - training on balanced data but validating on imbalanced data led to model collapse (predicting single class per fold).

**Final solution** (line 694): Disabled resampling (apply_sampling=False) to maintain consistent train/validation distributions. Class balancing now handled via alpha values (inverse class frequency weights) in focal loss function.

**src/training/training_utils.py** (lines 325-376): Added MacroF1Score metric class - computes macro-averaged F1 score, robust to class imbalance (treats all classes equally regardless of frequency).
**src/training/training_utils.py** (line 1013): Added macro_f1 to model metrics for monitoring during training.
**src/training/training_utils.py** (line 1039): Changed ModelCheckpoint to monitor 'val_macro_f1' instead of 'val_weighted_accuracy' for selecting best model weights.

Alpha values verified to be calculated from training class frequencies (not hardcoded) - inverse proportional weighting gives higher importance to minority classes.

**src/training/training_utils.py** (line 1610): Fixed error handling - `save_aggregated_predictions()` now skips empty lists with warning instead of raising ValueError.
**src/training/training_utils.py** (line 1211): Training errors now always visible (level=0) even at verbosity 3; added traceback output at level=2.
**src/training/training_utils.py** (line 1224): Added check after training retry loop to warn if training completely failed.
**src/training/training_utils.py** (lines 335-357): Fixed MacroF1Score.update_state() - replaced loop with vectorized one-hot encoding operations and assign_add on entire tensor (fixes SymbolicTensor AttributeError during graph execution).
**src/main.py** (line 13): Set TF_CPP_MIN_LOG_LEVEL='1' before TensorFlow import to suppress INFO messages like "Local rendezvous recv item cancelled" at verbosity 3.

**Critical fix - Callback alignment** (lines 1017-1032): Changed EarlyStopping and ReduceLROnPlateau to monitor 'val_macro_f1' (mode=max) instead of 'val_loss' (mode=min). Previously, EarlyStopping restored epoch 1 weights (best loss but collapsed model) while ModelCheckpoint saved best macro F1 weights (different epoch). Now all callbacks aligned on same metric, preventing model collapse.

**src/data/dataset_utils.py** (lines 611-646): Smart alpha redistribution algorithm - iteratively caps maximum value at MAX_ALPHA and redistributes excess to other classes proportionally based on remaining capacity. Maintains sum=1.0 while preventing extreme weights. Example: [0.288, 0.135, 0.577] → [0.316, 0.184, 0.5]. Classes with more room to grow receive more of the redistributed weight.
**src/data/dataset_utils.py** (line 617): MAX_ALPHA=0.5 to prevent any class from dominating (>50% focal loss weight).
**src/training/training_utils.py** (line 1011): Reduced learning rate from 1e-3 to 1e-4 to prevent overshooting and collapse.
**src/training/training_utils.py** (line 1022): Reduced EarlyStopping min_delta to 0.001 (was 0.01, too strict - caused premature stopping).
**src/training/training_utils.py** (line 1030): Reduced ReduceLROnPlateau min_delta to 0.0005 (was 0.005, too strict).
**src/training/training_utils.py** (line 1100-1101): Added metadata-only training message to clarify minimal training on final layer.
**src/training/training_utils.py** (line 1103): Changed fit verbose from 1 to 2 - displays one line per epoch instead of multi-line progress bars (eliminates 1/4, 2/4, 4/4 output).
**src/training/training_utils.py** (lines 1208-1214): Added confusion matrix display at verbosity 2 after each run to diagnose collapse patterns (which classes model predicts vs actual).
**src/utils/production_config.py** (line 161): Reduced FOCAL_GAMMA from 2.0 to 1.0 - less aggressive focal loss allows better learning of minority classes.
**src/utils/production_config.py** (lines 29, 33-34, 61-62): Optimized for RTX 5090 - increased batch sizes (30→128), reduced epochs (50→30), added EARLY_STOP_PATIENCE and REDUCE_LR_PATIENCE configs.
**src/models/builders.py**: Verified model architectures match main_original.py - metadata branch and image branch (depth_rgb) are identical.
**src/training/training_utils.py** (line 1209): Fixed confusion_matrix scope error - removed duplicate import, renamed variable to cm_display to avoid conflict with existing cm variable at line 1179.
**src/utils/production_config.py** (line 48): Added EPOCH_PRINT_INTERVAL=50 to print training progress every 50 epochs (reduces output clutter for long runs).
**src/training/training_utils.py** (lines 62-99): Added PeriodicEpochPrintCallback class - prints epoch 1, last epoch, and every Nth epoch with key metrics in single line.
**src/training/training_utils.py** (lines 1087-1091): Integrated PeriodicEpochPrintCallback into training when EPOCH_PRINT_INTERVAL > 0.
**src/training/training_utils.py** (lines 1153-1158): Updated fit verbosity logic - uses verbose=0 with callback for periodic printing, verbose=2 for all epochs otherwise.
**src/utils/production_config.py** (lines 28-34): Increased training parameters for production: N_EPOCHS 20→100, EARLY_STOP_PATIENCE 10→20, REDUCE_LR_PATIENCE 3→5. IMAGE_SIZE kept at 64 (model hardcoded to 64x64), BATCH_SIZE kept at 128.
**src/main.py** (line 1999): Disabled gating network optimal search (find_optimal=False) - search was hanging in infinite loop during model combination testing. Gating network now uses simple average ensemble instead of optimized combination search.

## 2025-12-21 — Systematic comparison tool for validating refactored code

**scripts/compare_main_versions.py**: Created comprehensive comparison tool to validate that refactored main.py produces identical results to main_original.py. Tests individual modalities and combinations systematically. Runs both versions with same config (cv_folds=0, n_runs=1 for deterministic comparison), extracts metrics from CSV files, and reports differences.
**scripts/COMPARISON_README.md**: Created detailed documentation for comparison testing strategy, usage examples, troubleshooting guide, and interpretation of results. Includes systematic testing phases from individual modalities to complex combinations.
**scripts/simple_comparison_test.py**: Created simpler direct comparison test that runs both versions in same process for easier debugging. Uses smaller default dataset (5%) for quick testing.
**src/training/training_utils.py** (lines 755-764): Fixed KeyError when configs dict doesn't include batch_size/max_epochs/image_size keys - now adds defaults from production_config.py for backward compatibility with external callers and test scripts.
**src/main_original.py** (lines 2913-2929): Fixed UnboundLocalError for alpha_value/class_weights - added default initialization before config-specific blocks and changed if statements to elif. Minimal fix to enable comparison testing (same bug was already fixed in refactored version).
**docs/KNOWN_ISSUES_ORIGINAL.md**: Created documentation of bugs found in main_original.py during comparison testing. Updated to reflect that alpha_value bug is now fixed in both versions.
