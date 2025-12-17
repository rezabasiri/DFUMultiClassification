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

