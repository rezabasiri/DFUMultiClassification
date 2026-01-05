# Change Log

Tracks major repository changes and refactors.

## 2026-01-04 — CRITICAL: Data leakage discovery in Phase 2 testing

### Phase Confidence (%) data leakage invalidates initial Phase 2 results
**Files**: All Phase 2 solutions in `agent_communication/rf_improvement/`
- **CRITICAL BUG**: "Phase Confidence (%)" was included as a feature in initial Phase 2 testing
- **IMPACT**: This column is the model's own confidence score - **MASSIVE DATA LEAKAGE**
- **DISCOVERY**: Solution 7 (Feature Selection) showed "Phase Confidence (%)" as #1 most informative feature (MI=0.0521)
- **VALIDATION**: Checked `src/main_original.py:1110` - Phase Confidence explicitly excluded in original code

**Root Cause**: Phase 2 solutions only excluded 4 columns (Patient#, Appt#, DFU#, Healing Phase Abs). Should exclude 30+ columns per original implementation.

**Additional leakage sources discovered**:
- Temperature measurements: Peri-Ulcer Temperature (°C), Wound Centre Temperature (°C)
- Individual Offloading categorical columns (No Offloading, Offloading: Therapeutic Footwear, etc.)
- Dressing variants (Dressing, Dressing Grouped)
- Type of Pain variants (Type of Pain, Type of Pain2, Type of Pain_Grouped2)
- Offloading Score
- Appt Days

**INVALIDATED RESULTS** (from initial Phase 2 run):
- Solution 7 (Feature Selection k=50): Kappa 0.2201 ← **INVALID** (used Phase Confidence)
- Solution 8 (Median Imputation): Kappa 0.2159 ← **INVALID**
- Solution 9 (Strategy B): Kappa 0.2124 ← **INVALID**
- Solution 6 (Bayesian): Kappa 0.2062 ← **INVALID** (also had wrong optimization objective)

**VALIDATED TRUE PERFORMANCE** (without Phase Confidence leakage):
- **Baseline: Kappa 0.176 ± 0.078** (was 0.220 ± 0.088 with leakage)
- **Performance drop: -20%** (-0.044 Kappa points)
- Feature count: 73 → 42 valid features (-31 leakage columns)
- Top feature changed: Phase Confidence → Height (cm)

**RE-RUN RESULTS** (after fixes):
- **Solution 6 (Bayesian - RE-FIXED)**: Kappa **0.205 ± 0.057** ← **NEW WINNER** (+16.5% vs baseline)
  - **Tightened search space validated**: max_depth [8,15] prevents shallow overfitting
  - **Found optimal params**: n_estimators=646, max_depth=14, min_samples_split=19, max_features='log2'
  - **Beats manual tuning**: 0.205 vs 0.176 baseline (+16.5%)
- Solution 7 (Feature Selection k=40): Kappa 0.202 ± 0.049 ← Runner-up (+14.8% vs baseline)
- Solution 8 (KNN k=3 Imputation): Kappa 0.201 ± 0.043 ← Third (+14.2% vs baseline)
- Solution 9 (Strategy A Decomp): Kappa 0.181 ± 0.061 (+2.8% vs baseline)
- Solution 11 (Feature Engineering): Kappa 0.125 ± 0.039 ← FAILED (added noise)

**Performance Ceiling Identified**: Kappa ~0.20-0.21 appears to be maximum achievable with metadata-only features

**FIXES APPLIED**:
1. Updated all Phase 2 solutions (6, 7, 8, 9) with correct 30+ column exclusion list
2. Created `solution_6_bayesian_optimization_fixed.py` - optimizes end-to-end 3-class Kappa (not binary separately)
3. Created `solution_11_feature_engineering.py` - 20+ domain-specific engineered features to overcome reduced feature count
4. Created `README_PHASE2_FIXES.md` with detailed explanation and re-run instructions

**Impact**: Phase 2 completely re-run with validated results. Previous Kappa 0.220 was artificially inflated by 20% due to Phase Confidence leakage. True baseline is 0.176.

**Production-Ready Configuration** (validated - **NEW WINNER: Bayesian-optimized**):
- **Feature Selection**: Top 40 features (Height, Onset, Weight, Smoking, Cancer History as top 5)
- **Imputation**: KNN k=3 (improves over current k=5)
- **RF Parameters**: **Bayesian-optimized** (validated Kappa 0.205):
  - n_estimators=646 (up from manual 500)
  - max_depth=14 (up from manual 10)
  - min_samples_split=19 (up from manual 10)
  - min_samples_leaf=2 (explicit)
  - max_features='log2' (changed from 'sqrt')
- **Expected Performance**: Kappa 0.20-0.21 ± 0.05 (robust, validated, no leakage)
- **Improvement**: +16.5% over baseline 0.176

**Lesson Learned**: Always validate feature exclusions against original implementation before running experiments. Phase Confidence being the top feature was a red flag that should have been caught immediately. The 20% performance drop from fixing leakage is expected and represents true model capability.

---

## 2026-01-04 — Implement validated RF hyperparameter tuning

### Tuned RF parameters for metadata classifier
**Files**: `src/data/dataset_utils.py` (lines 845-861, 907-925)
- **IMPROVEMENT**: Implemented tuned RF hyperparameters validated through patient-level 5-fold CV
- **VALIDATION**: Patient-level 5-fold CV with strict leakage prevention:
  - Kappa: 0.220 ± 0.088 (4/5 folds >0.2, target met)
  - Accuracy: 55.9% ± 4.2%
  - All classes functional (no zero F1 scores)
- **HYPERPARAMETERS TUNED**:
  - `n_estimators`: 300 → 500 (increased stability)
  - `max_depth`: None → 10 (prevent overfitting)
  - `min_samples_split`: 2 → 10 (require sufficient samples)
  - `max_features`: None → 'sqrt' (reduce variance)
- **TESTING METHODOLOGY**: `agent_communication/rf_improvement/`
  - Tested 5 approaches: baseline, SMOTE, tuned RF, feature selection, ensemble
  - Tuned RF achieved best cross-validated performance
  - Strict patient-level CV: patients split (not samples), imputer/scaler fit on train only
- **APPLIED TO**: Both TensorFlow Decision Forests (TFDF) and sklearn fallback paths

**Root Cause Analysis**: Original RF parameters (300 trees, unlimited depth, no min_samples constraints) were prone to overfitting on small patient cohorts. Limited regularization caused high variance across different patient distributions.

**Impact**: Metadata RF classifier improved from Kappa ~0.10 to 0.220 (validated, robust). Expected production performance: ~56% accuracy with all three classes functional. Eliminates zero F1 scores for minority class (R).

**Validation Details**: See `agent_communication/rf_improvement/cv_validation.csv` and `validate_cv.py` for full implementation.

---

## 2026-01-04 — Fix catastrophic metadata classifier failure

### Fixed RF probability normalization bug
**File**: `src/data/dataset_utils.py`
- **BUG**: StandardScaler was normalizing RF probability columns (`rf_prob_I`, `rf_prob_P`, `rf_prob_R`), converting valid probabilities to z-scores
- **IMPACT**: Probabilities that should sum to 1.0 became arbitrary z-scores with negative values, causing 10% accuracy
- **FIX**: Added RF probability columns to `exclude_cols` list to prevent normalization (line 988)
- **DIAGNOSIS**: Comprehensive test suite in `agent_communication/metadata_diagnosis/` isolated the bug through 6 sequential tests

**Root Cause**: Probabilities expected range [0-1] summing to 1.0, but received z-scores with mean≈0, std≈1, and negative values. Neural network couldn't learn from corrupted probability distributions.

**Impact**: Metadata modality should now achieve 40-50% accuracy instead of 10%. RF classifier still has issues with minority class (R), but probabilities are now valid.

---

## 2025-12-30 — Critical metadata preprocessing fixes

### Fixed data leakage in imputation
**File**: `src/data/caching.py`
- Fixed imputation to properly separate train/validation: `fit_transform` on training data, `transform` on validation data
- Previously used `fit_transform` on both train and validation, causing data leakage
- Added imputer and scaler parameters to `preprocess_split()` function signature
- Updated function calls to pass fitted imputer/scaler from training to validation preprocessing

### Added missing normalization step
**File**: `src/data/caching.py`
- Added StandardScaler normalization after imputation (was missing in refactored code)
- Properly separates train/validation: `fit_transform` on training data, `transform` on validation data
- Added missing imports: `KNNImputer` and `StandardScaler` from sklearn

### Fixed Random Forest parameters
**File**: `src/data/caching.py`
- Changed `num_trees` from 800 to 300 to match original implementation
- Fixed `random_seed` from fixed value `42` to varying `42 + run * (run + 3)` for model diversity across runs
- Added `run` parameter to `prepare_cached_datasets()` function signature to support varying seed

**Impact**: Metadata preprocessing now matches original implementation; eliminates data leakage and adds critical normalization step for proper feature scaling. RF models now vary across runs for better robustness.

---

## 2025-12-28 — Phase 2 optimization bug fixes and logging improvements

### Fixed CSV file reading and directory creation
**File**: `scripts/auto_polish_dataset_v2.py`
- Fixed CSV filename mismatch: optimization now checks `modality_combination_results.csv` (written by main.py) with fallback to `modality_results_averaged.csv`
- Ensured CSV directory exists before training to prevent write failures
- Added better error messages when CSV files not found

### Simplified Phase 2 optimization logging
**File**: `scripts/auto_polish_dataset_v2.py`
- Replaced temporary output file with timestamped persistent log: `phase2_optimization_YYYYMMDD_HHMMSS.log`
- Cumulative log retains ALL evaluation outputs with clear separators (never overwrites)
- Removed individual per-evaluation logs (excessive; single cumulative log is sufficient)
- Simplified output redirection (direct append to cumulative log)
- Added evaluation headers with timestamp, eval number, and thresholds

### Enhanced baseline performance display
**File**: `scripts/auto_polish_dataset_v2.py`
- Save per-class F1 scores (I, P, R) in `phase1_baseline.json`
- Display per-class F1 in all baseline output (not just min F1)
- Show baseline at end of Phase 1 (after saving metrics)
- Show baseline again at start of Phase 2 (for comparison during optimization)
- Enhanced "Using best baseline" message with per-class F1 scores

### Fixed misclassification count calculation
**File**: `scripts/auto_polish_dataset_v2.py`
- Corrected max misclassification count message to account for tracking mode:
  - `track_misclass='valid'`: max = runs (each sample in valid once per run)
  - `track_misclass='train'`: max = runs × (cv_folds - 1)
  - `track_misclass='both'`: max = runs × cv_folds (tracked from both datasets)
- Previous message incorrectly used total_runs without considering CV folds or tracking mode

**Impact**: Phase 2 optimization works correctly; cleaner logging; better baseline visibility for analysis.

---

## 2025-12-27 — Core data flag for optimized dataset filtering

### Integration of auto_polish_dataset_v2.py results with main.py

**Files Modified**:
- `src/main.py`: Added `--core-data` flag that automatically loads optimized thresholds from `results/bayesian_optimization_results.json` and applies them via the existing `filter_frequent_misclassifications()` function (lines 2279-2288, 2450-2483)
- `src/evaluation/metrics.py`: Fixed `filter_frequent_misclassifications()` to check multiple possible locations for misclassification CSV file (lines 172-188) - handles both `misclassifications/` and `misclassifications_saved/` subdirectories

**Usage**:
```bash
# Use optimized core dataset (excludes outlier samples)
python src/main.py --mode search --core-data

# Manual thresholds override core-data
python src/main.py --mode search --core-data --threshold_P 3
```

**What it does**:
- Reads best_thresholds from Bayesian optimization results (e.g., P=5, I=4, R=3)
- Filters out frequently misclassified samples that may result from human error
- Improves model performance by training on high-quality core data
- Manual threshold arguments (--threshold_I/P/R) override auto-loaded thresholds if specified

**Requirements**: Must run `scripts/auto_polish_dataset_v2.py` first to generate optimization results

---

## 2025-12-26 — Phase 2 optimization fixes and improvements

### Critical Bug Fixes in scripts/auto_polish_dataset_v2.py
- **Fixed subprocess TensorFlow context conflicts**: Replaced `subprocess.run()` with `os.system()` to avoid parent/child TensorFlow context conflicts that caused thread creation failures (lines 768-780, 1292-1340)
- **Fixed batch size adjustment**: Added `_calculate_phase2_batch_size()` method to prevent OOM by dividing batch size by number of image modalities in Phase 2 (lines 170-203, 1183, 1256-1260)
- **Fixed metric extraction for non-metadata modalities**: Changed CSV search from hardcoded 'metadata' to actual modalities being tested (line 1367), fixed column names from `Class {i} F1-score` to `{cls} F1-score` (lines 1377-1382)
- **Fixed timeout handling**: Added proper os.system() return code decoding, handle timeout exit code 124 gracefully (lines 1308-1340)
- **Removed metadata requirement**: Deleted unnecessary check forcing metadata inclusion (line 134-135 removed)
- **Fixed baseline corruption**: Phase 1 baseline now saved to `results/misclassifications/phase1_baseline.json` and loaded from there during Phase 2 to prevent using filtered results as baseline (lines 875-953)

### New Features in scripts/auto_polish_dataset_v2.py
- **Automatic logging**: All terminal output saved to timestamped files in `results/misclassifications/optimization_run_YYYYMMDD_HHMMSS.log` (lines 1410-1438)
- **Phase 1 baseline display**: Added `show_phase1_baseline()` to show performance metrics before optimization for comparison (lines 872-978)
- **CV folds arguments**: Added `--phase1-cv-folds` and `--phase2-cv-folds` command-line args for flexible cross-validation (lines 1498-1508, 1547, 1551)
- **Dataset size display fix**: Shows actual samples used based on `--phase1-data-percentage` instead of full dataset (lines 663-666)
- **Improved objective function for clinical applicability** (lines 527-594):
  - Added weighted F1 extraction from CSV (lines 1356, 1373)
  - New improvement-based scoring: `0.4×Δweighted_f1 + 0.3×Δmin_f1 + 0.2×Δkappa + 0.1×retention - penalties`
  - Focuses on beating baseline rather than absolute scores
  - Weighted F1 prioritized for imbalanced datasets (40% weight vs 30% for min F1)
  - Shows improvement deltas in output: `Weighted F1: 0.4491 (Δ+0.0123)`
- **Best baseline selection**: When multiple modalities tested in Phase 1, automatically selects best performing one (highest weighted F1) as optimization target (lines 923-949)

### Files Modified
- `scripts/auto_polish_dataset_v2.py`: All fixes and features above
- `agent_communication/phase2_subprocess_investigation/`: Investigation docs (FINDINGS.md, SOLUTION.md) and test script

### Key Parameters
- Timeout: 3600s (60 min) per evaluation to allow completion
- Default CV folds: Phase 1=1, Phase 2=3
- Batch size adjustment: `GLOBAL_BATCH_SIZE / num_image_modalities` in Phase 2
- Objective weights: 40% weighted F1, 30% min F1, 20% kappa, 10% data retention

### Output Files (all in results/misclassifications/)
- `phase1_baseline.json` - Preserved Phase 1 baseline metrics for all modalities
- `bayesian_optimization_results.json` - Phase 2 optimization history and best thresholds
- `optimization_run_YYYYMMDD_HHMMSS.log` - Timestamped execution logs
- `frequent_misclassifications_saved.csv` - Misclassification counts from Phase 1

---

## 2025-12-25 — Multi-GPU support implementation

### Multi-GPU Training with MirroredStrategy

**Files Modified**:
- `src/training/training_utils.py`: Wrapped dataset creation/processing in `strategy.scope()` for proper multi-GPU distribution (lines 926-927, 947-959, 1008-1021). Added multi-GPU logging (lines 786-794).
- `src/utils/gpu_config.py`: Creates MirroredStrategy for multi-GPU mode, sets CUDA_VISIBLE_DEVICES, filters GPUs by memory/display status.
- `agent_communication/gpu_setup/MULTI_GPU_GUIDE.md`: Created concise guide for multi-GPU usage.

**Key Changes**:
- Dataset operations now execute within `strategy.scope()` for proper distribution across GPUs
- Global batch size automatically split across GPU replicas (e.g., 128 → 64 per GPU with 2 GPUs)
- Model creation already wrapped in `strategy.scope()` (line 1070-1080)

**Usage**:
```bash
python src/main.py --device-mode multi        # Use all GPUs (>=8GB, non-display)
python src/main.py --device-mode single       # Single GPU (auto-select best)
python src/main.py --device-mode custom --custom-gpus 0 1  # Specific GPUs
```

**RTX 5090 Compatibility**: Requires TensorFlow 2.15.1+ for compute capability 12.0 support (see `INSTALL_RTX5090_FIX.md`)

---

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
**scripts/simple_comparison_test.py**: Updated to force fresh runs (cleanup_for_resume_mode) and read results from CSV files for proper comparison.
**docs/COMPARISON_RESULTS.md**: Documented successful validation test results - refactored code produces functionally equivalent results to original (<1% metric difference) with expected improvements (smart alpha capping). Metadata modality test: Original Acc 0.61/Kappa 0.1691, Refactored Acc 0.6053/Kappa 0.1691.

---

## 2025-12-23 — Systematic debugging resolves Min F1=0.0 issue (9 debugging phases)

### Context
Training produced catastrophic results: Min F1=0.0, Macro F1=0.21, model predicting only one class (usually majority class P). Conducted systematic 9-phase investigation to isolate root causes.

### Debugging Process

**Phase 1-2**: Verified data loading works; TensorFlow training works but model collapses to majority class.
**Phase 5**: Verified focal loss alpha weighting math is correct (9.27x weight for rare class R).
**Phase 6**: Found focal loss + alpha ALONE insufficient for 5.6:1 class imbalance (P:R ratio).
**Phase 7**: Enabled oversampling but discovered DOUBLE-CORRECTION BUG - model switched to predicting only minority class R.
**Phase 8**: Fixed double-correction but model stuck at 33.3% accuracy (random guessing) - NO LEARNING occurring.
**Phase 9**: **SUCCESS** - Feature normalization solves problem! Achieved 97.6% accuracy, Min F1=0.964 (metadata only).

### Root Causes Identified and Fixed

**Root Cause #1 - Oversampling Disabled**:
- **File**: `src/data/dataset_utils.py:714`
- **Bug**: `apply_sampling=False` disabled RandomOverSampler
- **Fix**: Changed to `apply_sampling=True` to balance training data (all classes → 1504 samples)

**Root Cause #2 - Double-Correction Bug**:
- **File**: `src/data/dataset_utils.py:676-687`
- **Bug**: Alpha values calculated from ORIGINAL imbalanced distribution [0.725, 0.344, 1.931], but applied to BALANCED data after oversampling
- **Problem**: Class R had same training samples as P, but 5.6x higher loss weight → over-prioritized
- **Fix**: Recalculate alpha from BALANCED distribution after oversampling → returns [1.0, 1.0, 1.0]
- **Code**: Lines 676-687 now recalculate frequencies from resampled data and compute balanced alpha values

**Root Cause #3 - Features Not Normalized (CRITICAL)**:
- **File**: `src/data/dataset_utils.py:965-995` (NEW CODE ADDED)
- **Bug**: Features ranged from -0.24 to 7071.0 without normalization; large-scale features (bounding boxes) dominated gradients
- **Symptom**: Phase 8 showed model stuck at initialization (train acc=33.3%, loss=10.8, no learning)
- **Fix**: Added StandardScaler normalization for all numeric metadata features
- **Code**: Fit scaler on train_data only (prevent leakage), transform both train and valid data
- **Result**: Phase 9 achieved 97.6% accuracy with normalized features (was 28.8% without)

### Changes Made

**src/data/dataset_utils.py**:
- Line 714: `apply_sampling=True` (enable oversampling)
- Lines 676-687: Recalculate alpha from balanced distribution after oversampling
- Lines 965-995: Added StandardScaler normalization for numeric metadata features (CRITICAL FIX)

**agent_communication/** (debugging artifacts):
- Created 9 debug scripts (debug_01 through debug_09) for systematic investigation
- `FINAL_ROOT_CAUSE.md`: Complete analysis showing all 3 root causes
- `results_09_normalized_features.txt`: Phase 9 success (97.6% acc, Min F1=0.964)

### Results

**Before fixes**:
- Min F1: 0.000 (at least one class never predicted)
- Macro F1: 0.21
- Model: Predicts only one class (P, R, or I depending on configuration)

**After fixes (Phase 9 - metadata only)**:
- Test Accuracy: **97.59%**
- F1 Macro: **0.9720**
- F1 per class: **[I=0.972, P=0.980, R=0.964]** - ALL CLASSES EXCELLENT
- Min F1: **0.964** (complete fix!)
- Prediction distribution matches actual distribution (I=28.5% vs 28.8%, P=60% vs 60.5%, R=11.6% vs 10.8%)

### Important Notes

1. **Phase 9 tested metadata only** (61 tabular features) - easiest modality. Image modalities need separate verification.
2. **Feature normalization is CRITICAL** - without it, model cannot learn (proven by Phase 8 vs Phase 9 comparison).
3. **Solution requires ALL THREE fixes**: oversampling + balanced alpha + feature normalization.
4. ~~**No data leaks detected**: StandardScaler fitted only on train data, train/val splits properly separated.~~ **INCORRECT - See Phase 9 Data Leakage Discovery below**

### Phase 9 Data Leakage Discovery (2025-12-24)

**CRITICAL FINDING**: Phase 9's 97.6% accuracy was **NOT due to real generalization** - it was due to **patient-level data leakage**.

**Root Cause**:
- **File**: `agent_communication/debug_09_normalized_features.py:61-63`
- **Bug**: Used `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` which splits by **SAMPLES**, not **PATIENTS**
- **Problem**: Same patient's visits appeared in both train and test sets → model learned patient-specific patterns (memorization), not generalizable wound healing features
- **Evidence**:
  - Phase 9 with sample-level split: 97.59% accuracy, Min F1=0.964
  - Same architecture with patient-level CV: **47.26% accuracy**, Min F1=0.245 (created via `test_metadata_phase9_cv.py`)
  - Performance drop: **50.33 percentage points**

**Verification Test**:
- Created `agent_communication/test_metadata_phase9_cv.py` implementing proper patient-level CV
- Results across 3 folds with patient-level splitting:
  - Fold 1: Acc=44.9%, F1_macro=0.378, Min_F1=0.231
  - Fold 2: Acc=52.0%, F1_macro=0.440, Min_F1=0.249
  - Fold 3: Acc=44.8%, F1_macro=0.404, Min_F1=0.256
  - **Average: 47.3% accuracy** (not 97.6%)
- Patient overlap verification: 0 patients shared between train/val in all folds ✓

**Correct Interpretation**:
1. **Feature normalization IS still critical** - enables learning (Phase 8: 33% → Phase 9 with normalization: 47%)
2. **True metadata performance**: ~47-50% accuracy (not 97.6%)
3. **All three fixes (oversampling + balanced alpha + normalization) are necessary** but not sufficient alone
4. **Patient-level CV is essential** to prevent overfitting to patient-specific characteristics
5. **CV test results (~50% accuracy) were CORRECT** - Phase 9 results were the outlier due to leakage

**Investigation Details**: See `agent_communication/test_metadata_phase9_cv_output.txt` and `metadata_investigation_summary.txt`

### Next Steps

- ✅ Test with image modalities (depth_rgb, depth_map, thermal_map) - COMPLETED via comprehensive CV test
- ✅ Run comprehensive CV test with all modality combinations - COMPLETED (results in `results_comprehensive_cv_test.txt`)
- ✅ Verify no data/model leaks in full pipeline - COMPLETED (thermal_map warning was false positive)
- **NEW**: Develop feature engineering strategies to improve true generalization performance beyond current ~47-50% baseline

---

## 2025-12-23 — StandardScaler scope bug fix (comprehensive CV test unblocked)

### Issue

Comprehensive CV test failed for ALL modalities with:
```
NameError: cannot access free variable 'StandardScaler' where it is not associated with a value in enclosing scope
File: src/data/dataset_utils.py, line 823
Function: preprocess_split()
```

### Root Cause

- StandardScaler already imported at module level (line 14)
- Redundant local import added at line 968 during Phase 9 implementation
- Created scope conflict: nested function `preprocess_split()` (defined at line 727) couldn't access StandardScaler because:
  - It's referenced at line 823 inside the nested function
  - The local import at line 968 happens AFTER the function definition
  - Python sees the variable will be defined in enclosing scope but not yet available

### Fix

**File**: `src/data/dataset_utils.py`
**Change**: Removed redundant import at line 968 (commit e37bdf7)
```python
# BEFORE (line 968)
from sklearn.preprocessing import StandardScaler

# AFTER
# Note: StandardScaler is already imported at module level (line 14)
```

### Impact

- Unblocks comprehensive CV test for all modalities
- No functional change - same StandardScaler is used, just from module-level import
- Local agent can now re-run `test_comprehensive_cv.py` to verify Phase 9 fix works across all modalities

---

## 2026-01-04 — Multi-modal fusion catastrophic failure fix (V5: Two-stage fine-tuning)

### Fusion architecture overfitting issue

**Files**: `src/models/builders.py` (lines 318-347), `src/training/training_utils.py` (lines 1090-1375)

**Problem**: Multi-modal fusion (metadata + thermal_map) failed catastrophically:
- Metadata-only: Kappa 0.254 ✅
- thermal_map-only: Kappa 0.145 ✅
- Fusion V1 (learned weights): Kappa 0.014 ❌ (Train 0.96, Val -0.02)
- Fusion V2 (fixed 70/30 weights): Kappa 0.023 ❌ (Train 0.96, Val 0.02)

**Root causes identified**:
1. BatchNormalization destroyed RF probability structure (negative values, wrong scale)
2. Image branch trained in fusion mode optimizes different objective than standalone
3. Image learns to overfit training data (complementary-but-weak features)

**Solution V5 - Two-stage fine-tuning**:
- **Stage 1** (30 epochs): Load pre-trained thermal_map weights → freeze image branch → train with fixed 70/30 fusion → establish stable baseline (Kappa ~0.22)
- **Stage 2** (up to 100 epochs): Unfreeze image → recompile with LR 1e-6 (100x lower) → fine-tune gently → allow potential synergy (Kappa 0.22-0.28)
- Early stopping reverts to Stage 1 if overfitting detected

**Expected results**:
- Stage 1: Kappa 0.22 (frozen baseline, no overfitting)
- Stage 2: Kappa 0.22-0.28 (potential improvement from synergy)
- Better than individual modalities if fusion benefits achieved

**Implementation**:
- Removed BatchNorm from metadata branch (was converting probabilities to z-scores)
- Fixed 70/30 fusion weights (RF dominates since Kappa 0.254 > Image 0.145)
- Two-stage training with pre-trained frozen → fine-tune pipeline
- Aggressive early stopping (patience=10) prevents catastrophic overfitting

**Note**: Fusion weight "alpha" (70/30 split) is DIFFERENT from class weight alphas (computed from class frequencies for focal loss). Class weight alphas remain dynamic.

**Automatic pre-training** (NEW): System now automatically trains image-only model if checkpoint missing
- Detects missing thermal_map checkpoint
- Trains thermal_map-only on SAME data split (prevents data leakage)
- Saves checkpoint with correct CV fold alignment
- Loads weights into fusion model and freezes
- Proceeds with two-stage fusion training
- **Single command execution** - no manual two-step workflow required!

**CRITICAL BUG FIX** (2026-01-04): RF probability normalization
- **Bug**: Ordinal RF probabilities didn't sum to 1.0 (summed to ~1.04-1.10)
- **Formula**: `prob_I + prob_P + prob_R = 1 + prob2(1-prob1)` ≠ 1.0
- **Impact**: Fusion catastrophically failed (Kappa -0.007, worse than random!)
- **Fix**: Added normalization step: divide each probability by total sum
- **Files**: `src/data/dataset_utils.py:1031-1037`, `src/data/caching.py:539-545`
- **Results** (32x32, 1-fold):
  - Before fix: Kappa -0.007 ❌
  - After fix: Kappa 0.3158 ✅ (441% better than thermal_map baseline 0.0584)
- **Details**: See `agent_communication/fusion_fix/ROOT_CAUSE_ANALYSIS.md`

**Testing protocol**: Just run fusion training - automatic pre-training handles everything!
```bash
# In production_config.py:
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map'),]

# Single command - auto pre-trains if needed
python src/main.py --mode search --cv_folds 3 --verbosity 3 --resume_mode fresh
```

---
