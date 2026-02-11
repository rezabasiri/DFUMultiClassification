# Production Config Audit Report

**Date**: 2026-02-11
**Files Analyzed**: `src/utils/production_config.py`, `src/main.py`, and all importing modules
**Total Configuration Variables**: 87
**Status**: ✅ 78 used (89.7%), ❌ 4 unused (4.6%), 🔧 6 helper functions never called

---

## Executive Summary

The comprehensive audit reveals that **production_config.py is largely well-utilized** with 89.7% of configurations actively used. However, there are **4 unused configs**, **6 helper functions never called**, and **~10 hardcoded values** that should be moved to configs for better maintainability.

### Critical Issues Found

1. **LR_SCHEDULE_EXPLORATION_EPOCHS (300) > N_EPOCHS (100)** - Inconsistent configuration
2. **FOCAL_GAMMA and FOCAL_ALPHA** - Dead code, never used anywhere
3. **10+ hardcoded neural network hyperparameters** - Should be in config
4. **6 helper functions defined but never invoked** - Code bloat

---

## Section 1: Configuration Usage by Category

### ✅ Fully Used Categories (All configs actively used)

#### 1. Training Parameters
- `IMAGE_SIZE` (256) → Used in 6+ files
- `GLOBAL_BATCH_SIZE` (320) → Batch size calculations
- `N_EPOCHS` (100) → Training duration
- `STAGE1_EPOCHS` (10) → Two-stage fusion training
- `RGB_BACKBONE` ('EfficientNetB3') → Model architecture selection
- `MAP_BACKBONE` ('EfficientNetB1') → Model architecture selection
- `DATA_PERCENTAGE` (100) → Data sampling control
- `SAMPLING_STRATEGY` ('combined') → Class balancing strategy
- `EARLY_STOP_PATIENCE` (20) → Early stopping callback
- `REDUCE_LR_PATIENCE` (10) → Learning rate reduction callback

#### 2. Outlier Detection
- `OUTLIER_REMOVAL` (True) → Controls outlier detection pipeline
- `OUTLIER_CONTAMINATION` (0.15) → Isolation Forest contamination parameter
- `OUTLIER_BATCH_SIZE` (64) → Batch size for feature extraction

#### 3. Augmentation Configuration
All 13 generative augmentation configs are properly wired:
- V3 (SDXL): `GENERATIVE_AUG_SDXL_MODEL_PATH`, `GENERATIVE_AUG_SDXL_RESOLUTION`, `GENERATIVE_AUG_SDXL_GUIDANCE_SCALE`
- V2 (SD 1.5): `GENERATIVE_AUG_MODEL_PATH`, `GENERATIVE_AUG_MAX_MODELS`
- Common: `USE_GENERATIVE_AUGMENTATION`, `GENERATIVE_AUG_VERSION`, `GENERATIVE_AUG_PROB`, etc.

#### 4. Gating Network Configuration
All 13 gating network parameters actively used:
- Architecture: `GATING_NUM_HEADS_MULTIPLIER`, `GATING_KEY_DIM_MULTIPLIER`
- Training: `GATING_LEARNING_RATE`, `GATING_EPOCHS`, `GATING_BATCH_SIZE`, `GATING_VERBOSE`
- Callbacks: `GATING_REDUCE_LR_*`, `GATING_EARLY_STOP_*`

#### 5. Hierarchical Gating Network
All 13 hierarchical parameters actively used in main.py lines 1360-1525

#### 6. Learning Rate Scheduler
All 7 LR scheduler parameters used (but see Issue #1 below)

#### 7. Attention Mechanisms
All 8 attention visualization and entropy parameters actively used

#### 8. Cross-Validation
All 3 CV parameters used: `CV_N_SPLITS`, `CV_RANDOM_STATE`, `CV_SHUFFLE`

#### 9. Modality Search
All 7 modality search parameters actively used

#### 10. Model Combination Search
All 5 search parameters used: `SEARCH_MIN_MODELS`, `SEARCH_MAX_TRIES`, etc.

#### 11. Data Cleaning
All 5 parameters used: `TRACK_MISCLASS`, `USE_CORE_DATA`, `THRESHOLD_I/P/R`

#### 12. File I/O
Both retry parameters used: `PROGRESS_MAX_RETRIES`, `PROGRESS_RETRY_DELAY`

#### 13. Environment Configuration
All 5 TensorFlow environment variables properly set via `apply_environment_config()`

---

### ❌ Unused/Dead Code

#### Unused Configuration Variables (Can be removed)

1. ~~**`FOCAL_GAMMA = 1.0`** (line 266)~~ ✅ **FIXED**
   - Status: ✅ DELETED
   - Note: Was legacy dead code, HIERARCHICAL_FOCAL_GAMMA is used instead

2. ~~**`FOCAL_ALPHA = 0.25`** (line 267)~~ ✅ **FIXED**
   - Status: ✅ DELETED
   - Note: Was legacy dead code, HIERARCHICAL_FOCAL_ALPHA is used instead

3. **`SHOW_PROGRESS_BAR = True`** (line 153)
   - Status: ❌ NEVER REFERENCED
   - Action: **DELETE** if progress bar feature was abandoned, or **IMPLEMENT** if intended

4. **`PROGRESS_BAR_UPDATE_INTERVAL = 1`** (line 154)
   - Status: ❌ NEVER REFERENCED
   - Action: **DELETE** if progress bar feature was abandoned, or **IMPLEMENT** if intended

#### Unused Helper Functions (Never called)

All 6 helper functions are defined but NEVER invoked anywhere:

1. **`get_gating_config()`** (lines 342-356)
   - Returns: Dictionary of 11 gating network parameters
   - Action: Either **USE IT** (recommended for consistency) or **DELETE IT**

2. **`get_hierarchical_config()`** (lines 358-374)
   - Returns: Dictionary of 13 hierarchical gating parameters
   - Action: Either **USE IT** or **DELETE IT**

3. **`get_lr_schedule_config()`** (lines 376-386)
   - Returns: Dictionary of 7 LR scheduler parameters
   - Action: Either **USE IT** or **DELETE IT**

4. **`get_search_config()`** (lines 388-396)
   - Returns: Dictionary of 5 model search parameters
   - Action: Either **USE IT** or **DELETE IT**

5. **`get_attention_config()`** (lines 398-410)
   - Returns: Dictionary of 8 attention/entropy parameters
   - Action: Either **USE IT** or **DELETE IT**

6. **`get_environment_config()`** (lines 412-420)
   - Returns: Dictionary of 5 environment variables
   - Action: Either **USE IT** or **DELETE IT**

**Recommendation**: These functions provide a cleaner API for accessing grouped configs. **Keep them and START USING them** in the codebase for better code organization.

---

## ~~Section 2: Hardcoded Values That Should Be Configs~~ ✅ **ALL FIXED**

### ~~High Priority (Neural Network Hyperparameters)~~ ✅ **ALL MOVED TO CONFIG**

All 10 hardcoded neural network hyperparameters have been moved to `production_config.py`:

| Config Variable | Value | Usage | Status |
|----------------|-------|-------|--------|
| `DROPOUT_RATE` | 0.3 | ResidualBlock dropout | ✅ ADDED |
| `ATTENTION_TEMPERATURE` | 0.1 | DualLevelAttentionLayer temperature init | ✅ ADDED |
| `GATING_DROPOUT_RATE` | 0.1 | MultiHeadAttention dropout | ✅ ADDED |
| `GATING_L2_REGULARIZATION` | 1e-4 | Gating network L2 regularizer | ✅ ADDED |
| `ATTENTION_CLASS_NUM_HEADS` | 8 | Class attention heads | ✅ ADDED |
| `ATTENTION_CLASS_KEY_DIM` | 32 | Class attention key dimension | ✅ ADDED |
| `TEMPERATURE_MIN_VALUE` | 0.01 | Temperature scaling divisor | ✅ ADDED |
| `TRANSFORMER_DROPOUT_RATE` | 0.1 | TransformerBlock dropout | ✅ ADDED |
| `LAYER_NORM_EPSILON` | 1e-6 | LayerNormalization epsilon | ✅ ADDED |
| `HIERARCHICAL_L2_REGULARIZATION` | 0.001 | Hierarchical gating L2 reg | ✅ ADDED |

**Changes Made**:
- Added 10 new config variables to `production_config.py` (lines 163-169, 198-200)
- Updated `ResidualBlock` to use `DROPOUT_RATE` and `LAYER_NORM_EPSILON`
- Updated `DualLevelAttentionLayer` to use all attention configs
- Updated `TransformerBlock` to use `TRANSFORMER_DROPOUT_RATE` and `LAYER_NORM_EPSILON`
- Updated `create_hierarchical_gating_network` to use `HIERARCHICAL_L2_REGULARIZATION` and `TRANSFORMER_DROPOUT_RATE`

**Result**: All neural network hyperparameters now centralized in production_config.py for easy tuning and experimentation!

---

## Section 3: Configuration Inconsistencies

### ~~Issue #1: LR Schedule vs Training Epochs~~ ✅ **FIXED**

**Problem**: Learning rate scheduler exploration period exceeded total training epochs (was 300 vs 100)

**Solution Implemented**: Changed LR_SCHEDULE_EXPLORATION_EPOCHS to automatically adjust with N_EPOCHS:
```python
# production_config.py line 212
LR_SCHEDULE_EXPLORATION_EPOCHS = int(N_EPOCHS * 0.10)  # Auto-adjusts to 10% of N_EPOCHS
```

**Result**:
- Exploration phase is now 10 epochs (10% of 100 total epochs)
- Automatically scales when N_EPOCHS is changed
- Matches the STAGE1_EPOCHS pattern (also 10% of N_EPOCHS)
- Updated comments to clarify exploration phase purpose

---

### ~~Issue #2: Duplicate Environment Variable Setting~~ ✅ **FIXED**

**Problem**: Environment variables were set in TWO places with hardcoded values

**Solution Implemented**:
1. Moved production_config import to line 43 (before TensorFlow import)
2. Call `apply_environment_config()` at line 44 (before TF import) to set all 5 environment variables from config
3. Removed duplicate hardcoded `setdefault()` calls (lines 23-25)
4. Removed redundant `apply_environment_config()` call at line 2548

**Result**:
- Single source of truth: all environment settings come from production_config.py
- No hardcoded values
- Applied before TensorFlow import (as required)
- Cleaner code with no duplication

---

### ~~Issue #3: Grid Search Parameter Duplication~~ ✅ **FIXED**

**Problem**: Grid search parameters were defined in config but `perform_grid_search()` function used hardcoded values

**Solution Implemented**: Updated `perform_grid_search()` function to use config variables:
```python
# main.py perform_grid_search() function
param_grid = {
    'ordinal_weight': GRID_SEARCH_ORDINAL_WEIGHTS,
    'gamma': GRID_SEARCH_GAMMAS,
    'alpha': GRID_SEARCH_ALPHAS
}
```

**Result**:
- Single source of truth: all grid search parameters defined in production_config.py
- Easy to modify search parameters without editing main.py
- Consistent with other config usage patterns

---

## Section 4: Recommendations Summary

### 🔴 HIGH PRIORITY (Fix Immediately)

1. ~~**Fix LR_SCHEDULE_EXPLORATION_EPOCHS**~~ ✅ **FIXED**
   - ~~Change from 300 to 100 (line 212) OR document why it differs~~
   - Now auto-calculates as 10% of N_EPOCHS
   - File: `src/utils/production_config.py:212`

2. ~~**Remove Dead Code**~~ ✅ **FIXED**
   - ~~Delete `FOCAL_GAMMA` (line 266)~~ - DELETED
   - ~~Delete `FOCAL_ALPHA` (line 267)~~ - DELETED
   - File: `src/utils/production_config.py`

3. ~~**Add Missing Configs for Hardcoded Hyperparameters**~~ ✅ **FIXED**
   - ~~Add 10+ new config variables (see Section 2)~~ - ADDED 10 new configs
   - ~~Update main.py to use these configs~~ - Updated ResidualBlock, DualLevelAttentionLayer, TransformerBlock, create_hierarchical_gating_network
   - Files: `src/utils/production_config.py`, `src/main.py`

---

### 🟡 MEDIUM PRIORITY (Improve Code Quality)

4. **Resolve Helper Function Status**
   - Option A: **START USING** the 6 helper functions (recommended)
   - Option B: **DELETE** the 6 helper functions if not needed
   - File: `src/utils/production_config.py:342-420`

5. **Remove Unused Progress Bar Configs**
   - Delete `SHOW_PROGRESS_BAR` (line 153)
   - Delete `PROGRESS_BAR_UPDATE_INTERVAL` (line 154)
   - OR implement the progress bar feature if intended
   - File: `src/utils/production_config.py`

6. ~~**Resolve Environment Variable Duplication**~~ ✅ **FIXED**
   - ~~Choose one method: direct setdefault() OR apply_environment_config()~~
   - Now uses only `apply_environment_config()` called before TF import
   - Files: `src/main.py:43-44`

7. ~~**Fix Grid Search Parameter Duplication**~~ ✅ **FIXED**
   - ~~Update `perform_grid_search()` to use GRID_SEARCH_* configs~~
   - Now uses config variables from production_config.py
   - File: `src/main.py:1680-1682`

---

### 🟢 LOW PRIORITY (Nice to Have)

8. **Add Documentation**
   - Document why `USE_GENERATIVE_AUGMENTATION = False` by default
   - Document why `USE_CORE_DATA = False` by default
   - Add usage examples for each config category

9. **Add Type Hints**
   - Add type annotations to all config variables
   - Improves IDE autocomplete and type checking

10. **Add Config Validation Function**
    - Create `validate_config()` to check for inconsistencies
    - Example: Verify `LR_SCHEDULE_EXPLORATION_EPOCHS <= N_EPOCHS`
    - Call during initialization to catch errors early

---

## Section 5: Overall Assessment

### Strengths ✅

1. **Excellent organization** - Configs grouped by category with clear comments
2. **High utilization rate** - 89.7% of configs actively used
3. **Good documentation** - Inline comments explain purpose of most parameters
4. **Modular design** - Helper functions provided (even if not used)

### Weaknesses ❌

1. **Dead code present** - 4 unused config variables
2. **Hardcoded hyperparameters** - ~10 values should be in config
3. **Inconsistent values** - LR_SCHEDULE_EXPLORATION_EPOCHS > N_EPOCHS
4. **Unused helper functions** - 6 functions defined but never invoked

### Conclusion

The `production_config.py` file is **generally well-designed and properly utilized**. The main issues are:
- Small amount of dead code (4 variables)
- Missing configs for hardcoded neural network hyperparameters
- One critical inconsistency (LR schedule vs N_EPOCHS)

**Recommended Action**: Address the HIGH PRIORITY items immediately. The MEDIUM and LOW priority items can be addressed incrementally as part of normal code maintenance.

---

## Appendix: Quick Reference

### Config Files
- **Main config**: `src/utils/production_config.py` (430 lines, 87 variables)
- **Primary consumer**: `src/main.py` (imports with `from src.utils.production_config import *`)

### Other Consumers
- `src/training/training_utils.py` - Uses 15+ configs
- `src/data/dataset_utils.py` - Uses augmentation configs
- `src/data/generative_augmentation_v3.py` - Uses SDXL configs
- `src/data/generative_augmentation_v2.py` - Uses SD 1.5 configs
- `src/models/builders.py` - Uses IMAGE_SIZE, RGB_BACKBONE, MAP_BACKBONE
- `src/utils/outlier_detection.py` - Uses outlier detection configs
- `src/utils/verbosity.py` - Uses VERBOSITY

### Configuration Statistics
- **Total variables**: 87
- **Actively used**: 78 (89.7%)
- **Never used**: 4 (4.6%)
- **Helper functions**: 6 (0% usage)
- **Hardcoded values that should be configs**: ~10

---

**Report Generated**: 2026-02-11
**Agent**: Explore agent (thorough codebase analysis)
**Status**: ✅ Investigation Complete
