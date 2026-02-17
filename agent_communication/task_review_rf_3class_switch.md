# Task: Review RF Switch from Ordinal Decomposition to Direct 3-Class

## Status: PENDING REVIEW

---

## Objective

Verify that the RF metadata classifier has been correctly switched from **ordinal decomposition** (2 binary RFs) to a **direct 3-class RF** (single multiclass RF). Ensure no references to the old binary approach remain, no downstream breakages exist, and the new config parameters are wired correctly.

---

## Why This Change Was Made

Diagnostic testing (`scripts/diagnose_loo_kappa.py`) showed direct 3-class RF consistently outperforms ordinal decomposition:

| Approach | Kappa |
|----------|-------|
| Ordinal, 300 trees | 0.08-0.12 |
| Ordinal, 50 trees | 0.12-0.17 |
| **Direct 3-class, 300 trees** | **0.16-0.17** |

The ordinal approach trained 2 binary RFs:
- `rf_model1`: I vs (P+R) → `prob1`
- `rf_model2`: (I+P) vs R → `prob2`
- Combined: `p_I = 1 - prob1`, `p_P = prob1 * (1 - prob2)`, `p_R = prob2`
- These probabilities did NOT sum to 1.0, requiring a normalization hack

The direct 3-class RF is simpler (1 model), produces properly normalized probabilities, and performs better.

---

## Files Modified

### 1. `src/utils/production_config.py`

**What was added** (after `TRAINING_CLASS_WEIGHT_MODE` line):
```python
RF_N_ESTIMATORS = 300         # Number of trees
RF_CLASS_WEIGHT = 'balanced'  # 'balanced' (sklearn auto), 'frequency' (use alpha_values), or None
RF_OOF_FOLDS = 5             # Internal OOF folds for training predictions
```

**Verify:**
- These 3 variables exist and are importable
- `RF_N_ESTIMATORS` is a positive integer
- `RF_CLASS_WEIGHT` is one of: `'balanced'`, `'frequency'`, `None`
- `RF_OOF_FOLDS` is a positive integer ≥ 2

### 2. `src/data/dataset_utils.py`

This is the main file with extensive changes. All changes are within the `prepare_cached_datasets()` function and its inner `preprocess_split()` function, plus the LOO filter functions.

---

## Detailed Changes to Verify

### A. `_make_rf()` function (inside `preprocess_split()`)

**Old:**
```python
def _make_rf():
    return RandomForestClassifier(
        n_estimators=300,                    # hardcoded
        random_state=42 + run * (run + 3),
        class_weight='balanced',             # hardcoded
        n_jobs=-1,
    )
```

**New:**
```python
def _make_rf():
    from src.utils.production_config import RF_N_ESTIMATORS, RF_CLASS_WEIGHT
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,        # from config
        random_state=42 + run * (run + 3),
        class_weight=RF_CLASS_WEIGHT,        # from config
        n_jobs=-1,
    )
```

**Verify:**
- `RF_N_ESTIMATORS` and `RF_CLASS_WEIGHT` are imported from production_config
- `random_state` still uses the run-dependent seed (unchanged)
- `n_jobs=-1` is preserved

### B. Binary label creation removed

**Old (should NOT exist anymore):**
```python
y_bin1 = (y > 0).astype(int)
y_bin2 = (y > 1).astype(int)
```

**Verify:**
- No references to `y_bin1` or `y_bin2` exist anywhere in the main RF pipeline (inside `preprocess_split`)
- The variable `y` (3-class labels: 0, 1, 2) is used directly

### C. OOF training loop (training path, `is_training=True`)

**Old:** Created `prob1_unique_oof` and `prob2_unique_oof` (two 1D arrays), trained `rf1_fold` and `rf2_fold` per fold.

**New:** Creates `probs_unique_oof` (one 2D array of shape `(N_unique, 3)`), trains single `rf_fold` per fold.

**Verify:**
- `probs_unique_oof = np.zeros((len(X_unique), 3))` — shape is `(N, 3)` not `(N,)`
- In the fold loop: `rf_fold.predict_proba(X_unique[oof_idx])` returns shape `(n_oof, 3)` and is assigned to `probs_unique_oof[oof_idx]`
- `y_unique = y.values[unique_indices]` — using 3-class labels, NOT binary
- `rf_fold.fit(X_unique[tr_idx], y_unique[tr_idx])` — fitting on 3-class labels
- OOF fold count uses `RF_OOF_FOLDS` from config (was hardcoded 5)

### D. OOF prediction mapping back to all rows

**Old:**
```python
prob1 = np.zeros(len(X))
prob2 = np.zeros(len(X))
for u_pos, orig_idxs in group_map.items():
    for oi in orig_idxs:
        prob1[oi] = prob1_unique_oof[u_pos]
        prob2[oi] = prob2_unique_oof[u_pos]
```

**New:**
```python
probs = np.zeros((len(X), 3))
for u_pos, orig_idxs in group_map.items():
    for oi in orig_idxs:
        probs[oi] = probs_unique_oof[u_pos]
```

**Verify:**
- `probs` shape is `(len(X), 3)` — 2D array
- Each row in `probs` gets all 3 class probabilities from its unique pattern

### E. Final RF model training

**Old:**
```python
rf_model1 = _make_rf()
rf_model2 = _make_rf()
rf_model1.fit(X_unique, y_bin1_unique)
rf_model2.fit(X_unique, y_bin2_unique)
```

**New:**
```python
rf_model = _make_rf()
rf_model.fit(X_unique, y_unique)
```

**Verify:**
- Single model `rf_model` (not `rf_model1`/`rf_model2`)
- Fitted on `y_unique` (3-class), not binary labels

### F. Validation prediction path (`is_training=False`)

**Old:**
```python
prob1 = rf_model1.predict_proba(X.values)[:, 1]
prob2 = rf_model2.predict_proba(X.values)[:, 1]
```

**New:**
```python
probs = rf_model.predict_proba(X.values)
```

**Verify:**
- Single model call, result is shape `(N, 3)`
- No `[:, 1]` indexing — we take all 3 columns

### G. Ordinal decomposition removed

**Old (should NOT exist anymore):**
```python
prob_I_unnorm = 1 - prob1
prob_P_unnorm = prob1 * (1 - prob2)
prob_R_unnorm = prob2
total = prob_I_unnorm + prob_P_unnorm + prob_R_unnorm
prob_I = prob_I_unnorm / total
prob_P = prob_P_unnorm / total
prob_R = prob_R_unnorm / total
```

**New:**
```python
prob_I = probs[:, 0]
prob_P = probs[:, 1]
prob_R = probs[:, 2]
```

**Verify:**
- No `prob_I_unnorm`, `prob_P_unnorm`, `prob_R_unnorm` variables
- No normalization step (RF `predict_proba` already sums to 1.0)
- `prob_I` is class 0, `prob_P` is class 1, `prob_R` is class 2
- **CRITICAL**: Confirm class ordering matches label encoding: `I=0, P=1, R=2`. This is set by `data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})` at the top of `prepare_cached_datasets()`. sklearn RF `predict_proba` returns columns in sorted class order (0, 1, 2), so `probs[:, 0]` = class 0 = I. This is correct.

### H. RF metrics calculation

**Old:**
```python
rf_preds = np.argmax(np.column_stack([prob_I_unnorm, prob_P_unnorm, prob_R_unnorm]), axis=1)
```

**New:**
```python
rf_preds = np.argmax(probs, axis=1)
```

**Verify:**
- `probs` is already shape `(N, 3)`, no need to column_stack
- Metrics (accuracy, F1-macro, Kappa) are computed on `rf_preds` vs `rf_true`

### I. DataFrame storage (should be unchanged)

```python
split_data['rf_prob_I'] = prob_I
split_data['rf_prob_P'] = prob_P
split_data['rf_prob_R'] = prob_R
```

**Verify:**
- Column names `rf_prob_I`, `rf_prob_P`, `rf_prob_R` are preserved
- These are used downstream in `map_row_to_features()` to create `metadata_input` tensor of shape `(3,)`

### J. Function signature changes

**Old:**
```python
def preprocess_split(split_data, is_training=True, class_weight_dict_binary1=None,
                     class_weight_dict_binary2=None, rf_model1=None, rf_model2=None, imputation_data=None):
    ...
    return split_data, rf_model1, rf_model2
```

**New:**
```python
def preprocess_split(split_data, is_training=True, rf_model=None, imputation_data=None):
    ...
    return split_data, rf_model
```

**Verify:**
- `class_weight_dict_binary1`, `class_weight_dict_binary2` parameters removed
- `rf_model1`, `rf_model2` replaced by single `rf_model`
- Return tuple is `(split_data, rf_model)` — 2 elements, not 3

### K. Call sites

**Old:**
```python
train_data, rf_model1, rf_model2 = preprocess_split(source_data, is_training=True,
    class_weight_dict_binary1=class_weight_dict_binary1, class_weight_dict_binary2=class_weight_dict_binary2)
valid_data, _, _ = preprocess_split(valid_data, is_training=False,
    rf_model1=rf_model1, rf_model2=rf_model2, imputation_data=source_data)
del rf_model1, rf_model2
```

**New:**
```python
train_data, rf_model = preprocess_split(source_data, is_training=True)
valid_data, _ = preprocess_split(valid_data, is_training=False, rf_model=rf_model, imputation_data=source_data)
del rf_model
```

**Verify:**
- Training call: no keyword args for class weights or RF models
- Validation call: `rf_model=rf_model` passes the single trained model
- `del rf_model` (not `del rf_model1, rf_model2`)
- No references to `class_weight_dict_binary1` or `class_weight_dict_binary2` remain anywhere in `prepare_cached_datasets()`

### L. Binary class weight computation block removed

**Old (should NOT exist anymore):**
```python
if 'metadata' in selected_modalities:
    unique_cases = train_data[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates().copy()
    unique_cases['label_bin1'] = (unique_cases['Healing Phase Abs'] > 0).astype(int)
    unique_cases['label_bin2'] = (unique_cases['Healing Phase Abs'] > 1).astype(int)
    class_weights_binary1 = compute_class_weight(...)
    class_weights_binary2 = compute_class_weight(...)
else:
    class_weight_dict_binary1 = None
    class_weight_dict_binary2 = None
```

**Verify:**
- This entire block (~35 lines) is gone
- No references to `class_weight_dict_binary1` or `class_weight_dict_binary2` anywhere in the file
- The `from sklearn.utils.class_weight import compute_class_weight` import at the top of the file should also be removed (it was only used by this block in dataset_utils.py)

### M. LOO filter function (`rf_loo_influence_filter`)

**Old signature:**
```python
def rf_loo_influence_filter(X_unique, y_bin1_unique, y_bin2_unique, y_unique_3class, seed, ...)
```

**New signature:**
```python
def rf_loo_influence_filter(X_unique, y_unique, seed, ...)
```

**Old `_compute_oof_kappa`:** trained 2 binary RFs, combined with ordinal formula, used `cohen_kappa_score(y_3c, preds)`.

**New `_compute_oof_kappa`:** trains 1 direct 3-class RF, uses `rf.predict()`, returns `cohen_kappa_score(y, preds)`.

**Verify:**
- Function signature has `y_unique` (not `y_bin1_unique, y_bin2_unique, y_unique_3class`)
- `_compute_oof_kappa(X, y)` takes 2 args (not 4)
- Inside `_compute_oof_kappa`: single `RandomForestClassifier` (not 2)
- Uses `rf.predict(X[val])` (direct class prediction) — no ordinal decomposition
- In the harmful pattern selection loop: `y_unique == cls` (not `y_unique_3class == cls`)
- LOO is currently **disabled** (`USE_RF_LOO_FILTERING = False`) but the code should still be correct for when it's re-enabled

### N. LOO integration block (inside `prepare_cached_datasets`)

**Old:**
```python
y_bin1_unique = (y_unique > 0).astype(int)
y_bin2_unique = (y_unique > 1).astype(int)
...
patterns_to_remove, influence_scores, baseline_kappa = rf_loo_influence_filter(
    X_unique, y_bin1_unique, y_bin2_unique, y_unique, seed, ...
)
```

**New:**
```python
patterns_to_remove, influence_scores, baseline_kappa = rf_loo_influence_filter(
    X_unique, y_unique, seed, ...
)
```

**Verify:**
- `y_bin1_unique` and `y_bin2_unique` creation lines are removed
- `rf_loo_influence_filter` called with `(X_unique, y_unique, seed, ...)` — 3 positional args before kwargs

---

## Things That Should NOT Have Changed

Verify these are still intact:

1. **`map_row_to_features()` function** — Still reads `rf_prob_I`, `rf_prob_P`, `rf_prob_R` and stacks into `metadata_input` shape `(3,)`
2. **`training_utils.py` line ~350** — `self.all_modality_shapes['metadata'] = (3,)` still hardcoded
3. **`builders.py`** — `ConfidenceBasedMetadataAttention` still expects `inputs[:, :3]` as RF probs
4. **Diagnostic/visualization code** — Bar charts with `['I', 'P', 'R']` labels
5. **`_preprocess_for_loo()` function** — Should be unchanged (it's preprocessing, not the RF itself)
6. **Oversampling** (`apply_mixed_sampling_to_df`) — Should still happen before `preprocess_split`
7. **Feature normalization** block after `preprocess_split` calls — Should be unchanged
8. **TF dataset creation** (`create_cached_dataset`) — Should be unchanged

---

## Potential Edge Cases to Check

1. **Class ordering**: RF `predict_proba` returns columns in sorted order of `rf.classes_`. Since labels are `{0, 1, 2}`, `probs[:, 0]` = class 0 = I, `probs[:, 1]` = class 1 = P, `probs[:, 2]` = class 2 = R. Verify that `rf.classes_` after fitting always equals `[0, 1, 2]`. This could break if a training fold is missing a class.

2. **Missing class in OOF fold**: If a KFold split has a fold where one class is absent from the training portion, `predict_proba` might return a `(N, 2)` array instead of `(N, 3)`. This would cause a shape mismatch when assigning to `probs_unique_oof[oof_idx]`. Check if `class_weight='balanced'` or the stratification prevents this. With ~414 unique patterns and 5 folds, each fold has ~83 patterns — all 3 classes should be present, but verify.

3. **`RF_CLASS_WEIGHT = 'frequency'`**: The config allows `'frequency'` as an option, but this is not a valid sklearn `class_weight` parameter. If used, `_make_rf()` would need to convert it to a dict using `alpha_values`. Currently `'balanced'` is set, but if someone changes it to `'frequency'`, it would crash. Either document this limitation or add handling.

---

## Verification Command

```bash
/venv/multimodal/bin/python src/main.py --data_percentage 100 --device-mode multi --verbosity 2 --resume_mode fresh --fold 1 --cv_folds 3
```

**Expected output:**
- `RF dedup: XXXX rows → ~414 unique patterns`
- `RF out-of-fold predictions (5-fold on ~414 unique rows)` — 5 from `RF_OOF_FOLDS`
- `RF standalone metrics (TRAIN (out-of-fold)):` — Kappa should be ~0.15-0.22 (was ~0.08-0.12 with ordinal)
- `RF standalone metrics (VALIDATION):` — Kappa should be ~0.12-0.18 (was ~0.05-0.10)
- No errors, shape mismatches, or import failures
- Training should complete all epochs and produce final metrics

**Failure indicators:**
- `ValueError: Input contains NaN` — missing class in a fold
- `ImportError: cannot import name 'RF_N_ESTIMATORS'` — config not saved
- Shape mismatch errors — predict_proba returning wrong shape
- `NameError: name 'rf_model1' is not defined` — leftover reference to old variable names
