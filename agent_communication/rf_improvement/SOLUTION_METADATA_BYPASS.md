# SOLUTION: Bypass Neural Network for Metadata-Only Mode

## Problem Statement

**Current Behavior:**
```python
# src/training/training_utils.py:1195-1215
if selected_modalities == ['metadata']:
    vprint("Metadata-only: Minimal training on final layer", level=2)

# BUG: Still trains NN for 300 epochs even in metadata-only mode!
history = model.fit(train_dataset_dis, epochs=max_epochs, ...)
```

**Result**: NN training degrades RF performance from Kappa 0.20 → 0.109 (-45%)

**Root Cause**: The fusion layer is designed for multi-modal flexibility, but when only metadata is present, training this layer is unnecessary and harmful.

---

## Solution Overview

Add bypass logic to use RF probabilities directly in metadata-only mode:

```python
if selected_modalities == ['metadata']:
    vprint("Metadata-only: Using RF predictions directly (no NN training)", level=2)
    # Use RF probabilities as final predictions
    # Skip model.fit() entirely
else:
    vprint("Multi-modal: Training fusion network", level=2)
    # Train model for fusion
    history = model.fit(...)
```

**Expected Impact**: Kappa 0.109 → 0.20 (+83%)

---

## Implementation Details

### Location
**File**: `src/training/training_utils.py`
**Lines**: ~1195-1280 (modify the training section)

### Current Flow
1. Check if metadata-only (line 1195) ✅
2. Print message about minimal training ✅
3. **Call model.fit() anyway** ❌
4. Evaluate model.predict() on train/valid ❌
5. Calculate metrics from NN predictions ❌

### New Flow (Metadata-Only)
1. Check if metadata-only ✅
2. Print "Using RF predictions directly" ✅
3. **Extract RF probabilities from dataset** 🆕
4. **Use argmax(RF probs) as predictions** 🆕
5. Calculate metrics from RF predictions 🆕
6. **Skip model.fit() entirely** 🆕

### New Flow (Multi-Modal)
1. Check if NOT metadata-only ✅
2. Build fusion model ✅
3. Train with model.fit() ✅
4. Evaluate with model.predict() ✅
5. Calculate metrics from NN predictions ✅

---

## Code Implementation

### Step 1: Extract RF Probabilities Function

Add helper function to extract RF probabilities from dataset:

```python
def extract_rf_predictions_from_dataset(dataset, steps):
    """
    Extract RF probabilities from metadata_input and convert to predictions.

    Args:
        dataset: TF dataset containing 'metadata_input' with [rf_prob_I, rf_prob_P, rf_prob_R]
        steps: Number of batches to process

    Returns:
        y_pred: Predicted classes (argmax of RF probabilities)
        probabilities: Raw RF probabilities [N, 3]
        y_true: True labels
        sample_ids: Sample identifiers
    """
    y_true = []
    probabilities = []
    sample_ids = []

    for batch in dataset.take(steps):
        batch_inputs, batch_labels = batch

        # Extract RF probabilities from metadata_input
        # metadata_input contains [rf_prob_I, rf_prob_P, rf_prob_R]
        rf_probs = batch_inputs['metadata_input'].numpy()  # Shape: [batch_size, 3]

        # Get predictions from argmax
        batch_pred_classes = np.argmax(rf_probs, axis=1)

        # Get true labels
        batch_true = np.argmax(batch_labels.numpy(), axis=1)

        # Get sample IDs
        batch_sample_ids = batch_inputs['sample_id'].numpy()

        # Store results
        y_true.extend(batch_true)
        probabilities.extend(rf_probs)
        sample_ids.extend(batch_sample_ids)

    y_pred = np.argmax(np.array(probabilities), axis=1)

    return y_pred, np.array(probabilities), np.array(y_true), np.array(sample_ids)
```

### Step 2: Modify Training Logic

Replace lines 1195-1280 with conditional bypass:

```python
# Training logic (around line 1195)
vprint(f"Total model trainable weights: {len(model.trainable_weights)}", level=2)

if selected_modalities == ['metadata']:
    # ==========================================
    # METADATA-ONLY: Use RF predictions directly
    # ==========================================
    vprint("Metadata-only: Using RF predictions directly (no NN training)", level=2)
    vprint("Bypassing neural network to preserve RF performance (Kappa ~0.20)", level=2)

    # Extract RF predictions from training dataset
    vprint("Extracting RF predictions from training data...", level=2)
    y_pred_t, probabilities_t, y_true_t, all_sample_ids_t = extract_rf_predictions_from_dataset(
        pre_aug_train_dataset, steps_per_epoch
    )

    # Store for gating network
    run_predictions_list_t.append(probabilities_t)
    if run_true_labels_t is None:
        run_true_labels_t = y_true_t

    # Save predictions
    save_run_predictions(run + 1, config_name, probabilities_t, y_true_t, ck_path, dataset_type='train')

    # Track misclassifications (if requested)
    if track_misclass in ['both', 'train']:
        track_misclassifications(y_true_t, y_pred_t, all_sample_ids_t, selected_modalities, misclass_path)

    # Extract RF predictions from validation dataset
    vprint("Extracting RF predictions from validation data...", level=2)
    valid_dataset_with_ids = filter_dataset_modalities(master_valid_dataset, selected_modalities)
    y_pred_v, probabilities_v, y_true_v, all_sample_ids_v = extract_rf_predictions_from_dataset(
        valid_dataset_with_ids, validation_steps
    )

    # Store for gating network
    run_predictions_list_v.append(probabilities_v)
    if run_true_labels_v is None:
        run_true_labels_v = y_true_v

    # Save predictions
    save_run_predictions(run + 1, config_name, probabilities_v, y_true_v, ck_path, dataset_type='valid')

    # Track misclassifications (if requested)
    if track_misclass in ['both', 'valid']:
        track_misclassifications(y_true_v, y_pred_v, all_sample_ids_v, selected_modalities, misclass_path)

    vprint("RF predictions extracted successfully (no NN training performed)", level=2)

else:
    # ==========================================
    # MULTI-MODAL: Train fusion network
    # ==========================================
    vprint("Multi-modal: Training fusion network for modality combination", level=2)

    # Train model (check for existing weights)
    checkpoint_path = create_checkpoint_filename(selected_modalities, run+1, config_name)
    if os.path.exists(checkpoint_path):
        with strategy.scope():
            model.load_weights(checkpoint_path)
        vprint("Loaded existing weights", level=1)
    else:
        vprint("No existing pretrained weights found", level=1)

        # Determine verbosity for model.fit()
        if EPOCH_PRINT_INTERVAL > 0 and get_verbosity() >= 2:
            fit_verbose = 0  # Callback will handle printing
        elif get_verbosity() >= 2:
            fit_verbose = 2  # Print every epoch
        else:
            fit_verbose = 0  # Silent

        # Train the fusion model
        history = model.fit(
            train_dataset_dis,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset_dis,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=fit_verbose
        )

    # Load best weights
    with strategy.scope():
        model.load_weights(create_checkpoint_filename(selected_modalities, run+1, config_name))

    # Evaluate training data using NN predictions
    y_true_t = []
    y_pred_t = []
    probabilities_t = []
    all_sample_ids_t = []

    for batch in pre_aug_train_dataset.take(steps_per_epoch):
        batch_inputs, batch_labels = batch
        sample_ids_batch = batch_inputs['sample_id'].numpy()
        model_inputs = {k: v for k, v in batch_inputs.items() if k != 'sample_id'}
        batch_pred = model.predict(model_inputs, verbose=0)
        y_true_t.extend(np.argmax(batch_labels, axis=1))
        y_pred_t.extend(np.argmax(batch_pred, axis=1))
        probabilities_t.extend(batch_pred)
        all_sample_ids_t.extend(sample_ids_batch)

        del batch_inputs, batch_labels, batch_pred, model_inputs, sample_ids_batch
        gc.collect()

    save_run_predictions(run + 1, config_name, np.array(probabilities_t), np.array(y_true_t), ck_path, dataset_type='train')
    run_predictions_list_t.append(np.array(probabilities_t))
    if run_true_labels_t is None:
        run_true_labels_t = np.array(y_true_t)

    if track_misclass in ['both', 'train']:
        sample_ids_t = np.array(all_sample_ids_t)
        track_misclassifications(np.array(y_true_t), np.array(y_pred_t), sample_ids_t, selected_modalities, misclass_path)

    # Evaluate validation data using NN predictions
    y_true_v = []
    y_pred_v = []
    probabilities_v = []
    all_sample_ids_v = []

    valid_dataset_with_ids = filter_dataset_modalities(master_valid_dataset, selected_modalities)

    for batch in valid_dataset_with_ids.take(validation_steps):
        batch_inputs, batch_labels = batch
        sample_ids_batch = batch_inputs['sample_id'].numpy()
        model_inputs = {k: v for k, v in batch_inputs.items() if k != 'sample_id'}
        batch_pred = model.predict(model_inputs, verbose=0)
        y_true_v.extend(np.argmax(batch_labels, axis=1))
        y_pred_v.extend(np.argmax(batch_pred, axis=1))
        probabilities_v.extend(batch_pred)
        all_sample_ids_v.extend(sample_ids_batch)

        del batch_inputs, batch_labels, batch_pred, model_inputs, sample_ids_batch
        gc.collect()

    save_run_predictions(run + 1, config_name, np.array(probabilities_v), np.array(y_true_v), ck_path, dataset_type='valid')
    run_predictions_list_v.append(np.array(probabilities_v))
    if run_true_labels_v is None:
        run_true_labels_v = np.array(y_true_v)

    if track_misclass in ['both', 'valid']:
        sample_ids_v = np.array(all_sample_ids_v)
        track_misclassifications(np.array(y_true_v), np.array(y_pred_v), sample_ids_v, selected_modalities, misclass_path)

# Continue with common code (metrics calculation, etc.)
# Lines after 1280 remain unchanged
```

---

## Validation

### Test 1: Metadata-Only Mode

```bash
# Set: included_modalities = [('metadata',)]
python src/main.py --mode search --cv_folds 5 --verbosity 2 --resume_mode fresh
```

**Expected Output:**
```
Metadata-only: Using RF predictions directly (no NN training)
Bypassing neural network to preserve RF performance (Kappa ~0.20)
Extracting RF predictions from training data...
Extracting RF predictions from validation data...
RF predictions extracted successfully (no NN training performed)
```

**NO "Epoch 1/300" messages!**

**Expected Performance:**
- Kappa: **0.20 ± 0.05** (up from 0.109)
- Accuracy: ~51-54%
- F1 Macro: ~0.42-0.45

### Test 2: Multi-Modal Mode (Ensure No Regression)

```bash
# Set: included_modalities = [('metadata', 'depth_rgb')]
python src/main.py --mode search --cv_folds 3 --verbosity 2
```

**Expected Output:**
```
Multi-modal: Training fusion network for modality combination
Epoch 1/300 - loss: ... - val_loss: ...
```

**Expected**: Normal multi-modal training works as before

---

## Success Criteria

✅ **Metadata-only mode:**
   - NO neural network training
   - Uses RF predictions directly
   - Kappa ≥ 0.19 (target: 0.20-0.21)

✅ **Multi-modal mode:**
   - Neural network training happens normally
   - Fusion works as designed
   - No regressions in performance

✅ **Code maintains flexibility:**
   - Easy to switch between metadata-only and multi-modal
   - Preserves original fusion architecture for multi-modal cases

---

## Summary

**Problem**: Production pipeline trains unnecessary NN on RF probabilities in metadata-only mode

**Solution**: Bypass NN training, use RF predictions directly when `selected_modalities == ['metadata']`

**Impact**: Kappa 0.109 → 0.20 (+83% improvement)

**Maintains**: Multi-modal fusion capability for when images are included

**If Kappa ≥ 0.19 after this fix** → ✅ **TASK COMPLETE - PRODUCTION READY!**
