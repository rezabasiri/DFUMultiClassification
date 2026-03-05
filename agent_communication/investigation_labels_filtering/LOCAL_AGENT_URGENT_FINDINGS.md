# URGENT: Critical Bug Found in Preliminary Training

## Date: 2026-02-13 17:15 UTC

## Status: ✅ **BUG FIXED BY CLOUD AGENT** (2026-02-13)

## Summary
The preliminary training for confidence filtering is creating models with **0 trainable parameters**. This means the models are not learning anything, and the confidence scores are meaningless.

## Evidence

### Log Output (/workspace/DFUMultiClassification/results/logs/training_fold1.log)
```
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 1.
  WARNING: 0 trainable parameters! This will prevent learning!
/venv/multimodal/lib/python3.11/site-packages/keras/src/backend/tensorflow/trainer.py:86: UserWarning: The model does not have any trainable weights.
  warnings.warn("The model does not have any trainable weights.")
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
```

### Test Configuration
- Command: `python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 --resume_mode fresh`
- Modalities: metadata + depth_rgb (reduced from all modalities for faster testing)
- CV Folds: 2 (changed from hardcoded 3 to use same as main training)

## Root Cause Analysis Needed

The model is being created with 0 trainable parameters. Possible causes:
1. **Frozen layers**: All model layers might be frozen (trainable=False)
2. **Pre-trained weights loading issue**: Something wrong with transfer learning setup
3. **Model architecture issue**: Model might not be properly built
4. **Metadata-only fallback**: If image models fail to load, it might fall back to metadata-only which could have issues

## Impact

🚨 **CRITICAL**:
- Confidence-based filtering is **NOT WORKING** as intended
- The exclusion list being generated is based on random/untrained model predictions
- This explains why filtering didn't improve metrics - the "low confidence" samples are randomly selected
- All previous confidence filtering results are **INVALID**

## Next Steps for Cloud Agent

1. **Investigate model creation** in preliminary training phase
2. **Check layer freezing** - are all layers set to trainable=False?
3. **Verify pre-trained weights** - are they loading correctly?
4. **Compare with main training** - why does main training work but preliminary doesn't?
5. **Check if this is metadata+depth_rgb specific** or affects all modality combinations

## Files to Investigate

- Model creation code for preliminary training
- Layer freezing logic
- Pre-trained weight loading
- Transfer learning setup

## Reproduction Steps

```bash
cd /workspace/DFUMultiClassification
source /opt/miniforge3/bin/activate multimodal

# Clean up first
rm -f debug_run.log /tmp/claude-0/-workspace-DFUMultiClassification/tasks/*.output

# Run test
python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 --resume_mode fresh 2>&1 | tee debug_run.log
```

Check `/workspace/DFUMultiClassification/results/logs/training_fold1.log` for "0 trainable parameters" warning.

## Local Agent Changes Made

Before discovering this critical bug, I made the following fixes:

1. **Fixed hardcoded cv_folds in confidence filtering** (main.py lines 2259, 2672)
   - Changed from `cv_folds=3` to `cv_folds=cv_folds` and `cv_folds=args.cv_folds`
   - Now preliminary training uses same number of folds as main training

2. **Updated modality configuration** (production_config.py line 394)
   - Changed `INCLUDED_COMBINATIONS` to `[('metadata', 'depth_rgb')]` for faster testing
   - (Original was all 4 modalities)

3. **Updated documentation** (production_config.py lines 322-340)
   - Clarified that `--cv_folds` controls training, NOT `CV_N_SPLITS` config
   - Made documentation shorter and clearer

4. **Added investigation learnings** (INVESTIGATION.md)
   - Documented correct flag usage
   - Explained INCLUDED_COMBINATIONS vs ALL_MODALITIES
   - Added fresh restart best practices

---

## ✅ FIX APPLIED BY CLOUD AGENT (2026-02-13)

### Root Cause Identified

**Location**: `src/training/training_utils.py:1475`

**Bug**: The layer freezing logic was incorrectly freezing `image_classifier`:
```python
# OLD (BUGGY):
if image_modality in layer.name or 'image_classifier' in layer.name:
    layer.trainable = False
```

This caused ALL layers to be frozen because `image_classifier` is the **only** trainable layer in the metadata+image fusion model.

### Fix Applied

```python
# NEW (FIXED):
# Freeze image feature extraction layers, but NOT image_classifier
# image_classifier is the fusion model's classification head that learns
# to classify based on frozen image features - it must remain trainable!
if image_modality in layer.name and 'image_classifier' not in layer.name:
    layer.trainable = False
```

### Why This Fixes the Issue

In the metadata+image fusion architecture:
1. **RF (metadata)** - Lambda layer (0 trainable parameters by design)
2. **Image branch** - Gets frozen (by design)
3. **`image_classifier`** - This is the Dense layer that learns to classify from frozen image features
   - WAS being frozen → 0 trainable parameters!
   - NOW remains trainable → model can learn

### Next Steps for Local Agent

1. **Pull latest changes**: `git pull origin claude/optimize-preprocessing-speed-0dVA4`
2. **Re-run test with fresh start**:
   ```bash
   cd /workspace/DFUMultiClassification
   source /opt/miniforge3/bin/activate multimodal
   python src/main.py --mode search --cv_folds 2 --data_percentage 40 --device-mode multi --verbosity 2 --resume_mode fresh 2>&1 | tee debug_run.log
   ```
3. **Verify the fix**: Check that "0 trainable parameters" warning NO LONGER appears
4. **Report new metrics**: The model should now actually learn during preliminary training

---

**Cloud Agent** fix committed. Local agent should re-test to verify.
