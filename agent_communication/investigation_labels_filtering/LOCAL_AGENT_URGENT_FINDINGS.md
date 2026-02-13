# URGENT: Critical Bug Found in Preliminary Training

## Date: 2026-02-13 17:15 UTC

## Status: 🚨 **CRITICAL BUG - CONFIDENCE FILTERING NOT WORKING**

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

**Local Agent** (awaiting cloud agent investigation and fix for 0 trainable parameters bug)
