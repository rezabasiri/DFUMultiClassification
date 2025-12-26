# Phase 2 Investigation Guide for Local AI Agent

**Date**: December 26, 2025
**Issue**: auto_polish_dataset_v2.py Phase 2 failing with 19/20 evaluations crashed
**Environment**: `source /opt/miniforge3/bin/activate multimodal`

---

## Problem Summary

### Current Situation
- **Phase 1**: All individual modality runs succeed ✅
  - metadata: 5/5 runs completed
  - depth_rgb: 5/5 runs completed
  - depth_map: 5/5 runs completed
  - thermal_map: 5/5 runs completed

- **Phase 2**: 19/20 evaluations failed ❌
  - Evaluations 1-12, 14-20: Silent failures (returncode != 0)
  - Evaluation 13: "Succeeded" but has catastrophic metrics
    - All class F1 = 0.0 (I=0.0, P=0.0, R=0.0)
    - Macro F1 = 0.35
    - Accuracy = 0.59
    - **Model is predicting only one class**

### Root Causes Identified

1. **Silent failures**: `subprocess.run()` captures stderr/stdout but never prints them
2. **Memory overload**: 4 image modalities with GLOBAL_BATCH_SIZE=320 is too large
3. **Batch size not adjusted**: Phase 2 uses same batch size as Phase 1, but Phase 1 tests ONE modality at a time

---

## Critical Fix #1: Batch Size Calculation

### Problem
Phase 1 runs with GLOBAL_BATCH_SIZE=320 for **one image modality at a time**.
Phase 2 runs with GLOBAL_BATCH_SIZE=320 for **FOUR image modalities simultaneously**.

**Memory requirement**:
- Phase 1: 320 samples × 1 image modality × 128×128 = ~5.2GB per batch
- Phase 2: 320 samples × 4 image modalities × 128×128 = ~20.8GB per batch (EXCEEDS GPU MEMORY!)

### Solution
**Divide batch size by number of image modalities in Phase 2.**

Formula:
```
phase2_batch_size = phase1_batch_size / num_image_modalities
```

Example:
- Phase 1: GLOBAL_BATCH_SIZE = 320 (1 modality)
- Phase 2 with 4 image modalities: GLOBAL_BATCH_SIZE = 320 / 4 = 80

---

## Fix Implementation

### Step 1: Add Error Logging to Phase 2

**File**: `scripts/auto_polish_dataset_v2.py`
**Location**: Lines 1179-1186 (in `train_with_thresholds` method)

**REPLACE**:
```python
print(f"⏳ Training with cv_folds={self.phase2_cv_folds} (fresh mode)...")
result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

if result.returncode != 0:
    return None
```

**WITH**:
```python
print(f"⏳ Training with cv_folds={self.phase2_cv_folds} (fresh mode)...")
result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

if result.returncode != 0:
    print(f"\n{'='*80}")
    print("❌ TRAINING SUBPROCESS FAILED")
    print(f"{'='*80}")
    print(f"Return code: {result.returncode}")
    print(f"\nCommand that failed:")
    print(' '.join(cmd))

    if result.stderr:
        print(f"\n{'─'*80}")
        print("STDERR (last 3000 chars):")
        print(f"{'─'*80}")
        print(result.stderr[-3000:])

    if result.stdout:
        print(f"\n{'─'*80}")
        print("STDOUT (last 3000 chars):")
        print(f"{'─'*80}")
        print(result.stdout[-3000:])

    print(f"{'='*80}\n")
    return None
```

### Step 2: Add Batch Size Adjustment Logic

**File**: `scripts/auto_polish_dataset_v2.py`
**Location**: Add new method after `__init__` (around line 156)

**ADD THIS METHOD**:
```python
def _calculate_phase2_batch_size(self):
    """
    Calculate appropriate batch size for Phase 2 based on number of image modalities.

    Phase 1 tests ONE modality at a time with GLOBAL_BATCH_SIZE.
    Phase 2 tests MULTIPLE modalities simultaneously, requiring proportional reduction.

    Returns:
        int: Adjusted batch size for Phase 2
    """
    from src.utils.production_config import GLOBAL_BATCH_SIZE

    # Count image modalities (exclude metadata which is small)
    image_modalities = [m for m in self.modalities if m != 'metadata']
    num_image_modalities = len(image_modalities)

    if num_image_modalities == 0:
        # Metadata only - use full batch size
        return GLOBAL_BATCH_SIZE

    # Divide batch size by number of image modalities to maintain similar memory usage
    adjusted_batch_size = max(16, GLOBAL_BATCH_SIZE // num_image_modalities)

    print(f"\n{'='*80}")
    print("BATCH SIZE ADJUSTMENT FOR PHASE 2")
    print(f"{'='*80}")
    print(f"Phase 1 batch size (1 modality at a time): {GLOBAL_BATCH_SIZE}")
    print(f"Phase 2 modalities: {self.modalities}")
    print(f"  Image modalities: {image_modalities} (count: {num_image_modalities})")
    print(f"  Adjusted batch size: {GLOBAL_BATCH_SIZE} / {num_image_modalities} = {adjusted_batch_size}")
    print(f"  Memory reduction: ~{num_image_modalities}x less per batch")
    print(f"{'='*80}\n")

    return adjusted_batch_size
```

### Step 3: Use Adjusted Batch Size in Phase 2

**File**: `scripts/auto_polish_dataset_v2.py`
**Location**: Line 1131-1154 (in `train_with_thresholds` method)

**FIND** the section that modifies production_config.py:
```python
# Temporarily override config
config_path = project_root / 'src' / 'utils' / 'production_config.py'
backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

with open(config_path, 'r') as f:
    original_config = f.read()
with open(backup_path, 'w') as f:
    f.write(original_config)

try:
    import re
    # Build the modality tuple string, e.g., "('metadata', 'depth_rgb')"
    modality_tuple = "(" + ", ".join(f"'{m}'" for m in self.modalities) + ",)"
    modified_config = re.sub(
        r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
        f"INCLUDED_COMBINATIONS = [\n    {modality_tuple},  # Temporary: Phase 2 evaluation\n]",
        original_config
    )
    with open(config_path, 'w') as f:
        f.write(modified_config)
```

**REPLACE WITH**:
```python
# Temporarily override config
config_path = project_root / 'src' / 'utils' / 'production_config.py'
backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

with open(config_path, 'r') as f:
    original_config = f.read()
with open(backup_path, 'w') as f:
    f.write(original_config)

try:
    import re

    # Calculate adjusted batch size for Phase 2
    adjusted_batch_size = self._calculate_phase2_batch_size()

    # Build the modality tuple string, e.g., "('metadata', 'depth_rgb')"
    modality_tuple = "(" + ", ".join(f"'{m}'" for m in self.modalities) + ",)"

    # Modify INCLUDED_COMBINATIONS
    modified_config = re.sub(
        r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
        f"INCLUDED_COMBINATIONS = [\n    {modality_tuple},  # Temporary: Phase 2 evaluation\n]",
        original_config
    )

    # Modify GLOBAL_BATCH_SIZE
    modified_config = re.sub(
        r'GLOBAL_BATCH_SIZE\s*=\s*\d+',
        f"GLOBAL_BATCH_SIZE = {adjusted_batch_size}",
        modified_config
    )

    with open(config_path, 'w') as f:
        f.write(modified_config)
```

---

## Testing Procedure

### Test 1: Metadata Only (Baseline)

This should work since metadata is small. If this fails, there's a deeper issue.

```bash
# Activate environment
source /opt/miniforge3/bin/activate multimodal

# Navigate to project
cd /workspace/DFUMultiClassification

# Run Phase 2 only with metadata (using existing Phase 1 data)
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata \
  --phase2-only \
  --n-evaluations 5 \
  --device-mode multi \
  2>&1 | tee results/misclassifications_saved/phase2_test_metadata_only.log
```

**Expected**: All 5 evaluations should succeed
**If fails**: Deeper issue with subprocess execution or multi-GPU

### Test 2: Two Image Modalities

```bash
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata+depth_rgb \
  --phase2-only \
  --n-evaluations 5 \
  --device-mode multi \
  2>&1 | tee results/misclassifications_saved/phase2_test_two_modalities.log
```

**Expected**: Should succeed with adjusted batch size (320 / 1 = 320 for 1 image modality)
**If fails**: Check error logs added in Fix #1

### Test 3: Four Image Modalities (Original Problem)

```bash
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata+depth_rgb+depth_map+thermal_map \
  --phase2-only \
  --n-evaluations 5 \
  --device-mode multi \
  2>&1 | tee results/misclassifications_saved/phase2_test_four_modalities.log
```

**Expected**: Should succeed with adjusted batch size (320 / 3 = 106 for 3 image modalities)
**If fails**: May need further batch size reduction or single GPU mode

### Test 4: Full Run with Fixes

```bash
# Run complete Phase 1 + Phase 2 with fixes
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata+depth_rgb+depth_map+thermal_map \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase1-n-runs 5 \
  --n-evaluations 20 \
  --device-mode multi \
  2>&1 | tee results/misclassifications_saved/phase_full_run_with_fixes.log
```

---

## Monitoring During Tests

### GPU Memory Usage

In a separate terminal:
```bash
watch -n 1 nvidia-smi
```

**Look for**:
- Both GPUs should show usage during training
- Memory should not exceed 30GB per GPU
- If you see "out of memory" errors, reduce batch size further

### Real-time Log Monitoring

The `tee` command in the test commands saves output to both terminal and file.

**To view logs later**:
```bash
cd results/misclassifications_saved
ls -lht *.log  # List log files by modification time
tail -n 100 phase2_test_metadata_only.log  # View last 100 lines
grep "❌" *.log  # Find all failures
grep "STDERR" *.log  # Find error messages
```

---

## Expected Outcomes

### If Batch Size Fix Works

You should see:
```
BATCH SIZE ADJUSTMENT FOR PHASE 2
================================================================================
Phase 1 batch size (1 modality at a time): 320
Phase 2 modalities: ['metadata', 'depth_rgb', 'depth_map', 'thermal_map']
  Image modalities: ['depth_rgb', 'depth_map', 'thermal_map'] (count: 3)
  Adjusted batch size: 320 / 3 = 106
  Memory reduction: ~3x less per batch
================================================================================

EVALUATION 1/5
──────────────────────────────────────────────────────────────────────
⏳ Training with cv_folds=3 (fresh mode)...
✅ Metrics extracted
```

### If Still Failing After Batch Size Fix

The error logging will show one of:
1. **OOM (Out of Memory)**: Reduce batch size further (try 64 or 32)
2. **Multi-GPU specific error**: Test with `--device-mode single`
3. **Data loading error**: Check that thresholds aren't filtering too much data
4. **Model creation error**: Issue with 4-modality model architecture

---

## Alternative Fixes if Batch Size Doesn't Solve It

### Option A: Force Single GPU for Phase 2

Multi-GPU may have bugs with 4 modalities. Try:
```bash
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata+depth_rgb+depth_map+thermal_map \
  --phase2-only \
  --n-evaluations 5 \
  --device-mode single \
  2>&1 | tee results/misclassifications_saved/phase2_single_gpu.log
```

### Option B: Reduce Image Resolution for Phase 2

If memory is still an issue, temporarily reduce image size.

**Add to the config override section** (after GLOBAL_BATCH_SIZE modification):
```python
# Modify IMAGE_SIZE for Phase 2 memory reduction
modified_config = re.sub(
    r'IMAGE_SIZE\s*=\s*\d+',
    f"IMAGE_SIZE = 64",  # Reduce from 128 to 64
    modified_config
)
```

### Option C: Test Phase 2 with Single Modality

Phase 2 optimization should work with just metadata since misclassification counts come from Phase 1 individual modalities anyway.

```bash
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata \
  --phase2-only \
  --n-evaluations 20 \
  --device-mode multi \
  2>&1 | tee results/misclassifications_saved/phase2_metadata_final.log
```

**Rationale**: The goal is to find thresholds that filter out frequently misclassified samples. The misclassification counts were generated from individual modality runs in Phase 1, so optimizing on metadata alone should still find valid thresholds.

---

## What to Report Back

After running tests, push the following to the repo:

1. **Log files**: All `*.log` files in `results/misclassifications_saved/`
2. **Modified code**: Updated `scripts/auto_polish_dataset_v2.py` with fixes
3. **Results JSON**: Any new `bayesian_optimization_results.json` files
4. **Summary**: Brief text file with:
   - Which tests passed/failed
   - Any error messages observed
   - Final recommended configuration

---

## Quick Reference Commands

```bash
# Activate environment
source /opt/miniforge3/bin/activate multimodal

# Navigate to project
cd /workspace/DFUMultiClassification

# Quick diagnostic: Check current batch size
python -c "from src.utils.production_config import GLOBAL_BATCH_SIZE, IMAGE_SIZE; print(f'Batch: {GLOBAL_BATCH_SIZE}, Image: {IMAGE_SIZE}')"

# Check GPU memory
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# View recent logs
ls -lht results/misclassifications_saved/*.log | head -5

# Search for errors in logs
grep -i "error\|fail\|exception" results/misclassifications_saved/*.log
```

---

## Debugging Checklist

- [ ] Implemented Fix #1 (error logging)
- [ ] Implemented Fix #2 (batch size calculation method)
- [ ] Implemented Fix #3 (batch size override in config)
- [ ] Ran Test 1 (metadata only) - Result: ______
- [ ] Ran Test 2 (two modalities) - Result: ______
- [ ] Ran Test 3 (four modalities) - Result: ______
- [ ] Checked nvidia-smi during training
- [ ] Saved all log files to results/misclassifications_saved/
- [ ] Reviewed error messages (if any)
- [ ] Determined root cause: ______
- [ ] Pushed logs and results back to repo

---

**End of Investigation Guide**
