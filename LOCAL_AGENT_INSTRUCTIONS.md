# Instructions for Local AI Agent

**Date**: December 26, 2025
**Task**: Fix Phase 2 failures in auto_polish_dataset_v2.py
**Environment**: `source /opt/miniforge3/bin/activate multimodal`

---

## Quick Start (TL;DR)

```bash
# 1. Activate environment
source /opt/miniforge3/bin/activate multimodal
cd /workspace/DFUMultiClassification

# 2. Apply code fixes from PHASE2_CODE_FIXES.py to scripts/auto_polish_dataset_v2.py
#    (See detailed instructions in PHASE2_CODE_FIXES.py)

# 3. Make test script executable
chmod +x PHASE2_QUICK_TEST.sh

# 4. Run automated tests
./PHASE2_QUICK_TEST.sh

# 5. Push results
git add results/misclassifications_saved/*.log
git add scripts/auto_polish_dataset_v2.py  # if you modified it
git commit -m "Phase 2 fixes and test results"
git push
```

---

## Problem Summary

**What's failing**: Phase 2 of auto_polish_dataset_v2.py fails 19 out of 20 evaluations

**Root cause**: Batch size not adjusted for multiple image modalities
- Phase 1: Tests ONE modality at a time with GLOBAL_BATCH_SIZE=320
- Phase 2: Tests FOUR modalities simultaneously with GLOBAL_BATCH_SIZE=320
- Result: 4x memory usage → GPU OOM or crashes

**Solution**: Divide batch size by number of image modalities in Phase 2

---

## Files Created for You

1. **PHASE2_INVESTIGATION_GUIDE.md** - Comprehensive analysis and testing procedures
2. **PHASE2_CODE_FIXES.py** - Exact code changes to apply (3 fixes)
3. **PHASE2_QUICK_TEST.sh** - Automated test script
4. **LOCAL_AGENT_INSTRUCTIONS.md** - This file

---

## Step-by-Step Instructions

### Step 1: Read the Investigation Guide

```bash
cat PHASE2_INVESTIGATION_GUIDE.md
```

This explains:
- Why Phase 2 is failing
- What the fixes do
- Expected outcomes

### Step 2: Apply Code Fixes

Open `scripts/auto_polish_dataset_v2.py` and apply the 3 fixes from `PHASE2_CODE_FIXES.py`:

**Fix #1**: Add `_calculate_phase2_batch_size()` method after `__init__`
- Location: Line ~156
- What it does: Calculates adjusted batch size = GLOBAL_BATCH_SIZE / num_image_modalities

**Fix #2**: Use adjusted batch size in config override
- Location: Line ~1145 in `train_with_thresholds()` method
- What it does: Modifies GLOBAL_BATCH_SIZE in production_config.py during Phase 2

**Fix #3**: Add error logging to subprocess failures
- Location: Line ~1180 in `train_with_thresholds()` method
- What it does: Prints stderr/stdout when training subprocess fails

To view the exact code:
```bash
python PHASE2_CODE_FIXES.py
```

### Step 3: Verify Fixes Are Applied

```bash
# Check for Fix #1
grep "_calculate_phase2_batch_size" scripts/auto_polish_dataset_v2.py

# Check for Fix #2
grep "adjusted_batch_size = self._calculate_phase2_batch_size()" scripts/auto_polish_dataset_v2.py

# Check for Fix #3
grep "TRAINING SUBPROCESS FAILED" scripts/auto_polish_dataset_v2.py
```

All three should return matches.

### Step 4: Run Automated Tests

```bash
chmod +x PHASE2_QUICK_TEST.sh
./PHASE2_QUICK_TEST.sh
```

This will run 3 tests:
1. **Test 1**: Metadata only (baseline, should always work)
2. **Test 2**: Metadata + depth_rgb (tests batch size adjustment with 1 image modality)
3. **Test 3**: Metadata + 3 image modalities (original failing case)

Each test runs 3 evaluations and saves logs to `results/misclassifications_saved/`

**Expected runtime**: ~15-30 minutes total

### Step 5: Review Results

```bash
# View all test logs
ls -lh results/misclassifications_saved/test*.log

# Check for failures
grep -i "error\|fail\|❌" results/misclassifications_saved/test*.log

# Check for successes
grep -i "✅" results/misclassifications_saved/test*.log

# View specific log
cat results/misclassifications_saved/test3_four_modalities.log
```

**What to look for**:
- "BATCH SIZE ADJUSTMENT FOR PHASE 2" messages showing adjusted batch sizes
- Evaluations completing without "❌ Training failed"
- Valid metrics (not all F1=0.0)

### Step 6: Run Full Optimization (If Tests Pass)

```bash
# Full 20-evaluation run with fixes
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata+depth_rgb+depth_map+thermal_map \
  --phase2-only \
  --n-evaluations 20 \
  --device-mode multi \
  2>&1 | tee results/misclassifications_saved/phase2_full_run_fixed.log
```

**Expected runtime**: ~2-3 hours

### Step 7: Push Results to Repo

```bash
# Add log files
git add results/misclassifications_saved/*.log

# Add modified code (if you changed it)
git add scripts/auto_polish_dataset_v2.py

# Add any new results JSON
git add results/bayesian_optimization_results.json

# Commit
git commit -m "Phase 2 fixes applied and tested

- Added batch size adjustment for multi-modality Phase 2
- Added error logging for subprocess failures
- Test results: [summarize here]
- Full optimization: [success/failure]"

# Push
git push origin claude/restore-weighted-f1-metrics-5PNy8
```

---

## Expected Batch Size Adjustments

With GLOBAL_BATCH_SIZE = 320 in production_config.py:

| Phase 2 Modalities | Image Modalities | Calculation | Adjusted Batch Size |
|-------------------|------------------|-------------|---------------------|
| metadata | 0 | 320 (no reduction) | 320 |
| metadata+depth_rgb | 1 | 320 / 1 | 320 |
| metadata+depth_rgb+depth_map | 2 | 320 / 2 | 160 |
| metadata+depth_rgb+depth_map+thermal_map | 3 | 320 / 3 | 106 |
| All 5 modalities | 4 | 320 / 4 | 80 |

**Memory per batch** (approximate):
- 1 image modality @ batch 320: ~5.2 GB
- 3 image modalities @ batch 106: ~5.5 GB (similar!)
- 4 image modalities @ batch 80: ~4.2 GB (reduced)

---

## Troubleshooting

### If Test 1 (metadata only) fails:
- Deeper issue not related to batch size
- Check error logs for actual cause
- Try with `--device-mode single` to isolate multi-GPU issues

### If Test 2 passes but Test 3 fails:
- Reduce batch size further (edit `_calculate_phase2_batch_size` to use `// (num_image_modalities + 1)`)
- Or try `--device-mode single` for Phase 2

### If you see "out of memory" errors:
- Reduce IMAGE_SIZE to 64 temporarily
- Or manually set adjusted_batch_size to 32 or 16

### If subprocess still fails silently:
- Error logging (Fix #3) should now print stderr/stdout
- Check the printed error messages for root cause
- Common issues:
  - Data loading errors
  - Model architecture incompatibility
  - Multi-GPU bugs

---

## What to Report Back

Create a summary file `results/misclassifications_saved/PHASE2_TEST_SUMMARY.txt`:

```
Phase 2 Fix Testing Summary
Date: [date]
Environment: multimodal

Fixes Applied:
[ ] Fix #1: Batch size calculator
[ ] Fix #2: Config override
[ ] Fix #3: Error logging

Test Results:
Test 1 (metadata only): PASS/FAIL - [details]
Test 2 (2 modalities): PASS/FAIL - [details]
Test 3 (4 modalities): PASS/FAIL - [details]

Batch Size Adjustments Observed:
[paste the BATCH SIZE ADJUSTMENT output from logs]

Final Recommendation:
[What configuration works best]

Issues Encountered:
[Any problems or errors]
```

Then push everything:
```bash
git add results/misclassifications_saved/
git commit -m "Phase 2 test results and summary"
git push
```

---

## Alternative Approaches (If Fixes Don't Work)

### Approach A: Use Metadata Only for Phase 2

The misclassification counts come from Phase 1 individual modalities, so optimizing thresholds on metadata alone is valid:

```bash
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata \
  --phase2-only \
  --n-evaluations 20 \
  --device-mode multi
```

### Approach B: Single GPU for Phase 2

Multi-GPU might have compatibility issues with 4-modality training:

```bash
python scripts/auto_polish_dataset_v2.py \
  --phase2-modalities metadata+depth_rgb+depth_map+thermal_map \
  --phase2-only \
  --n-evaluations 20 \
  --device-mode single
```

### Approach C: Reduce Image Resolution

Temporarily use 64×64 images for Phase 2 optimization:
- Modify `_calculate_phase2_batch_size()` to also override IMAGE_SIZE
- Set IMAGE_SIZE = 64 in the config override

---

## Quick Reference Commands

```bash
# Activate environment
source /opt/miniforge3/bin/activate multimodal

# Check GPU memory
nvidia-smi

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# View recent log
tail -f results/misclassifications_saved/test3_four_modalities.log

# Search for errors
grep -i "error\|exception\|fail" results/misclassifications_saved/*.log

# Count successful evaluations
grep "✅ Metrics extracted" results/misclassifications_saved/*.log | wc -l

# View batch size adjustments
grep "BATCH SIZE ADJUSTMENT" results/misclassifications_saved/*.log
```

---

## Success Criteria

Phase 2 is fixed when:
- [ ] All 3 quick tests pass (Tests 1-3)
- [ ] No "❌ Training failed" messages
- [ ] Metrics are valid (not all F1=0.0)
- [ ] At least 15/20 evaluations succeed in full run
- [ ] Best score is reasonable (not -10.0)
- [ ] Log files show proper batch size adjustments

---

**End of Instructions**
