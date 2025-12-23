# Instructions for Local AI Agent

## IMPORTANT CONTEXT

**Environment**: `/home/rezab/projects/enviroments/multimodal/bin`
**Project**: `/home/rezab/projects/DFUMultiClassification`
**Platform**: Linux (Ubuntu/WSL)

## THE REAL PROBLEM

You fixed bugs in `auto_polish_dataset_v2.py` - **that's good, but not the main issue**.

The **core problem** is that **training in main.py produces terrible results**:
```
Macro F1: 0.2092
Min F1: 0.0000  ← At least one class gets ZERO predictions!
Kappa: 0.0342
```

This is catastrophic. The model is NOT learning. No amount of data filtering can fix broken training.

## YOUR TASK

Run 4 debug scripts in order. Each takes 2-15 minutes. After EACH script:
1. Check if output file was created
2. Push to repo (so remote agent can analyze)
3. Wait for next instructions

## SETUP

```bash
# Activate environment
source /home/rezab/projects/enviroments/multimodal/bin/activate

# Go to project
cd /home/rezab/projects/DFUMultiClassification

# Pull latest scripts (IMPORTANT: Data loading was fixed!)
git pull origin claude/restore-weighted-f1-metrics-5PNy8

# Make scripts executable
chmod +x agent_communication/debug_*.py
```

## CONTEXT REFRESH (for your memory)

**Project**: DFU (Diabetic Foot Ulcer) multimodal classification
**Platform**: Linux (Ubuntu/WSL)
**Environment**: `/home/rezab/projects/enviroments/multimodal/bin`
**Project Dir**: `/home/rezab/projects/DFUMultiClassification`
**Goal**: Fix catastrophic training failure (Min F1=0.0, model only predicts one class)

**Dataset**:
- 3107 samples, 3 classes: I=Inflammation (28.7%), P=Proliferation (60.5%), R=Remodeling (10.8%)
- SEVERE imbalance: P:R = 5.6:1 ratio
- Target metric: Weighted F1 score with inverse frequency alpha

---

## 🎉 BREAKTHROUGH: Phase 9 SUCCESS!

✅ **All 9 debugging phases complete**
✅ **Root causes identified and FIXED**
✅ **Phase 9: 97.6% accuracy, Min F1=0.964** (metadata only)

## Complete Solution Implemented

**Three critical fixes applied to src/data/dataset_utils.py:**

1. **Enable oversampling** (line 725): `apply_sampling=True`
   - Balances training data (all classes → 1504 samples)

2. **Fix double-correction** (lines 676-687): Recalculate alpha after oversampling
   - Returns [1.0, 1.0, 1.0] instead of imbalanced [0.725, 0.344, 1.931]

3. **Feature normalization** (lines 965-995): StandardScaler for metadata
   - **CRITICAL** - Without this, model can't learn at all
   - Phase 8: 33.3% acc (stuck at init) → Phase 9: 97.6% acc (learning!)

## ⚠️  IMPORTANT WARNING

**Phase 9 only tested METADATA** (61 tabular features) - the easiest modality!

The 97.6% accuracy is suspiciously good. Need to verify:
- Does fix work with IMAGE modalities (depth_rgb, depth_map, thermal_map)?
- Are there data leaks (train/val overlap)?
- Are there model leaks (weights carrying over between folds)?

**Next step**: Run comprehensive CV test to verify fix generalizes

---

## COMPREHENSIVE CV TEST (30 minutes)

**Verify Phase 9 fix works across all modalities with proper leak detection**

```bash
cd /home/rezab/projects/DFUMultiClassification
source /home/rezab/projects/enviroments/multimodal/bin/activate
git pull origin claude/restore-weighted-f1-metrics-5PNy8

# Run comprehensive CV test (all modalities + combinations, 3-fold CV)
python agent_communication/test_comprehensive_cv.py

# Commit results
git add agent_communication/results_comprehensive_cv_test.txt agent_communication/results_comprehensive_cv_test.json
git commit -m "Comprehensive CV test results: All modalities with leak detection"
```

**What this tests**:
- All modalities: metadata, depth_rgb, depth_map, thermal_map
- Key combinations: metadata+depth_rgb, metadata+depth_map, metadata+thermal_map
- 3-fold cross-validation (proper evaluation)
- Data leak detection (ensures train/val don't overlap)
- Model leak detection (ensures model reset between folds)
- Reduced parameters for speed (image_size=32, epochs=10)

**Expected outcomes**:
- **If all pass**: Phase 9 fix generalizes! Solution complete ✅
- **If metadata passes, images fail**: Fix works for tabular, need image-specific solution
- **If leaks detected**: Need to fix data/model handling

Then notify: "Comprehensive CV test complete - awaiting manual push"

---

**What this tests**:
- StandardScaler (mean=0, std=1 for all features)
- Plain cross-entropy (no focal loss - unnecessary with balanced data)
- Oversampling (balanced training data)

**Expected**:
- Training loss DECREASES (not plateau at 10.8)
- Training accuracy INCREASES above 33.3%
- Model predicts all 3 classes
- Min F1 > 0.3 (actual learning!)

**If PASS**: Complete solution found! Apply to main.py
**If FAIL**: May need architecture changes

---

## PHASE 1: Data Sanity Check (2 minutes)

Tests if raw data loads correctly via `prepare_dataset()` function.

```bash
python agent_communication/debug_01_data_sanity.py
```

**Expected**: Creates `agent_communication/results_01_data_sanity.txt`

**After completion**:
```bash
# Check file exists
ls -lh agent_communication/results_01_data_sanity.txt

# Push result
git add agent_communication/results_01_data_sanity.txt
git commit -m "Debug Phase 1: Data sanity check results"
git push origin claude/restore-weighted-f1-metrics-5PNy8
```

**Then**: Tell user "Phase 1 complete, results pushed. Awaiting analysis."

---

## PHASE 2: Minimal Training Test (10 minutes)

Tests if basic training works WITHOUT using any main.py code.

```bash
python agent_communication/debug_02_minimal_training.py
```

**Expected**: Creates `agent_communication/results_02_minimal_training.txt`

**What this tests**: If this PASSES but main.py FAILS, the bug is in main.py's training pipeline.

**After completion**:
```bash
git add agent_communication/results_02_minimal_training.txt
git commit -m "Debug Phase 2: Minimal training test results"
git push origin claude/restore-weighted-f1-metrics-5PNy8
```

**Then**: Tell user "Phase 2 complete, results pushed. Awaiting analysis."

---

## PHASE 3: Main.py Data Loading (5 minutes)

Tests if main.py's `prepare_dataset()` function loads data correctly.

```bash
python agent_communication/debug_03_main_data_loading.py
```

**Expected**: Creates `agent_communication/results_03_main_data_loading.txt`

**After completion**:
```bash
git add agent_communication/results_03_main_data_loading.txt
git commit -m "Debug Phase 3: Main.py data loading test results"
git push origin claude/restore-weighted-f1-metrics-5PNy8
```

**Then**: Tell user "Phase 3 complete, results pushed. Awaiting analysis."

---

## PHASE 4: Instrumented Training (10-15 minutes)

Runs one full training iteration with maximum verbosity to capture all debug info.

```bash
python agent_communication/debug_04_instrumented_training.py
```

**Expected**: Creates `agent_communication/results_04_training_debug.txt`
**Warning**: This file will be LARGE (thousands of lines).

**After completion**:
```bash
git add agent_communication/results_04_training_debug.txt
git commit -m "Debug Phase 4: Instrumented training output"
git push origin claude/restore-weighted-f1-metrics-5PNy8
```

**Then**: Tell user "Phase 4 complete, results pushed. This is the full training output for analysis."

---

## IF ANY SCRIPT FAILS

1. Check the output file (it will contain error messages)
2. Push it anyway:
   ```bash
   git add agent_communication/results_*.txt
   git commit -m "Debug failed at phase X - see results file"
   git push origin claude/restore-weighted-f1-metrics-5PNy8
   ```
3. Tell user: "Phase X failed - results pushed for analysis"

## TIMELINE

- Phase 1: 2 min
- Phase 2: 10 min
- Phase 3: 5 min
- Phase 4: 15 min
- **Total**: ~32 minutes

## IMPORTANT

- Run scripts **in order** (1 → 2 → 3 → 4)
- **Push results after EACH phase**
- **Don't skip ahead** even if early phases pass
- The remote agent needs to see ALL results to diagnose the issue

## WHAT WE'RE LOOKING FOR

The scripts will reveal WHERE the training breaks:
- ✅ Phase 1 PASS, ❌ Phase 2 FAIL → Basic data/TensorFlow issue
- ✅ Phase 2 PASS, ❌ Phase 3 FAIL → prepare_dataset() broken
- ✅ Phase 3 PASS, ❌ Phase 4 FAIL → Training pipeline bug in main.py
- ✅ All PASS → Need to add more instrumentation to find subtle bug

Start with Phase 1 now!
