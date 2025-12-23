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

## PROGRESS UPDATE

✅ **Phase 1**: Data loads correctly
✅ **Phase 2**: Training works, but predicts only P (majority) → Min F1=0.0
✅ **Phase 5**: Focal loss math correct, alpha weights work
✅ **Phase 6**: Focal loss + alpha ALONE insufficient → Min F1=0.0
✅ **Phase 7**: Oversampling enabled, but DOUBLE-CORRECTION bug → predicts only R (minority) → Min F1=0.0

## 🎯 ROOT CAUSE #2: DOUBLE-CORRECTION BUG

**Phase 7 revealed**: Oversampling worked, but created **over-correction**!
- Before: Predicts 100% class P (majority)
- After: Predicts 100% class R (minority) ← Over-corrected!

**File**: `src/data/dataset_utils.py:676`
**Bug**: Alpha values calculated from ORIGINAL distribution, but applied to BALANCED data after oversampling

**The Problem**:
1. Data-level: Oversampling balances classes (all → 1504 samples)
2. Loss-level: Alpha still uses original imbalance weights [I=0.725, P=0.344, R=1.931]
3. Result: Class R has same samples AS P, but 5.6x higher loss weight → over-prioritized

**Fix Applied**: Recalculate alpha from BALANCED distribution after oversampling (gives ~[1, 1, 1])

**Next Action**: Test if double-correction fix works

---

## PHASE 8: Test Balanced Alpha Fix (5 minutes)

**VERIFICATION TEST**: Confirm oversampling + balanced alpha solves Min F1=0.

```bash
python agent_communication/debug_08_test_balanced_alpha.py

# Commit results
git add agent_communication/results_08_balanced_alpha.txt
git commit -m "Debug Phase 8: Test balanced alpha fix (no double-correction)"
```

**What to expect**:
- Training data balanced: ~1504 samples per class
- Alpha values balanced: ~[1.000, 1.000, 1.000]
- Model predicts ALL 3 classes (not just one)
- Min F1 > 0.0 (SUCCESS!)

**If PASS**: Double-correction fixed, ready to test main.py
**If FAIL**: Need to investigate normalization/architecture

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
