# Instructions for Local AI Agent

## IMPORTANT CONTEXT

**Environment**: `/Users/rezabasiri/env/multimodal/bin`
**Project**: `/Users/rezabasiri/DFUMultiClassification`

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
source /Users/rezabasiri/env/multimodal/bin/activate

# Go to project
cd /Users/rezabasiri/DFUMultiClassification

# Pull latest scripts
git pull origin claude/restore-weighted-f1-metrics-5PNy8

# Make scripts executable
chmod +x agent_communication/debug_*.py
```

## PHASE 1: Data Sanity Check (2 minutes)

Tests if raw data loads correctly.

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
