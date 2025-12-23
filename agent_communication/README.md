# Agent Communication Directory

## ESSENTIAL CONTEXT (READ FIRST - EVERY TIME)

### Environment
- **Virtual Environment**: `/home/rezab/projects/enviroments/multimodal/bin`
- **Project Directory**: `/home/rezab/projects/DFUMultiClassification`
- **Activate Command**: `source /home/rezab/projects/enviroments/multimodal/bin/activate`
- **Git Branch**: `claude/restore-weighted-f1-metrics-5PNy8`
- **Platform**: Linux (Ubuntu/WSL)

### Project Overview
**DFU Multi-Classification**: Diabetic Foot Ulcer healing phase classification using multimodal data (metadata, depth images, thermal images).

**Classes**:
- 0 = I (Inflammation)
- 1 = P (Proliferation)
- 2 = R (Remodeling)

**Dataset**:
- File: `balanced_combined_healing_phases.csv` (in project root)
- ~600 samples across 3 classes
- Multiple modalities: metadata, depth_rgb, depth_map, thermal_map
- Dataset is VERIFIED as correct - the issue is in the code

### Git Operations
- **Pull**: `git pull origin claude/restore-weighted-f1-metrics-5PNy8`
- **Add files**: `git add agent_communication/results_*.txt`
- **Commit**: `git commit -m "Debug Phase X: [description]"`
- **Push**: `git push origin claude/restore-weighted-f1-metrics-5PNy8`
  - Note: Push may fail due to permissions - if so, tell user to push manually

---

## CURRENT PROBLEM: Training Catastrophically Broken

**Symptoms** (observed across ALL training runs):
```
Macro F1: 0.2092
Min F1: 0.0000  ← At least one class gets ZERO correct predictions!
Kappa: 0.0342
Balance: 0.64
```

**What this means**:
- Model is essentially random guessing (barely better than 33% accuracy for 3 classes)
- At least one class NEVER gets predicted correctly (Min F1 = 0)
- This happens REGARDLESS of data filtering thresholds
- This is NOT a data quality issue - it's a **training pipeline bug**

**Location**: The bug is somewhere in `src/main.py` or its dependencies (`src/training/training_utils.py`, `src/data/dataset_utils.py`, etc.)

---

## INVESTIGATION PLAN

### Objective
Systematically test each component of the training pipeline to identify WHERE it breaks.

### Scripts Overview

| Phase | Script | Duration | Tests | Output File |
|-------|--------|----------|-------|-------------|
| 1 | `debug_01_data_sanity.py` | 2 min | Raw data loads correctly | `results_01_data_sanity.txt` |
| 2 | `debug_02_minimal_training.py` | 10 min | Basic TensorFlow works in isolation | `results_02_minimal_training.txt` |
| 3 | `debug_03_main_data_loading.py` | 5 min | `prepare_dataset()` works | `results_03_main_data_loading.txt` |
| 4 | `debug_04_instrumented_training.py` | 15 min | Full training with verbose output | `results_04_training_debug.txt` |

**Total**: ~32 minutes

### Expected Diagnosis Patterns

| Scenario | Meaning |
|----------|---------|
| ✅ Phase 1, ❌ Phase 2 | TensorFlow/environment issue |
| ✅ Phase 2, ❌ Phase 3 | `prepare_dataset()` broken |
| ✅ Phase 3, ❌ Phase 4 | Training pipeline bug in main.py |
| ✅ All pass | Subtle bug - need deeper instrumentation |

---

## HOW TO RUN INVESTIGATION

### Setup (First Time)
```bash
# Activate environment
source /home/rezab/projects/enviroments/multimodal/bin/activate

# Navigate to project
cd /home/rezab/projects/DFUMultiClassification

# Pull latest scripts
git pull origin claude/restore-weighted-f1-metrics-5PNy8

# Make scripts executable
chmod +x agent_communication/debug_*.py
```

### Run Each Phase
```bash
# Phase 1
python agent_communication/debug_01_data_sanity.py

# Phase 2
python agent_communication/debug_02_minimal_training.py

# Phase 3
python agent_communication/debug_03_main_data_loading.py

# Phase 4
python agent_communication/debug_04_instrumented_training.py
```

### After Each Phase

1. **Check output file exists**:
   ```bash
   ls -lh agent_communication/results_0X_*.txt
   ```

2. **Stage and commit**:
   ```bash
   git add agent_communication/results_0X_*.txt
   git commit -m "Debug Phase X: [brief description of results]"
   ```

3. **Try to push** (may fail - that's okay):
   ```bash
   git push origin claude/restore-weighted-f1-metrics-5PNy8
   ```

4. **Report to user**:
   - If push succeeded: "Phase X complete, results pushed. Awaiting analysis."
   - If push failed: "Phase X complete, results ready. Please push manually."

5. **Wait for remote agent's analysis and next instructions**

---

## WORKFLOW EXAMPLE

```bash
# Activate environment (do this ONCE per session)
source /home/rezab/projects/enviroments/multimodal/bin/activate
cd /home/rezab/projects/DFUMultiClassification

# Phase 1
python agent_communication/debug_01_data_sanity.py
git add agent_communication/results_01_data_sanity.txt
git commit -m "Debug Phase 1: Data sanity check complete"
git push origin claude/restore-weighted-f1-metrics-5PNy8
# → Tell user: "Phase 1 complete, results pushed"

# Wait for remote agent to analyze...

# Phase 2 (only run when instructed)
python agent_communication/debug_02_minimal_training.py
git add agent_communication/results_02_minimal_training.txt
git commit -m "Debug Phase 2: Minimal training test complete"
git push origin claude/restore-weighted-f1-metrics-5PNy8
# → Tell user: "Phase 2 complete, results pushed"

# And so on...
```

---

## IMPORTANT NOTES

### For Local Agent
- **Run phases SEQUENTIALLY** - don't skip ahead
- **Always push results** after each phase (even if tests fail)
- **Don't modify the debug scripts** - they're designed to be run as-is
- **Environment must be activated** before running any script
- If a phase **fails**, that's valuable information - push the results anyway

### For User
- Local agent will commit results but may not be able to push
- If push fails, manually run: `git push origin claude/restore-weighted-f1-metrics-5PNy8`
- Remote agent (me) will pull, analyze results, and provide next steps

### For Remote Agent
- After receiving results, pull and analyze
- Identify root cause based on which phase failed
- Either provide fix or request additional instrumentation
- Update this README with new instructions as needed

---

## File Locations

### Input Files (Required)
- `balanced_combined_healing_phases.csv` - Main dataset (project root)
- `data/raw/DataMaster_Processed_V12_WithMissing.csv` - Raw data
- `data/raw/bb_depth_annotation.csv` - Depth bounding boxes
- `data/raw/bb_thermal_annotation.csv` - Thermal bounding boxes

### Output Files (Generated)
- `results_01_data_sanity.txt` - Phase 1 results
- `results_02_minimal_training.txt` - Phase 2 results
- `results_03_main_data_loading.txt` - Phase 3 results
- `results_04_training_debug.txt` - Phase 4 results (large file)

### Code Being Tested
- `src/main.py` - Main training orchestration
- `src/training/training_utils.py` - Training loop implementation
- `src/data/dataset_utils.py` - Data preparation and batching
- `src/data/image_processing.py` - Data loading (`prepare_dataset()`)
- `src/models/losses.py` - Custom metrics and losses

---

## Communication Protocol

### Local Agent → User
After each phase:
```
Phase [N] complete - [PASS/FAIL]
Output file: agent_communication/results_0[N]_*.txt
Commit: [commit hash]
Push: [SUCCESS/FAILED - please push manually]
Status: [Awaiting analysis / Ready for Phase N+1]
```

### User → Remote Agent
Notify when results are pushed (or ready to be pulled)

### Remote Agent → User → Local Agent
After analysis:
- If bug found: Provide fix instructions
- If need more info: Provide additional debug script or instrumentation
- If phase passed: Approve moving to next phase

---

## Version History

- **v1.0** (2025-12-23): Initial investigation framework created
  - 4-phase systematic testing approach
  - Focus on identifying training pipeline bug causing Min F1=0

---

## Quick Reference

**Always remember**:
1. Environment: `/home/rezab/projects/enviroments/multimodal/bin`
2. Project: `/home/rezab/projects/DFUMultiClassification`
3. Branch: `claude/restore-weighted-f1-metrics-5PNy8`
4. Platform: Linux (Ubuntu/WSL)
5. Run phases in order: 1 → 2 → 3 → 4
6. Push results after EACH phase
7. Dataset is correct - bug is in training code
