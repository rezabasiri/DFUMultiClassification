# Agent Communication Directory

## Purpose
This directory is for communication between the remote Claude agent and local AI agent.

## Current Investigation: Main Training Pipeline Bugs

**Problem**: Training produces catastrophic results regardless of data filtering:
- Min F1 = 0.0 (at least one class gets ZERO predictions)
- Macro F1 = 0.21 (barely better than random)
- Kappa = 0.03 (terrible agreement)

**This indicates the training in `src/main.py` is fundamentally broken.**

## Investigation Plan

Run these scripts in order and report results:

### Phase 1: Data Sanity
```bash
python agent_communication/debug_01_data_sanity.py
```
Creates: `agent_communication/results_01_data_sanity.txt`

### Phase 2: Minimal Training Test
```bash
python agent_communication/debug_02_minimal_training.py
```
Creates: `agent_communication/results_02_minimal_training.txt`

### Phase 3: Main.py Data Loading Test
```bash
python agent_communication/debug_03_main_data_loading.py
```
Creates: `agent_communication/results_03_main_data_loading.txt`

### Phase 4: Single Training Run with Debug
```bash
python agent_communication/debug_04_instrumented_training.py
```
Creates: `agent_communication/results_04_training_debug.txt`

## After Each Phase

1. Run the script
2. Check if output file was created
3. Push to repo so remote agent can analyze
4. Wait for next instructions

## Expected Timeline

- Phase 1: 2 minutes
- Phase 2: 10 minutes
- Phase 3: 5 minutes
- Phase 4: 15 minutes

Total: ~30 minutes to identify root cause
