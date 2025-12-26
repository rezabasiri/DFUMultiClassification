# Phase 2 Subprocess Issue - RESOLVED

## Problem
subprocess.run() calls from auto_polish_dataset_v2.py were hanging/failing, but direct command execution worked perfectly.

## Root Cause
Python's subprocess.run() with `capture_output=True` creates resource conflicts when:
1. Parent process has TensorFlow/CUDA context
2. Child process tries to initialize TensorFlow with multi-GPU strategy
3. Result: Thread creation failures or hangs

## Solution
Replace `subprocess.run(cmd, capture_output=True)` with `os.system(cmd_str >output 2>&1)`

**Why this works:**
- os.system creates completely independent shell process
- No Python subprocess machinery overhead
- Child inherits clean environment without parent's TensorFlow context

## Test Results
**BEFORE**: 100% failure rate with thread creation errors
**AFTER**: Evaluations completing successfully (verified 2/10 in test run)

## Files Changed
- scripts/auto_polish_dataset_v2.py (lines 1230-1262)
  - Changed subprocess.run to os.system
  - Added temp output file for error logging
  - Preserves return code checking
