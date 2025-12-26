# Phase 2 Subprocess Failure Investigation

## Problem
Phase 2 of auto_polish_dataset_v2.py fails 100% of evaluations with thread creation errors, but the same command runs successfully when executed directly in terminal.

## Key Evidence

### 1. Direct Execution: WORKS
```bash
/venv/multimodal/bin/python src/main.py --mode search --cv_folds 3 \
  --data_percentage 100 --verbosity 0 --resume_mode fresh \
  --threshold_I 17 --threshold_P 18 --threshold_R 19 \
  --device-mode multi --min-gpu-memory 8.0
```
- Starts training normally
- GPU utilization > 0%
- Completes successfully

### 2. Subprocess.run from Python: FAILS/HANGS
```python
subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
```
- Either times out (hangs forever)
- OR returns exit code -6 with thread creation error
- GPU utilization = 0% (never reaches GPU initialization)

### 3. Thread Creation Error
```
Check failed: ret == 0 (11 vs. 0)Thread tf_ creation via pthread_create() failed.
```

## Root Cause Analysis

The subprocess call is **waiting for the child process**, but the child process appears to be:
1. **Deadlocked** waiting for something
2. **Blocked** on I/O that never completes
3. **Hitting resource limits** that don't apply to direct execution

## Critical Observation
- **Before batch size fixes**: Phase 2 had 19/20 failures
- **After batch size fixes**: Same failures, but now we see they're NOT OOM-related
- **GPU utility = 0%**: Failure happens BEFORE GPU work begins
- **Direct execution works**: Problem is specific to subprocess invocation

## Hypothesis: TensorFlow Strategy Scope Issue

When auto_polish_dataset_v2.py runs as a **parent process**, it may be:
1. Already initializing TensorFlow/CUDA in some way
2. Setting environment variables that conflict with child process
3. Holding resources that prevent child from initializing properly

### Evidence
- Parent script imports: `import pandas as pd`, `import numpy as np`
- Parent may be importing TensorFlow indirectly through other imports
- Child process (main.py) tries to initialize TensorFlow with MirroredStrategy
- **Conflict**: Can't initialize distributed TF strategy in subprocess if parent has TF context

## Potential Solutions

### Solution A: Use os.system() or shell=True
```python
import os
os.system(' '.join(cmd))
```
Starts completely independent process, no Python subprocess machinery.

### Solution B: Extract metrics without subprocess
Call the training function directly instead of via subprocess.

### Solution C: Add environment isolation
```python
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = ''  # Clear for parent
# Then child can set it fresh
subprocess.run(cmd, env=child_env, ...)
```

### Solution D: Use multiprocessing.Process instead
```python
from multiprocessing import Process
p = Process(target=run_training, args=(...))
p.start()
p.join()
```

## Recommended Fix
**Try Solution A first** (os.system) - simplest and most likely to work since it's closest to manual execution.

If that doesn't work, need to investigate if parent process (auto_polish_dataset_v2.py) is inadvertently initializing TensorFlow.

## Next Steps for Remote Agent
1. Check if auto_polish_dataset_v2.py imports anything that loads TensorFlow
2. Consider refactoring Phase 2 to not use subprocess at all
3. Or ensure complete environment isolation between parent and child processes
