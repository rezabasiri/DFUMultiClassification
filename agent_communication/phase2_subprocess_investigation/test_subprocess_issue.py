#!/usr/bin/env python3
"""
Test to identify the subprocess issue in auto_polish_dataset_v2.py

Hypothesis: capture_output=True causes resource exhaustion when training
produces large output, leading to thread creation failures.
"""

import subprocess
import sys
from pathlib import Path

project_root = Path('/workspace/DFUMultiClassification')

# Test command - quick run with reduced data
cmd = [
    sys.executable, 'src/main.py',
    '--mode', 'search',
    '--cv_folds', '2',
    '--data_percentage', '20',  # Use only 20% of data for quick test
    '--verbosity', '0',
    '--resume_mode', 'fresh',
    '--threshold_I', '17',
    '--threshold_P', '18',
    '--threshold_R', '19',
    '--device-mode', 'multi',
    '--min-gpu-memory', '8.0'
]

print("="*80)
print("TEST 1: WITH capture_output=True (current implementation)")
print("="*80)
print("This should FAIL with thread creation error")
print()

result1 = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

if result1.returncode != 0:
    print(f"❌ FAILED with return code: {result1.returncode}")
    print(f"Error snippet: {result1.stderr[-500:]}")
else:
    print("✅ SUCCESS")

print("\n" + "="*80)
print("TEST 2: WITHOUT capture_output (let output stream normally)")
print("="*80)
print("This should SUCCEED")
print()

result2 = subprocess.run(cmd, cwd=project_root)

if result2.returncode != 0:
    print(f"❌ FAILED with return code: {result2.returncode}")
else:
    print("✅ SUCCESS")

print("\n" + "="*80)
print("TEST 3: With PIPE but actively consumed (better approach)")
print("="*80)
print("This should also SUCCEED")
print()

# This approach reads output incrementally instead of buffering it all
result3 = subprocess.Popen(
    cmd,
    cwd=project_root,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1  # Line buffered
)

# Consume output to prevent buffer overflow
for line in result3.stdout:
    pass  # Discard output for this test

result3.wait()

if result3.returncode != 0:
    print(f"❌ FAILED with return code: {result3.returncode}")
else:
    print("✅ SUCCESS")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If TEST 1 fails but TEST 2 or 3 succeeds, the issue is")
print("capture_output=True buffering too much output in memory,")
print("causing resource exhaustion and thread creation failures.")
print("="*80)
