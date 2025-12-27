#!/usr/bin/env python3
"""
Test script to verify that main.py and auto_polish_dataset_v2.py produce matching results.
Uses reduced parameters for quick testing.
"""

import subprocess
import os
import sys
import pandas as pd
import shutil
from pathlib import Path

# Test parameters (reduced for speed)
DATA_PCT = 100  # Use 100% to ensure same data
CV_FOLDS = 2
N_RUNS = 1
DEVICE_MODE = "single"
MODALITY = "metadata"

print("=" * 60)
print("RESULTS MATCHING TEST")
print("=" * 60)
print()
print(f"Test Parameters:")
print(f"  Data: {DATA_PCT}%")
print(f"  CV Folds: {CV_FOLDS}")
print(f"  Runs: {N_RUNS}")
print(f"  Device: {DEVICE_MODE}")
print(f"  Modality: {MODALITY}")
print()

# Paths
root_dir = Path("/workspace/DFUMultiClassification")
test_dir = root_dir / "agent_communication" / "results_matching_investigation"
csv_dir = root_dir / "results" / "csv"

def clean_results():
    """Clean up previous test results"""
    print("Cleaning up previous test results...")
    patterns = [
        csv_dir / "modality_results_run_*.csv",
        csv_dir / "modality_combination_results.csv",
        csv_dir / "modality_results_averaged.csv"
    ]
    for pattern in patterns:
        for f in csv_dir.glob(pattern.name):
            f.unlink(missing_ok=True)

def count_run_files():
    """Count number of run CSV files (excluding _list files)"""
    all_files = list(csv_dir.glob("modality_results_run_*.csv"))
    # Exclude _list.csv files
    non_list_files = [f for f in all_files if not f.name.endswith("_list.csv")]
    return len(non_list_files)

def save_results(prefix):
    """Save results with given prefix"""
    files_saved = []

    # Save combination results
    src = csv_dir / "modality_combination_results.csv"
    if src.exists():
        dst = test_dir / f"{prefix}_combination_results.csv"
        shutil.copy2(src, dst)
        files_saved.append(dst.name)

    # Save averaged results
    src = csv_dir / "modality_results_averaged.csv"
    if src.exists():
        dst = test_dir / f"{prefix}_averaged.csv"
        shutil.copy2(src, dst)
        files_saved.append(dst.name)

    # Save all run files
    for src in csv_dir.glob("modality_results_run_*.csv"):
        dst = test_dir / f"{prefix}_{src.name}"
        shutil.copy2(src, dst)
        files_saved.append(dst.name)

    return files_saved

def run_main_py():
    """Run src/main.py"""
    print("=" * 60)
    print("TEST 1: Running src/main.py")
    print("=" * 60)
    print()

    # Temporarily modify production_config.py to test our modality
    config_file = root_dir / "src" / "utils" / "production_config.py"
    config_backup = config_file.read_text()

    try:
        # Update INCLUDED_COMBINATIONS to use our modality
        new_config = config_backup.replace(
            "INCLUDED_COMBINATIONS = [\n    ('thermal_map',),  # Temporary: Phase 2 evaluation\n]",
            f"INCLUDED_COMBINATIONS = [\n    ('{MODALITY}',),  # Temporary: Test\n]"
        )
        config_file.write_text(new_config)

        cmd = [
        sys.executable, str(root_dir / "src" / "main.py"),
        "--mode", "search",
        "--data_percentage", str(DATA_PCT),
        "--cv_folds", str(CV_FOLDS),
        "--resume_mode", "fresh",
        "--verbosity", "1",
        "--device-mode", DEVICE_MODE
        # NOTE: Don't pass thresholds - we want NO filtering (full dataset)
    ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=root_dir)

        # Save output
        with open(test_dir / "main_py_output.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode != 0:
            print(f"❌ main.py failed with return code {result.returncode}")
            print(result.stderr[-500:])
            return False

        print(f"✅ main.py completed successfully")
        num_runs = count_run_files()
        print(f"Number of run files created: {num_runs}")

        files = save_results("main_py")
        print(f"Saved {len(files)} result files")
        print()

        return True
    finally:
        # Restore original config
        config_file.write_text(config_backup)

def run_auto_polish():
    """Run auto_polish_dataset_v2.py"""
    print("=" * 60)
    print("TEST 2: Running auto_polish_dataset_v2.py")
    print("=" * 60)
    print()

    cmd = [
        sys.executable, str(root_dir / "scripts" / "auto_polish_dataset_v2.py"),
        "--phase1-modalities", MODALITY,
        "--phase1-n-runs", str(N_RUNS),
        "--phase1-cv-folds", str(CV_FOLDS),
        "--device-mode", DEVICE_MODE,
        "--phase1-only",
        "--phase1-data-percentage", str(DATA_PCT),
        "--phase2-modalities", MODALITY
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root_dir)

    # Save output
    with open(test_dir / "auto_polish_output.log", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"❌ auto_polish failed with return code {result.returncode}")
        print(result.stderr[-500:])
        return False

    print(f"✅ auto_polish completed successfully")
    num_runs = count_run_files()
    print(f"Number of run files created: {num_runs}")

    files = save_results("auto_polish")
    print(f"Saved {len(files)} result files")
    print()

    return True

def compare_results():
    """Compare the averaged results"""
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print()

    main_file = test_dir / "main_py_averaged.csv"
    auto_file = test_dir / "auto_polish_averaged.csv"

    if not main_file.exists() or not auto_file.exists():
        print("❌ Could not find averaged results files")
        return False

    df_main = pd.read_csv(main_file)
    df_auto = pd.read_csv(auto_file)

    print("Main.py averaged results:")
    print(df_main.to_string())
    print()

    print("Auto-polish averaged results:")
    print(df_auto.to_string())
    print()

    # Extract macro F1 scores
    main_f1 = float(df_main['Macro Avg F1-score (Mean)'].iloc[0])
    auto_f1 = float(df_auto['Macro Avg F1-score (Mean)'].iloc[0])

    print(f"Macro F1 Comparison:")
    print(f"  main.py:      {main_f1:.6f}")
    print(f"  auto_polish:  {auto_f1:.6f}")
    print()

    diff = abs(main_f1 - auto_f1)
    print(f"Absolute difference: {diff:.6f}")

    if diff < 0.001:
        print("✅ RESULTS MATCH (difference < 0.001)")
        return True
    else:
        print("❌ RESULTS DON'T MATCH (difference >= 0.001)")
        return False

def main():
    os.chdir(root_dir)

    clean_results()

    if not run_main_py():
        return 1

    clean_results()

    if not run_auto_polish():
        return 1

    if not compare_results():
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
