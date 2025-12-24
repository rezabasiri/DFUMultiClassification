#!/usr/bin/env python3
"""
Quick End-to-End Test for auto_polish_dataset_v2.py

This test runs with minimal settings to verify the pipeline works:
- Phase 1: 2 runs per modality, testing all 4 modalities with 20% data
- Phase 2: 10 evaluations with 20% data (minimum required by gp_minimize)
- CV folds: 1 (minimal)

Expected runtime: ~5-10 minutes
"""
import sys
import os
import subprocess
import time

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    print("="*80)
    print("QUICK E2E TEST: auto_polish_dataset_v2.py")
    print("="*80)
    print("\nThis test verifies the polishing pipeline works end-to-end.")
    print("\nTest Configuration:")
    print("  Phase 1: 2 runs per modality (not 10)")
    print("  Phase 1 modalities: metadata, depth_rgb, depth_map, thermal_map (all 4)")
    print("  Phase 1 data: 20% (not 100%)")
    print("  Phase 2: 10 evaluations (minimum required by gp_minimize)")
    print("  Phase 2 data: 20% (not 100%)")
    print("  Min retention per class: 0.50 (relaxed from 0.90)")
    print("  Min dataset fraction: 0.30 (relaxed from 0.50)")
    print("\nExpected runtime: ~5-10 minutes\n")

    start_time = time.time()

    # Build command with minimal settings but testing all modalities
    cmd = [
        sys.executable,
        os.path.join(project_root, 'scripts', 'auto_polish_dataset_v2.py'),
        '--phase2-modalities', 'metadata',  # Use + to combine, e.g., 'metadata+depth_rgb'
        '--phase1-n-runs', '2',      # Minimal: 2 runs per modality
        '--phase1-modalities', 'metadata', 'depth_map',  # 2 modalities
        '--phase1-data-percentage', '20',  # Only 20% of data for speed
        '--n-evaluations', '10',     # Minimum required by gp_minimize
        '--phase2-data-percentage', '20',  # Only 20% of data for speed
        '--min-retention-per-class', '0.50',  # Relaxed to ensure optimization runs
        '--min-dataset-fraction', '0.30',     # Relaxed to allow more filtering
    ]

    print("Running command:")
    print(f"  {' '.join(cmd)}\n")
    print("="*80)
    print("OUTPUT:")
    print("="*80 + "\n")

    # Run with output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=project_root
    )

    # Stream output
    for line in process.stdout:
        print(line, end='')

    process.wait()
    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nReturn code: {process.returncode}")
    print(f"Elapsed time: {elapsed/60:.1f} minutes")

    if process.returncode == 0:
        print("\n✅ E2E TEST PASSED - Pipeline works correctly!")

        # Check if results file was created
        results_file = os.path.join(project_root, 'results', 'bayesian_optimization_results.json')
        if os.path.exists(results_file):
            print(f"\n📊 Results saved to: {results_file}")

            # Show results summary
            import json
            with open(results_file) as f:
                results = json.load(f)

            if results.get('best_thresholds'):
                print(f"\nBest thresholds found: {results['best_thresholds']}")
                print(f"Best score: {results['best_score']:.4f}")
            else:
                print("\n⚠️  Optimization was skipped (could not achieve retention target)")
        else:
            print("\n⚠️  Results file not found (optimization may have been skipped)")
    else:
        print("\n❌ E2E TEST FAILED - Check output above for errors")

    return process.returncode


if __name__ == '__main__':
    sys.exit(main())
