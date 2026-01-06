#!/usr/bin/env python3
"""
Automated backbone comparison script for DFU classification.
Tests different CNN backbones for RGB and map image branches.

Usage:
    python agent_communication/image_backbone/test_backbones.py

Results saved to: agent_communication/image_backbone/BACKBONE_RESULTS.txt
"""

import os
import sys
import subprocess
import time
import re
from datetime import datetime
from pathlib import Path

# Add project root to path (we're in agent_communication/image_backbone/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
RGB_BACKBONES = ['SimpleCNN', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB3']
MAP_BACKBONES = ['SimpleCNN', 'EfficientNetB0', 'EfficientNetB1']

# Test parameters
DATA_PERCENTAGE = 30
IMAGE_SIZE = 32
DEVICE_MODE = 'single'
RESUME_MODE = 'fresh'

# Paths
PRODUCTION_CONFIG = project_root / 'src/utils/production_config.py'
RESULTS_FILE = project_root / 'agent_communication/image_backbone/BACKBONE_RESULTS.txt'

def update_backbone_config(rgb_backbone, map_backbone):
    """Update RGB_BACKBONE and MAP_BACKBONE in production_config.py"""
    with open(PRODUCTION_CONFIG, 'r') as f:
        content = f.read()

    # Replace backbone configurations
    content = re.sub(
        r"RGB_BACKBONE = '[^']*'",
        f"RGB_BACKBONE = '{rgb_backbone}'",
        content
    )
    content = re.sub(
        r"MAP_BACKBONE = '[^']*'",
        f"MAP_BACKBONE = '{map_backbone}'",
        content
    )

    with open(PRODUCTION_CONFIG, 'w') as f:
        f.write(content)

    print(f"✓ Updated config: RGB={rgb_backbone}, MAP={map_backbone}")

def run_training(rgb_backbone, map_backbone, test_num, total_tests):
    """Run training with specified backbones"""
    print(f"\n{'='*80}")
    print(f"TEST {test_num}/{total_tests}: RGB={rgb_backbone}, MAP={map_backbone}")
    print(f"{'='*80}\n")

    cmd = [
        'python', 'src/main.py',
        '--mode', 'search',
        '--device-mode', DEVICE_MODE,
        '--resume-mode', RESUME_MODE,
        '--data-percentage', str(DATA_PERCENTAGE)
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        runtime = time.time() - start_time

        # Parse results from output
        output = result.stdout + result.stderr

        # Save raw output for debugging
        log_file = project_root / f"agent_communication/image_backbone/test_{test_num:02d}_{rgb_backbone}_{map_backbone}.log"
        with open(log_file, 'w') as f:
            f.write(output)

        # Try multiple patterns for each metric
        kappa = (extract_metric(output, r"Kappa[:\s]+(\d+\.\d+)") or
                extract_metric(output, r"Cohen'?s?\s+Kappa[:\s]+(\d+\.\d+)") or
                extract_metric(output, r"kappa[:\s]+(\d+\.\d+)"))

        accuracy = (extract_metric(output, r"Accuracy[:\s]+(\d+\.\d+)") or
                   extract_metric(output, r"accuracy[:\s]+(\d+\.\d+)"))

        f1_macro = (extract_metric(output, r"Macro\s+F1[:\s]+(\d+\.\d+)") or
                   extract_metric(output, r"F1\s+Macro[:\s]+(\d+\.\d+)") or
                   extract_metric(output, r"f1_macro[:\s]+(\d+\.\d+)"))

        f1_weighted = (extract_metric(output, r"Weighted\s+F1[:\s]+(\d+\.\d+)") or
                      extract_metric(output, r"F1\s+Weighted[:\s]+(\d+\.\d+)") or
                      extract_metric(output, r"f1_weighted[:\s]+(\d+\.\d+)"))

        return {
            'rgb_backbone': rgb_backbone,
            'map_backbone': map_backbone,
            'kappa': kappa,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'runtime_min': runtime / 60,
            'success': True,
            'error': None,
            'log_file': str(log_file)
        }

    except subprocess.TimeoutExpired:
        return {
            'rgb_backbone': rgb_backbone,
            'map_backbone': map_backbone,
            'success': False,
            'error': 'TIMEOUT (30 min)',
            'runtime_min': 30.0
        }
    except Exception as e:
        return {
            'rgb_backbone': rgb_backbone,
            'map_backbone': map_backbone,
            'success': False,
            'error': str(e),
            'runtime_min': (time.time() - start_time) / 60
        }

def extract_metric(text, pattern):
    """Extract metric value from output text"""
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None

def format_results(all_results):
    """Format results into a readable report"""
    lines = []
    lines.append("="*100)
    lines.append("BACKBONE COMPARISON RESULTS")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Configuration: {DATA_PERCENTAGE}% data, {IMAGE_SIZE}x{IMAGE_SIZE} images, single GPU")
    lines.append("="*100)
    lines.append("")

    # Summary table
    lines.append("RESULTS SUMMARY")
    lines.append("-"*100)
    lines.append(f"{'#':<4} {'RGB Backbone':<18} {'MAP Backbone':<18} {'Kappa':<8} {'Accuracy':<10} {'F1 Macro':<10} {'F1 Wtd':<10} {'Time (min)':<10}")
    lines.append("-"*100)

    for i, result in enumerate(all_results, 1):
        if result['success']:
            lines.append(
                f"{i:<4} {result['rgb_backbone']:<18} {result['map_backbone']:<18} "
                f"{result['kappa']:<8.4f} {result['accuracy']:<10.4f} {result['f1_macro']:<10.4f} "
                f"{result['f1_weighted']:<10.4f} {result['runtime_min']:<10.1f}"
            )
        else:
            lines.append(
                f"{i:<4} {result['rgb_backbone']:<18} {result['map_backbone']:<18} "
                f"FAILED: {result['error']}"
            )

    lines.append("-"*100)
    lines.append("")

    # Best performers
    successful_results = [r for r in all_results if r['success'] and r['kappa'] is not None]

    if successful_results:
        lines.append("BEST PERFORMERS")
        lines.append("-"*100)

        best_kappa = max(successful_results, key=lambda x: x['kappa'])
        lines.append(f"Best Kappa: {best_kappa['kappa']:.4f} - RGB={best_kappa['rgb_backbone']}, MAP={best_kappa['map_backbone']}")

        best_accuracy = max(successful_results, key=lambda x: x['accuracy'])
        lines.append(f"Best Accuracy: {best_accuracy['accuracy']:.4f} - RGB={best_accuracy['rgb_backbone']}, MAP={best_accuracy['map_backbone']}")

        fastest = min(successful_results, key=lambda x: x['runtime_min'])
        lines.append(f"Fastest: {fastest['runtime_min']:.1f} min - RGB={fastest['rgb_backbone']}, MAP={fastest['map_backbone']}")

        lines.append("-"*100)
        lines.append("")

    # Analysis
    if successful_results:
        lines.append("ANALYSIS")
        lines.append("-"*100)

        # Baseline comparison
        baseline = next((r for r in successful_results if r['rgb_backbone'] == 'SimpleCNN' and r['map_backbone'] == 'SimpleCNN'), None)

        if baseline:
            lines.append(f"Baseline (SimpleCNN/SimpleCNN): Kappa {baseline['kappa']:.4f}")
            lines.append("")

            for result in successful_results:
                if result != baseline:
                    kappa_diff = result['kappa'] - baseline['kappa']
                    kappa_pct = (kappa_diff / baseline['kappa']) * 100

                    lines.append(
                        f"  {result['rgb_backbone']}/{result['map_backbone']}: "
                        f"Kappa {result['kappa']:.4f} ({kappa_diff:+.4f}, {kappa_pct:+.1f}%)"
                    )

        lines.append("-"*100)

    lines.append("")
    lines.append(f"Total tests: {len(all_results)}")
    lines.append(f"Successful: {len(successful_results)}")
    lines.append(f"Failed: {len(all_results) - len(successful_results)}")
    lines.append("")
    lines.append("="*100)

    return "\n".join(lines)

def main():
    print("\n" + "="*80)
    print("AUTOMATED BACKBONE COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  RGB Backbones: {RGB_BACKBONES}")
    print(f"  MAP Backbones: {MAP_BACKBONES}")
    print(f"  Data: {DATA_PERCENTAGE}%")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Device: {DEVICE_MODE}")
    print(f"")

    # Generate test combinations
    test_combinations = [
        (rgb, map_b)
        for rgb in RGB_BACKBONES
        for map_b in MAP_BACKBONES
    ]

    total_tests = len(test_combinations)
    print(f"Total tests to run: {total_tests}")
    print(f"Estimated time: {total_tests * 10} - {total_tests * 15} minutes\n")

    input("Press Enter to start tests...")

    # Run all tests
    all_results = []

    for i, (rgb_backbone, map_backbone) in enumerate(test_combinations, 1):
        # Update configuration
        update_backbone_config(rgb_backbone, map_backbone)

        # Run training
        result = run_training(rgb_backbone, map_backbone, i, total_tests)
        all_results.append(result)

        # Print immediate result
        if result['success']:
            if result['kappa'] is not None:
                print(f"✓ Result: Kappa={result['kappa']:.4f}, Time={result['runtime_min']:.1f} min")
            else:
                print(f"⚠ Completed but metrics not found. Check output manually. Time={result['runtime_min']:.1f} min")
        else:
            print(f"✗ FAILED: {result['error']}")

    # Generate report
    print("\n\nGenerating report...")
    report = format_results(all_results)

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)

    print(f"\n✓ Results saved to: {RESULTS_FILE}")
    print("\n" + report)

    # Restore baseline configuration
    print("\nRestoring baseline configuration (SimpleCNN/SimpleCNN)...")
    update_backbone_config('SimpleCNN', 'SimpleCNN')

    print("\n✓ Backbone comparison complete!")

if __name__ == '__main__':
    main()
