#!/usr/bin/env python3
"""
Automated backbone comparison script for DFU classification.
Tests different CNN backbones for RGB and map image branches.

Usage:
    python agent_communication/image_backbone/test_backbones.py

Output:
    - Real-time output to console and log files
    - Master log: agent_communication/image_backbone/MASTER_LOG_<timestamp>.txt
    - Test logs: agent_communication/image_backbone/test_XX_<backbone>_<backbone>.txt
    - Results: agent_communication/image_backbone/BACKBONE_RESULTS.txt
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

def run_training(rgb_backbone, map_backbone, test_num, total_tests, master_log):
    """Run training with specified backbones"""
    header = f"\n{'='*80}\nTEST {test_num}/{total_tests}: RGB={rgb_backbone}, MAP={map_backbone}\n{'='*80}\n"
    print(header)
    master_log.write(header)
    master_log.flush()

    cmd = [
        'python', 'src/main.py',
        '--mode', 'search',
        '--device-mode', DEVICE_MODE,
        '--resume_mode', RESUME_MODE,
        '--data_percentage', str(DATA_PERCENTAGE)
    ]

    start_time = time.time()

    # Create test-specific log file
    test_log_file = project_root / f"agent_communication/image_backbone/test_{test_num:02d}_{rgb_backbone}_{map_backbone}.txt"

    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []

        # Read output line by line and write to both console and files
        with open(test_log_file, 'w') as test_log:
            for line in process.stdout:
                # Write to console
                print(line, end='')
                # Write to master log
                master_log.write(line)
                master_log.flush()
                # Write to test-specific log
                test_log.write(line)
                test_log.flush()
                # Store for parsing
                output_lines.append(line)

        # Wait for process to complete
        return_code = process.wait(timeout=3600)
        runtime = time.time() - start_time

        # Combine output for parsing
        output = ''.join(output_lines)

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

        success = return_code == 0
        error = None if success else f"Process exited with code {return_code}"

        return {
            'rgb_backbone': rgb_backbone,
            'map_backbone': map_backbone,
            'kappa': kappa,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'runtime_min': runtime / 60,
            'success': success,
            'error': error,
            'log_file': str(test_log_file)
        }

    except subprocess.TimeoutExpired:
        process.kill()
        error_msg = "TIMEOUT (60 min)\n"
        print(error_msg)
        master_log.write(error_msg)
        master_log.flush()
        return {
            'rgb_backbone': rgb_backbone,
            'map_backbone': map_backbone,
            'success': False,
            'error': 'TIMEOUT (60 min)',
            'runtime_min': 60.0
        }
    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n"
        print(error_msg)
        master_log.write(error_msg)
        master_log.flush()
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
    # Create master log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_log_file = project_root / f"agent_communication/image_backbone/MASTER_LOG_{timestamp}.txt"

    with open(master_log_file, 'w') as master_log:
        header = "\n" + "="*80 + "\n" + "AUTOMATED BACKBONE COMPARISON\n" + "="*80 + "\n"
        print(header)
        master_log.write(header)

        config_info = f"\nConfiguration:\n  RGB Backbones: {RGB_BACKBONES}\n  MAP Backbones: {MAP_BACKBONES}\n  Data: {DATA_PERCENTAGE}%\n  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n  Device: {DEVICE_MODE}\n\n"
        print(config_info)
        master_log.write(config_info)

        # Generate test combinations
        test_combinations = [
            (rgb, map_b)
            for rgb in RGB_BACKBONES
            for map_b in MAP_BACKBONES
        ]

        total_tests = len(test_combinations)
        test_info = f"Total tests to run: {total_tests}\nEstimated time: {total_tests * 10} - {total_tests * 15} minutes\n\n"
        print(test_info)
        master_log.write(test_info)
        master_log.write(f"Master log file: {master_log_file}\n\n")
        master_log.flush()

        # Run all tests
        all_results = []

        for i, (rgb_backbone, map_backbone) in enumerate(test_combinations, 1):
            # Update configuration
            update_backbone_config(rgb_backbone, map_backbone)

            # Run training
            result = run_training(rgb_backbone, map_backbone, i, total_tests, master_log)
            all_results.append(result)

            # Print immediate result
            result_msg = ""
            if result['success']:
                if result['kappa'] is not None:
                    result_msg = f"✓ Result: Kappa={result['kappa']:.4f}, Time={result['runtime_min']:.1f} min\n"
                else:
                    result_msg = f"⚠ Completed but metrics not found. Check log file: {result['log_file']}\n"
            else:
                result_msg = f"✗ FAILED: {result['error']}\n"

            print(result_msg)
            master_log.write(result_msg + "\n")
            master_log.flush()

        # Generate report
        report_msg = "\n\nGenerating report...\n"
        print(report_msg)
        master_log.write(report_msg)
        report = format_results(all_results)

        # Save results
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            f.write(report)

        final_msg = f"\n✓ Results saved to: {RESULTS_FILE}\n✓ Master log saved to: {master_log_file}\n\n{report}\n"
        print(final_msg)
        master_log.write(final_msg)

        # Restore baseline configuration
        restore_msg = "\nRestoring baseline configuration (SimpleCNN/SimpleCNN)...\n"
        print(restore_msg)
        master_log.write(restore_msg)
        update_backbone_config('SimpleCNN', 'SimpleCNN')

        complete_msg = "\n✓ Backbone comparison complete!\n"
        print(complete_msg)
        master_log.write(complete_msg)

if __name__ == '__main__':
    main()
