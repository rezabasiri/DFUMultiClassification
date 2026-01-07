#!/usr/bin/env python3
"""
Automated backbone comparison script for DFU classification.
Tests different CNN backbones for RGB and map image branches.

Usage:
    python agent_communication/image_backbone/test_backbones.py

Output:
    - Clean status updates to console
    - Live detailed log: agent_communication/image_backbone/backbone_test.log
    - Results: agent_communication/image_backbone/BACKBONE_RESULTS.txt
"""

import os
import sys
import subprocess
import time
import re
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path (we're in agent_communication/image_backbone/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging - console for status, file for everything including subprocess output
LOG_FILE = project_root / 'agent_communication/image_backbone/backbone_test.log'

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler - clean status messages only
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# File handler - detailed with timestamps
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Patterns to filter out from log file (noisy but harmless warnings)
FILTER_PATTERNS = [
    'cache_dataset_ops.cc',
    'The calling iterator did not fully read the dataset',
    'loop_optimizer.cc',
    'Skipping loop optimization for Merge node',
    'node_def_util.cc',
    'use_unbounded_threadpool which is not in the op definition',
    'Unknown attributes will be ignored',
]

def should_filter_line(line):
    """Check if line should be filtered from log"""
    return any(pattern in line for pattern in FILTER_PATTERNS)

def log_to_file_only(message):
    """Write detailed output to log file only, not to console"""
    if not should_filter_line(message):
        with open(LOG_FILE, 'a') as f:
            f.write(message + '\n')

# Configuration
RGB_BACKBONES = ['SimpleCNN', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB3']
MAP_BACKBONES = ['SimpleCNN', 'EfficientNetB0', 'EfficientNetB1']

# Test parameters
DATA_PERCENTAGE = 50
IMAGE_SIZE = 32
DEVICE_MODE = 'single'
RESUME_MODE = 'fresh'

# Paths
PRODUCTION_CONFIG = project_root / 'src/utils/production_config.py'
RESULTS_FILE = project_root / 'agent_communication/image_backbone/BACKBONE_RESULTS.txt'

# Store original config to restore later
ORIGINAL_COMBINATIONS = None

def save_original_combinations():
    """Save original INCLUDED_COMBINATIONS before tests"""
    global ORIGINAL_COMBINATIONS
    with open(PRODUCTION_CONFIG, 'r') as f:
        content = f.read()
    match = re.search(r"INCLUDED_COMBINATIONS = \[([\s\S]*?)\n\]", content)
    if match:
        ORIGINAL_COMBINATIONS = match.group(0)
    logger.info(f"Saved original INCLUDED_COMBINATIONS")

def restore_original_combinations():
    """Restore original INCLUDED_COMBINATIONS after tests"""
    if ORIGINAL_COMBINATIONS:
        with open(PRODUCTION_CONFIG, 'r') as f:
            content = f.read()
        content = re.sub(
            r"INCLUDED_COMBINATIONS = \[[\s\S]*?\n\]",
            ORIGINAL_COMBINATIONS,
            content
        )
        with open(PRODUCTION_CONFIG, 'w') as f:
            f.write(content)
        logger.info("Restored original INCLUDED_COMBINATIONS")

def update_backbone_config(rgb_backbone, map_backbone):
    """Update RGB_BACKBONE, MAP_BACKBONE, and INCLUDED_COMBINATIONS in production_config.py"""
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

    # Set INCLUDED_COMBINATIONS to use both RGB and MAP modalities
    # Use depth_rgb (RGB modality) + thermal_map (MAP modality) to test both backbones
    content = re.sub(
        r"INCLUDED_COMBINATIONS = \[[\s\S]*?\n\]",
        "INCLUDED_COMBINATIONS = [\n    ('depth_rgb', 'thermal_map',),\n]",
        content
    )

    with open(PRODUCTION_CONFIG, 'w') as f:
        f.write(content)

    logger.info(f"Updated config: RGB={rgb_backbone}, MAP={map_backbone}, modalities=depth_rgb+thermal_map")

def run_training(rgb_backbone, map_backbone, test_num, total_tests):
    """Run training with specified backbones - streams output live to log file"""
    logger.info("="*80)
    logger.info(f"TEST {test_num}/{total_tests}: RGB={rgb_backbone}, MAP={map_backbone}")
    logger.info("="*80)

    cmd = [
        'python', '-u', 'src/main.py',  # -u for unbuffered output
        '--mode', 'search',
        '--device-mode', DEVICE_MODE,
        '--resume_mode', RESUME_MODE,
        '--data_percentage', str(DATA_PERCENTAGE)
    ]

    start_time = time.time()
    output_lines = []

    try:
        # Write header to log file
        log_to_file_only(f"\n{'='*80}")
        log_to_file_only(f"SUBPROCESS OUTPUT - TEST {test_num}: RGB={rgb_backbone}, MAP={map_backbone}")
        log_to_file_only(f"{'='*80}")

        # Use Popen to stream output live
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1  # Line buffered
        )

        # Stream output line by line to log file
        for line in process.stdout:
            line = line.rstrip('\n')
            output_lines.append(line)
            log_to_file_only(line)

        process.wait(timeout=3600)  # 60 minute timeout

        log_to_file_only(f"{'='*80}\n")

        runtime = time.time() - start_time
        output = '\n'.join(output_lines)

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
            'error': None
        }

    except subprocess.TimeoutExpired:
        process.kill()
        log_to_file_only(f"TIMEOUT - Process killed after 60 minutes")
        log_to_file_only(f"{'='*80}\n")
        return {
            'rgb_backbone': rgb_backbone,
            'map_backbone': map_backbone,
            'success': False,
            'error': 'TIMEOUT (60 min)',
            'runtime_min': 60.0
        }
    except Exception as e:
        log_to_file_only(f"ERROR: {str(e)}")
        log_to_file_only(f"{'='*80}\n")
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
    logger.info("="*80)
    logger.info("AUTOMATED BACKBONE COMPARISON")
    logger.info("="*80)
    logger.info(f"Log file: {LOG_FILE}")

    # Save original configuration
    save_original_combinations()

    logger.info(f"Configuration:")
    logger.info(f"  RGB Backbones: {RGB_BACKBONES}")
    logger.info(f"  MAP Backbones: {MAP_BACKBONES}")
    logger.info(f"  Data: {DATA_PERCENTAGE}%")
    logger.info(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info(f"  Device: {DEVICE_MODE}")
    logger.info(f"  Test modalities: depth_rgb + thermal_map")

    # Generate test combinations
    test_combinations = [
        (rgb, map_b)
        for rgb in RGB_BACKBONES
        for map_b in MAP_BACKBONES
    ]

    total_tests = len(test_combinations)
    logger.info(f"Total tests to run: {total_tests}")
    logger.info(f"Estimated time: {total_tests * 10} - {total_tests * 15} minutes")

    # Run all tests
    all_results = []

    for i, (rgb_backbone, map_backbone) in enumerate(test_combinations, 1):
        # Update configuration
        update_backbone_config(rgb_backbone, map_backbone)

        # Run training
        result = run_training(rgb_backbone, map_backbone, i, total_tests)
        all_results.append(result)

        # Log immediate result
        if result['success']:
            if result['kappa'] is not None:
                logger.info(f"Result: Kappa={result['kappa']:.4f}, Time={result['runtime_min']:.1f} min")
            else:
                logger.warning(f"Completed but metrics not found. Check output manually. Time={result['runtime_min']:.1f} min")
        else:
            logger.error(f"FAILED: {result['error']}")

    # Generate report
    logger.info("Generating report...")
    report = format_results(all_results)

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)

    logger.info(f"Results saved to: {RESULTS_FILE}")
    logger.info(report)

    # Restore baseline configuration
    logger.info("Restoring baseline configuration...")
    update_backbone_config('SimpleCNN', 'SimpleCNN')
    restore_original_combinations()

    logger.info("Backbone comparison complete!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error in backbone test: {e}")
        raise
