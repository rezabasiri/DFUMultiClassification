#!/usr/bin/env python3
"""
Automated generative augmentation effectiveness test for DFU classification.
Compares performance with and without Stable Diffusion-based augmentation.

Usage:
    python agent_communication/generative_augmentation/test_generative_aug.py          # Resume previous run
    python agent_communication/generative_augmentation/test_generative_aug.py --fresh  # Start from scratch
    python agent_communication/generative_augmentation/test_generative_aug.py --quick  # Quick test with minimal settings

Output:
    - Clean status updates to console
    - Live detailed log: agent_communication/generative_augmentation/genaug_test.log
    - Progress tracking: agent_communication/generative_augmentation/GENGEN_PROGRESS.json
    - Final report: agent_communication/generative_augmentation/GENGEN_REPORT.txt
"""

# CRITICAL: Disable tqdm BEFORE any other imports
import os
os.environ['TQDM_DISABLE'] = '1'

import sys
import subprocess
import time
import re
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
LOG_FILE = project_root / 'agent_communication/generative_augmentation/gengen_test.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all debug messages

# Console handler - clean status messages
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# File handler - detailed with timestamps
# Use mode='w' to overwrite log file each run (no need to manually delete)
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Patterns to filter from log
FILTER_PATTERNS = [
    'cache_dataset_ops.cc',
    'The calling iterator did not fully read the dataset',
    'loop_optimizer.cc',
    'Skipping loop optimization',
    'node_def_util.cc',
    'use_unbounded_threadpool',
    'Unknown attributes will be ignored',
    '/step',
    '━',
]

def should_filter_line(line):
    """Check if line should be filtered from log"""
    return any(pattern in line for pattern in FILTER_PATTERNS)

def log_to_file_only(message):
    """Write to log file only, not console"""
    if not should_filter_line(message):
        with open(LOG_FILE, 'a') as f:
            f.write(message + '\n')

# Test configurations
TEST_CONFIGS = [
    {
        'name': 'baseline',
        'description': 'Baseline (no generative augmentation)',
        'use_gen_aug': False,
    },
    {
        'name': 'gengen_enabled',
        'description': 'With generative augmentation (depth_rgb)',
        'use_gen_aug': True,
    }
]

# Test modalities (fixed for all tests)
# IMPORTANT: Must match order in production_config.py ALL_MODALITIES
# ALL_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
TEST_MODALITIES = ['metadata', 'depth_rgb', 'depth_map', 'thermal_map']

# Test parameters (will be overridden by --quick mode)
QUICK_MODE = False
DATA_PERCENTAGE = 100
N_EPOCHS = 300
IMAGE_SIZE = 64

# Quick mode settings
QUICK_DATA_PERCENTAGE = 30.0  # Reduced for faster quick testing
QUICK_N_EPOCHS = 3  # Minimal epochs for quick error checking
QUICK_IMAGE_SIZE = 32
QUICK_STAGE1_EPOCHS = 1  # Stage 1 pre-training for quick mode (just 1 epoch to verify pipeline)
QUICK_EARLY_STOP_PATIENCE = 3  # Quick early stopping
QUICK_REDUCE_LR_PATIENCE = 1  # Reduce LR after 1 epoch without improvement
QUICK_BATCH_SIZE = 256  # Large batch size for quick mode with 64x64 images to maximize GPU utilization

# Quick mode SDXL settings (significantly faster generation)
QUICK_GENERATIVE_AUG_INFERENCE_STEPS = 10  # Reduced from 50 - much faster, quality doesn't matter for quick test
QUICK_GENERATIVE_AUG_BATCH_LIMIT = 4  # Reduced from 8 - fewer images per generation call
QUICK_GENERATIVE_AUG_PROB = 0.2  # Reduced from 0.3 - generate less frequently in quick mode

# Paths
PRODUCTION_CONFIG = project_root / 'src/utils/production_config.py'
PROGRESS_FILE = project_root / 'agent_communication/generative_augmentation/GENGEN_PROGRESS.json'
REPORT_FILE = project_root / 'agent_communication/generative_augmentation/GENGEN_REPORT.txt'

# Store original config
ORIGINAL_CONFIG = {}

def save_original_config():
    """Save original production_config values"""
    global ORIGINAL_CONFIG
    with open(PRODUCTION_CONFIG, 'r') as f:
        content = f.read()

    # Extract key values
    patterns = {
        'USE_GENERATIVE_AUGMENTATION': r'USE_GENERATIVE_AUGMENTATION = (True|False)',
        'INCLUDED_COMBINATIONS': r'INCLUDED_COMBINATIONS = \[[\s\S]*?\n\]',
        'DATA_PERCENTAGE': r'DATA_PERCENTAGE = ([\d.]+)',
        'N_EPOCHS': r'N_EPOCHS = (\d+)',
        'IMAGE_SIZE': r'IMAGE_SIZE = (\d+)',
        'STAGE1_EPOCHS': r'STAGE1_EPOCHS = (\d+)',
        'EARLY_STOP_PATIENCE': r'EARLY_STOP_PATIENCE = (\d+)',
        'REDUCE_LR_PATIENCE': r'REDUCE_LR_PATIENCE = (\d+)',
        'LR_SCHEDULE_EXPLORATION_EPOCHS': r'LR_SCHEDULE_EXPLORATION_EPOCHS = (\d+)',
        'GLOBAL_BATCH_SIZE': r'GLOBAL_BATCH_SIZE = (\d+)',
        # SDXL generative augmentation settings
        'GENERATIVE_AUG_INFERENCE_STEPS': r'GENERATIVE_AUG_INFERENCE_STEPS = (\d+)',
        'GENERATIVE_AUG_BATCH_LIMIT': r'GENERATIVE_AUG_BATCH_LIMIT = (\d+)',
        'GENERATIVE_AUG_PROB': r'GENERATIVE_AUG_PROB = ([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            ORIGINAL_CONFIG[key] = match.group(0)

    logger.info("Saved original production_config values")

def restore_original_config():
    """Restore original production_config values"""
    if not ORIGINAL_CONFIG:
        return

    with open(PRODUCTION_CONFIG, 'r') as f:
        content = f.read()

    for key, original_value in ORIGINAL_CONFIG.items():
        if key == 'USE_GENERATIVE_AUGMENTATION':
            content = re.sub(r'USE_GENERATIVE_AUGMENTATION = (True|False)', original_value, content)
        elif key == 'INCLUDED_COMBINATIONS':
            content = re.sub(r'INCLUDED_COMBINATIONS = \[[\s\S]*?\n\]', original_value, content)
        elif key == 'DATA_PERCENTAGE':
            content = re.sub(r'DATA_PERCENTAGE = [\d.]+', original_value, content)
        elif key == 'N_EPOCHS':
            content = re.sub(r'N_EPOCHS = \d+', original_value, content)
        elif key == 'IMAGE_SIZE':
            content = re.sub(r'IMAGE_SIZE = \d+', original_value, content)
        elif key == 'STAGE1_EPOCHS':
            content = re.sub(r'STAGE1_EPOCHS = \d+', original_value, content)
        elif key == 'EARLY_STOP_PATIENCE':
            content = re.sub(r'EARLY_STOP_PATIENCE = \d+', original_value, content)
        elif key == 'REDUCE_LR_PATIENCE':
            content = re.sub(r'REDUCE_LR_PATIENCE = \d+', original_value, content)
        elif key == 'LR_SCHEDULE_EXPLORATION_EPOCHS':
            content = re.sub(r'LR_SCHEDULE_EXPLORATION_EPOCHS = \d+', original_value, content)
        elif key == 'GLOBAL_BATCH_SIZE':
            content = re.sub(r'GLOBAL_BATCH_SIZE = \d+', original_value, content)
        # SDXL generative augmentation settings
        elif key == 'GENERATIVE_AUG_INFERENCE_STEPS':
            content = re.sub(r'GENERATIVE_AUG_INFERENCE_STEPS = \d+', original_value, content)
        elif key == 'GENERATIVE_AUG_BATCH_LIMIT':
            content = re.sub(r'GENERATIVE_AUG_BATCH_LIMIT = \d+', original_value, content)
        elif key == 'GENERATIVE_AUG_PROB':
            content = re.sub(r'GENERATIVE_AUG_PROB = [\d.]+', original_value, content)

    with open(PRODUCTION_CONFIG, 'w') as f:
        f.write(content)

    logger.info("Restored original production_config values")

def update_config_for_test(use_gen_aug):
    """Update production_config for test run"""
    logger.debug(f"[DEBUG] update_config_for_test called with use_gen_aug={use_gen_aug}")
    with open(PRODUCTION_CONFIG, 'r') as f:
        content = f.read()
    logger.debug(f"[DEBUG] Read production_config ({len(content)} bytes)")

    # Update USE_GENERATIVE_AUGMENTATION
    content = re.sub(
        r'USE_GENERATIVE_AUGMENTATION = (True|False)',
        f'USE_GENERATIVE_AUGMENTATION = {use_gen_aug}',
        content
    )

    # Update INCLUDED_COMBINATIONS with test modalities
    modalities_str = ', '.join([f"'{m}'" for m in TEST_MODALITIES])
    new_combinations = f"INCLUDED_COMBINATIONS = [\n    ({modalities_str},),\n]"
    # Match from INCLUDED_COMBINATIONS to the first ] that's at the start of a line or followed by whitespace and #
    content = re.sub(
        r'INCLUDED_COMBINATIONS = \[[\s\S]*?\n\]',
        new_combinations,
        content
    )

    # Update test parameters if in quick mode
    if QUICK_MODE:
        content = re.sub(r'DATA_PERCENTAGE = [\d.]+', f'DATA_PERCENTAGE = {QUICK_DATA_PERCENTAGE}', content)
        content = re.sub(r'N_EPOCHS = \d+', f'N_EPOCHS = {QUICK_N_EPOCHS}', content)
        content = re.sub(r'IMAGE_SIZE = \d+', f'IMAGE_SIZE = {QUICK_IMAGE_SIZE}', content)
        content = re.sub(r'STAGE1_EPOCHS = \d+', f'STAGE1_EPOCHS = {QUICK_STAGE1_EPOCHS}', content)
        content = re.sub(r'EARLY_STOP_PATIENCE = \d+', f'EARLY_STOP_PATIENCE = {QUICK_EARLY_STOP_PATIENCE}', content)
        content = re.sub(r'REDUCE_LR_PATIENCE = \d+', f'REDUCE_LR_PATIENCE = {QUICK_REDUCE_LR_PATIENCE}', content)
        content = re.sub(r'LR_SCHEDULE_EXPLORATION_EPOCHS = \d+', f'LR_SCHEDULE_EXPLORATION_EPOCHS = {QUICK_N_EPOCHS}', content)
        content = re.sub(r'GLOBAL_BATCH_SIZE = \d+', f'GLOBAL_BATCH_SIZE = {QUICK_BATCH_SIZE}', content)
        # SDXL settings for quick mode - significantly faster generation
        content = re.sub(r'GENERATIVE_AUG_INFERENCE_STEPS = \d+', f'GENERATIVE_AUG_INFERENCE_STEPS = {QUICK_GENERATIVE_AUG_INFERENCE_STEPS}', content)
        content = re.sub(r'GENERATIVE_AUG_BATCH_LIMIT = \d+', f'GENERATIVE_AUG_BATCH_LIMIT = {QUICK_GENERATIVE_AUG_BATCH_LIMIT}', content)
        content = re.sub(r'GENERATIVE_AUG_PROB = [\d.]+', f'GENERATIVE_AUG_PROB = {QUICK_GENERATIVE_AUG_PROB}', content)
    else:
        content = re.sub(r'DATA_PERCENTAGE = [\d.]+', f'DATA_PERCENTAGE = {DATA_PERCENTAGE}', content)
        content = re.sub(r'N_EPOCHS = \d+', f'N_EPOCHS = {N_EPOCHS}', content)
        content = re.sub(r'IMAGE_SIZE = \d+', f'IMAGE_SIZE = {IMAGE_SIZE}', content)

    with open(PRODUCTION_CONFIG, 'w') as f:
        f.write(content)
    logger.debug(f"[DEBUG] Config updated: USE_GENERATIVE_AUGMENTATION={use_gen_aug}, QUICK_MODE={QUICK_MODE}")
    if QUICK_MODE:
        logger.debug(f"[DEBUG] Quick mode config: DATA_PERCENTAGE={QUICK_DATA_PERCENTAGE}, N_EPOCHS={QUICK_N_EPOCHS}, IMAGE_SIZE={QUICK_IMAGE_SIZE}, BATCH_SIZE={QUICK_BATCH_SIZE}")
        logger.debug(f"[DEBUG] Quick mode SDXL: INFERENCE_STEPS={QUICK_GENERATIVE_AUG_INFERENCE_STEPS}, BATCH_LIMIT={QUICK_GENERATIVE_AUG_BATCH_LIMIT}, PROB={QUICK_GENERATIVE_AUG_PROB}")

def load_progress():
    """Load progress from file"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': {}}

def save_progress(progress):
    """Save progress to file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def run_test(config_name, config_desc, use_gen_aug):
    """Run a single test configuration"""
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST: {config_desc}")
    logger.info(f"{'='*80}")
    logger.debug(f"[DEBUG] run_test called: config_name={config_name}, use_gen_aug={use_gen_aug}")
    logger.debug(f"[DEBUG] QUICK_MODE={QUICK_MODE}, DATA_PERCENTAGE={DATA_PERCENTAGE}, N_EPOCHS={N_EPOCHS}, IMAGE_SIZE={IMAGE_SIZE}")

    # Delete cached cleaned datasets to force regeneration with correct DATA_PERCENTAGE
    # This is critical because:
    # 1. Cached datasets bypass DATA_PERCENTAGE parameter
    # 2. Both baseline and gen-aug tests would use the same cached dataset otherwise
    combo_name = '_'.join(sorted(TEST_MODALITIES))
    cleaned_dir = project_root / 'data/cleaned'
    if cleaned_dir.exists():
        for cache_file in cleaned_dir.glob(f"{combo_name}_*.csv"):
            cache_file.unlink()
            logger.info(f"Deleted cached dataset: {cache_file.name}")
        for outlier_file in cleaned_dir.glob(f"outliers_{combo_name}_*.csv"):
            outlier_file.unlink()
            logger.info(f"Deleted cached outliers: {outlier_file.name}")

    # Update config
    update_config_for_test(use_gen_aug)
    logger.info(f"Config updated: USE_GENERATIVE_AUGMENTATION={use_gen_aug}")
    logger.info(f"Modalities: {', '.join(TEST_MODALITIES)}")

    # Record log file position before running test (to read only new content)
    log_start_pos = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0

    # Run main.py (using conda environment)
    # Pass data_percentage as command-line argument (not just config file)
    # Use 'fresh' resume mode to avoid loading old checkpoints
    # Use 'multi' device mode to leverage both GPUs
    start_time = time.time()
    data_pct = DATA_PERCENTAGE if not QUICK_MODE else QUICK_DATA_PERCENTAGE
    cmd = [
        '/venv/multimodal/bin/python', 'src/main.py',
        '--data_percentage', str(data_pct),
        '--resume_mode', 'fresh',
        '--device-mode', 'multi'
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    logger.debug(f"[DEBUG] Working directory: {project_root}")
    logger.debug(f"[DEBUG] Log start position: {log_start_pos}")
    log_to_file_only(f"\nCOMMAND: {' '.join(cmd)}\n")

    try:
        logger.debug("[DEBUG] Starting subprocess...")
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        logger.debug(f"[DEBUG] Subprocess started with PID: {process.pid}")

        # Stream output with periodic status updates
        line_count = 0
        last_status_time = time.time()
        status_interval = 60  # Print status every 60 seconds

        for line in process.stdout:
            line = line.rstrip()
            line_count += 1
            log_to_file_only(line)

            # Show important messages on console
            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'kappa', 'accuracy', 'epoch', 'fold', 'stage', 'training', 'pre-training']):
                logger.info(f"  {line}")

            # Periodic status update to show the process is still running
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                elapsed_so_far = current_time - start_time
                logger.info(f"  [STATUS] Still running... {elapsed_so_far/60:.1f} min elapsed, {line_count} lines processed")
                logger.debug(f"[DEBUG] Last output line: {line[:100]}...")
                last_status_time = current_time

        logger.debug(f"[DEBUG] Subprocess output finished, waiting for exit...")
        process.wait()
        elapsed = time.time() - start_time
        logger.debug(f"[DEBUG] Subprocess exited with code: {process.returncode}")
        logger.debug(f"[DEBUG] Total lines processed: {line_count}")

        if process.returncode != 0:
            logger.error(f"Test FAILED with return code {process.returncode}")
            return None, None

        logger.info(f"Test completed in {elapsed/60:.1f} minutes ({line_count} log lines)")
        return elapsed, log_start_pos

    except Exception as e:
        logger.error(f"Error running test: {e}")
        import traceback
        logger.debug(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None

def extract_metrics_from_logs(log_start_pos=0):
    """Extract final metrics from main.py output logs

    Args:
        log_start_pos: File position to start reading from (to avoid stale metrics from previous runs)
    """
    metrics = {'kappa': None, 'accuracy': None, 'f1_macro': None, 'f1_weighted': None}

    # Try method 1: Read from CSV file
    results_dir = project_root / 'results'
    csv_pattern = results_dir / 'modality_combination_results.csv'

    if csv_pattern.exists():
        # Read CSV and get last line (latest result)
        with open(csv_pattern, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                try:
                    metrics = {
                        'kappa': float(last_line[1]) if len(last_line) > 1 else None,
                        'accuracy': float(last_line[2]) if len(last_line) > 2 else None,
                        'f1_macro': float(last_line[3]) if len(last_line) > 3 else None,
                        'f1_weighted': float(last_line[4]) if len(last_line) > 4 else None,
                    }
                    if metrics['kappa'] is not None:
                        return metrics
                except:
                    pass

    # Try method 2: Parse from log file (only read content after log_start_pos to avoid stale metrics)
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            # Seek to the position where this test started
            f.seek(log_start_pos)
            log_content = f.read()

            # Look for metrics - use findall and take last match to get final results from THIS test run
            kappa_matches = re.findall(r'Kappa:\s+(\d+\.\d+)', log_content)
            accuracy_matches = re.findall(r'Accuracy:\s+(\d+\.\d+)\s+±', log_content)
            f1_macro_matches = re.findall(r'F1\s+Macro:\s+(\d+\.\d+)', log_content)

            if kappa_matches:
                metrics['kappa'] = float(kappa_matches[-1])
            if accuracy_matches:
                metrics['accuracy'] = float(accuracy_matches[-1])
            if f1_macro_matches:
                metrics['f1_macro'] = float(f1_macro_matches[-1])

    return metrics

def generate_report(progress):
    """Generate final comparison report"""
    logger.info("\nGenerating final report...")

    results = progress['results']

    def safe_format(value, default=0.0, fmt='.4f'):
        """Safely format a value that might be None"""
        if value is None:
            value = default
        return f"{value:{fmt}}"

    with open(REPORT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATIVE AUGMENTATION EFFECTIVENESS TEST - RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modalities: {', '.join(TEST_MODALITIES)}\n")
        f.write(f"Mode: {'QUICK TEST' if QUICK_MODE else 'FULL PRODUCTION'}\n")
        if QUICK_MODE:
            f.write(f"Settings: {DATA_PERCENTAGE}% data, {N_EPOCHS} epochs, {IMAGE_SIZE}x{IMAGE_SIZE} images\n")
        else:
            f.write(f"Settings: {DATA_PERCENTAGE}% data, {N_EPOCHS} epochs, {IMAGE_SIZE}x{IMAGE_SIZE} images\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        # Table header
        f.write(f"{'Configuration':<30} {'Kappa':>10} {'Accuracy':>10} {'F1 Macro':>10} {'Runtime':>10}\n")
        f.write("-" * 80 + "\n")

        baseline = results.get('baseline', {})
        gengen = results.get('gengen_enabled', {})

        # Baseline row
        f.write(f"{'Baseline (no gen aug)':<30} "
                f"{safe_format(baseline.get('kappa'))}   "
                f"{safe_format(baseline.get('accuracy'))}   "
                f"{safe_format(baseline.get('f1_macro'))}   "
                f"{safe_format(baseline.get('runtime_min'), fmt='.1f')} min\n")

        # Gen aug row
        f.write(f"{'With gen aug (depth_rgb)':<30} "
                f"{safe_format(gengen.get('kappa'))}   "
                f"{safe_format(gengen.get('accuracy'))}   "
                f"{safe_format(gengen.get('f1_macro'))}   "
                f"{safe_format(gengen.get('runtime_min'), fmt='.1f')} min\n")

        f.write("\n")

        # Calculate improvements
        if baseline.get('kappa') and gengen.get('kappa'):
            kappa_improvement = ((gengen['kappa'] - baseline['kappa']) / baseline['kappa']) * 100
            f.write(f"Kappa Improvement: {kappa_improvement:+.1f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        if baseline.get('kappa') and gengen.get('kappa'):
            if gengen['kappa'] > baseline['kappa'] * 1.05:  # >5% improvement
                f.write("✓ EFFECTIVE: Generative augmentation shows significant improvement.\n")
                f.write("  Recommendation: Enable USE_GENERATIVE_AUGMENTATION in production.\n")
            elif gengen['kappa'] > baseline['kappa']:
                f.write("⚠ MARGINAL: Generative augmentation shows minor improvement.\n")
                f.write("  Recommendation: Consider cost/benefit (48 GB models, slower training).\n")
            else:
                f.write("✗ NOT EFFECTIVE: Generative augmentation does not improve performance.\n")
                f.write("  Recommendation: Keep USE_GENERATIVE_AUGMENTATION = False.\n")

        f.write("\n")
        f.write("Detailed logs: agent_communication/generative_augmentation/gengen_test.log\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Report saved to: {REPORT_FILE}")

    # Display report on console
    with open(REPORT_FILE, 'r') as f:
        print("\n" + f.read())

def main():
    global QUICK_MODE

    logger.debug("[DEBUG] main() started")
    logger.debug(f"[DEBUG] Python version: {sys.version}")
    logger.debug(f"[DEBUG] Project root: {project_root}")
    logger.debug(f"[DEBUG] Log file: {LOG_FILE}")

    parser = argparse.ArgumentParser(description='Test generative augmentation effectiveness')
    parser.add_argument('--fresh', action='store_true', help='Start from scratch')
    parser.add_argument('--quick', action='store_true', help='Quick test with minimal settings')
    args = parser.parse_args()
    logger.debug(f"[DEBUG] Parsed arguments: fresh={args.fresh}, quick={args.quick}")

    global QUICK_MODE, DATA_PERCENTAGE, N_EPOCHS, IMAGE_SIZE
    QUICK_MODE = args.quick

    if QUICK_MODE:
        # Update global variables for quick mode
        DATA_PERCENTAGE = QUICK_DATA_PERCENTAGE
        N_EPOCHS = QUICK_N_EPOCHS
        IMAGE_SIZE = QUICK_IMAGE_SIZE
        logger.info("=" * 80)
        logger.info(f"QUICK TEST MODE: {DATA_PERCENTAGE}% data, {N_EPOCHS} epochs, {IMAGE_SIZE}x{IMAGE_SIZE} images")
        logger.info("This is for error checking only - results not production-ready")
        logger.info("=" * 80 + "\n")
        logger.debug(f"[DEBUG] Quick mode settings: QUICK_BATCH_SIZE={QUICK_BATCH_SIZE}, QUICK_STAGE1_EPOCHS={QUICK_STAGE1_EPOCHS}")
        logger.debug(f"[DEBUG] Quick mode settings: QUICK_EARLY_STOP_PATIENCE={QUICK_EARLY_STOP_PATIENCE}, QUICK_REDUCE_LR_PATIENCE={QUICK_REDUCE_LR_PATIENCE}")

    # Save original config
    logger.debug("[DEBUG] Saving original config...")
    save_original_config()
    logger.debug(f"[DEBUG] Original config saved: {list(ORIGINAL_CONFIG.keys())}")

    # Load or reset progress
    if args.fresh or not PROGRESS_FILE.exists():
        logger.info("Starting fresh test run")
        logger.debug(f"[DEBUG] Fresh run requested or progress file doesn't exist: {PROGRESS_FILE}")
        progress = {'completed': [], 'results': {}}
        save_progress(progress)
    else:
        progress = load_progress()
        logger.info(f"Resuming test run - {len(progress['completed'])}/2 tests completed")
        logger.debug(f"[DEBUG] Loaded progress: completed={progress['completed']}, results_keys={list(progress['results'].keys())}")

    try:
        # Run each test
        logger.debug(f"[DEBUG] Starting test loop with {len(TEST_CONFIGS)} configs")
        for i, config in enumerate(TEST_CONFIGS):
            config_name = config['name']
            logger.debug(f"[DEBUG] Processing config {i+1}/{len(TEST_CONFIGS)}: {config_name}")

            if config_name in progress['completed']:
                logger.info(f"\nSkipping {config['description']} (already completed)")
                logger.debug(f"[DEBUG] Skipping {config_name} - already in completed list")
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"Starting test {len(progress['completed'])+1}/2")
            logger.info(f"{'='*80}")
            logger.debug(f"[DEBUG] About to call run_test for {config_name}")
            logger.debug(f"[DEBUG] Config: {config}")

            runtime, log_start_pos = run_test(config_name, config['description'], config['use_gen_aug'])
            logger.debug(f"[DEBUG] run_test returned: runtime={runtime}, log_start_pos={log_start_pos}")

            if runtime is None:
                logger.error("Test failed - stopping")
                logger.debug("[DEBUG] runtime is None, test failed")
                return 1

            # Extract metrics (only from this test run using log_start_pos)
            logger.debug(f"[DEBUG] Extracting metrics from log_start_pos={log_start_pos}")
            metrics = extract_metrics_from_logs(log_start_pos)
            metrics['runtime_min'] = runtime / 60
            metrics['success'] = True
            logger.debug(f"[DEBUG] Extracted metrics: {metrics}")

            # Save results
            progress['results'][config_name] = metrics
            progress['completed'].append(config_name)
            save_progress(progress)
            logger.debug(f"[DEBUG] Progress saved: completed={progress['completed']}")

            kappa_value = metrics.get('kappa')
            if kappa_value is not None:
                logger.info(f"✓ Test completed - Kappa: {kappa_value:.4f}")
            else:
                logger.warning("✓ Test completed - Kappa: N/A (metric extraction failed)")
                logger.debug(f"[DEBUG] Kappa was None, metrics={metrics}")

        # Generate final report
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETE")
        logger.info("="*80)
        generate_report(progress)

        return 0

    finally:
        # Always restore original config
        restore_original_config()
        logger.info("\nOriginal configuration restored")

if __name__ == '__main__':
    sys.exit(main())
