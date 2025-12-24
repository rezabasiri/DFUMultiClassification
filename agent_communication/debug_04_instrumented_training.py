#!/usr/bin/env python3
"""DEBUG 4: Run single training with heavy instrumentation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import tempfile

def main():
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 4: INSTRUMENTED TRAINING RUN")
    log("="*80)

    log("\nThis will run a single training iteration with:")
    log("  - Verbosity = 2 (maximum output)")
    log("  - cv_folds = 1 (single train/val split)")
    log("  - Modalities = metadata only")
    log("  - Duration: ~5-10 minutes")

    log("\nStarting training...")
    log("-" * 80)

    try:
        # Run main.py with maximum verbosity
        cmd = [
            sys.executable, 'src/main.py',
            '--mode', 'search',
            '--cv_folds', '1',
            '--verbosity', '2',
            '--modalities', 'metadata',
            '--resume_mode', 'fresh'
        ]

        log(f"Command: {' '.join(cmd)}\n")

        # Capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Save stdout
        if result.stdout:
            log("\n" + "="*80)
            log("TRAINING OUTPUT:")
            log("="*80)
            log(result.stdout)

        # Save stderr
        if result.stderr:
            log("\n" + "="*80)
            log("TRAINING ERRORS/WARNINGS:")
            log("="*80)
            log(result.stderr)

        # Check return code
        log("\n" + "="*80)
        log(f"Training completed with return code: {result.returncode}")
        log("="*80)

        success = (result.returncode == 0)

        # Look for key metrics in output
        if success:
            log("\nSearching for final metrics in output...")
            for line in result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['f1', 'accuracy', 'kappa', 'macro']):
                    log(f"  {line.strip()}")

        return success, output

    except subprocess.TimeoutExpired:
        log("\n❌ TIMEOUT: Training took longer than 10 minutes")
        return False, output

    except Exception as e:
        log(f"\n❌ EXCEPTION: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False, output

if __name__ == "__main__":
    success, output = main()

    # Save output
    output_file = 'agent_communication/results_04_training_debug.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    print(f"\n✓ Results saved to: {output_file}")
    print(f"  File size: {len(output)} lines")
    sys.exit(0 if success else 1)
