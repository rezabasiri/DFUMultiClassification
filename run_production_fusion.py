#!/usr/bin/env python3
"""
Production Fusion Training - End-to-End Pipeline
================================================

This script runs the complete production fusion pipeline:
1. Detects and removes outliers (15% contamination)
2. Applies cleaned dataset
3. Runs fusion training with optimized parameters
4. Reports final results

Expected Performance: Kappa 0.27 ± 0.08 (+63% vs baseline)

Based on Phase 7 investigation results.
See: agent_communication/fusion_fix/FUSION_FIX_GUIDE.md
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{msg.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}\n")

def print_step(step, msg):
    print(f"{Colors.BOLD}{Colors.CYAN}[Step {step}]{Colors.END} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def run_command(cmd, description, capture_output=False):
    """Run a shell command with error handling"""
    print(f"\n{Colors.BLUE}Running: {' '.join(cmd)}{Colors.END}")

    try:
        if capture_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        else:
            subprocess.run(cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed!")
        if capture_output and e.stderr:
            print(e.stderr)
        sys.exit(1)

def check_config():
    """Verify production_config.py has correct settings"""
    print_step(0, "Verifying production configuration...")

    config_path = Path("src/utils/production_config.py")
    if not config_path.exists():
        print_error("production_config.py not found!")
        sys.exit(1)

    with open(config_path) as f:
        config = f.read()

    errors = []

    # Check IMAGE_SIZE
    if "IMAGE_SIZE = 32" not in config:
        errors.append("IMAGE_SIZE should be 32 (not 128)")

    # Check SAMPLING_STRATEGY
    if "SAMPLING_STRATEGY = 'combined'" not in config:
        errors.append("SAMPLING_STRATEGY should be 'combined'")

    if errors:
        print_error("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix src/utils/production_config.py")
        sys.exit(1)

    print_success("Configuration verified")
    print(f"  ✓ IMAGE_SIZE = 32")
    print(f"  ✓ SAMPLING_STRATEGY = 'combined'")

def detect_outliers(contamination=0.15, seed=42, skip_if_exists=True):
    """Run outlier detection"""
    print_step(1, f"Detecting outliers ({contamination*100:.0f}% contamination)...")

    output_dir = Path("data/cleaned")
    cleaned_file = output_dir / f"metadata_cleaned_{int(contamination*100):02d}pct.csv"

    if skip_if_exists and cleaned_file.exists():
        print_warning(f"Cleaned dataset already exists: {cleaned_file}")
        print("  Use --force to regenerate")
        return

    cmd = [
        "python", "agent_communication/fusion_fix/scripts_production/detect_outliers.py",
        "--contamination", str(contamination),
        "--output", "data/cleaned",
        "--seed", str(seed)
    ]

    run_command(cmd, "Outlier detection")
    print_success(f"Outliers detected and saved to {output_dir}")

def apply_cleaned_dataset(contamination=0.15):
    """Apply cleaned dataset"""
    print_step(2, "Applying cleaned dataset...")

    pct = f"{int(contamination*100):02d}pct"
    cmd = [
        "python", "agent_communication/fusion_fix/scripts_production/test_cleaned_data.py",
        pct
    ]

    run_command(cmd, "Apply cleaned dataset")
    print_success("Cleaned dataset applied")

def run_training(cv_folds=3, verbosity=2, device_mode="multi"):
    """Run fusion training"""
    print_step(3, "Running fusion training...")

    cmd = [
        "python", "src/main.py",
        "--mode", "search",
        "--cv_folds", str(cv_folds),
        "--verbosity", str(verbosity),
        "--resume_mode", "fresh",
        "--device-mode", device_mode
    ]

    run_command(cmd, "Fusion training")
    print_success("Training completed")

def show_results():
    """Display final results"""
    print_step(4, "Extracting results...")

    results_file = Path("results/csv/modality_combination_results.csv")
    if not results_file.exists():
        print_warning("Results file not found, training may have failed")
        return

    # Try to extract final Kappa
    try:
        with open(results_file) as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1]
                fields = last_line.split(',')
                if len(fields) >= 4:
                    kappa = float(fields[3])
                    print_success(f"Final Kappa: {kappa:.4f}")

                    if kappa >= 0.25:
                        print_success("✓ Performance meets expectations (>= 0.25)")
                    elif kappa >= 0.20:
                        print_warning("⚠ Performance slightly below target (0.20-0.25)")
                    else:
                        print_error("✗ Performance below expectations (< 0.20)")
                        print("  Check that 'combined' sampling was applied correctly")
    except Exception as e:
        print_warning(f"Could not parse results: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run production fusion training with outlier removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (recommended)
  python run_production_fusion.py

  # Quick test with existing cleaned dataset
  python run_production_fusion.py --skip-outlier-detection

  # Force regenerate cleaned dataset
  python run_production_fusion.py --force

  # Skip training, just prepare data
  python run_production_fusion.py --skip-training

Expected Performance:
  Kappa: 0.27 ± 0.08
  Training time: ~30 min on 8x RTX 4090

See agent_communication/fusion_fix/FUSION_FIX_GUIDE.md for details.
        """
    )

    parser.add_argument("--contamination", type=float, default=0.15,
                       help="Outlier contamination rate (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for outlier detection (default: 42)")
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Number of CV folds (default: 3)")
    parser.add_argument("--verbosity", type=int, default=2, choices=[0, 1, 2, 3],
                       help="Verbosity level (default: 2)")
    parser.add_argument("--device-mode", choices=["single", "multi"], default="multi",
                       help="GPU mode (default: multi)")
    parser.add_argument("--skip-outlier-detection", action="store_true",
                       help="Skip outlier detection (use existing cleaned dataset)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (only prepare data)")
    parser.add_argument("--force", action="store_true",
                       help="Force regenerate cleaned dataset")

    args = parser.parse_args()

    print_header("DFU FUSION PRODUCTION PIPELINE")

    print(f"{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Outlier removal: {args.contamination*100:.0f}%")
    print(f"  Random seed: {args.seed}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Verbosity: {args.verbosity}")
    print(f"  Device mode: {args.device_mode}")
    print()

    # Step 0: Check config
    check_config()

    # Step 1: Detect outliers
    if not args.skip_outlier_detection:
        detect_outliers(
            contamination=args.contamination,
            seed=args.seed,
            skip_if_exists=not args.force
        )
    else:
        print_warning("Skipping outlier detection (--skip-outlier-detection)")

    # Step 2: Apply cleaned dataset
    apply_cleaned_dataset(contamination=args.contamination)

    # Step 3: Run training
    if not args.skip_training:
        run_training(
            cv_folds=args.cv_folds,
            verbosity=args.verbosity,
            device_mode=args.device_mode
        )

        # Step 4: Show results
        show_results()
    else:
        print_warning("Skipping training (--skip-training)")

    print_header("PIPELINE COMPLETE")

    if not args.skip_training:
        print(f"\n{Colors.BOLD}Expected Results:{Colors.END}")
        print(f"  Kappa: 0.27 ± 0.08")
        print(f"  vs Baseline (no outlier removal): 0.1664")
        print(f"  Improvement: +63%")
        print()
        print(f"{Colors.BOLD}Results saved to:{Colors.END}")
        print(f"  results/csv/modality_combination_results.csv")
        print()

    print(f"{Colors.BOLD}To restore original dataset:{Colors.END}")
    print(f"  python agent_communication/fusion_fix/scripts_production/test_cleaned_data.py restore")
    print()

if __name__ == "__main__":
    main()
