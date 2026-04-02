#!/usr/bin/env python3
"""
Export full-data trained models for inference deployment.

Retrains each of the 4 optimized ensemble combos on 100% of the training
data (no CV split) and saves one weight file per combo to inference/weights/.

This script should be run ONCE before deploying inference.py.
It also exports the RF pipeline.

Usage:
  python export_full_models.py

Requires: the project training environment with all dependencies.
Outputs:
  weights/metadata.weights.h5
  weights/metadata+depth_map.weights.h5
  weights/metadata+thermal_map.weights.h5
  weights/metadata+depth_rgb+depth_map.weights.h5
  weights/rf_pipeline.joblib

TODO: Implement full-data training loop. Currently a placeholder.
      To implement, adapt src/main.py train_single_combination() to
      use 100% of data (no validation split) with the production config
      hyperparameters. Key steps:
        1. Load and preprocess the full dataset (all 3,108 samples)
        2. Train RF on full data and export pipeline
        3. For each of the 4 combos:
           a. Build model via src/models/builders.py
           b. Stage 1: Train fusion layers (frozen backbone), 500 epochs, LR 1.72e-3
           c. Stage 2: Unfreeze top 5% backbone, 30 epochs, LR 5e-6
           d. Save weights to inference/weights/{combo_name}.weights.h5
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')

COMBOS = [
    'metadata',
    'metadata+depth_map',
    'metadata+thermal_map',
    'metadata+depth_rgb+depth_map',
]

EXPECTED_FILES = [f'{combo}.weights.h5' for combo in COMBOS] + ['rf_pipeline.joblib']


def main():
    print("=" * 70)
    print("FUSE4DFU: Export Full-Data Models for Inference")
    print("=" * 70)
    print()
    print("STATUS: Not yet implemented. This is a placeholder.")
    print()
    print("To create deployment weights, you need to retrain each combo")
    print("on 100% of the training data. Adapt the training pipeline in")
    print("src/main.py to skip cross-validation and train on all data.")
    print()
    print("Expected output files in inference/weights/:")
    for f in EXPECTED_FILES:
        path = os.path.join(WEIGHTS_DIR, f)
        exists = "EXISTS" if os.path.exists(path) else "MISSING"
        print(f"  [{exists}] {f}")
    print()

    # --- Alternatively: copy 5-fold weights for ensemble inference ---
    print("ALTERNATIVE: Use existing 5-fold CV weights.")
    print("Copy from results/models/ to inference/weights/ with:")
    print()
    models_dir = os.path.join(PROJ_ROOT, 'results', 'models')
    fold_patterns = {
        'metadata': 'metadata_{fold}_metadata.weights.h5',
        'metadata+depth_map': 'depth_map_metadata_{fold}_metadata+depth_map.weights.h5',
        'metadata+thermal_map': 'metadata_thermal_map_{fold}_metadata+thermal_map.weights.h5',
        'metadata+depth_rgb+depth_map': 'depth_map_depth_rgb_metadata_{fold}_metadata+depth_rgb+depth_map.weights.h5',
    }
    for combo, pattern in fold_patterns.items():
        for fold in range(1, 6):
            fname = pattern.format(fold=fold)
            src = os.path.join(models_dir, fname)
            exists = "EXISTS" if os.path.exists(src) else "MISSING"
            print(f"  [{exists}] {fname}")

    print()
    print("See inference/README.txt for details.")


if __name__ == '__main__':
    main()
