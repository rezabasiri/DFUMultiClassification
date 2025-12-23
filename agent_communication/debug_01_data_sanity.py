#!/usr/bin/env python3
"""DEBUG 1: Verify main.py's data loading pipeline works"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths

def main():
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 1: DATA LOADING TEST (Using main.py's actual pipeline)")
    log("="*80)

    try:
        # Use same paths as main.py
        directory, result_dir, root = get_project_paths()

        log("\n1. Project paths:")
        log(f"  Directory: {directory}")
        log(f"  Root: {root}")

        # File paths (same as main.py)
        depth_bb_file = os.path.join(root, 'raw', 'bb_depth_annotation.csv')
        thermal_bb_file = os.path.join(root, 'raw', 'bb_thermal_annotation.csv')
        csv_file = os.path.join(root, 'raw', 'DataMaster_Processed_V12_WithMissing.csv')

        log("\n2. Checking raw data files:")
        for name, path in [("Depth BB", depth_bb_file),
                           ("Thermal BB", thermal_bb_file),
                           ("CSV", csv_file)]:
            exists = os.path.exists(path)
            log(f"  {name}: {'✓ EXISTS' if exists else '✗ MISSING'}")
            if not exists:
                log(f"    Path: {path}")
                return False, output

        log("\n3. Loading data via prepare_dataset() (metadata only)...")
        log("   This is the ACTUAL method main.py uses...")

        data = prepare_dataset(
            depth_bb_file=depth_bb_file,
            thermal_bb_file=thermal_bb_file,
            csv_file=csv_file,
            selected_modalities=['metadata']
        )

        log(f"\n4. Data loaded successfully:")
        log(f"  Shape: {data.shape}")
        log(f"  Columns: {len(data.columns)}")

        # Check for label column
        if 'Healing Phase Abs' not in data.columns:
            log(f"\n  ❌ FAIL: Missing 'Healing Phase Abs' column!")
            log(f"  Available columns: {list(data.columns[:20])}")
            return False, output

        # Check labels
        labels = data['Healing Phase Abs'].values
        unique_labels = np.unique(labels)

        log(f"\n5. Labels:")
        log(f"  Count: {len(labels)}")
        log(f"  Unique values: {unique_labels}")

        if not set(unique_labels).issubset({0, 1, 2}):
            log(f"  ❌ FAIL: Unexpected label values: {unique_labels}")
            return False, output

        # Class distribution
        log(f"\n6. Class distribution:")
        for cls in [0, 1, 2]:
            count = np.sum(labels == cls)
            pct = count / len(labels) * 100 if len(labels) > 0 else 0
            log(f"  Class {cls}: {count} samples ({pct:.1f}%)")

        # Check features
        exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        log(f"\n7. Features:")
        log(f"  Found {len(feature_cols)} numeric features")
        log(f"  First 10: {feature_cols[:10].tolist()}")

        if len(feature_cols) == 0:
            log("  ❌ FAIL: No features found!")
            return False, output

        # Prepare feature matrix
        X = data[feature_cols].fillna(0).values

        log(f"\n8. Feature matrix:")
        log(f"  Shape: {X.shape}")
        log(f"  Min: {X.min():.4f}")
        log(f"  Max: {X.max():.4f}")
        log(f"  Mean: {X.mean():.4f}")
        log(f"  Std: {X.std():.4f}")

        # Check for invalid values
        if np.isnan(X).any():
            log("  ❌ FAIL: NaN values in feature matrix!")
            return False, output

        if np.isinf(X).any():
            log("  ❌ FAIL: Inf values in feature matrix!")
            return False, output

        log("\n" + "="*80)
        log("✅ PASS: Data loads correctly via main.py pipeline")
        log("="*80)
        log(f"\nSummary: {len(data)} samples, {len(feature_cols)} features, 3 classes")

        return True, output

    except Exception as e:
        log(f"\n❌ EXCEPTION: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False, output

if __name__ == "__main__":
    success, output = main()

    # Save output
    output_file = 'agent_communication/results_01_data_sanity.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    print(f"\n✓ Results saved to: {output_file}")
    sys.exit(0 if success else 1)
