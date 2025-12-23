#!/usr/bin/env python3
"""DEBUG 3: Test if main.py loads data correctly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths

def main():
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 3: MAIN.PY DATA LOADING TEST")
    log("="*80)

    try:
        # Get paths (same as main.py)
        directory, result_dir, root = get_project_paths()
        log(f"\nProject paths:")
        log(f"  Directory: {directory}")
        log(f"  Root: {root}")

        # File paths
        depth_bb_file = os.path.join(root, 'raw', 'bb_depth_annotation.csv')
        thermal_bb_file = os.path.join(root, 'raw', 'bb_thermal_annotation.csv')
        csv_file = os.path.join(root, 'raw', 'DataMaster_Processed_V12_WithMissing.csv')

        log(f"\n1. Checking files exist:")
        log(f"  Depth BB: {os.path.exists(depth_bb_file)}")
        log(f"  Thermal BB: {os.path.exists(thermal_bb_file)}")
        log(f"  CSV: {os.path.exists(csv_file)}")

        # Load data using main.py's pipeline
        log(f"\n2. Loading data via prepare_dataset()...")
        data = prepare_dataset(
            depth_bb_file=depth_bb_file,
            thermal_bb_file=thermal_bb_file,
            csv_file=csv_file,
            selected_modalities=['metadata']
        )

        log(f"  Shape: {data.shape}")
        log(f"  Columns: {len(data.columns)}")
        log(f"  First 10 columns: {list(data.columns[:10])}")

        # Check labels
        log(f"\n3. Checking labels:")
        if 'Healing Phase Abs' not in data.columns:
            log(f"  ❌ FAIL: No 'Healing Phase Abs' column!")
            log(f"  Available columns: {list(data.columns)}")
            return False, output

        labels = data['Healing Phase Abs'].values
        log(f"  Count: {len(labels)}")
        log(f"  Unique: {np.unique(labels)}")
        log(f"  Distribution: {np.bincount(labels.astype(int))}")

        # Check features
        log(f"\n4. Checking features:")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
        feature_cols = [c for c in numeric_cols if c not in exclude]

        log(f"  Found {len(feature_cols)} numeric features")
        log(f"  First 10: {feature_cols[:10].tolist()}")

        if len(feature_cols) == 0:
            log("  ❌ FAIL: No features found!")
            return False, output

        features = data[feature_cols].fillna(0).values
        log(f"  Feature matrix shape: {features.shape}")
        log(f"  Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")

        # Check alignment
        if len(features) != len(labels):
            log(f"  ❌ FAIL: Feature/label mismatch: {len(features)} vs {len(labels)}")
            return False, output

        log(f"\n" + "="*80)
        log(f"✅ PASS: Main.py data loading works")
        log(f"="*80)

        return True, output

    except Exception as e:
        log(f"\n❌ EXCEPTION: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False, output

if __name__ == "__main__":
    success, output = main()

    # Save output
    output_file = 'agent_communication/results_03_main_data_loading.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    print(f"\n✓ Results saved to: {output_file}")
    sys.exit(0 if success else 1)
