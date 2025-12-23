#!/usr/bin/env python3
"""DEBUG 1: Verify raw data loads correctly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def main():
    output = []
    def log(msg):
        print(msg)
        output.append(msg)

    log("="*80)
    log("DEBUG 1: DATA SANITY CHECK")
    log("="*80)

    try:
        # Load balanced data
        log("\n1. Loading balanced data...")
        data_path = 'balanced_combined_healing_phases.csv'

        if not os.path.exists(data_path):
            log(f"❌ FAIL: File not found: {data_path}")
            return False, output

        data = pd.read_csv(data_path)
        log(f"✓ Loaded: {len(data)} rows, {len(data.columns)} columns")

        # Check columns
        log("\n2. Checking required columns...")
        required = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
        missing = [col for col in required if col not in data.columns]

        if missing:
            log(f"❌ FAIL: Missing columns: {missing}")
            return False, output

        log("✓ All required columns present")

        # Check labels
        log("\n3. Checking labels...")
        labels = data['Healing Phase Abs'].values
        unique_labels = np.unique(labels)
        log(f"  Unique labels: {unique_labels}")

        if not set(unique_labels) == {0, 1, 2}:
            log(f"❌ FAIL: Expected [0,1,2], got {unique_labels}")
            return False, output

        for cls in [0, 1, 2]:
            count = np.sum(labels == cls)
            pct = count / len(labels) * 100
            log(f"  Class {cls}: {count} samples ({pct:.1f}%)")

        # Check features
        log("\n4. Checking features...")
        exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        log(f"  Found {len(feature_cols)} numeric features")
        log(f"  First 10: {feature_cols[:10].tolist()}")

        if len(feature_cols) == 0:
            log("❌ FAIL: No features found!")
            return False, output

        # Prepare data
        X = data[feature_cols].fillna(0).values

        log(f"\n5. Feature statistics:")
        log(f"  Shape: {X.shape}")
        log(f"  Min: {X.min():.4f}")
        log(f"  Max: {X.max():.4f}")
        log(f"  Mean: {X.mean():.4f}")
        log(f"  Std: {X.std():.4f}")

        if np.isnan(X).any() or np.isinf(X).any():
            log("❌ FAIL: NaN or Inf in features!")
            return False, output

        log("\n" + "="*80)
        log("✅ PASS: Data sanity check passed")
        log("="*80)

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
