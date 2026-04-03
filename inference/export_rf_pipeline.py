#!/usr/bin/env python3
"""
Export the Random Forest pipeline for inference.

Trains an RF model on the full training dataset (not OOF) and saves
the complete pipeline (imputer, scaler, feature selector, RF model)
as a single .joblib file for use with inference.py.

Usage:
  python export_rf_pipeline.py --csv ../data/raw/DataMaster_Processed_V12_WithMissing.csv \
                                --output weights/rf_pipeline.joblib
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# RF parameters (from production_config.py)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 5
RF_MIN_SAMPLES_SPLIT = 10
RF_FEATURE_SELECTION_K = 80

# Columns to exclude (treatment info causes data leakage)
EXCLUDE_PREFIXES = ['treatment', 'dressing', 'offloading', 'antibiotic']
EXCLUDE_COLS = ['target_class', 'Healing Phase Abs', 'Healing Phase', 'Phase Confidence (%)',
                'label',
                'assessment_id', 'patient_id', 'Patient#', 'ID', 'filename',
                'depth_rgb_filename', 'depth_map_filename', 'thermal_map_filename',
                'thermal_rgb_filename', 'wound_id', 'visit_date', 'sample_id',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
TARGET_COL = 'Healing Phase Abs'


def get_metadata_features(df):
    """Extract numeric metadata feature columns, excluding leaky and ID columns."""
    exclude = set(EXCLUDE_COLS)
    for col in df.columns:
        for prefix in EXCLUDE_PREFIXES:
            if col.lower().startswith(prefix):
                exclude.add(col)

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    return feature_cols


def export_pipeline(csv_path, output_path):
    """Train and export the RF pipeline."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        print(f"ERROR: '{TARGET_COL}' column not found in CSV.")
        sys.exit(1)

    # Map target classes
    class_map = {'I': 0, 'P': 1, 'R': 2,
                 'Inflammatory': 0, 'Proliferative': 1, 'Remodeling': 2}
    if df[TARGET_COL].dtype == object:
        df['label'] = df[TARGET_COL].map(class_map)
    else:
        df['label'] = df[TARGET_COL]

    df = df.dropna(subset=['label'])
    y = df['label'].astype(int).values
    print(f"  {len(df)} samples, classes: {np.bincount(y)}")

    # Get metadata features
    feature_cols = get_metadata_features(df)
    X = df[feature_cols].values.astype(np.float32)
    print(f"  {len(feature_cols)} candidate features")

    # Handle missing values
    missing_pct = np.isnan(X).mean() * 100
    print(f"  Missing values: {missing_pct:.1f}%")
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Feature selection via Mutual Information
    k = min(RF_FEATURE_SELECTION_K, X.shape[1])
    print(f"  Selecting top {k} features via Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_indices = np.argsort(mi_scores)[-k:]
    selected_features = [feature_cols[i] for i in top_indices]
    X_selected = X[:, top_indices]
    print(f"  Selected {len(selected_features)} features")

    # Train RF on full dataset
    print(f"  Training Random Forest (n_estimators={RF_N_ESTIMATORS})...")
    # Compute class weights
    counts = np.bincount(y)
    total = len(y)
    class_weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_selected, y)

    train_acc = rf.score(X_selected, y)
    print(f"  Training accuracy: {train_acc:.3f}")

    # Save pipeline
    pipeline = {
        'feature_names': selected_features,
        'all_feature_names': feature_cols,
        'selected_indices': top_indices.tolist(),
        'imputer': imputer,
        'scaler': scaler,
        'model': rf,
        'class_names': ['Inflammatory', 'Proliferative', 'Remodeling'],
        'class_short': ['I', 'P', 'R'],
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    joblib.dump(pipeline, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nPipeline saved to {output_path} ({size_mb:.1f} MB)")
    print(f"  Features: {len(selected_features)}")
    print(f"  RF estimators: {RF_N_ESTIMATORS}")


def main():
    parser = argparse.ArgumentParser(
        description='Export FUSE4DFU Random Forest Pipeline for Inference')
    parser.add_argument('--csv', required=True,
                        help='Path to training CSV with metadata and target_class')
    parser.add_argument('--output', default='weights/rf_pipeline.joblib',
                        help='Output path for the pipeline file')
    args = parser.parse_args()
    export_pipeline(args.csv, args.output)


if __name__ == '__main__':
    main()
