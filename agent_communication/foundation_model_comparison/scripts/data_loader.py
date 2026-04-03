"""
Shared data loader for foundation model comparison experiments.
Reuses the same dataset, preprocessing, and patient-stratified CV splits
as the main FUSE4DFU pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BEST_MATCHING_CSV = os.path.join(PROJ_ROOT, 'results', 'best_matching.csv')
IMAGE_DIRS = {
    'depth_rgb': os.path.join(PROJ_ROOT, 'data', 'raw', 'Depth_RGB'),
    'depth_map': os.path.join(PROJ_ROOT, 'data', 'raw', 'Depth_Map_IMG'),
    'thermal_map': os.path.join(PROJ_ROOT, 'data', 'raw', 'Thermal_Map_IMG'),
}

MISCLASS_DIR = os.path.join(PROJ_ROOT, 'results', 'misclassifications_saved')

LABEL_MAP = {'I': 0, 'P': 1, 'R': 2}
LABEL_MAP_REV = {0: 'I', 1: 'P', 2: 'R'}
CLASS_NAMES = ['I', 'P', 'R']
N_FOLDS = 5
CV_SEED = 42

# Core data thresholds (identical to production_config.py)
USE_CORE_DATA = True
THRESHOLD_I = 53
THRESHOLD_P = 84
THRESHOLD_R = 70


def _apply_core_data_filter(df):
    """
    Remove frequently misclassified samples using the same thresholds
    as FUSE4DFU (USE_CORE_DATA=True). Matches filter_frequent_misclassifications()
    in src/evaluation/metrics.py.
    """
    thresholds = {'I': THRESHOLD_I, 'P': THRESHOLD_P, 'R': THRESHOLD_R}

    # Build sample IDs matching the format in misclassification CSVs
    df['Sample_ID'] = df.apply(
        lambda r: f"P{int(r['Patient#']):03d}A{int(r['Appt#']):02d}D{int(r['DFU#'])}",
        axis=1
    )

    # Collect sample IDs to remove from all misclassification files
    remove_ids = set()
    mc_files = [f for f in os.listdir(MISCLASS_DIR)
                if f.startswith('frequent_misclassifications_') and f.endswith('_saved.csv')]

    if not mc_files:
        print("  WARNING: No misclassification files found, skipping core data filter")
        return df

    for mc_file in mc_files:
        mc = pd.read_csv(os.path.join(MISCLASS_DIR, mc_file))
        for _, row in mc.iterrows():
            phase = row['True_Label']
            threshold = thresholds.get(phase, 999)
            if row['Misclass_Count'] >= threshold:
                remove_ids.add(row['Sample_ID'])

    before = len(df)
    df = df[~df['Sample_ID'].isin(remove_ids)].reset_index(drop=True)
    after = len(df)
    print(f"  Core data filter: {before} -> {after} samples "
          f"({before - after} removed, {after/before*100:.1f}% retained)")

    return df


def load_dataset():
    """
    Load the pre-matched best_matching.csv (3,108 multimodal samples),
    apply core data filtering, and return the same dataset as FUSE4DFU.
    """
    df = pd.read_csv(BEST_MATCHING_CSV)

    # Target labels
    df['label'] = df['Healing Phase Abs'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"Before filtering: {len(df)} samples, {df['Patient#'].nunique()} patients")

    # Apply core data filter (same as FUSE4DFU)
    if USE_CORE_DATA:
        df = _apply_core_data_filter(df)

    print(f"Dataset: {len(df)} samples, {df['Patient#'].nunique()} patients")
    print(f"  Classes: {dict(df['label'].value_counts().sort_index())}")
    return df


def get_patient_folds(df, n_folds=N_FOLDS, seed=CV_SEED):
    """Create patient-stratified CV folds matching the main pipeline."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = df['label'].values
    groups = df['Patient#'].values
    folds = list(sgkf.split(df, labels, groups))
    return folds


def adjust_depth_bb(xmin, ymin, xmax, ymax):
    """FOV adjustment for depth maps (3% correction)."""
    w = xmax - xmin
    h = ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    nw, nh = w * 1.03, h * 1.03
    return int(cx - nw/2), int(cy - nh/2), int(cx + nw/2), int(cy + nh/2)


def crop_and_resize(filepath, bb, modality, target_size=224):
    """
    Load, crop to bounding box, resize preserving aspect ratio, pad to square.
    Returns RGB numpy array of shape (target_size, target_size, 3) in [0, 255] uint8.
    """
    try:
        img = Image.open(filepath).convert('RGB')
    except Exception:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    img_w, img_h = img.size
    xmin, ymin, xmax, ymax = [int(c) for c in bb]

    if modality == 'depth_map':
        xmin, ymin, xmax, ymax = adjust_depth_bb(xmin, ymin, xmax, ymax)
    elif modality == 'thermal_map':
        xmin, ymin, xmax, ymax = xmin - 30, ymin - 30, xmax + 30, ymax + 30

    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(xmin + 1, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(ymin + 1, min(ymax, img_h))

    cropped = img.crop((xmin, ymin, xmax, ymax))
    cw, ch = cropped.size
    if cw == 0 or ch == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    aspect = cw / ch
    if aspect > 1:
        new_w = target_size
        new_h = max(1, int(target_size / aspect))
    else:
        new_h = target_size
        new_w = max(1, int(target_size * aspect))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    final = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    final.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return np.array(final, dtype=np.uint8)


def load_images(df, modality, target_size=224):
    """Load and preprocess all images for a given modality."""
    images = np.zeros((len(df), target_size, target_size, 3), dtype=np.uint8)

    # In best_matching.csv: depth_rgb, depth_map, thermal_rgb, thermal_map are filename cols
    if modality in ('depth_rgb', 'depth_map'):
        fn_col = modality  # 'depth_rgb' or 'depth_map' column holds the filename
        bb_cols = ['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']
    else:
        fn_col = modality  # 'thermal_map' column holds the filename
        bb_cols = ['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']

    img_dir = IMAGE_DIRS[modality]
    skipped = 0

    for i, row in df.iterrows():
        fn = str(row[fn_col])
        filepath = os.path.join(img_dir, fn)
        if not os.path.exists(filepath):
            skipped += 1
            continue
        bb = [row[c] for c in bb_cols]
        if any(pd.isna(b) for b in bb):
            skipped += 1
            continue
        images[i] = crop_and_resize(filepath, bb, modality, target_size)

    if skipped > 0:
        print(f"  {modality}: {skipped}/{len(df)} images skipped (missing)")
    return images


def get_metadata_features(df):
    """Extract numeric metadata features, excluding leaky/ID columns."""
    exclude_prefixes = ['treatment', 'dressing', 'offloading', 'antibiotic']
    exclude_cols = {
        'Healing Phase Abs', 'Healing Phase', 'Phase Confidence (%)',
        'label', 'target_class',
        'Patient#', 'Appt#', 'DFU#', 'ID', 'assessment_id', 'patient_id',
        'wound_id', 'visit_date', 'sample_id', 'filename',
        'depth_filename', 'thermal_filename',
        'depth_rgb_filename', 'depth_map_filename',
        'thermal_map_filename', 'thermal_rgb_filename',
        'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
        'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax',
    }

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if any(c.lower().startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype in ['float64', 'float32', 'int64', 'int32']:
            feature_cols.append(c)

    return feature_cols
