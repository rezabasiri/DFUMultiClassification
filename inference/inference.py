#!/usr/bin/env python3
"""
FUSE4DFU Inference Script

Performs healing phase classification (Inflammatory / Proliferative / Remodeling)
using the optimized ensemble of 4 modality combinations averaged across 5 CV folds.

Ensemble members:
  1. Metadata (RF probabilities passed through directly)
  2. Metadata + Depth Map
  3. Metadata + Thermal Map
  4. Metadata + RGB + Depth Map

Usage:
  python inference.py --csv data.csv --image_dir ./images --output predictions.csv
  python inference.py --csv data.csv --image_dir ./images --folds 1  # single fold (faster)
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Lambda,
    GlobalAveragePooling2D, concatenate
)
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128
N_CLASSES = 3
CLASS_NAMES = ['Inflammatory', 'Proliferative', 'Remodeling']
CLASS_SHORT = ['I', 'P', 'R']
N_FOLDS = 5

# The 4 combos in the optimized ensemble
ENSEMBLE_COMBOS = [
    ('metadata',),
    ('metadata', 'depth_map'),
    ('metadata', 'thermal_map'),
    ('metadata', 'depth_rgb', 'depth_map'),
]

# Full-data weight files (single model per combo, trained on 100% data)
WEIGHT_NAMES_FULL = {
    ('metadata',): 'metadata.weights.h5',
    ('metadata', 'depth_map'): 'metadata+depth_map.weights.h5',
    ('metadata', 'thermal_map'): 'metadata+thermal_map.weights.h5',
    ('metadata', 'depth_rgb', 'depth_map'): 'metadata+depth_rgb+depth_map.weights.h5',
}

# 5-fold CV weight files (legacy: averaged at inference time)
WEIGHT_PATTERNS_FOLD = {
    ('metadata',): 'metadata_{fold}_metadata.weights.h5',
    ('metadata', 'depth_map'): 'depth_map_metadata_{fold}_metadata+depth_map.weights.h5',
    ('metadata', 'thermal_map'): 'metadata_thermal_map_{fold}_metadata+thermal_map.weights.h5',
    ('metadata', 'depth_rgb', 'depth_map'): 'depth_map_depth_rgb_metadata_{fold}_metadata+depth_rgb+depth_map.weights.h5',
}

# DenseNet121 projection head dimensions (from production_config)
HEAD_DIMS = {
    'depth_rgb': [128, 32],
    'depth_map': [128],
    'thermal_map': [128],
}


# ---------------------------------------------------------------------------
# Model building (mirrors src/models/builders.py)
# ---------------------------------------------------------------------------

def create_image_branch(input_shape, modality_name):
    """Create a DenseNet121 image branch with projection head."""
    inp = Input(shape=input_shape, name=f'{modality_name}_input')

    backbone = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet',
        input_shape=input_shape,
        name=f'densenet121_{modality_name}'
    )
    x = backbone(inp)
    x = GlobalAveragePooling2D(name=f'{modality_name}_gap')(x)

    head_layers = HEAD_DIMS.get(modality_name, [128])
    for i, dim in enumerate(head_layers):
        x = Dense(dim, activation='relu', kernel_initializer='he_normal',
                  name=f'{modality_name}_head_dense_{i}')(x)
        x = BatchNormalization(name=f'{modality_name}_head_bn_{i}')(x)
        x = Dropout(0.3, name=f'{modality_name}_head_dropout_{i}')(x)

    return inp, x


def create_metadata_branch(input_shape):
    """Metadata branch: pass RF probabilities through unchanged."""
    inp = Input(shape=input_shape, name='metadata_input')
    out = Lambda(lambda x: tf.cast(x, tf.float32), name='metadata_cast')(inp)
    return inp, out


def build_model(modalities):
    """Build a FUSE4DFU model for a given modality combination."""
    input_shapes = {
        'metadata': (N_CLASSES,),
        'depth_rgb': (IMAGE_SIZE, IMAGE_SIZE, 3),
        'depth_map': (IMAGE_SIZE, IMAGE_SIZE, 3),
        'thermal_map': (IMAGE_SIZE, IMAGE_SIZE, 3),
    }

    inputs = {}
    branches = []
    metadata_idx = None

    for i, mod in enumerate(modalities):
        if mod == 'metadata':
            metadata_idx = i
            inp, branch = create_metadata_branch(input_shapes[mod])
            inputs['metadata_input'] = inp
            branches.append(branch)
        else:
            inp, branch = create_image_branch(input_shapes[mod], mod)
            inputs[f'{mod}_input'] = inp
            branches.append(branch)

    has_metadata = metadata_idx is not None

    if len(modalities) == 1 and has_metadata:
        # Metadata only: RF probs passed through directly
        output = Lambda(lambda x: tf.identity(x), name='output')(branches[0])
    elif len(modalities) >= 2 and has_metadata:
        # Feature concatenation fusion
        rf_probs = branches[metadata_idx]
        image_branches = [b for i, b in enumerate(branches) if i != metadata_idx]
        if len(image_branches) > 1:
            image_features = concatenate(image_branches, name='concat_image_features')
        else:
            image_features = image_branches[0]
        fused = concatenate([rf_probs, image_features], name='fusion_concat')
        output = Dense(3, activation='softmax', name='output', dtype='float32',
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(fused)
    else:
        raise ValueError(f"Unsupported modality combination: {modalities}")

    return Model(inputs=inputs, outputs=output)


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors src/data/image_processing.py)
# ---------------------------------------------------------------------------

def adjust_depth_bounding_box(xmin, ymin, xmax, ymax):
    """FOV adjustment for depth maps (3% correction)."""
    width = xmax - xmin
    height = ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    new_w = width * 1.03
    new_h = height * 1.03
    return (int(cx - new_w / 2), int(cy - new_h / 2),
            int(cx + new_w / 2), int(cy + new_h / 2))


def preprocess_image(filepath, bb_coords, modality, target_size=IMAGE_SIZE):
    """Load, crop, resize, and pad an image for inference."""
    img = Image.open(filepath).convert('RGB')
    img_w, img_h = img.size
    xmin, ymin, xmax, ymax = [int(c) for c in bb_coords]

    # Modality-specific bounding box adjustment
    if modality == 'depth_map':
        xmin, ymin, xmax, ymax = adjust_depth_bounding_box(xmin, ymin, xmax, ymax)
    elif modality == 'thermal_map':
        xmin, ymin, xmax, ymax = xmin - 30, ymin - 30, xmax + 30, ymax + 30

    # Clamp to image bounds
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(xmin + 1, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(ymin + 1, min(ymax, img_h))

    # Crop
    cropped = img.crop((xmin, ymin, xmax, ymax))

    # Resize maintaining aspect ratio
    cw, ch = cropped.size
    aspect = cw / ch
    if aspect > 1:
        new_w = target_size
        new_h = max(1, int(target_size / aspect))
    else:
        new_h = target_size
        new_w = max(1, int(target_size * aspect))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center-pad to target_size x target_size
    final = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    final.paste(resized, (left, top))

    # Return as float32 array in [0, 255] range (DenseNet handles normalization)
    return np.array(final, dtype=np.float32)


# ---------------------------------------------------------------------------
# RF model loading
# ---------------------------------------------------------------------------

def load_rf_pipeline(rf_path):
    """Load a saved Random Forest pipeline (joblib)."""
    import joblib
    pipeline = joblib.load(rf_path)
    return pipeline


def get_rf_probabilities(metadata_df, rf_pipeline):
    """Generate 3-class RF probabilities from metadata features."""
    features = rf_pipeline['feature_names']
    X = metadata_df[features].values

    # Impute missing values
    X = rf_pipeline['imputer'].transform(X)
    # Scale
    X = rf_pipeline['scaler'].transform(X)
    # Feature selection
    if 'selector' in rf_pipeline:
        X = rf_pipeline['selector'].transform(X)
    # Predict probabilities
    probs = rf_pipeline['model'].predict_proba(X)
    return probs  # shape: (n_samples, 3)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def resolve_weight_files(weights_dir, combo, folds):
    """Find weight files for a combo: prefer full-data, fall back to fold weights."""
    # Check for full-data weight
    full_name = WEIGHT_NAMES_FULL[combo]
    full_path = os.path.join(weights_dir, full_name)
    if os.path.exists(full_path):
        return [('full', full_path)]

    # Fall back to fold weights
    found = []
    for fold in folds:
        fold_name = WEIGHT_PATTERNS_FOLD[combo].format(fold=fold)
        fold_path = os.path.join(weights_dir, fold_name)
        if os.path.exists(fold_path):
            found.append((f'fold{fold}', fold_path))
    return found


def run_inference(csv_path, image_dir, weights_dir, rf_pipeline_path,
                  folds=None, batch_size=32, output_path=None):
    """
    Run FUSE4DFU inference on new data.

    Parameters
    ----------
    csv_path : str
        Path to CSV with metadata features, image filenames, and bounding boxes.
    image_dir : str
        Root directory containing image subdirectories
        (Depth_RGB/, Depth_Map_IMG/, Thermal_Map_IMG/).
    weights_dir : str
        Directory containing model weight files. Accepts either:
        - Full-data weights: {combo_name}.weights.h5 (1 per combo)
        - 5-fold CV weights: {modalities}_{fold}_{combo}.weights.h5 (5 per combo)
        Full-data weights are preferred when both exist.
    rf_pipeline_path : str
        Path to saved RF pipeline (.joblib).
    folds : list of int or None
        Which fold models to use if using CV weights (1-5). None = all 5 folds.
    batch_size : int
        Batch size for neural network inference.
    output_path : str or None
        Path to save predictions CSV. None = print to stdout.
    """
    if folds is None:
        folds = list(range(1, N_FOLDS + 1))

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    n_samples = len(df)
    print(f"  {n_samples} samples")

    # --- Step 1: RF probabilities ---
    print(f"Loading RF pipeline from {rf_pipeline_path}...")
    rf_pipeline = load_rf_pipeline(rf_pipeline_path)
    rf_probs = get_rf_probabilities(df, rf_pipeline)
    print(f"  RF probabilities shape: {rf_probs.shape}")

    # --- Step 2: Preprocess images ---
    bb_depth_cols = ['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']
    bb_thermal_cols = ['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']

    image_modalities = ['depth_rgb', 'depth_map', 'thermal_map']
    image_data = {}

    for mod in image_modalities:
        print(f"  Preprocessing {mod} images...")
        images = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        for i, row in df.iterrows():
            if mod == 'depth_rgb':
                subdir = 'Depth_RGB'
                bb_cols = bb_depth_cols
            elif mod == 'depth_map':
                subdir = 'Depth_Map_IMG'
                bb_cols = bb_depth_cols
            elif mod == 'thermal_map':
                subdir = 'Thermal_Map_IMG'
                bb_cols = bb_thermal_cols

            filename = row.get(f'{mod}_filename', row.get('filename', ''))
            filepath = os.path.join(image_dir, subdir, str(filename))

            if os.path.exists(filepath) and all(c in df.columns for c in bb_cols):
                bb = [row[c] for c in bb_cols]
                images[i] = preprocess_image(filepath, bb, mod)
            else:
                images[i] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        image_data[mod] = images

    # --- Step 3: Run models and collect predictions ---
    all_predictions = []

    for combo in ENSEMBLE_COMBOS:
        combo_name = '+'.join(combo)

        if combo == ('metadata',):
            # Metadata-only: use RF probs directly
            all_predictions.append(rf_probs.copy())
            print(f"  {combo_name}: using RF probabilities directly")
            continue

        weight_files = resolve_weight_files(weights_dir, combo, folds)
        if not weight_files:
            print(f"  WARNING: No weights found for {combo_name}, skipping")
            continue

        combo_preds = []
        for label, weight_path in weight_files:
            print(f"  Loading {combo_name} ({label})...")
            model = build_model(combo)
            model.load_weights(weight_path)

            feed = {'metadata_input': rf_probs}
            for mod in combo:
                if mod != 'metadata':
                    feed[f'{mod}_input'] = image_data[mod]

            preds = model.predict(feed, batch_size=batch_size, verbose=0)
            combo_preds.append(preds)

            del model
            tf.keras.backend.clear_session()

        avg = np.mean(combo_preds, axis=0)
        all_predictions.append(avg)
        print(f"  {combo_name}: averaged {len(combo_preds)} model(s)")

    if not all_predictions:
        print("ERROR: No predictions generated. Check weight files.")
        sys.exit(1)

    # --- Step 4: Ensemble (simple average across combos) ---
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_classes = np.argmax(ensemble_probs, axis=1)

    # --- Step 5: Output ---
    results = pd.DataFrame({
        'sample_index': range(n_samples),
        'predicted_class': [CLASS_SHORT[c] for c in ensemble_classes],
        'predicted_label': [CLASS_NAMES[c] for c in ensemble_classes],
        'prob_I': ensemble_probs[:, 0],
        'prob_P': ensemble_probs[:, 1],
        'prob_R': ensemble_probs[:, 2],
        'confidence': np.max(ensemble_probs, axis=1),
    })

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
    else:
        print("\n" + results.to_string(index=False))

    print(f"\nPrediction summary:")
    for i, name in enumerate(CLASS_NAMES):
        count = (ensemble_classes == i).sum()
        print(f"  {name} ({CLASS_SHORT[i]}): {count} ({count/n_samples*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='FUSE4DFU Inference: DFU Healing Phase Classification')
    parser.add_argument('--csv', required=True,
                        help='Path to input CSV with metadata and image filenames')
    parser.add_argument('--image_dir', required=True,
                        help='Root directory containing Depth_RGB/, Depth_Map_IMG/, Thermal_Map_IMG/')
    parser.add_argument('--weights_dir', default='weights',
                        help='Directory with model weight files (default: weights/)')
    parser.add_argument('--rf_pipeline', default='weights/rf_pipeline.joblib',
                        help='Path to saved RF pipeline (default: weights/rf_pipeline.joblib)')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Which folds to use (1-5). Default: all 5')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path. If omitted, prints to stdout')

    args = parser.parse_args()

    run_inference(
        csv_path=args.csv,
        image_dir=args.image_dir,
        weights_dir=args.weights_dir,
        rf_pipeline_path=args.rf_pipeline,
        folds=args.folds,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
