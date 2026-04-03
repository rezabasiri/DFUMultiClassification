#!/usr/bin/env python3
"""
DINOv2 Foundation Model Prototype for DFU Healing Phase Classification.

Approach:
  1. Extract frozen DINOv2 features from each image modality (RGB, depth, thermal)
  2. Concatenate features across modalities + optional metadata
  3. Train a lightweight classifier head (logistic regression or small MLP)
  4. Evaluate with patient-stratified 5-fold CV

DINOv2 ViT-B/14 produces 768-dim CLS embeddings per image.
With 3 modalities: 768*3 = 2304 dim feature vector per sample.

Usage:
  python run_dinov2.py
  python run_dinov2.py --modalities depth_rgb              # single modality
  python run_dinov2.py --modalities depth_rgb thermal_map  # subset
  python run_dinov2.py --include_metadata                  # add RF metadata probs
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import json
from datetime import datetime

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
LOGS_DIR = os.path.join(SCRIPT_DIR, '..', 'logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from data_loader import (load_dataset, get_patient_folds, load_images,
                          get_metadata_features, CLASS_NAMES, N_FOLDS)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (cohen_kappa_score, accuracy_score, f1_score,
                              classification_report)


def extract_dinov2_features(images, model, transform, device, batch_size=64):
    """Extract CLS token features from DINOv2 for a batch of images."""
    import torch

    n = len(images)
    features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]

        # Convert uint8 [0,255] HWC numpy to PIL-like tensors
        tensors = []
        for img in batch:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            tensors.append(transform(pil_img))

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            out = model(batch_tensor)
            # DINOv2 returns CLS token embedding directly
            features.append(out.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"    Extracted {end}/{n} features", flush=True)

    return np.concatenate(features, axis=0)


def run_experiment(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOGS_DIR, f'dinov2_{timestamp}.log')

    print("=" * 70)
    print("DINOv2 Foundation Model Experiment")
    print(f"  Modalities: {args.modalities}")
    print(f"  Include metadata: {args.include_metadata}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Model variant: {args.variant}")
    print("=" * 70)

    # --- Load data ---
    df = load_dataset()
    folds = get_patient_folds(df)
    labels = df['label'].values

    # --- Load DINOv2 ---
    import torch
    from torchvision import transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    variant_map = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14',
        'large': 'dinov2_vitl14',
    }
    model_name = variant_map[args.variant]
    print(f"Loading {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)
    model.eval()

    # DINOv2 expects 224x224, ImageNet normalized
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Extract features per modality ---
    modality_features = {}
    for mod in args.modalities:
        print(f"\nLoading {mod} images...")
        images = load_images(df, mod, target_size=224)
        print(f"  Extracting DINOv2 features...")
        feats = extract_dinov2_features(images, model, transform, device,
                                        batch_size=args.batch_size)
        modality_features[mod] = feats
        print(f"  {mod}: {feats.shape}")
        del images  # free memory

    # Concatenate all modality features
    all_features = np.concatenate(list(modality_features.values()), axis=1)
    print(f"\nConcatenated features: {all_features.shape}")

    # Optionally add metadata
    if args.include_metadata:
        meta_cols = get_metadata_features(df)
        meta_values = df[meta_cols].values.astype(np.float32)
        imputer = KNNImputer(n_neighbors=5)
        meta_values = imputer.fit_transform(meta_values)
        scaler_meta = StandardScaler()
        meta_values = scaler_meta.fit_transform(meta_values)
        all_features = np.concatenate([all_features, meta_values], axis=1)
        print(f"  + metadata ({len(meta_cols)} features) -> {all_features.shape}")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Cross-validation ---
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        X_train = all_features[train_idx]
        X_val = all_features[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train classifier
        if args.classifier == 'logreg':
            clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight='balanced',
                solver='lbfgs', multi_class='multinomial', random_state=42
            )
        elif args.classifier == 'mlp':
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 64), max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                random_state=42, learning_rate='adaptive',
                alpha=1e-3
            )
        else:
            raise ValueError(f"Unknown classifier: {args.classifier}")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val) if hasattr(clf, 'predict_proba') else None

        kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')
        acc = accuracy_score(y_val, y_pred)
        f1_per = f1_score(y_val, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

        fold_results.append({
            'fold': fold_idx + 1,
            'kappa': kappa,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_I': f1_per[0],
            'f1_P': f1_per[1],
            'f1_R': f1_per[2],
        })

        print(f"  Kappa: {kappa:.4f}  Acc: {acc:.4f}  "
              f"F1: I={f1_per[0]:.3f} P={f1_per[1]:.3f} R={f1_per[2]:.3f}")

    # --- Summary ---
    results_df = pd.DataFrame(fold_results)
    means = results_df.drop(columns='fold').mean()
    stds = results_df.drop(columns='fold').std()

    print(f"\n{'='*70}")
    print(f"DINOv2 ({args.variant}) + {args.classifier} | "
          f"Modalities: {'+'.join(args.modalities)}"
          f"{' +metadata' if args.include_metadata else ''}")
    print(f"{'='*70}")
    print(f"  Kappa:    {means['kappa']:.4f} +/- {stds['kappa']:.4f}")
    print(f"  Accuracy: {means['accuracy']:.4f} +/- {stds['accuracy']:.4f}")
    print(f"  F1-macro: {means['f1_macro']:.4f}")
    print(f"  F1-I:     {means['f1_I']:.4f}  F1-P: {means['f1_P']:.4f}  "
          f"F1-R: {means['f1_R']:.4f}")

    # Save results
    config_name = (f"dinov2_{args.variant}_{args.classifier}_"
                   f"{'_'.join(args.modalities)}"
                   f"{'_meta' if args.include_metadata else ''}")
    csv_path = os.path.join(RESULTS_DIR, f'{config_name}.csv')
    results_df.to_csv(csv_path, index=False)

    summary = {
        'model': f'DINOv2-{args.variant}',
        'classifier': args.classifier,
        'modalities': args.modalities,
        'include_metadata': args.include_metadata,
        'n_folds': N_FOLDS,
        'mean_kappa': round(float(means['kappa']), 4),
        'std_kappa': round(float(stds['kappa']), 4),
        'mean_accuracy': round(float(means['accuracy']), 4),
        'mean_f1_macro': round(float(means['f1_macro']), 4),
        'mean_f1_I': round(float(means['f1_I']), 4),
        'mean_f1_P': round(float(means['f1_P']), 4),
        'mean_f1_R': round(float(means['f1_R']), 4),
        'feature_dim': int(all_features.shape[1]),
        'timestamp': timestamp,
    }
    json_path = os.path.join(RESULTS_DIR, f'{config_name}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {csv_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='DINOv2 DFU Classification')
    parser.add_argument('--modalities', nargs='+',
                        default=['depth_rgb', 'depth_map', 'thermal_map'],
                        choices=['depth_rgb', 'depth_map', 'thermal_map'],
                        help='Image modalities to use')
    parser.add_argument('--include_metadata', action='store_true',
                        help='Include clinical metadata features')
    parser.add_argument('--classifier', default='logreg',
                        choices=['logreg', 'mlp'],
                        help='Classifier head type')
    parser.add_argument('--variant', default='base',
                        choices=['small', 'base', 'large'],
                        help='DINOv2 model size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
