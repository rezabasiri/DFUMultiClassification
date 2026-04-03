#!/usr/bin/env python3
"""
BiomedCLIP Foundation Model Prototype for DFU Healing Phase Classification.

Approach:
  1. Extract frozen BiomedCLIP image features from each modality
  2. Optionally encode clinical metadata as text and extract text features
  3. Concatenate image + text features
  4. Train a lightweight classifier head
  5. Evaluate with patient-stratified 5-fold CV

BiomedCLIP (ViT-B/16 + PubMedBERT) produces 512-dim embeddings.

Usage:
  python run_biomedclip.py
  python run_biomedclip.py --modalities depth_rgb
  python run_biomedclip.py --include_metadata          # encode metadata as text
  python run_biomedclip.py --include_metadata_numeric   # append raw metadata features
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
from sklearn.metrics import (cohen_kappa_score, accuracy_score, f1_score)


def build_metadata_text(row, feature_cols):
    """
    Convert a row of clinical metadata into a natural language description
    for BiomedCLIP text encoding.
    """
    parts = []

    # Demographics
    if 'Age' in row and not pd.isna(row.get('Age')):
        parts.append(f"Age {int(row['Age'])}")
    if 'Sex' in row and not pd.isna(row.get('Sex')):
        parts.append(f"{'male' if row['Sex'] == 1 else 'female'}")

    # Wound characteristics
    for col in feature_cols[:20]:  # Top features only to keep text concise
        val = row.get(col)
        if val is not None and not pd.isna(val):
            clean_name = col.replace('_', ' ').replace('#', '').strip()
            if isinstance(val, float) and val == int(val):
                val = int(val)
            parts.append(f"{clean_name}: {val}")

    text = "Diabetic foot ulcer patient. " + ", ".join(parts[:15]) + "."
    return text


def extract_biomedclip_image_features(images, model, processor, device,
                                       batch_size=64):
    """Extract image features from BiomedCLIP."""
    import torch
    from PIL import Image as PILImage

    n = len(images)
    features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]

        pil_images = [PILImage.fromarray(img) for img in batch]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()
                  if k.startswith('pixel')}

        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            features.append(img_features.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"    Extracted {end}/{n} image features", flush=True)

    return np.concatenate(features, axis=0)


def extract_biomedclip_text_features(texts, model, tokenizer, device,
                                      batch_size=64):
    """Extract text features from BiomedCLIP."""
    import torch

    n = len(texts)
    features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]

        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            features.append(text_features.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"    Extracted {end}/{n} text features", flush=True)

    return np.concatenate(features, axis=0)


def run_experiment(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("BiomedCLIP Foundation Model Experiment")
    print(f"  Modalities: {args.modalities}")
    print(f"  Metadata as text: {args.include_metadata}")
    print(f"  Metadata numeric: {args.include_metadata_numeric}")
    print(f"  Classifier: {args.classifier}")
    print("=" * 70)

    # --- Load data ---
    df = load_dataset()
    folds = get_patient_folds(df)
    labels = df['label'].values

    # --- Load BiomedCLIP ---
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"Loading {model_id}...")

    from open_clip import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = model.to(device)
    model.eval()

    # --- Extract image features per modality ---
    modality_features = {}
    for mod in args.modalities:
        print(f"\nLoading {mod} images...")
        images = load_images(df, mod, target_size=224)
        print(f"  Extracting BiomedCLIP image features...")

        feats = _extract_openclip_image_features(
            images, model, preprocess, device, args.batch_size)
        modality_features[mod] = feats
        print(f"  {mod}: {feats.shape}")
        del images

    all_features = np.concatenate(list(modality_features.values()), axis=1)
    print(f"\nConcatenated image features: {all_features.shape}")

    # Optionally add metadata as text embeddings
    if args.include_metadata:
        meta_cols = get_metadata_features(df)
        print(f"\nEncoding metadata as text ({len(meta_cols)} features)...")
        texts = [build_metadata_text(row, meta_cols) for _, row in df.iterrows()]
        text_feats = _extract_openclip_text_features(
            texts, model, tokenizer, device, args.batch_size)
        all_features = np.concatenate([all_features, text_feats], axis=1)
        print(f"  + text features -> {all_features.shape}")

    # Optionally add raw metadata numeric features
    if args.include_metadata_numeric:
        meta_cols = get_metadata_features(df)
        meta_values = df[meta_cols].values.astype(np.float32)
        imputer = KNNImputer(n_neighbors=5)
        meta_values = imputer.fit_transform(meta_values)
        scaler_meta = StandardScaler()
        meta_values = scaler_meta.fit_transform(meta_values)
        all_features = np.concatenate([all_features, meta_values], axis=1)
        print(f"  + numeric metadata ({len(meta_cols)}) -> {all_features.shape}")

    # Free GPU
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

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        if args.classifier == 'logreg':
            clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight='balanced',
                solver='lbfgs', multi_class='multinomial', random_state=42
            )
        elif args.classifier == 'mlp':
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 64), max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                random_state=42, learning_rate='adaptive', alpha=1e-3
            )
        else:
            raise ValueError(f"Unknown classifier: {args.classifier}")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')
        acc = accuracy_score(y_val, y_pred)
        f1_per = f1_score(y_val, y_pred, labels=[0, 1, 2], average=None,
                          zero_division=0)
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
    print(f"BiomedCLIP | Modalities: {'+'.join(args.modalities)}"
          f"{'  +text_meta' if args.include_metadata else ''}"
          f"{'  +num_meta' if args.include_metadata_numeric else ''}")
    print(f"{'='*70}")
    print(f"  Kappa:    {means['kappa']:.4f} +/- {stds['kappa']:.4f}")
    print(f"  Accuracy: {means['accuracy']:.4f} +/- {stds['accuracy']:.4f}")
    print(f"  F1-macro: {means['f1_macro']:.4f}")
    print(f"  F1-I:     {means['f1_I']:.4f}  F1-P: {means['f1_P']:.4f}  "
          f"F1-R: {means['f1_R']:.4f}")

    # Save
    config_name = (f"biomedclip_{args.classifier}_"
                   f"{'_'.join(args.modalities)}"
                   f"{'_textmeta' if args.include_metadata else ''}"
                   f"{'_nummeta' if args.include_metadata_numeric else ''}")
    csv_path = os.path.join(RESULTS_DIR, f'{config_name}.csv')
    results_df.to_csv(csv_path, index=False)

    summary = {
        'model': 'BiomedCLIP',
        'classifier': args.classifier,
        'modalities': args.modalities,
        'include_metadata_text': args.include_metadata,
        'include_metadata_numeric': args.include_metadata_numeric,
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


def _extract_openclip_image_features(images, model, preprocess, device,
                                      batch_size=64):
    """Extract image features using open_clip API."""
    import torch
    from PIL import Image as PILImage

    n = len(images)
    features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]

        tensors = []
        for img in batch:
            pil_img = PILImage.fromarray(img)
            tensors.append(preprocess(pil_img))
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            img_features = model.encode_image(batch_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            features.append(img_features.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"    Extracted {end}/{n} image features", flush=True)

    return np.concatenate(features, axis=0)


def _extract_openclip_text_features(texts, model, tokenizer, device,
                                     batch_size=64):
    """Extract text features using open_clip API."""
    import torch

    n = len(texts)
    features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]

        tokens = tokenizer(batch).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            features.append(text_features.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"    Extracted {end}/{n} text features", flush=True)

    return np.concatenate(features, axis=0)


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP DFU Classification')
    parser.add_argument('--modalities', nargs='+',
                        default=['depth_rgb', 'depth_map', 'thermal_map'],
                        choices=['depth_rgb', 'depth_map', 'thermal_map'],
                        help='Image modalities to use')
    parser.add_argument('--include_metadata', action='store_true',
                        help='Encode metadata as text via BiomedCLIP text encoder')
    parser.add_argument('--include_metadata_numeric', action='store_true',
                        help='Append raw numeric metadata features')
    parser.add_argument('--classifier', default='logreg',
                        choices=['logreg', 'mlp'],
                        help='Classifier head type')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
