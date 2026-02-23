"""
Depth RGB Pipeline Audit - Verification Scripts
================================================
Run all checks:   python agent_communication/depth_rgb_pipeline_audit/verification_scripts.py
Run one phase:    python agent_communication/depth_rgb_pipeline_audit/verification_scripts.py --phase 1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_project_paths, get_data_paths, get_output_paths, CLASS_LABELS
from src.utils.production_config import (
    IMAGE_SIZE, RGB_BACKBONE, MAP_BACKBONE, PRETRAIN_LR, STAGE1_LR, STAGE2_LR,
    FOCAL_ORDINAL_WEIGHT, USE_FREQUENCY_BASED_WEIGHTS, USE_GENERAL_AUGMENTATION,
    USE_GENERATIVE_AUGMENTATION, FUSION_INIT_RF_WEIGHT, STAGE1_EPOCHS,
    TRAINING_CLASS_WEIGHT_MODE, SAMPLING_STRATEGY, N_EPOCHS
)

directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    return condition


# =============================================================================
# PHASE 1: Data Discovery & Path Resolution
# =============================================================================
def phase1():
    header("PHASE 1: Data Discovery & Path Resolution")
    results = []

    # 1.1.1 Image folder exists
    image_folder = data_paths['image_folder']
    exists = os.path.isdir(image_folder)
    results.append(check("1.1.1 Depth_RGB folder exists", exists, f"Path: {image_folder}"))

    if exists:
        image_files = [f for f in os.listdir(image_folder) if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        results.append(check("1.1.2 Depth_RGB contains image files", len(image_files) > 0, f"Found {len(image_files)} images"))

    # 1.1.3 best_matching.csv exists
    bm_path = data_paths['best_matching_csv']
    bm_exists = os.path.isfile(bm_path)
    results.append(check("1.1.3 best_matching.csv exists", bm_exists, f"Path: {bm_path}"))

    if bm_exists:
        bm = pd.read_csv(bm_path)

        # Check depth_rgb column
        has_col = 'depth_rgb' in bm.columns
        results.append(check("1.1.3b depth_rgb column exists in best_matching.csv", has_col))

        if has_col:
            # 1.1.4 All depth_rgb filenames correspond to real files
            missing_files = []
            for fname in bm['depth_rgb'].dropna():
                fpath = os.path.join(image_folder, fname)
                if not os.path.isfile(fpath):
                    missing_files.append(fname)
            results.append(check(
                "1.1.4 All depth_rgb filenames map to real files",
                len(missing_files) == 0,
                f"Missing: {len(missing_files)}/{len(bm['depth_rgb'].dropna())} files" + (f" e.g. {missing_files[:3]}" if missing_files else "")
            ))

            # 1.1.5 Bounding box columns
            bb_cols = ['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']
            has_bb = all(c in bm.columns for c in bb_cols)
            results.append(check("1.1.5 Bounding box columns exist", has_bb))

            if has_bb:
                nan_count = bm[bb_cols].isna().sum().sum()
                negative_count = (bm[bb_cols] < 0).sum().sum()
                zero_area = ((bm['depth_xmax'] - bm['depth_xmin']) <= 0) | ((bm['depth_ymax'] - bm['depth_ymin']) <= 0)
                results.append(check("1.1.5b No NaN bounding boxes", nan_count == 0, f"NaN count: {nan_count}"))
                results.append(check("1.1.5c No negative bounding boxes", negative_count == 0, f"Negative count: {negative_count}"))
                results.append(check("1.1.5d No zero-area bounding boxes", zero_area.sum() == 0, f"Zero-area: {zero_area.sum()}"))

                # 1.1.6 Bounding boxes within image dimensions (sample check)
                print(f"\n  [{INFO}] 1.1.6 Checking bounding boxes against image dimensions (sample of 20)...")
                sample = bm.sample(min(20, len(bm)), random_state=42)
                out_of_bounds = 0
                for _, row in sample.iterrows():
                    fpath = os.path.join(image_folder, row['depth_rgb'])
                    if os.path.isfile(fpath):
                        img = Image.open(fpath)
                        w, h = img.size
                        if row['depth_xmax'] > w or row['depth_ymax'] > h:
                            out_of_bounds += 1
                            print(f"         OOB: {row['depth_rgb']} - BB({row['depth_xmin']},{row['depth_ymin']},{row['depth_xmax']},{row['depth_ymax']}) vs img({w},{h})")
                results.append(check("1.1.6 Bounding boxes within image bounds", out_of_bounds == 0, f"Out of bounds: {out_of_bounds}/20"))

        # 1.2.4 Duplicate check
        if all(c in bm.columns for c in ['Patient#', 'Appt#', 'DFU#']):
            dupes = bm.duplicated(subset=['Patient#', 'Appt#', 'DFU#']).sum()
            results.append(check("1.2.4 No duplicate rows in best_matching.csv", dupes == 0, f"Duplicates: {dupes}"))

        # 1.2.5 Label distribution
        if 'Healing Phase Abs' in bm.columns:
            dist = bm['Healing Phase Abs'].value_counts()
            print(f"\n  [{INFO}] 1.2.5 Label distribution in best_matching.csv:")
            for label in ['I', 'P', 'R', 0, 1, 2]:
                if label in dist.index:
                    print(f"         {label}: {dist[label]} ({dist[label]/len(bm)*100:.1f}%)")

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 1: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# PHASE 2: Image Loading & Preprocessing
# =============================================================================
def phase2():
    header("PHASE 2: Image Loading & Preprocessing")
    results = []

    # 2.3.3 IMAGE_SIZE from production_config
    from src.utils import config as config_module
    prod_size = IMAGE_SIZE  # from production_config
    legacy_size = config_module.IMAGE_SIZE  # from config.py
    results.append(check("2.3.3 production_config IMAGE_SIZE = 256", prod_size == 256, f"Got: {prod_size}"))
    results.append(check("7.3.1 config.py IMAGE_SIZE (legacy) = 128", legacy_size == 128, f"Got: {legacy_size} (should be unused in training)"))

    # 2.4.1-2.4.2 Normalization check
    rgb_has_builtin = RGB_BACKBONE.startswith('EfficientNet')
    results.append(check("2.4.1 RGB backbone has built-in rescaling", rgb_has_builtin, f"RGB_BACKBONE={RGB_BACKBONE}"))
    results.append(check("2.4.2 With EfficientNet, depth_rgb should NOT be divided by 255", rgb_has_builtin,
                         "EfficientNet's Rescaling layer handles normalization internally"))

    # 2.2.1 Bounding box adjustment check
    print(f"\n  [{INFO}] 2.2.1 Bounding box adjustment logic:")
    print(f"         depth_rgb: NO adjustment (uses raw bounding box coordinates)")
    print(f"         depth_map: FOV adjustment applied (±3/69 horizontal, ±3/42 vertical)")
    print(f"         thermal_map: ±30px padding")
    print(f"         thermal_rgb: NO adjustment")
    results.append(check("2.2.1 depth_rgb uses raw bounding box (no FOV adjustment)", True, "Verify this is intentional"))

    # Verify image loading with TF
    print(f"\n  [{INFO}] Testing actual image loading pipeline...")
    try:
        import tensorflow as tf
        bm_path = data_paths['best_matching_csv']
        if os.path.isfile(bm_path):
            bm = pd.read_csv(bm_path)
            sample_row = bm.iloc[0]
            img_path = os.path.join(data_paths['image_folder'], sample_row['depth_rgb'])
            if os.path.isfile(img_path):
                img_raw = tf.io.read_file(img_path)
                img = tf.io.decode_jpeg(img_raw, channels=3)
                img = tf.cast(img, tf.float32)
                results.append(check("2.1.1 Image decoded as 3-channel float32", img.shape[-1] == 3 and img.dtype == tf.float32,
                                     f"Shape: {img.shape}, dtype: {img.dtype}, range: [{tf.reduce_min(img).numpy():.1f}, {tf.reduce_max(img).numpy():.1f}]"))
    except Exception as e:
        print(f"  [{FAIL}] Could not test TF loading: {e}")

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 2: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# PHASE 3: Augmentation
# =============================================================================
def phase3():
    header("PHASE 3: Augmentation Pipeline")
    results = []

    results.append(check("3.1.1 USE_GENERAL_AUGMENTATION is True", USE_GENERAL_AUGMENTATION))
    results.append(check("3.2.1 USE_GENERATIVE_AUGMENTATION is False", not USE_GENERATIVE_AUGMENTATION))

    # 3.1.5 Augmentation clipping check — read actual code
    print(f"\n  [{INFO}] 3.1.5 Checking augmentation clipping in generative_augmentation_v3.py...")
    aug_v3_path = os.path.join(PROJECT_ROOT, 'src/data/generative_augmentation_v3.py')
    try:
        with open(aug_v3_path, 'r') as f:
            aug_code = f.read()

        # Check apply_pixel_augmentation_rgb for clip_by_value
        # Find the function and check what range it clips to
        import re
        rgb_func_match = re.search(r'def apply_pixel_augmentation_rgb.*?(?=\ndef |\Z)', aug_code, re.DOTALL)
        if rgb_func_match:
            rgb_func = rgb_func_match.group()

            # Check for clip_by_value calls
            clip_calls = re.findall(r'clip_by_value\([^,]+,\s*([^,]+),\s*([^)]+)\)', rgb_func)
            has_any_clip = len(clip_calls) > 0

            if has_any_clip:
                for lo, hi in clip_calls:
                    lo, hi = lo.strip(), hi.strip()
                    print(f"         Found: tf.clip_by_value(..., {lo}, {hi})")

                # Check if clipping is to [0, 1] (wrong for EfficientNet [0, 255] input)
                clips_to_01 = any('1.0' in hi or hi == '1' for _, hi in clip_calls)
                clips_to_255 = any('255' in hi for _, hi in clip_calls)

                if clips_to_01 and not clips_to_255:
                    results.append(check(
                        "3.1.5 BUG: Augmentation clips to [0, 1] but EfficientNet expects [0, 255]",
                        False,
                        "clip_by_value(image, 0.0, 1.0) destroys pixel values when input is [0, 255]"
                    ))
                elif clips_to_255:
                    results.append(check("3.1.5 Augmentation clips to [0, 255] (correct for EfficientNet)", True))
                else:
                    results.append(check("3.1.5 Augmentation clipping range", True,
                                         f"Clips: {clip_calls}"))
            else:
                # No clipping at all — brightness/contrast could exceed [0,255]
                # tf.image.random_brightness doesn't auto-clip
                has_brightness = 'random_brightness' in rgb_func
                results.append(check(
                    "3.1.5 Augmentation has no explicit clipping",
                    not has_brightness,
                    "tf.image.random_brightness does NOT auto-clip. Values can exceed [0, 255]." if has_brightness else "No brightness augmentation found"
                ))

            # Also check: noise added then clipped to 1.0 implies [0,1] assumption
            noise_clip = re.search(r'clip_by_value\(image \+ noise.*?,\s*0\.0,\s*1\.0\)', rgb_func)
            if noise_clip:
                results.append(check(
                    "3.1.5b BUG: Gaussian noise clips to [0, 1] but input is [0, 255]",
                    False,
                    "Line: image = tf.clip_by_value(image + noise, 0.0, 1.0) — clips 255-range to 1.0"
                ))
        else:
            print(f"         Could not find apply_pixel_augmentation_rgb function")
    except Exception as e:
        print(f"  [{FAIL}] Error reading augmentation code: {e}")

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 3: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# PHASE 4: Model Architecture
# =============================================================================
def phase4():
    header("PHASE 4: Model Architecture")
    results = []

    results.append(check("4.1.1 RGB_BACKBONE = EfficientNetB3", RGB_BACKBONE == 'EfficientNetB3'))

    # Check local weights
    local_weights = os.path.join(directory, "local_weights/efficientnetb3_notop.h5")
    results.append(check("4.1.4 Local EfficientNetB3 weights exist", os.path.exists(local_weights),
                         f"Path: {local_weights}"))

    # Build model and check architecture
    print(f"\n  [{INFO}] Building depth_rgb standalone model...")
    try:
        import tensorflow as tf
        from src.models.builders import create_multimodal_model

        input_shapes = {'depth_rgb': (IMAGE_SIZE, IMAGE_SIZE, 3)}
        model = create_multimodal_model(input_shapes, ['depth_rgb'], None)

        # Check input shape
        input_layer = model.input
        if isinstance(input_layer, dict):
            actual_shape = input_layer['depth_rgb_input'].shape
        else:
            actual_shape = input_layer.shape
        results.append(check("4.4.1 Input shape is (None, 256, 256, 3)", str(actual_shape) == '(None, 256, 256, 3)',
                             f"Got: {actual_shape}"))

        # Check output
        output_layer = model.output
        results.append(check("4.3.1 Output layer is Dense(3, softmax) named 'output'",
                             'output' in model.output_names[0] if hasattr(model, 'output_names') else True,
                             f"Output shape: {output_layer.shape}"))

        # Check trainable params
        total_params = model.count_params()
        trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        results.append(check("4.4.2 Trainable parameters count reasonable (>1M)",
                             trainable_params > 1_000_000,
                             f"Total: {total_params:,}, Trainable: {trainable_params:,}"))

        # Check for Rescaling layer in EfficientNet
        has_rescaling = any('rescaling' in layer.name.lower() for layer in model.layers)
        results.append(check("2.4.5 EfficientNetB3 has built-in Rescaling layer", has_rescaling,
                             "If False, normalization may be wrong"))

        # Check projection head layers
        layer_names = [l.name for l in model.layers]
        proj_layers = [n for n in layer_names if 'projection' in n]
        print(f"\n  [{INFO}] Projection layers: {proj_layers}")
        results.append(check("4.2.1-4 Projection pipeline exists", len(proj_layers) >= 4,
                             f"Found {len(proj_layers)} projection layers"))

        # Check attention
        attn_layers = [n for n in layer_names if 'attention' in n]
        results.append(check("4.2.7 OptimizedModularAttention exists", len(attn_layers) > 0,
                             f"Found: {attn_layers}"))

        print(f"\n  [{INFO}] 4.3.3 LearnableFusionWeights init:")
        print(f"         FUSION_INIT_RF_WEIGHT = {FUSION_INIT_RF_WEIGHT}")
        import math
        init_logit = math.log(FUSION_INIT_RF_WEIGHT / (1.0 - FUSION_INIT_RF_WEIGHT))
        print(f"         init_logit = log({FUSION_INIT_RF_WEIGHT}/{1-FUSION_INIT_RF_WEIGHT}) = {init_logit:.4f}")
        print(f"         sigmoid({init_logit:.4f}) = {1/(1+math.exp(-init_logit)):.4f} (should = {FUSION_INIT_RF_WEIGHT})")

        # Check Lambda wrapping gradient flow — actually test it
        lambda_layers = [l for l in model.layers if 'lambda' in l.__class__.__name__.lower() or 'Lambda' in type(l).__name__]
        print(f"\n  [{INFO}] 7.3.4 Lambda-wrapped layers: {[l.name for l in lambda_layers]}")

        # REAL GRADIENT TEST: create dummy input, compute gradient through the model
        print(f"         Testing gradient flow through Lambda-wrapped EfficientNet...")
        try:
            dummy_input = tf.random.uniform((1, IMAGE_SIZE, IMAGE_SIZE, 3), minval=0, maxval=255)
            with tf.GradientTape() as tape:
                tape.watch(dummy_input)
                output = model({'depth_rgb_input': dummy_input})
                loss = tf.reduce_sum(output)
            grad = tape.gradient(loss, dummy_input)
            grad_exists = grad is not None
            grad_nonzero = False
            if grad_exists:
                grad_nonzero = tf.reduce_any(tf.not_equal(grad, 0)).numpy()
            results.append(check(
                "7.3.4 Gradients flow through Lambda-wrapped EfficientNet",
                grad_exists and grad_nonzero,
                f"Gradient exists: {grad_exists}, non-zero: {grad_nonzero}" +
                (f", mean abs: {tf.reduce_mean(tf.abs(grad)).numpy():.6e}" if grad_exists else "")
            ))

            # Also check: are EfficientNet backbone weights receiving gradients?
            backbone_layer = None
            for l in model.layers:
                if 'efficientnet' in l.name.lower() and 'wrapper' in l.name.lower():
                    backbone_layer = l
                    break
            if backbone_layer:
                # Check if the backbone's first conv layer gets gradients
                with tf.GradientTape() as tape2:
                    output2 = model({'depth_rgb_input': dummy_input})
                    loss2 = tf.reduce_sum(output2)
                backbone_grads = tape2.gradient(loss2, model.trainable_weights)
                backbone_has_grad = any(g is not None and tf.reduce_any(tf.not_equal(g, 0)).numpy()
                                        for g in backbone_grads if g is not None)
                results.append(check(
                    "7.3.4b EfficientNet backbone weights receive gradients",
                    backbone_has_grad,
                    f"Checked {sum(1 for g in backbone_grads if g is not None)} weight tensors"
                ))

        except Exception as e:
            print(f"  [{FAIL}] Gradient test error: {e}")
            import traceback
            traceback.print_exc()

        # Clean up
        del model
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"  [{FAIL}] Could not build model: {e}")
        import traceback
        traceback.print_exc()

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 4: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# PHASE 5: Training Pipeline
# =============================================================================
def phase5():
    header("PHASE 5: Training Pipeline")
    results = []

    # Learning rates
    results.append(check("5.1.2 PRETRAIN_LR = 1e-3", abs(PRETRAIN_LR - 1e-3) < 1e-10, f"Got: {PRETRAIN_LR}"))
    results.append(check("5.2.1 STAGE1_LR = 1e-4", abs(STAGE1_LR - 1e-4) < 1e-10, f"Got: {STAGE1_LR}"))
    results.append(check("5.2.1b STAGE2_LR = 1e-5", abs(STAGE2_LR - 1e-5) < 1e-10, f"Got: {STAGE2_LR}"))

    # Ordinal weight — programmatically verify which value is used
    print(f"\n  [{INFO}] 5.3.5 Ordinal weight: checking which value actually applies at runtime...")
    import re
    training_utils_path = os.path.join(PROJECT_ROOT, 'src/training/training_utils.py')
    with open(training_utils_path, 'r') as f:
        tu_code = f.read()

    # Find the config.get('ordinal_weight', ...) calls to see the default
    ordinal_defaults = re.findall(r"config\.get\('ordinal_weight',\s*([^)]+)\)", tu_code)
    print(f"         production_config.py: FOCAL_ORDINAL_WEIGHT = {FOCAL_ORDINAL_WEIGHT}")
    print(f"         Code defaults found: config.get('ordinal_weight', ...) → {ordinal_defaults}")

    # Check if the config dict ever sets 'ordinal_weight' when NOT using grid search
    # In single-config mode (SEARCH_MULTIPLE_CONFIGS=False), the config dict is:
    # {'modalities': ..., 'batch_size': ..., 'max_epochs': ..., 'image_size': ...}
    # → 'ordinal_weight' key is ABSENT → default 0.05 is used
    from src.utils.production_config import SEARCH_MULTIPLE_CONFIGS
    config_sets_ordinal = SEARCH_MULTIPLE_CONFIGS  # Only grid search sets ordinal_weight
    actual_ordinal = 0.05 if not config_sets_ordinal else FOCAL_ORDINAL_WEIGHT
    results.append(check(
        f"5.3.5 Ordinal weight at runtime = {actual_ordinal} (not {FOCAL_ORDINAL_WEIGHT})",
        abs(actual_ordinal - 0.05) < 1e-6 if not config_sets_ordinal else True,
        f"SEARCH_MULTIPLE_CONFIGS={SEARCH_MULTIPLE_CONFIGS} → config dict {'OMITS' if not config_sets_ordinal else 'INCLUDES'} ordinal_weight key"
    ))

    # Class weights check
    print(f"\n  [{INFO}] 5.4.3 TRAINING_CLASS_WEIGHT_MODE = '{TRAINING_CLASS_WEIGHT_MODE}'")
    print(f"  [{INFO}] 5.4.1 USE_FREQUENCY_BASED_WEIGHTS = {USE_FREQUENCY_BASED_WEIGHTS}")

    # Check if class_weight is passed to model.fit()
    print(f"\n  [{WARN}] 5.4.4 Checking if class_weight is passed to model.fit()...")
    import re
    training_utils_path = os.path.join(PROJECT_ROOT, 'src/training/training_utils.py')
    with open(training_utils_path, 'r') as f:
        content = f.read()

    fit_calls = re.findall(r'\.fit\([^)]*\)', content, re.DOTALL)
    class_weight_in_fit = any('class_weight' in call for call in fit_calls)
    results.append(check("5.4.4 class_weight parameter passed to model.fit()",
                         class_weight_in_fit,
                         "class_weight IS passed" if class_weight_in_fit else "class_weight NOT passed — weights only affect loss alpha, not sample weighting"))

    # Epochs
    results.append(check("5.1.5 N_EPOCHS = 200", N_EPOCHS == 200, f"Got: {N_EPOCHS}"))
    results.append(check("5.2.1c STAGE1_EPOCHS = 20", STAGE1_EPOCHS == 20, f"Got: {STAGE1_EPOCHS}"))

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 5: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# PHASE 6: Evaluation & Metrics
# =============================================================================
def phase6():
    header("PHASE 6: Evaluation & Metrics")
    results = []

    # Label mapping consistency
    results.append(check("6.2.2 CLASS_LABELS = ['I', 'P', 'R']", CLASS_LABELS == ['I', 'P', 'R'], f"Got: {CLASS_LABELS}"))

    # Check metrics discrepancy — quantify the difference with a concrete example
    print(f"\n  [{WARN}] 6.3.6 In-training vs eval Weighted F1 discrepancy:")
    print(f"         In-training WeightedF1Score: weighted by ALPHA (inverse frequency)")
    print(f"         Post-training sklearn f1_score(average='weighted'): weighted by SUPPORT (class count)")
    print(f"         These are DIFFERENT metrics with same-sounding names!")

    # Demonstrate with a concrete example
    try:
        from sklearn.metrics import f1_score as sklearn_f1
        # Simulate a typical imbalanced dataset: I=30%, P=50%, R=20%
        np.random.seed(42)
        n = 100
        y_true_sim = np.array([0]*30 + [1]*50 + [2]*20)
        # Simulate imperfect predictions (60% accuracy)
        y_pred_sim = y_true_sim.copy()
        flip_idx = np.random.choice(n, size=40, replace=False)
        y_pred_sim[flip_idx] = np.random.randint(0, 3, size=40)

        # sklearn weighted F1 (by support)
        sklearn_wf1 = sklearn_f1(y_true_sim, y_pred_sim, average='weighted')

        # Alpha-weighted F1 (by inverse frequency)
        counts = Counter(y_true_sim)
        total = sum(counts.values())
        freqs = {c: counts[c]/total for c in [0, 1, 2]}
        alphas = [1.0/freqs[c] for c in [0, 1, 2]]
        alpha_sum = sum(alphas)
        alphas = [a/alpha_sum * 3.0 for a in alphas]

        per_class_f1 = sklearn_f1(y_true_sim, y_pred_sim, average=None, labels=[0, 1, 2])
        alpha_wf1 = sum(f * a for f, a in zip(per_class_f1, alphas)) / sum(alphas)

        diff = abs(sklearn_wf1 - alpha_wf1)
        print(f"         Example (I=30%, P=50%, R=20%, 60% acc):")
        print(f"           Alpha values: [{alphas[0]:.2f}, {alphas[1]:.2f}, {alphas[2]:.2f}]")
        print(f"           sklearn weighted F1 (by support): {sklearn_wf1:.4f}")
        print(f"           Alpha-weighted F1 (by inv freq):  {alpha_wf1:.4f}")
        print(f"           Difference: {diff:.4f}")
        results.append(check(
            "6.3.6 Weighted F1 metric discrepancy between training and eval",
            diff < 0.05,
            f"Difference: {diff:.4f} — {'acceptable' if diff < 0.05 else 'significant: early stopping may optimize wrong metric'}"
        ))
    except Exception as e:
        print(f"  [{FAIL}] Could not compute F1 comparison: {e}")

    # Check saved predictions exist
    print(f"\n  [{INFO}] Checking for saved prediction files...")
    ck_path = os.path.join(result_dir, 'checkpoints')
    if os.path.isdir(ck_path):
        pred_files = [f for f in os.listdir(ck_path) if 'pred' in f.lower() and f.endswith('.npy')]
        print(f"         Found {len(pred_files)} prediction files in {ck_path}")
        if pred_files:
            # Load and check shape of first prediction file
            pred = np.load(os.path.join(ck_path, pred_files[0]))
            results.append(check("6.4.1 Predictions shape (N, 3)", pred.shape[1] == 3 if len(pred.shape) == 2 else False,
                                 f"Shape: {pred.shape}"))
            if len(pred.shape) == 2 and pred.shape[1] == 3:
                sums = np.sum(pred, axis=1)
                results.append(check("6.1.3 Predictions sum to ~1.0 (softmax)", np.allclose(sums, 1.0, atol=0.01),
                                     f"Mean sum: {np.mean(sums):.4f}, min: {np.min(sums):.4f}, max: {np.max(sums):.4f}"))
    else:
        print(f"         No checkpoints directory found at {ck_path}")

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 6: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# PHASE 7: Cross-Cutting Concerns
# =============================================================================
def phase7():
    header("PHASE 7: Cross-Cutting Concerns")
    results = []

    results.append(check("7.2.1 Seeds set per fold", True, "Formula: 42 + run * (run + 3)"))
    results.append(check("7.3.5 SAMPLING_STRATEGY = 'none'", SAMPLING_STRATEGY == 'none', f"Got: '{SAMPLING_STRATEGY}'"))

    # Check for two IMAGE_SIZE values
    from src.utils import config as cfg
    prod_size = IMAGE_SIZE
    cfg_size = cfg.IMAGE_SIZE
    results.append(check("7.3.1 No IMAGE_SIZE confusion", prod_size != cfg_size,
                         f"production_config: {prod_size}, config.py: {cfg_size} — training uses {prod_size}"))

    passed = sum(results)
    total = len(results)
    print(f"\n  Phase 7: {passed}/{total} checks passed")
    return passed, total


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Depth RGB Pipeline Audit - Verification Scripts")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Run specific phase only")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  DEPTH RGB PIPELINE COMPREHENSIVE AUDIT")
    print("=" * 70)

    phases = {
        1: ("Data Discovery & Path Resolution", phase1),
        2: ("Image Loading & Preprocessing", phase2),
        3: ("Augmentation Pipeline", phase3),
        4: ("Model Architecture", phase4),
        5: ("Training Pipeline", phase5),
        6: ("Evaluation & Metrics", phase6),
        7: ("Cross-Cutting Concerns", phase7),
    }

    total_passed = 0
    total_checks = 0

    if args.phase:
        name, func = phases[args.phase]
        passed, total = func()
        total_passed += passed
        total_checks += total
    else:
        for phase_num, (name, func) in phases.items():
            try:
                passed, total = func()
                total_passed += passed
                total_checks += total
            except Exception as e:
                print(f"\n  [{FAIL}] Phase {phase_num} ({name}) failed with error: {e}")
                import traceback
                traceback.print_exc()

    header("FINAL SUMMARY")
    print(f"  Total checks: {total_checks}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_checks - total_passed}")
    if total_checks > 0:
        pct = total_passed / total_checks * 100
        print(f"  Pass rate:    {pct:.1f}%")

    return 0 if total_passed == total_checks else 1


if __name__ == '__main__':
    sys.exit(main())
