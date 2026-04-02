FUSE4DFU Inference
==================

Healing phase classification for diabetic foot ulcers using the optimized
4-combination ensemble: metadata, metadata+depth_map, metadata+thermal_map,
and metadata+RGB+depth_map.

Output: 3-class prediction (Inflammatory / Proliferative / Remodeling) with
per-class probabilities.


Directory Structure
-------------------

inference/
  inference.py            Main inference script
  export_rf_pipeline.py   Export RF model from training data
  export_full_models.py   Retrain models on full data (placeholder)
  README.txt              This file
  weights/                Model weight files (created by user)
    rf_pipeline.joblib      Random Forest pipeline
    *.weights.h5            Neural network weights


Prerequisites
-------------

Python 3.10+
  tensorflow >= 2.15
  scikit-learn >= 1.4
  numpy
  pandas
  Pillow
  joblib


Setup (Before First Use)
------------------------

Step 1: Export the RF pipeline

  python export_rf_pipeline.py \
    --csv ../data/raw/DataMaster_Processed_V12_WithMissing.csv \
    --output weights/rf_pipeline.joblib

Step 2: Provide neural network weights (choose one option)

  Option A: Full-data weights (recommended for deployment)
    Run export_full_models.py after implementing the full-data training
    loop (see placeholder TODO in that file). This produces:
      weights/metadata.weights.h5
      weights/metadata+depth_map.weights.h5
      weights/metadata+thermal_map.weights.h5
      weights/metadata+depth_rgb+depth_map.weights.h5

  Option B: Use existing 5-fold CV weights
    Copy from results/models/ into inference/weights/:
      metadata_1_metadata.weights.h5 ... metadata_5_metadata.weights.h5
      depth_map_metadata_1_metadata+depth_map.weights.h5 ... (x5)
      metadata_thermal_map_1_metadata+thermal_map.weights.h5 ... (x5)
      depth_map_depth_rgb_metadata_1_metadata+depth_rgb+depth_map.weights.h5 ... (x5)

    The inference script auto-detects which format is available.
    Full-data weights are preferred when both exist.


Running Inference
-----------------

  python inference.py \
    --csv input_data.csv \
    --image_dir /path/to/images \
    --output predictions.csv

  Arguments:
    --csv           Input CSV (required). Must contain metadata feature columns
                    and image filename/bounding box columns.
    --image_dir     Root directory with image subdirectories (required).
                    Expected subdirectories:
                      Depth_RGB/        RGB images from depth camera
                      Depth_Map_IMG/    Depth map images
                      Thermal_Map_IMG/  Thermal map images
    --weights_dir   Directory with weight files (default: weights/)
    --rf_pipeline   Path to RF pipeline joblib (default: weights/rf_pipeline.joblib)
    --folds         Which CV folds to use, e.g. --folds 1 2 3 (default: all 5)
                    Ignored when full-data weights are present.
    --batch_size    Batch size for NN inference (default: 32)
    --output        Output CSV path. Omit to print to stdout.


Input CSV Format
----------------

The input CSV must contain:

  Metadata columns:
    Numeric clinical features (demographics, wound scores, temperature
    measurements, etc.). The RF pipeline selects the 80 most informative
    features automatically. Missing values are imputed via KNN.

  Image filename columns (at least one of):
    depth_rgb_filename    Filename of the RGB image in Depth_RGB/
    depth_map_filename    Filename of the depth map in Depth_Map_IMG/
    thermal_map_filename  Filename of the thermal map in Thermal_Map_IMG/

    If a single 'filename' column exists, it is used for all modalities.

  Bounding box columns:
    depth_xmin, depth_ymin, depth_xmax, depth_ymax
    thermal_xmin, thermal_ymin, thermal_xmax, thermal_ymax


Output Format
-------------

  sample_index     Row index (0-based)
  predicted_class  Short label: I, P, or R
  predicted_label  Full label: Inflammatory, Proliferative, or Remodeling
  prob_I           Probability of Inflammatory class
  prob_P           Probability of Proliferative class
  prob_R           Probability of Remodeling class
  confidence       Maximum probability (highest of prob_I, prob_P, prob_R)


Image Preprocessing
-------------------

Images are preprocessed identically to training:
  1. Load image as RGB
  2. Crop to bounding box with modality-specific adjustments:
       depth_map: 3% FOV expansion
       thermal_map: 30-pixel symmetric padding
       depth_rgb: no adjustment
  3. Resize to 128x128 preserving aspect ratio (Lanczos interpolation)
  4. Center-pad with black to 128x128
  5. Keep [0, 255] float32 range (DenseNet121 handles normalization internally)


Tunable Parameters
------------------

  Parameter            Default    Description
  ---------            -------    -----------
  IMAGE_SIZE           128        Input image resolution (must match training)
  N_FOLDS              5          Number of CV folds (for fold-based weights)
  batch_size           32         NN inference batch size (reduce if OOM)
  RF_N_ESTIMATORS      300        RF trees (only for export_rf_pipeline.py)
  RF_MAX_DEPTH         10         RF max depth (only for export_rf_pipeline.py)
  RF_FEATURE_SEL_K     80         Number of MI-selected features

  These are set in the scripts and generally should not be changed unless
  retraining with different hyperparameters.


Model Architecture Summary
--------------------------

  Metadata branch:
    Random Forest (300 trees, depth 10, 80 features) -> 3-class probabilities
    Passed through to fusion layer unchanged (no neural layers).

  Image branches (per modality):
    DenseNet121 (ImageNet pretrained) -> GlobalAvgPool -> projection head
    Projection head sizes:
      depth_rgb:   Dense(128) -> Dense(32)
      depth_map:   Dense(128)
      thermal_map: Dense(128)

  Fusion:
    Concatenate [RF probs (3-dim), image features] -> Dense(3, softmax)

  Ensemble:
    Simple average of predictions from the 4 modality combinations.


Performance (5-fold patient-stratified CV)
------------------------------------------

  Optimized Ensemble (4 combos):
    Accuracy: 80.6%
    Cohen's kappa: 0.58
    F1-I: 0.73, F1-P: 0.86, F1-R: 0.59

  Best single combination (metadata + RGB + thermal):
    Accuracy: 71.2%
    Cohen's kappa: 0.61

  Classes: Inflammatory (23%), Proliferative (63%), Remodeling (13%)
