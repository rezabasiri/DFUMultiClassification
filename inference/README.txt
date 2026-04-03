FUSE4DFU Inference
==================

Classifies diabetic foot ulcer healing phase: Inflammatory (I), Proliferative (P), or Remodeling (R).

Uses a 4-model ensemble: metadata-only, metadata+depth_map, metadata+thermal_map, metadata+RGB+depth_map.


Quick Start (Demo)
------------------

  python inference.py --demo demo_data.npz --output predictions.csv

  This runs on 292 pre-packed demo samples. No images or CSV needed.


Full Inference (New Data)
-------------------------

  python inference.py \
    --csv input_data.csv \
    --image_dir /path/to/images \
    --output predictions.csv

  The CSV must contain:
    - Numeric clinical metadata columns (demographics, wound scores, temperatures, etc.)
    - Image filename columns: depth_rgb, depth_map, thermal_map
    - Bounding boxes: depth_xmin/ymin/xmax/ymax, thermal_xmin/ymin/xmax/ymax

  The image_dir must contain subdirectories:
    Depth_RGB/        RGB wound images
    Depth_Map_IMG/    Depth map images
    Thermal_Map_IMG/  Thermal map images


Output
------

  predictions.csv columns:
    sample_index      Row number (0-based)
    predicted_class   I, P, or R
    predicted_label   Inflammatory, Proliferative, or Remodeling
    prob_I            Probability of Inflammatory
    prob_P            Probability of Proliferative
    prob_R            Probability of Remodeling
    confidence        Highest probability among the three classes
    true_class        Ground truth label (demo mode only)

  Terminal also prints a summary table with per-class counts and accuracy.


Files
-----

  inference.py            Main script (run this)
  demo_data.npz           Pre-packed demo dataset (292 samples, binary)
  export_rf_pipeline.py   Re-export RF pipeline if retraining
  weights/
    rf_pipeline.joblib    Random Forest pipeline (metadata classifier)
    *.weights.h5          Neural network weights (4 ensemble models)


Requirements
------------

  Python 3.10+, tensorflow >= 2.15, scikit-learn >= 1.4, numpy, pandas, Pillow, joblib
