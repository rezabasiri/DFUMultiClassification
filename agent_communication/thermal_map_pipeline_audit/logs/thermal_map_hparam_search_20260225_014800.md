Logging to: /workspace/DFUMultiClassification/agent_communication/thermal_map_pipeline_audit/logs/thermal_map_hparam_search_20260225_014800.log
================================================================================
THERMAL MAP HYPERPARAMETER SEARCH
================================================================================

============================================================
FILTERING SUMMARY
============================================================
Thresholds: I=18, P=16, R=26

Excluded samples per class:
  Class I: 65 samples
  Class P: 143 samples
  Class R: 8 samples

Total unique samples to exclude: 216

Dataset size (rows): 3108 -> 2072 (66.7%)
Unique samples: 648 -> 432 (removed 216)

Class distribution after filtering:
  Class I: 595 rows
  Class P: 1181 rows
  Class R: 296 rows
============================================================

Loaded 2072 samples for thermal_map
RESUME MODE — no previous results found, starting from scratch

################################################################################
ROUND 1: BACKBONE + FREEZE STRATEGY
################################################################################

================================================================================
CONFIG [thermal_map]: R1_EfficientNetB0_frozen
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3634 at epoch 23/38
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3642 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3634, acc=0.4304, f1=0.4180
  Confusion matrix:
[[134  54  11]
 [161 118 115]
 [ 14  38  45]]
  Time: 225s

================================================================================
CONFIG [thermal_map]: R1_EfficientNetB0_partial_unfreeze
  backbone=EfficientNetB0, freeze=partial_unfreeze
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=0
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 48/320
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 50: early stopping
Restoring model weights from the end of the best epoch: 35.
  Stage 1 best: val_kappa=0.3511 at epoch 35/50
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3487, acc=0.4029, f1=0.3942
  Confusion matrix:
[[137  50  12]
 [175  94 125]
 [ 17  33  47]]
  Time: 215s

================================================================================
CONFIG [thermal_map]: R1_EfficientNetB2_frozen
  backbone=EfficientNetB2, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.4111 at epoch 16/31
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.4043 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4043, acc=0.4507, f1=0.4451
  Confusion matrix:
[[139  48  12]
 [145 114 135]
 [ 15  24  58]]
  Time: 217s

================================================================================
CONFIG [thermal_map]: R1_EfficientNetB2_partial_unfreeze
  backbone=EfficientNetB2, freeze=partial_unfreeze
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=0
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 68/448
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.3857 at epoch 16/31
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3822, acc=0.4159, f1=0.4110
  Confusion matrix:
[[148  32  19]
 [180  78 136]
 [ 15  21  61]]
  Time: 149s

================================================================================
CONFIG [thermal_map]: R1_DenseNet121_frozen
  backbone=DenseNet121, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 1 best: val_kappa=0.3690 at epoch 19/34
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3748 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3748, acc=0.3971, f1=0.3917
  Confusion matrix:
[[150  31  18]
 [198  64 132]
 [ 15  22  60]]
  Time: 216s

================================================================================
CONFIG [thermal_map]: R1_DenseNet121_partial_unfreeze
  backbone=DenseNet121, freeze=partial_unfreeze
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=0
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 78/612
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.3519 at epoch 9/24
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3519, acc=0.3725, f1=0.3678
  Confusion matrix:
[[139  24  36]
 [154  46 194]
 [ 15  10  72]]
  Time: 136s

================================================================================
CONFIG [thermal_map]: R1_ResNet50V2_frozen
  backbone=ResNet50V2, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 25.
  Stage 1 best: val_kappa=0.3796 at epoch 25/40
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.3671 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3671, acc=0.4406, f1=0.4313
  Confusion matrix:
[[154  34  11]
 [199 100  95]
 [ 21  26  50]]
  Time: 236s

================================================================================
CONFIG [thermal_map]: R1_ResNet50V2_partial_unfreeze
  backbone=ResNet50V2, freeze=partial_unfreeze
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=0
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 42/278
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.3800 at epoch 5/20
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3800, acc=0.4203, f1=0.4069
  Confusion matrix:
[[176  13  10]
 [233  54 107]
 [ 29   8  60]]
  Time: 94s

  BEST from this round: R1_EfficientNetB2_frozen (kappa=0.4043, s1_kappa=0.4111)

################################################################################
ROUND 2: HEAD ARCHITECTURE
################################################################################

================================================================================
CONFIG [thermal_map]: R2_tiny
  backbone=EfficientNetB2, freeze=frozen
  head=[32], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.3855 at epoch 7/22
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.3850 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3850, acc=0.3739, f1=0.3582
  Confusion matrix:
[[162  21  16]
 [201  32 161]
 [ 20  13  64]]
  Time: 180s

================================================================================
CONFIG [thermal_map]: R2_small
  backbone=EfficientNetB2, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.3988 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3839 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3839, acc=0.3754, f1=0.3471
  Confusion matrix:
[[182   7  10]
 [256  12 126]
 [ 28   4  65]]
  Time: 174s

================================================================================
CONFIG [thermal_map]: R2_medium
  backbone=EfficientNetB2, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.4157 at epoch 16/31
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.3971 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3952, acc=0.4493, f1=0.4308
  Confusion matrix:
[[149  44   6]
 [165 119 110]
 [ 15  40  42]]
  Time: 220s

================================================================================
CONFIG [thermal_map]: R2_large
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 37: early stopping
Restoring model weights from the end of the best epoch: 22.
  Stage 1 best: val_kappa=0.4301 at epoch 22/37
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3995 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3995, acc=0.4826, f1=0.4661
  Confusion matrix:
[[111  76  12]
 [ 99 170 125]
 [  8  37  52]]
  Time: 221s

================================================================================
CONFIG [thermal_map]: R2_two_layer
  backbone=EfficientNetB2, freeze=frozen
  head=[128, 32], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4001 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3928 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3907, acc=0.4043, f1=0.3982
  Confusion matrix:
[[151  28  20]
 [171  58 165]
 [ 19   8  70]]
  Time: 186s

  BEST from this round: R2_large (kappa=0.3995, s1_kappa=0.4301)

################################################################################
ROUND 3: LOSS + REGULARIZATION
################################################################################

================================================================================
CONFIG [thermal_map]: R3_focal_g2_d03
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4141 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.3917 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3924, acc=0.4362, f1=0.4293
  Confusion matrix:
[[146  43  10]
 [166 100 128]
 [ 18  24  55]]
  Time: 210s

================================================================================
CONFIG [thermal_map]: R3_focal_g3_d03
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.4251 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3922 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3922, acc=0.3696, f1=0.3492
  Confusion matrix:
[[162  18  19]
 [197  18 179]
 [ 22   0  75]]
  Time: 151s

================================================================================
CONFIG [thermal_map]: R3_cce_d03
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=cce, gamma=0.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.3929 at epoch 6/21
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3966 at epoch 4/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3959, acc=0.3899, f1=0.3793
  Confusion matrix:
[[157  24  18]
 [179  45 170]
 [ 18  12  67]]
  Time: 167s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_d05
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.5, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.3902 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 2 best: val_kappa=0.3928 at epoch 10/20
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3933, acc=0.4000, f1=0.3918
  Confusion matrix:
[[157  31  11]
 [195  59 140]
 [ 19  18  60]]
  Time: 210s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_d02
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.2, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.4024 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3939 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3939, acc=0.3739, f1=0.3481
  Confusion matrix:
[[169  11  19]
 [195  16 183]
 [ 24   0  73]]
  Time: 151s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_ls01
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.1, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.3929 at epoch 8/23
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 2 best: val_kappa=0.3804 at epoch 18/28
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3808, acc=0.3884, f1=0.3789
  Confusion matrix:
[[156  27  16]
 [197  51 146]
 [ 18  18  61]]
  Time: 237s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_l2_1e3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.001
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 37: early stopping
Restoring model weights from the end of the best epoch: 22.
  Stage 1 best: val_kappa=0.3820 at epoch 22/37
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3846 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3846, acc=0.4449, f1=0.4372
  Confusion matrix:
[[132  57  10]
 [144 122 128]
 [ 16  28  53]]
  Time: 223s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_mixup02
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.859, 0.432, 1.709]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.4070 at epoch 17/32
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3863 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3863, acc=0.4203, f1=0.4182
  Confusion matrix:
[[153  40   6]
 [203  78 113]
 [ 23  15  59]]
  Time: 212s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha0
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=0.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (UNIFORM — no class weighting): [1.0, 1.0, 1.0]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4402 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3982 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3982, acc=0.5652, f1=0.4918
  Confusion matrix:
[[ 81 116   2]
 [ 55 275  64]
 [  3  60  34]]
  Time: 187s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha1
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.4418 at epoch 13/19
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.4107 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4107, acc=0.4594, f1=0.4501
  Confusion matrix:
[[141  49   9]
 [136 122 136]
 [ 16  27  54]]
  Time: 182s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha5
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.431, 0.72, 2.848]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3893 at epoch 15/30
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.3899 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3899, acc=0.4014, f1=0.3907
  Confusion matrix:
[[159  28  12]
 [188  60 146]
 [ 20  19  58]]
  Time: 218s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha8
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=8.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=8.0): [2.29, 1.152, 4.557]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.4019 at epoch 5/20
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3815 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3815, acc=0.3768, f1=0.3654
  Confusion matrix:
[[156  25  18]
 [194  38 162]
 [ 19  12  66]]
  Time: 167s

  BEST from this round: R3_focal_g2_alpha1 (kappa=0.4107, s1_kappa=0.4418)

################################################################################
ROUND 4: TRAINING DYNAMICS
################################################################################

================================================================================
CONFIG [thermal_map]: R4_lr5e4_b64_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.4083 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3932 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3918, acc=0.3971, f1=0.4000
  Confusion matrix:
[[132  53  14]
 [154  70 170]
 [ 18   7  72]]
  Time: 163s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b64_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4042 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4050 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4053, acc=0.4333, f1=0.4280
  Confusion matrix:
[[148  36  15]
 [142  89 163]
 [ 18  17  62]]
  Time: 188s

================================================================================
CONFIG [thermal_map]: R4_lr3e3_b64_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.003, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.4093 at epoch 7/22
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4097 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4097, acc=0.4043, f1=0.3936
  Confusion matrix:
[[167  23   9]
 [213  45 136]
 [ 22   8  67]]
  Time: 163s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b32_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.4162 at epoch 2/17
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4329 at epoch 1/11

  POST-EVAL: kappa=0.4329, acc=0.4652, f1=0.4431
  Confusion matrix:
[[171  18  10]
 [175 105 114]
 [ 12  40  45]]
  Time: 147s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b64_cosine
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 1 best: val_kappa=0.3964 at epoch 1/21
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4002 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4002, acc=0.3638, f1=0.3346
  Confusion matrix:
[[166   3  30]
 [165   6 223]
 [ 18   0  79]]
  Time: 159s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b64_e100
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.4228 at epoch 4/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3974 at epoch 11/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3969, acc=0.4333, f1=0.4329
  Confusion matrix:
[[101  86  12]
 [102 134 158]
 [  8  25  64]]
  Time: 171s

================================================================================
CONFIG [thermal_map]: R4_adamw_wd1e4
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.0001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.4004 at epoch 5/25
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.3832 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3832, acc=0.4203, f1=0.4165
  Confusion matrix:
[[139  47  13]
 [158  93 143]
 [ 17  22  58]]
  Time: 206s

================================================================================
CONFIG [thermal_map]: R4_adamw_wd1e3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.4001 at epoch 4/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4489 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4489, acc=0.4261, f1=0.4249
  Confusion matrix:
[[151  34  14]
 [162  67 165]
 [ 11  10  76]]
  Time: 172s

  BEST from this round: R4_adamw_wd1e3 (kappa=0.4489, s1_kappa=0.4001)

################################################################################
ROUND 5: AUGMENTATION + IMAGE SIZE
################################################################################

================================================================================
CONFIG [thermal_map]: R5_aug_on_128
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 25.
  Stage 1 best: val_kappa=0.4253 at epoch 25/40
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3990 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3990, acc=0.4290, f1=0.4239
  Confusion matrix:
[[137  50  12]
 [149 104 141]
 [ 12  30  55]]
  Time: 108s

================================================================================
CONFIG [thermal_map]: R5_aug_off_128
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4367 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4110 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4110, acc=0.4304, f1=0.4342
  Confusion matrix:
[[108  70  21]
 [ 94 122 178]
 [  2  28  67]]
  Time: 86s

================================================================================
CONFIG [thermal_map]: R5_aug_on_256
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4033 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4009 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4009, acc=0.3710, f1=0.3524
  Confusion matrix:
[[166  19  14]
 [201  26 167]
 [ 18  15  64]]
  Time: 171s

================================================================================
CONFIG [thermal_map]: R5_aug_off_256
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.4337 at epoch 20/35
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3802 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3802, acc=0.4449, f1=0.4415
  Confusion matrix:
[[123  61  15]
 [140 125 129]
 [ 13  25  59]]
  Time: 195s

  BEST from this round: R5_aug_off_128 (kappa=0.4110, s1_kappa=0.4367)

################################################################################
ROUND 6: FINE-TUNING STRATEGY
################################################################################

================================================================================
CONFIG [thermal_map]: R6_ft_top20_30ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4367 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4110 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4110, acc=0.4304, f1=0.4342
  Confusion matrix:
[[108  70  21]
 [ 94 122 178]
 [  2  28  67]]
  Time: 78s

================================================================================
CONFIG [thermal_map]: R6_ft_top40_30ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4367 at epoch 9/24
  Stage 2: unfreezing top 40% (137/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3969 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3966, acc=0.4246, f1=0.4263
  Confusion matrix:
[[105  73  21]
 [ 91 126 177]
 [  2  33  62]]
  Time: 83s

================================================================================
CONFIG [thermal_map]: R6_ft_top50_50ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.5, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4367 at epoch 9/24
  Stage 2: unfreezing top 50% (171/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4292 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4292, acc=0.4333, f1=0.4385
  Confusion matrix:
[[115  64  20]
 [100 115 179]
 [  1  27  69]]
  Time: 87s

================================================================================
CONFIG [thermal_map]: R6_ft_top20_50ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.57]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4367 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4165 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4157, acc=0.4319, f1=0.4292
  Confusion matrix:
[[126  60  13]
 [126 116 152]
 [  5  36  56]]
  Time: 81s

  BEST from this round: R6_ft_top50_50ep (kappa=0.4292, s1_kappa=0.4367)

################################################################################
TOP 5 SELECTION: 5-FOLD VALIDATION OF BEST CONFIGS
################################################################################

Top 5 configs by fold-0 kappa:
  1. R4_adamw_wd1e3 → kappa=0.4489
  2. R4_lr1e3_b32_plateau → kappa=0.4329
  3. R6_ft_top50_50ep → kappa=0.4292
  4. R6_ft_top20_50ep → kappa=0.4157
  5. R5_aug_off_128 → kappa=0.4110

  BEST from this round: R4_adamw_wd1e3 (kappa=0.4489, s1_kappa=0.4001)

================================================================================
CONFIG [thermal_map]: TOP5_1_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.143, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4544 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4442 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3659, acc=0.3696, f1=0.3463
  Confusion matrix:
[[171  17  11]
 [219  30 145]
 [ 26  17  54]]
  Time: 198s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.283, 0.143, 0.575]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 37: early stopping
Restoring model weights from the end of the best epoch: 22.
  Stage 1 best: val_kappa=0.4686 at epoch 22/37
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.4638 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4638, acc=0.5000, f1=0.5039
  Confusion matrix:
[[82 30  7]
 [97 85 55]
 [ 3 17 42]]
  Time: 258s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.3989 at epoch 24/39
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 2 best: val_kappa=0.4065 at epoch 13/23
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4065, acc=0.4517, f1=0.4493
  Confusion matrix:
[[76 34  9]
 [80 74 83]
 [ 6 15 37]]
  Time: 318s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 1 best: val_kappa=0.5027 at epoch 19/34
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.5030 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.5030, acc=0.4843, f1=0.4897
  Confusion matrix:
[[82 32  6]
 [79 71 85]
 [ 1 10 47]]
  Time: 263s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.284, 0.144, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 45: early stopping
Restoring model weights from the end of the best epoch: 30.
  Stage 1 best: val_kappa=0.4027 at epoch 30/45
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3907 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3907, acc=0.3850, f1=0.3783
  Confusion matrix:
[[ 92  19   7]
 [125  33  78]
 [  8  17  34]]
  Time: 295s

  BEST from this round: R4_lr1e3_b32_plateau (kappa=0.4329, s1_kappa=0.4162)

================================================================================
CONFIG [thermal_map]: TOP5_2_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.143, 0.572]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.4757 at epoch 8/23
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4669 at epoch 1/11

  POST-EVAL: kappa=0.3848, acc=0.3971, f1=0.3843
  Confusion matrix:
[[164  25  10]
 [206  52 136]
 [ 23  16  58]]
  Time: 184s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.283, 0.143, 0.575]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 25.
  Stage 1 best: val_kappa=0.4455 at epoch 25/40
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4374 at epoch 11/13

  POST-EVAL: kappa=0.4370, acc=0.5120, f1=0.5130
  Confusion matrix:
[[ 88  27   4]
 [110  91  36]
 [  8  19  35]]
  Time: 265s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.144, 0.571]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4063 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3736 at epoch 1/11

  POST-EVAL: kappa=0.3736, acc=0.4541, f1=0.4491
  Confusion matrix:
[[66 43 10]
 [77 86 74]
 [ 6 16 36]]
  Time: 188s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.571]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.5246 at epoch 14/29
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.5254 at epoch 2/12

  POST-EVAL: kappa=0.5254, acc=0.5303, f1=0.5341
  Confusion matrix:
[[95 19  6]
 [97 78 60]
 [ 2 10 46]]
  Time: 213s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.284, 0.144, 0.572]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.3755 at epoch 5/20
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.4094 at epoch 4/14

  POST-EVAL: kappa=0.4094, acc=0.3850, f1=0.3797
  Confusion matrix:
[[ 90  20   8]
 [108  28 100]
 [  9   9  41]]
  Time: 182s

  BEST from this round: R6_ft_top50_50ep (kappa=0.4292, s1_kappa=0.4367)

================================================================================
CONFIG [thermal_map]: TOP5_3_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.143, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4372 at epoch 9/24
  Stage 2: unfreezing top 50% (171/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4658 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4426, acc=0.4986, f1=0.4778
  Confusion matrix:
[[137  53   9]
 [135 162  97]
 [  3  49  45]]
  Time: 89s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=1
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.283, 0.143, 0.575]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4426 at epoch 10/25
  Stage 2: unfreezing top 50% (171/341 layers, BN frozen)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 2 best: val_kappa=0.4387 at epoch 8/18
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4387, acc=0.5407, f1=0.5266
  Confusion matrix:
[[ 81  31   7]
 [ 83 111  43]
 [  6  22  34]]
  Time: 111s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=2
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4703 at epoch 13/28
  Stage 2: unfreezing top 50% (171/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4337 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4337, acc=0.4541, f1=0.4585
  Confusion matrix:
[[70 40  9]
 [84 76 77]
 [ 1 15 42]]
  Time: 102s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=3
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.5456 at epoch 6/21
  Stage 2: unfreezing top 50% (171/341 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.5217 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.5217, acc=0.5133, f1=0.5083
  Confusion matrix:
[[91 26  3]
 [80 83 72]
 [ 1 19 38]]
  Time: 94s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=4
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.284, 0.144, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3758 at epoch 20/35
  Stage 2: unfreezing top 50% (171/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3534 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3548, acc=0.5303, f1=0.4950
  Confusion matrix:
[[ 78  32   8]
 [ 80 117  39]
 [ 11  24  24]]
  Time: 110s

  BEST from this round: R6_ft_top20_50ep (kappa=0.4157, s1_kappa=0.4367)

================================================================================
CONFIG [thermal_map]: TOP5_4_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.143, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4372 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4621 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4323, acc=0.4826, f1=0.4662
  Confusion matrix:
[[121  70   8]
 [121 164 109]
 [  2  47  48]]
  Time: 80s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.283, 0.143, 0.575]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4426 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.4579 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4579, acc=0.5359, f1=0.5290
  Confusion matrix:
[[ 81  32   6]
 [ 89 107  41]
 [  4  22  36]]
  Time: 89s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4703 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4590 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4590, acc=0.4541, f1=0.4581
  Confusion matrix:
[[80 33  6]
 [91 67 79]
 [ 2 15 41]]
  Time: 83s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.5456 at epoch 6/21
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.5094 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.5094, acc=0.5157, f1=0.5156
  Confusion matrix:
[[80 33  7]
 [66 89 80]
 [ 0 14 44]]
  Time: 75s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.284, 0.144, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3758 at epoch 20/35
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3580 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3580, acc=0.4891, f1=0.4694
  Confusion matrix:
[[74 34 10]
 [79 99 58]
 [ 9 21 29]]
  Time: 92s

  BEST from this round: R5_aug_off_128 (kappa=0.4110, s1_kappa=0.4367)

================================================================================
CONFIG [thermal_map]: TOP5_5_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.143, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4372 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4621 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4323, acc=0.4826, f1=0.4662
  Confusion matrix:
[[121  70   8]
 [121 164 109]
 [  2  47  48]]
  Time: 80s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.283, 0.143, 0.575]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4426 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.4579 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4579, acc=0.5359, f1=0.5290
  Confusion matrix:
[[ 81  32   6]
 [ 89 107  41]
 [  4  22  36]]
  Time: 89s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.285, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4703 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4590 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4590, acc=0.4541, f1=0.4581
  Confusion matrix:
[[80 33  6]
 [91 67 79]
 [ 2 15 41]]
  Time: 84s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.286, 0.144, 0.571]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.5456 at epoch 6/21
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.5094 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.5094, acc=0.5157, f1=0.5156
  Confusion matrix:
[[80 33  7]
 [66 89 80]
 [ 0 14 44]]
  Time: 75s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.284, 0.144, 0.572]
  Trainable weights: 6/448
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3758 at epoch 20/35
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3580 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3580, acc=0.4891, f1=0.4694
  Confusion matrix:
[[74 34 10]
 [79 99 58]
 [ 9 21 29]]
  Time: 92s

################################################################################
BASELINE: EfficientNetB0 FROZEN ON ALL 5 FOLDS
################################################################################

================================================================================
CONFIG [thermal_map]: BASELINE_fold1
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.854, 0.43, 1.716]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4492 at epoch 19/28
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 2 best: val_kappa=0.4595 at epoch 18/28
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3798, acc=0.4478, f1=0.4374
  Confusion matrix:
[[130  59  10]
 [142 129 123]
 [ 15  32  50]]
  Time: 280s

================================================================================
CONFIG [thermal_map]: BASELINE_fold2
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.848, 0.428, 1.725]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 37: early stopping
Restoring model weights from the end of the best epoch: 22.
  Stage 1 best: val_kappa=0.4272 at epoch 22/37
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.4422 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4422, acc=0.5144, f1=0.5157
  Confusion matrix:
[[80 31  8]
 [90 92 55]
 [ 7 12 43]]
  Time: 246s

================================================================================
CONFIG [thermal_map]: BASELINE_fold3
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.856, 0.432, 1.712]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.4408 at epoch 24/39
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4454 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4454, acc=0.5097, f1=0.4982
  Confusion matrix:
[[ 76  40   3]
 [ 82 103  52]
 [  4  22  32]]
  Time: 251s

================================================================================
CONFIG [thermal_map]: BASELINE_fold4
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.858, 0.431, 1.712]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 25.
  Stage 1 best: val_kappa=0.5098 at epoch 30/40
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.5261 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.5246, acc=0.5327, f1=0.5318
  Confusion matrix:
[[85 27  8]
 [59 89 87]
 [ 1 11 46]]
  Time: 292s

================================================================================
CONFIG [thermal_map]: BASELINE_fold5
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.853, 0.431, 1.717]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.4121 at epoch 24/39
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4212 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4159, acc=0.4673, f1=0.4625
  Confusion matrix:
[[77 36  5]
 [93 83 60]
 [ 5 21 33]]
  Time: 259s

================================================================================
THERMAL MAP SEARCH COMPLETE — SUMMARY
================================================================================
  R1: Backbone+Freeze: R1_EfficientNetB2_frozen → kappa=0.4043
  R2: Head: R2_large → kappa=0.3995
  R3: Loss+Reg+Alpha: R3_focal_g2_alpha1 → kappa=0.4107
  R4: Training+Optim: R4_adamw_wd1e3 → kappa=0.4489
  R5: Aug+ImgSize: R5_aug_off_128 → kappa=0.4110
  R6: FineTuning: R6_ft_top50_50ep → kappa=0.4292

────────────────────────────────────────────────────────────
TOP 5 CONFIGS — 5-FOLD RESULTS
────────────────────────────────────────────────────────────

  Rank   Config                           Fold0       Mean±Std
  ------------------------------------------------------------
  1      R6_ft_top20_50ep                0.4157 0.4433±0.0495
  2      R5_aug_off_128                  0.4110 0.4433±0.0495
  3      R6_ft_top50_50ep                0.4292 0.4383±0.0529
  4      R4_lr1e3_b32_plateau            0.4329 0.4261±0.0543
  5      R4_adamw_wd1e3                  0.4489 0.4260±0.0502

────────────────────────────────────────────────────────────
DETAILED RESULTS
────────────────────────────────────────────────────────────

#1: R6_ft_top20_50ep (TOP5_4):
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+50ft (unfreeze 20%), warmup=0
  aug=False, mixup=False(alpha=0.0), img_size=128
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4323     0.4826       0.4662
  Fold 2     0.4579     0.5359       0.5290
  Fold 3     0.4590     0.4541       0.4581
  Fold 4     0.5094     0.5157       0.5156
  Fold 5     0.3580     0.4891       0.4694
  ----------------------------------------
  Mean       0.4433     0.4955       0.4877
  Std        0.0495     0.0281       0.0289

#2: R5_aug_off_128 (TOP5_5):
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=False, mixup=False(alpha=0.0), img_size=128
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4323     0.4826       0.4662
  Fold 2     0.4579     0.5359       0.5290
  Fold 3     0.4590     0.4541       0.4581
  Fold 4     0.5094     0.5157       0.5156
  Fold 5     0.3580     0.4891       0.4694
  ----------------------------------------
  Mean       0.4433     0.4955       0.4877
  Std        0.0495     0.0281       0.0289

#3: R6_ft_top50_50ep (TOP5_3):
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+50ft (unfreeze 50%), warmup=0
  aug=False, mixup=False(alpha=0.0), img_size=128
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4426     0.4986       0.4778
  Fold 2     0.4387     0.5407       0.5266
  Fold 3     0.4337     0.4541       0.4585
  Fold 4     0.5217     0.5133       0.5083
  Fold 5     0.3548     0.5303       0.4950
  ----------------------------------------
  Mean       0.4383     0.5074       0.4933
  Std        0.0529     0.0303       0.0236

#4: R4_lr1e3_b32_plateau (TOP5_2):
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, schedule=plateau, batch=32
  optimizer=adam, weight_decay=0.0
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3848     0.3971       0.3843
  Fold 2     0.4370     0.5120       0.5130
  Fold 3     0.3736     0.4541       0.4491
  Fold 4     0.5254     0.5303       0.5341
  Fold 5     0.4094     0.3850       0.3797
  ----------------------------------------
  Mean       0.4261     0.4557       0.4521
  Std        0.0543     0.0586       0.0637

#5: R4_adamw_wd1e3 (TOP5_1):
  backbone=EfficientNetB2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3659     0.3696       0.3463
  Fold 2     0.4638     0.5000       0.5039
  Fold 3     0.4065     0.4517       0.4493
  Fold 4     0.5030     0.4843       0.4897
  Fold 5     0.3907     0.3850       0.3783
  ----------------------------------------
  Mean       0.4260     0.4381       0.4335
  Std        0.0502     0.0523       0.0617

────────────────────────────────────────────────────────────

BASELINE (EfficientNetB0 frozen):
  backbone=EfficientNetB0, freeze=frozen
  head=[128], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0001
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3798     0.4478       0.4374
  Fold 2     0.4422     0.5144       0.5157
  Fold 3     0.4454     0.5097       0.4982
  Fold 4     0.5246     0.5327       0.5318
  Fold 5     0.4159     0.4673       0.4625
  ----------------------------------------
  Mean       0.4416     0.4944       0.4891
  Std        0.0477     0.0316       0.0346

────────────────────────────────────────────────────────────
STATISTICAL COMPARISON: #1 R6_ft_top20_50ep vs BASELINE
────────────────────────────────────────────────────────────
  Mean Kappa diff:    +0.0017  (0.4416 → 0.4433)
  Mean Accuracy diff: +0.0011  (0.4944 → 0.4955)
  Mean F1 diff:       -0.0015  (0.4891 → 0.4877)

  Paired t-test on kappa (n=5 folds):
    t-statistic = 0.0937
    p-value     = 0.9299
    → NOT statistically significant (p >= 0.05)

Results saved to: /workspace/DFUMultiClassification/agent_communication/thermal_map_pipeline_audit/thermal_map_search_results.csv
Best config saved to: /workspace/DFUMultiClassification/agent_communication/thermal_map_pipeline_audit/thermal_map_best_config.json
