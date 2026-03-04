Logging to: /workspace/DFUMultiClassification/agent_communication/depth_rgb_pipeline_audit/logs/depth_rgb_hparam_search_20260225_193121.log
================================================================================
DEPTH RGB HYPERPARAMETER SEARCH
================================================================================
Loaded 3108 samples

################################################################################
ROUND 1: BACKBONE + FREEZE STRATEGY
################################################################################

================================================================================
CONFIG: R1_EfficientNetB0_frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.3197 at epoch 8/23
  Stage 2: unfreezing top 20% (48/239 layers)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3311 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3311, acc=0.4287, f1=0.4238
  Confusion matrix:
[[140 101  60]
 [153 211 265]
 [  4  14  97]]
  Time: 292s

================================================================================
CONFIG: R1_EfficientNetB0_partial_unfreeze
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 48/320
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Restoring model weights from the end of the best epoch: 38.
  Stage 1 best: val_kappa=0.3263 at epoch 38/50
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3263, acc=0.5110, f1=0.4716
  Confusion matrix:
[[162 110  29]
 [184 313 132]
 [ 13  43  59]]
  Time: 303s

================================================================================
CONFIG: R1_EfficientNetB2_frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.3594 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3341 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3355, acc=0.4785, f1=0.4545
  Confusion matrix:
[[187  74  40]
 [212 247 170]
 [ 12  37  66]]
  Time: 270s

================================================================================
CONFIG: R1_EfficientNetB2_partial_unfreeze
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 68/448
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3003 at epoch 25/35
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2966, acc=0.5139, f1=0.4674
  Confusion matrix:
[[160 107  34]
 [185 323 121]
 [ 15  46  54]]
  Time: 320s

================================================================================
CONFIG: R1_DenseNet121_frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3125 at epoch 15/30
  Stage 2: unfreezing top 20% (86/428 layers)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3085 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3085, acc=0.3952, f1=0.3932
  Confusion matrix:
[[168  67  66]
 [204 159 266]
 [  6  23  86]]
  Time: 406s

================================================================================
CONFIG: R1_DenseNet121_partial_unfreeze
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 78/612
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Restoring model weights from the end of the best epoch: 36.
  Stage 1 best: val_kappa=0.2974 at epoch 36/50
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2974, acc=0.4938, f1=0.4493
  Confusion matrix:
[[159 108  34]
 [198 306 125]
 [  9  55  51]]
  Time: 430s

================================================================================
CONFIG: R1_ResNet50V2_frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.2813 at epoch 6/21
  Stage 2: unfreezing top 20% (39/191 layers)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.2765 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2765, acc=0.4124, f1=0.3986
  Confusion matrix:
[[178  76  47]
 [248 187 194]
 [ 18  31  66]]
  Time: 222s

================================================================================
CONFIG: R1_ResNet50V2_partial_unfreeze
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 42/278
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Restoring model weights from the end of the best epoch: 39.
  Stage 1 best: val_kappa=0.2852 at epoch 39/50
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2852, acc=0.5665, f1=0.4793
  Confusion matrix:
[[129 148  24]
 [151 425  53]
 [  5  72  38]]
  Time: 335s

  BEST from this round: R1_EfficientNetB2_frozen (kappa=0.3355, s1_kappa=0.3594)

################################################################################
ROUND 2: HEAD ARCHITECTURE
################################################################################

================================================================================
CONFIG: R2_tiny
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3466 at epoch 18/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3359 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3360, acc=0.5656, f1=0.4834
  Confusion matrix:
[[119 169  13]
 [110 426  93]
 [  7  62  46]]
  Time: 339s

================================================================================
CONFIG: R2_small
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 45: early stopping
Restoring model weights from the end of the best epoch: 30.
  Stage 1 best: val_kappa=0.3480 at epoch 30/45
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2988 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2994, acc=0.4708, f1=0.4433
  Confusion matrix:
[[140 107  54]
 [140 279 210]
 [  9  33  73]]
  Time: 385s

================================================================================
CONFIG: R2_medium
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.3594 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3342 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3342, acc=0.4775, f1=0.4531
  Confusion matrix:
[[190  70  41]
 [215 244 170]
 [ 12  38  65]]
  Time: 238s

================================================================================
CONFIG: R2_large
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.3407 at epoch 5/20
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2996 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2980, acc=0.3684, f1=0.3665
  Confusion matrix:
[[188  48  65]
 [221 111 297]
 [ 19  10  86]]
  Time: 257s

================================================================================
CONFIG: R2_two_layer
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.3536 at epoch 21/36
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3449 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3449, acc=0.5043, f1=0.4719
  Confusion matrix:
[[189  96  16]
 [231 283 115]
 [ 19  41  55]]
  Time: 340s

  BEST from this round: R2_two_layer (kappa=0.3449, s1_kappa=0.3536)

################################################################################
ROUND 3: LOSS + REGULARIZATION
################################################################################

================================================================================
CONFIG: R3_focal_g2_d03
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.3536 at epoch 21/36
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3449 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3449, acc=0.5043, f1=0.4719
  Confusion matrix:
[[189  96  16]
 [231 283 115]
 [ 19  41  55]]
  Time: 358s

================================================================================
CONFIG: R3_focal_g3_d03
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 25.
  Stage 1 best: val_kappa=0.3339 at epoch 25/40
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3120 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3120, acc=0.4421, f1=0.4291
  Confusion matrix:
[[153  89  59]
 [165 228 236]
 [  7  27  81]]
  Time: 347s

================================================================================
CONFIG: R3_cce_d03
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=cce, gamma=0.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.3390 at epoch 2/17
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2622 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2622, acc=0.4048, f1=0.3946
  Confusion matrix:
[[114 108  79]
 [105 222 302]
 [  8  20  87]]
  Time: 197s

================================================================================
CONFIG: R3_focal_g2_d05
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.5, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.3588 at epoch 6/21
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2960 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2960, acc=0.4278, f1=0.4179
  Confusion matrix:
[[147  89  65]
 [133 219 277]
 [ 12  22  81]]
  Time: 238s

================================================================================
CONFIG: R3_focal_g2_d02
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.2, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3640 at epoch 18/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3348 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3355, acc=0.5062, f1=0.4638
  Confusion matrix:
[[151 124  26]
 [165 320 144]
 [  9  48  58]]
  Time: 305s

================================================================================
CONFIG: R3_focal_g2_ls01
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.1, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.3330 at epoch 2/17
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2628 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2616, acc=0.4287, f1=0.4132
  Confusion matrix:
[[124  96  81]
 [125 240 264]
 [  6  25  84]]
  Time: 194s

================================================================================
CONFIG: R3_focal_g2_l2_1e3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.001
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3623 at epoch 20/35
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3198 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3193, acc=0.4316, f1=0.4219
  Confusion matrix:
[[192  71  38]
 [254 184 191]
 [ 20  20  75]]
  Time: 285s

================================================================================
CONFIG: R3_focal_g2_mixup02
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3487 at epoch 23/38
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3225 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3225, acc=0.4450, f1=0.4279
  Confusion matrix:
[[201  63  37]
 [255 198 176]
 [ 18  31  66]]
  Time: 301s

================================================================================
CONFIG: R3_focal_g3_alpha0
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=0.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (UNIFORM — no class weighting): [1.0, 1.0, 1.0]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.3292 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2655 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2669, acc=0.5464, f1=0.4451
  Confusion matrix:
[[107 182  12]
 [128 431  70]
 [  8  74  33]]
  Time: 204s

================================================================================
CONFIG: R3_focal_g3_alpha1
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.24, 0.114, 0.646]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.3359 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3404 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3404, acc=0.4565, f1=0.4365
  Confusion matrix:
[[192  73  36]
 [195 219 215]
 [ 18  31  66]]
  Time: 234s

================================================================================
CONFIG: R3_focal_g3_alpha5
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.201, 0.568, 3.231]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3386 at epoch 24/30
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3373 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3373, acc=0.4699, f1=0.4458
  Confusion matrix:
[[152 110  39]
 [164 268 197]
 [  6  38  71]]
  Time: 260s

================================================================================
CONFIG: R3_focal_g3_alpha8
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=8.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=8.0): [1.921, 0.909, 5.17]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.3263 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.3106 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3106, acc=0.4632, f1=0.4400
  Confusion matrix:
[[168 101  32]
 [227 253 149]
 [ 13  39  63]]
  Time: 286s

  BEST from this round: R3_focal_g2_d03 (kappa=0.3449, s1_kappa=0.3536)

################################################################################
ROUND 4: TRAINING DYNAMICS
################################################################################

================================================================================
CONFIG: R4_lr5e4_b64_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3754 at epoch 23/38
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3540 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3540, acc=0.4986, f1=0.4664
  Confusion matrix:
[[205  70  26]
 [215 258 156]
 [ 20  37  58]]
  Time: 300s

================================================================================
CONFIG: R4_lr1e3_b64_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.3536 at epoch 21/36
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3431 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3437, acc=0.5024, f1=0.4732
  Confusion matrix:
[[182 101  18]
 [225 284 120]
 [ 18  38  59]]
  Time: 289s

================================================================================
CONFIG: R4_lr3e3_b64_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.003, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3536 at epoch 15/30
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3429 at epoch 3/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3429, acc=0.4823, f1=0.4554
  Confusion matrix:
[[164 101  36]
 [181 273 175]
 [  7  41  67]]
  Time: 259s

================================================================================
CONFIG: R4_lr1e3_b32_plateau
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.3456 at epoch 5/20
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2896 at epoch 1/11

  POST-EVAL: kappa=0.2896, acc=0.4900, f1=0.4519
  Confusion matrix:
[[195  92  14]
 [279 275  75]
 [ 25  48  42]]
  Time: 226s

================================================================================
CONFIG: R4_lr1e3_b64_cosine
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3428 at epoch 20/40
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3208 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3225, acc=0.4172, f1=0.4125
  Confusion matrix:
[[186  60  55]
 [233 164 232]
 [ 13  16  86]]
  Time: 309s

================================================================================
CONFIG: R4_lr1e3_b64_e100
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 41: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.3536 at epoch 21/41
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3486 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3486, acc=0.5062, f1=0.4770
  Confusion matrix:
[[184 101  16]
 [225 286 118]
 [ 19  37  59]]
  Time: 314s

================================================================================
CONFIG: R4_adamw_wd1e4
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
  Using AdamW (weight_decay=0.0001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 41: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.3280 at epoch 21/41
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2944 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2934, acc=0.4297, f1=0.4175
  Confusion matrix:
[[182  85  34]
 [257 198 174]
 [ 24  22  69]]
  Time: 316s

================================================================================
CONFIG: R4_adamw_wd1e3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3468 at epoch 20/40
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3289 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3289, acc=0.4823, f1=0.4599
  Confusion matrix:
[[170  90  41]
 [198 262 169]
 [ 12  31  72]]
  Time: 309s

  BEST from this round: R4_lr5e4_b64_plateau (kappa=0.3540, s1_kappa=0.3754)

################################################################################
ROUND 5: AUGMENTATION + IMAGE SIZE
################################################################################

================================================================================
CONFIG: R5_aug_on_256
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3754 at epoch 23/38
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3540 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3540, acc=0.4986, f1=0.4664
  Confusion matrix:
[[205  70  26]
 [215 258 156]
 [ 20  37  58]]
  Time: 298s

================================================================================
CONFIG: R5_aug_off_256
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3457 at epoch 19/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3009 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3023, acc=0.4632, f1=0.4353
  Confusion matrix:
[[173  91  37]
 [217 252 160]
 [ 14  42  59]]
  Time: 276s

================================================================================
CONFIG: R5_aug_on_384
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=384, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 384, 384, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 384, 384, 3))']
  warnings.warn(msg)
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.3404 at epoch 24/39
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2875 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 384, 384, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2875, acc=0.4230, f1=0.4105
  Confusion matrix:
[[209  64  28]
 [325 173 131]
 [ 26  29  60]]
  Time: 678s

================================================================================
CONFIG: R5_aug_off_384
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=384, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 384, 384, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 384, 384, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3368 at epoch 28/35
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3348 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 384, 384, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3348, acc=0.4478, f1=0.4316
  Confusion matrix:
[[203  65  33]
 [247 197 185]
 [ 20  27  68]]
  Time: 546s

  BEST from this round: R5_aug_on_256 (kappa=0.3540, s1_kappa=0.3754)

################################################################################
ROUND 6: FINE-TUNING STRATEGY
################################################################################

================================================================================
CONFIG: R6_ft_top20_30ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3680 at epoch 23/38
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3585 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3585, acc=0.4947, f1=0.4650
  Confusion matrix:
[[183  85  33]
 [185 271 173]
 [ 10  42  63]]
  Time: 304s

================================================================================
CONFIG: R6_ft_top40_30ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3754 at epoch 23/38
  Stage 2: unfreezing top 40% (137/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3499 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3499, acc=0.4957, f1=0.4627
  Confusion matrix:
[[194  86  21]
 [206 267 156]
 [ 21  37  57]]
  Time: 315s

================================================================================
CONFIG: R6_ft_top50_50ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.5, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3754 at epoch 23/38
  Stage 2: unfreezing top 50% (171/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3395 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3359, acc=0.4842, f1=0.4507
  Confusion matrix:
[[180  91  30]
 [181 269 179]
 [ 16  42  57]]
  Time: 326s

================================================================================
CONFIG: R6_ft_top20_50ep
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.3488 at epoch 24/39
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3357 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3358, acc=0.4813, f1=0.4535
  Confusion matrix:
[[167  98  36]
 [186 271 172]
 [  9  41  65]]
  Time: 300s

  BEST from this round: R6_ft_top20_30ep (kappa=0.3585, s1_kappa=0.3680)

################################################################################
TOP 5 SELECTION: 5-FOLD VALIDATION OF BEST CONFIGS
################################################################################

Top 5 configs by fold-0 kappa:
  1. R6_ft_top20_30ep → kappa=0.3585
  2. R6_ft_top40_30ep → kappa=0.3499
  3. R4_lr1e3_b64_e100 → kappa=0.3486
  4. R2_two_layer → kappa=0.3449
  5. R3_focal_g2_d03 → kappa=0.3449

  BEST from this round: R6_ft_top20_30ep (kappa=0.3585, s1_kappa=0.3680)

================================================================================
CONFIG: TOP5_1_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3750 at epoch 18/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3787 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3344, acc=0.4909, f1=0.4579
  Confusion matrix:
[[168 106  27]
 [200 286 143]
 [ 11  45  59]]
  Time: 308s

================================================================================
CONFIG: TOP5_1_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.3210 at epoch 17/32
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3050 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3050, acc=0.3932, f1=0.3892
  Confusion matrix:
[[116  30  32]
 [173  79 122]
 [  7  11  48]]
  Time: 292s

================================================================================
CONFIG: TOP5_1_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.2352 at epoch 13/28
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2025 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2025, acc=0.4272, f1=0.3892
  Confusion matrix:
[[ 69  69  39]
 [ 99 162 113]
 [  4  30  33]]
  Time: 273s

================================================================================
CONFIG: TOP5_1_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.2756 at epoch 14/29
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2497 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2498, acc=0.4662, f1=0.4245
  Confusion matrix:
[[104  53  21]
 [139 159  81]
 [ 14  24  27]]
  Time: 275s

================================================================================
CONFIG: TOP5_1_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 42: early stopping
Restoring model weights from the end of the best epoch: 27.
  Stage 1 best: val_kappa=0.1827 at epoch 27/42
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2290 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2290, acc=0.4659, f1=0.4169
  Confusion matrix:
[[ 82  75  20]
 [117 178  78]
 [ 11  28  27]]
  Time: 349s

  BEST from this round: R6_ft_top40_30ep (kappa=0.3499, s1_kappa=0.3754)

================================================================================
CONFIG: TOP5_2_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3750 at epoch 18/33
  Stage 2: unfreezing top 40% (137/341 layers)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3098 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2672, acc=0.4909, f1=0.4371
  Confusion matrix:
[[156 117  28]
 [182 313 134]
 [ 21  50  44]]
  Time: 331s

================================================================================
CONFIG: TOP5_2_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.3210 at epoch 17/32
  Stage 2: unfreezing top 40% (137/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2866 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2866, acc=0.3754, f1=0.3711
  Confusion matrix:
[[113  32  33]
 [173  73 128]
 [  8  12  46]]
  Time: 310s

================================================================================
CONFIG: TOP5_2_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.2352 at epoch 13/28
  Stage 2: unfreezing top 40% (137/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2091 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2091, acc=0.4434, f1=0.3987
  Confusion matrix:
[[ 66  74  37]
 [ 89 175 110]
 [  4  30  33]]
  Time: 286s

================================================================================
CONFIG: TOP5_2_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.2756 at epoch 14/29
  Stage 2: unfreezing top 40% (137/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2577 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2577, acc=0.4437, f1=0.4113
  Confusion matrix:
[[101  63  14]
 [152 147  80]
 [ 14  23  28]]
  Time: 300s

================================================================================
CONFIG: TOP5_2_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 42: early stopping
Restoring model weights from the end of the best epoch: 27.
  Stage 1 best: val_kappa=0.1827 at epoch 27/42
  Stage 2: unfreezing top 40% (137/341 layers)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 2 best: val_kappa=0.1801 at epoch 11/21
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1793, acc=0.4172, f1=0.3720
  Confusion matrix:
[[ 80  69  28]
 [127 154  92]
 [ 10  33  23]]
  Time: 440s

  BEST from this round: R4_lr1e3_b64_e100 (kappa=0.3486, s1_kappa=0.3536)

================================================================================
CONFIG: TOP5_3_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3921 at epoch 18/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3351 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3144, acc=0.4938, f1=0.4609
  Confusion matrix:
[[160 104  37]
 [184 292 153]
 [ 13  38  64]]
  Time: 291s

================================================================================
CONFIG: TOP5_3_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3461 at epoch 15/30
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3265 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3265, acc=0.4693, f1=0.4466
  Confusion matrix:
[[108  46  24]
 [128 143 103]
 [  7  20  39]]
  Time: 274s

================================================================================
CONFIG: TOP5_3_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.2283 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2217 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2217, acc=0.4126, f1=0.3955
  Confusion matrix:
[[ 86  50  41]
 [123 128 123]
 [ 10  16  41]]
  Time: 243s

================================================================================
CONFIG: TOP5_3_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.3010 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2676 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2676, acc=0.4727, f1=0.4326
  Confusion matrix:
[[109  41  28]
 [124 156  99]
 [ 12  24  29]]
  Time: 222s

================================================================================
CONFIG: TOP5_3_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 1 best: val_kappa=0.1631 at epoch 19/34
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.2007 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1967, acc=0.4286, f1=0.3986
  Confusion matrix:
[[ 82  62  33]
 [122 148 103]
 [ 14  18  34]]
  Time: 328s

  BEST from this round: R2_two_layer (kappa=0.3449, s1_kappa=0.3536)

================================================================================
CONFIG: TOP5_4_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3921 at epoch 18/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3351 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3144, acc=0.4938, f1=0.4609
  Confusion matrix:
[[160 104  37]
 [184 292 153]
 [ 13  38  64]]
  Time: 293s

================================================================================
CONFIG: TOP5_4_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3046 at epoch 23/38
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3059 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3059, acc=0.4337, f1=0.4229
  Confusion matrix:
[[ 90  58  30]
 [121 130 123]
 [  5  13  48]]
  Time: 314s

================================================================================
CONFIG: TOP5_4_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.2283 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2217 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2217, acc=0.4126, f1=0.3955
  Confusion matrix:
[[ 86  50  41]
 [123 128 123]
 [ 10  16  41]]
  Time: 246s

================================================================================
CONFIG: TOP5_4_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.3010 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2676 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2676, acc=0.4727, f1=0.4326
  Confusion matrix:
[[109  41  28]
 [124 156  99]
 [ 12  24  29]]
  Time: 219s

================================================================================
CONFIG: TOP5_4_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 1 best: val_kappa=0.1631 at epoch 19/34
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.2007 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1967, acc=0.4286, f1=0.3986
  Confusion matrix:
[[ 82  62  33]
 [122 148 103]
 [ 14  18  34]]
  Time: 324s

  BEST from this round: R3_focal_g2_d03 (kappa=0.3449, s1_kappa=0.3536)

================================================================================
CONFIG: TOP5_5_fold1
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.3921 at epoch 18/33
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3351 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3144, acc=0.4938, f1=0.4609
  Confusion matrix:
[[160 104  37]
 [184 292 153]
 [ 13  38  64]]
  Time: 292s

================================================================================
CONFIG: TOP5_5_fold2
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3461 at epoch 15/30
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3265 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3265, acc=0.4693, f1=0.4466
  Confusion matrix:
[[108  46  24]
 [128 143 103]
 [  7  20  39]]
  Time: 275s

================================================================================
CONFIG: TOP5_5_fold3
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.2283 at epoch 10/25
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2217 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2217, acc=0.4126, f1=0.3955
  Confusion matrix:
[[ 86  50  41]
 [123 128 123]
 [ 10  16  41]]
  Time: 244s

================================================================================
CONFIG: TOP5_5_fold4
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.3010 at epoch 4/19
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2676 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2676, acc=0.4727, f1=0.4326
  Confusion matrix:
[[109  41  28]
 [124 156  99]
 [ 12  24  29]]
  Time: 219s

================================================================================
CONFIG: TOP5_5_fold5
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 10/454
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 1 best: val_kappa=0.1631 at epoch 19/34
  Stage 2: unfreezing top 20% (69/341 layers)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.2007 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1967, acc=0.4286, f1=0.3986
  Confusion matrix:
[[ 82  62  33]
 [122 148 103]
 [ 14  18  34]]
  Time: 321s

################################################################################
BASELINE: EfficientNetB0 FROZEN ON ALL 5 FOLDS
################################################################################

================================================================================
CONFIG: BASELINE_fold1
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
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Restoring model weights from the end of the best epoch: 36.
  Stage 1 best: val_kappa=0.3492 at epoch 36/50
  Stage 2: unfreezing top 20% (48/239 layers)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 2 best: val_kappa=0.3140 at epoch 11/21
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2843, acc=0.4641, f1=0.4223
  Confusion matrix:
[[146 126  29]
 [201 290 138]
 [  8  58  49]]
  Time: 418s

================================================================================
CONFIG: BASELINE_fold2
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
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3325 at epoch 20/35
  Stage 2: unfreezing top 20% (48/239 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2797 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2797, acc=0.3285, f1=0.3160
  Confusion matrix:
[[125   9  44]
 [196  23 155]
 [ 10   1  55]]
  Time: 287s

================================================================================
CONFIG: BASELINE_fold3
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
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.2585 at epoch 5/20
  Stage 2: unfreezing top 20% (48/239 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2223 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2223, acc=0.4741, f1=0.4383
  Confusion matrix:
[[ 94  52  31]
 [127 165  82]
 [ 15  18  34]]
  Time: 199s

================================================================================
CONFIG: BASELINE_fold4
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
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.3293 at epoch 9/24
  Stage 2: unfreezing top 20% (48/239 layers)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2130 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2130, acc=0.3521, f1=0.3192
  Confusion matrix:
[[150  11  17]
 [242  41  96]
 [ 34   3  28]]
  Time: 222s

================================================================================
CONFIG: BASELINE_fold5
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
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.2044 at epoch 16/31
  Stage 2: unfreezing top 20% (48/239 layers)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2253 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_rgb_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2253, acc=0.4205, f1=0.3832
  Confusion matrix:
[[ 72  75  30]
 [ 90 157 126]
 [  7  29  30]]
  Time: 272s

================================================================================
SEARCH COMPLETE — SUMMARY
================================================================================
  R1: Backbone+Freeze: R1_EfficientNetB2_frozen → kappa=0.3355
  R2: Head: R2_two_layer → kappa=0.3449
  R3: Loss+Reg+Alpha: R3_focal_g2_d03 → kappa=0.3449
  R4: Training+Optim: R4_lr5e4_b64_plateau → kappa=0.3540
  R5: Aug+ImgSize: R5_aug_on_256 → kappa=0.3540
  R6: FineTuning: R6_ft_top20_30ep → kappa=0.3585

────────────────────────────────────────────────────────────
TOP 5 CONFIGS — 5-FOLD RESULTS
────────────────────────────────────────────────────────────

  Rank   Config                           Fold0       Mean±Std
  ------------------------------------------------------------
  1      R4_lr1e3_b64_e100               0.3486 0.2654±0.0505
  2      R3_focal_g2_d03                 0.3449 0.2654±0.0505
  3      R6_ft_top20_30ep                0.3585 0.2641±0.0487
  4      R2_two_layer                    0.3449 0.2613±0.0460
  5      R6_ft_top40_30ep                0.3499 0.2400±0.0397

────────────────────────────────────────────────────────────
DETAILED RESULTS
────────────────────────────────────────────────────────────

#1: R4_lr1e3_b64_e100 (TOP5_3):
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3144     0.4938       0.4609
  Fold 2     0.3265     0.4693       0.4466
  Fold 3     0.2217     0.4126       0.3955
  Fold 4     0.2676     0.4727       0.4326
  Fold 5     0.1967     0.4286       0.3986
  ----------------------------------------
  Mean       0.2654     0.4554       0.4268
  Std        0.0505     0.0300       0.0259

#2: R3_focal_g2_d03 (TOP5_5):
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0001
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3144     0.4938       0.4609
  Fold 2     0.3265     0.4693       0.4466
  Fold 3     0.2217     0.4126       0.3955
  Fold 4     0.2676     0.4727       0.4326
  Fold 5     0.1967     0.4286       0.3986
  ----------------------------------------
  Mean       0.2654     0.4554       0.4268
  Std        0.0505     0.0300       0.0259

#3: R6_ft_top20_30ep (TOP5_1):
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3344     0.4909       0.4579
  Fold 2     0.3050     0.3932       0.3892
  Fold 3     0.2025     0.4272       0.3892
  Fold 4     0.2498     0.4662       0.4245
  Fold 5     0.2290     0.4659       0.4169
  ----------------------------------------
  Mean       0.2641     0.4487       0.4155
  Std        0.0487     0.0344       0.0256

#4: R2_two_layer (TOP5_4):
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0001
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.3144     0.4938       0.4609
  Fold 2     0.3059     0.4337       0.4229
  Fold 3     0.2217     0.4126       0.3955
  Fold 4     0.2676     0.4727       0.4326
  Fold 5     0.1967     0.4286       0.3986
  ----------------------------------------
  Mean       0.2613     0.4483       0.4221
  Std        0.0460     0.0301       0.0240

#5: R6_ft_top40_30ep (TOP5_2):
  backbone=EfficientNetB2, freeze=frozen
  head=[256, 64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0
  epochs=50+30ft (unfreeze 40%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.2672     0.4909       0.4371
  Fold 2     0.2866     0.3754       0.3711
  Fold 3     0.2091     0.4434       0.3987
  Fold 4     0.2577     0.4437       0.4113
  Fold 5     0.1793     0.4172       0.3720
  ----------------------------------------
  Mean       0.2400     0.4341       0.3980
  Std        0.0397     0.0378       0.0249

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
  Fold 1     0.2843     0.4641       0.4223
  Fold 2     0.2797     0.3285       0.3160
  Fold 3     0.2223     0.4741       0.4383
  Fold 4     0.2130     0.3521       0.3192
  Fold 5     0.2253     0.4205       0.3832
  ----------------------------------------
  Mean       0.2449     0.4078       0.3758
  Std        0.0306     0.0585       0.0508

────────────────────────────────────────────────────────────
STATISTICAL COMPARISON: #1 R4_lr1e3_b64_e100 vs BASELINE
────────────────────────────────────────────────────────────
  Mean Kappa diff:    +0.0205  (0.2449 → 0.2654)
  Mean Accuracy diff: +0.0475  (0.4078 → 0.4554)
  Mean F1 diff:       +0.0511  (0.3758 → 0.4268)

  Paired t-test on kappa (n=5 folds):
    t-statistic = 1.3201
    p-value     = 0.2573
    → NOT statistically significant (p >= 0.05)
    Note: With 5 folds (4 df), moderate effect sizes can reach significance.

Results saved to: /workspace/DFUMultiClassification/agent_communication/depth_rgb_pipeline_audit/depth_rgb_search_results.csv
Best config saved to: /workspace/DFUMultiClassification/agent_communication/depth_rgb_pipeline_audit/depth_rgb_best_config.json
