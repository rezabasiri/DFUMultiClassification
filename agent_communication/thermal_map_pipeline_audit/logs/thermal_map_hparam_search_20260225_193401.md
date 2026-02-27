Logging to: /workspace/DFUMultiClassification/agent_communication/thermal_map_pipeline_audit/logs/thermal_map_hparam_search_20260225_193401.log
================================================================================
THERMAL MAP HYPERPARAMETER SEARCH
================================================================================
Loaded 3108 samples for thermal_map

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.3954 at epoch 15/30
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3656 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3674, acc=0.4058, f1=0.3994
  Confusion matrix:
[[145  36  18]
 [176  79 139]
 [ 16  25  56]]
  Time: 285s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
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
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.3321 at epoch 20/35
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3339, acc=0.3826, f1=0.3794
  Confusion matrix:
[[133  41  25]
 [163  73 158]
 [ 17  22  58]]
  Time: 229s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.3842 at epoch 6/21
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3897 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3897, acc=0.3667, f1=0.3513
  Confusion matrix:
[[159  23  17]
 [195  26 173]
 [ 19  10  68]]
  Time: 260s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
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
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.3570 at epoch 16/31
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3538, acc=0.4174, f1=0.4066
  Confusion matrix:
[[151  32  16]
 [171  83 140]
 [ 25  18  54]]
  Time: 269s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.3676 at epoch 23/38
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.3731 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3738, acc=0.4188, f1=0.4083
  Confusion matrix:
[[152  32  15]
 [186  86 122]
 [ 16  30  51]]
  Time: 417s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
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
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.3645 at epoch 9/24
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3645, acc=0.4522, f1=0.4375
  Confusion matrix:
[[140  49  10]
 [174 128  92]
 [ 16  37  44]]
  Time: 227s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.4031 at epoch 9/24
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4036 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4036, acc=0.4217, f1=0.4200
  Confusion matrix:
[[150  39  10]
 [185  76 133]
 [ 19  13  65]]
  Time: 245s

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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
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
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.3746 at epoch 11/26
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3746, acc=0.4507, f1=0.4337
  Confusion matrix:
[[135  53  11]
 [138 132 124]
 [ 15  38  44]]
  Time: 173s

  BEST from this round: R1_ResNet50V2_frozen (kappa=0.4036, s1_kappa=0.4031)

################################################################################
ROUND 2: HEAD ARCHITECTURE
################################################################################

================================================================================
CONFIG [thermal_map]: R2_tiny
  backbone=ResNet50V2, freeze=frozen
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
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.4028 at epoch 14/29
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 2 best: val_kappa=0.3702 at epoch 10/20
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3702, acc=0.4493, f1=0.4368
  Confusion matrix:
[[136  51  12]
 [145 125 124]
 [ 18  30  49]]
  Time: 320s

================================================================================
CONFIG [thermal_map]: R2_small
  backbone=ResNet50V2, freeze=frozen
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
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4163 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4224 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4224, acc=0.4145, f1=0.4066
  Confusion matrix:
[[161  23  15]
 [192  50 152]
 [ 18   4  75]]
  Time: 245s

================================================================================
CONFIG [thermal_map]: R2_medium
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
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.4009 at epoch 15/30
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 12.
  Stage 2 best: val_kappa=0.3858 at epoch 12/22
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3858, acc=0.4638, f1=0.4535
  Confusion matrix:
[[132  60   7]
 [154 138 102]
 [ 16  31  50]]
  Time: 330s

================================================================================
CONFIG [thermal_map]: R2_large
  backbone=ResNet50V2, freeze=frozen
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
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.3993 at epoch 11/21
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.4214 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4232, acc=0.4783, f1=0.4768
  Confusion matrix:
[[143  45  11]
 [153 120 121]
 [ 18  12  67]]
  Time: 244s

================================================================================
CONFIG [thermal_map]: R2_two_layer
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 10/284
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.4185 at epoch 17/32
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3981 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3963, acc=0.4319, f1=0.4288
  Confusion matrix:
[[155  34  10]
 [198  82 114]
 [ 20  16  61]]
  Time: 300s

  BEST from this round: R2_large (kappa=0.4232, s1_kappa=0.3993)

################################################################################
ROUND 3: LOSS + REGULARIZATION
################################################################################

================================================================================
CONFIG [thermal_map]: R3_focal_g2_d03
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.4073 at epoch 21/36
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.3632 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3632, acc=0.4522, f1=0.4401
  Confusion matrix:
[[130  63   6]
 [161 136  97]
 [ 18  33  46]]
  Time: 348s

================================================================================
CONFIG [thermal_map]: R3_focal_g3_d03
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4097 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3880 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3880, acc=0.4406, f1=0.4350
  Confusion matrix:
[[144  43  12]
 [167 103 124]
 [ 18  22  57]]
  Time: 266s

================================================================================
CONFIG [thermal_map]: R3_cce_d03
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4053 at epoch 10/25
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3857 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3857, acc=0.4652, f1=0.4598
  Confusion matrix:
[[132  52  15]
 [143 129 122]
 [ 17  20  60]]
  Time: 241s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_d05
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.4353 at epoch 7/22
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4159 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4159, acc=0.4507, f1=0.4483
  Confusion matrix:
[[142  45  12]
 [145 104 145]
 [ 17  15  65]]
  Time: 238s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_d02
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.4006 at epoch 6/21
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.3907 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3907, acc=0.4377, f1=0.4365
  Confusion matrix:
[[137  46  16]
 [154 100 140]
 [ 17  15  65]]
  Time: 255s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_ls01
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.4142 at epoch 21/36
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3803 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3803, acc=0.4449, f1=0.4434
  Confusion matrix:
[[117  67  15]
 [130 127 137]
 [ 14  20  63]]
  Time: 310s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_l2_1e3
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4324 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.4237 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4237, acc=0.5043, f1=0.5016
  Confusion matrix:
[[124  63  12]
 [126 155 113]
 [ 15  13  69]]
  Time: 263s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_mixup02
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4329 at epoch 10/25
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4260 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4260, acc=0.4551, f1=0.4577
  Confusion matrix:
[[146  42  11]
 [171  95 128]
 [ 18   6  73]]
  Time: 247s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha0
  backbone=ResNet50V2, freeze=frozen
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
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.3536 at epoch 10/25
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 2 best: val_kappa=0.3122 at epoch 15/25
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3122, acc=0.5188, f1=0.4404
  Confusion matrix:
[[ 87 109   3]
 [ 79 247  68]
 [ 11  62  24]]
  Time: 315s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha1
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=1.0): [0.24, 0.114, 0.646]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4009 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3948 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3948, acc=0.4203, f1=0.4143
  Confusion matrix:
[[156  30  13]
 [190  71 133]
 [ 20  14  63]]
  Time: 237s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha5
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=5.0): [1.201, 0.568, 3.231]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.4160 at epoch 10/25
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 12.
  Stage 2 best: val_kappa=0.3826 at epoch 12/22
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3826, acc=0.4638, f1=0.4587
  Confusion matrix:
[[138  46  15]
 [155 122 117]
 [ 19  18  60]]
  Time: 297s

================================================================================
CONFIG [thermal_map]: R3_focal_g2_alpha8
  backbone=ResNet50V2, freeze=frozen
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
  Alpha values (sum=8.0): [1.921, 0.909, 5.17]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4154 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3919 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3919, acc=0.4580, f1=0.4550
  Confusion matrix:
[[136  52  11]
 [158 120 116]
 [ 18  19  60]]
  Time: 239s

  BEST from this round: R3_focal_g2_mixup02 (kappa=0.4260, s1_kappa=0.4329)

################################################################################
ROUND 4: TRAINING DYNAMICS
################################################################################

================================================================================
CONFIG [thermal_map]: R4_lr5e4_b64_plateau
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Restoring model weights from the end of the best epoch: 36.
  Stage 1 best: val_kappa=0.4071 at epoch 36/50
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 2 best: val_kappa=0.4235 at epoch 18/28
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4235, acc=0.4377, f1=0.4370
  Confusion matrix:
[[151  34  14]
 [174  76 144]
 [ 18   4  75]]
  Time: 481s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b64_plateau
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4224 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.4187 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4187, acc=0.4290, f1=0.4294
  Confusion matrix:
[[147  40  12]
 [172  76 146]
 [ 18   6  73]]
  Time: 275s

================================================================================
CONFIG [thermal_map]: R4_lr3e3_b64_plateau
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.003, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.4238 at epoch 23/38
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.4191 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4191, acc=0.4058, f1=0.4050
  Confusion matrix:
[[147  38  14]
 [179  56 159]
 [ 16   4  77]]
  Time: 344s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b32_plateau
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 46: early stopping
Restoring model weights from the end of the best epoch: 31.
  Stage 1 best: val_kappa=0.4271 at epoch 31/46
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 2 best: val_kappa=0.4194 at epoch 8/18

  POST-EVAL: kappa=0.4194, acc=0.4304, f1=0.4288
  Confusion matrix:
[[154  34  11]
 [185  72 137]
 [ 19   7  71]]
  Time: 378s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b64_cosine
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 41: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.4209 at epoch 21/41
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.4280 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4280, acc=0.4130, f1=0.4037
  Confusion matrix:
[[168  22   9]
 [211  47 136]
 [ 19   8  70]]
  Time: 332s

================================================================================
CONFIG [thermal_map]: R4_lr1e3_b64_e100
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4387 at epoch 11/31
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 2 best: val_kappa=0.4269 at epoch 20/30
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4269, acc=0.4319, f1=0.4305
  Confusion matrix:
[[148  33  18]
 [154  72 168]
 [ 16   3  78]]
  Time: 379s

================================================================================
CONFIG [thermal_map]: R4_adamw_wd1e4
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.0001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 41: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.4250 at epoch 21/41
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.4127 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4127, acc=0.4290, f1=0.4197
  Confusion matrix:
[[170  20   9]
 [213  64 117]
 [ 21  14  62]]
  Time: 339s

================================================================================
CONFIG [thermal_map]: R4_adamw_wd1e3
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 37: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.4275 at epoch 17/37
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.4298 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4298, acc=0.4493, f1=0.4494
  Confusion matrix:
[[155  34  10]
 [184  83 127]
 [ 19   6  72]]
  Time: 326s

  BEST from this round: R4_adamw_wd1e3 (kappa=0.4298, s1_kappa=0.4275)

################################################################################
ROUND 5: AUGMENTATION + IMAGE SIZE
################################################################################

================================================================================
CONFIG [thermal_map]: R5_aug_on_128
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.4145 at epoch 20/35
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 2 best: val_kappa=0.4343 at epoch 13/23
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4343, acc=0.3971, f1=0.3782
  Confusion matrix:
[[172  12  15]
 [213  25 156]
 [ 16   4  77]]
  Time: 128s

================================================================================
CONFIG [thermal_map]: R5_aug_off_128
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 1 best: val_kappa=0.3988 at epoch 19/34
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 2 best: val_kappa=0.4147 at epoch 8/18
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4147, acc=0.4406, f1=0.4409
  Confusion matrix:
[[150  39  10]
 [184  87 123]
 [ 18  12  67]]
  Time: 90s

================================================================================
CONFIG [thermal_map]: R5_aug_on_256
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.4354 at epoch 17/32
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.4281 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4281, acc=0.4362, f1=0.4385
  Confusion matrix:
[[138  42  19]
 [125  84 185]
 [ 15   3  79]]
  Time: 310s

================================================================================
CONFIG [thermal_map]: R5_aug_off_256
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=False, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.3639 at epoch 17/32
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3662 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3662, acc=0.4145, f1=0.4213
  Confusion matrix:
[[132  54  13]
 [187  84 123]
 [ 22   5  70]]
  Time: 180s

  BEST from this round: R5_aug_on_128 (kappa=0.4343, s1_kappa=0.4145)

################################################################################
ROUND 6: FINE-TUNING STRATEGY
################################################################################

================================================================================
CONFIG [thermal_map]: R6_ft_top20_30ep
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 56: early stopping
Restoring model weights from the end of the best epoch: 41.
  Stage 1 best: val_kappa=0.4136 at epoch 41/56
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 2 best: val_kappa=0.4236 at epoch 20/30
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4236, acc=0.3884, f1=0.3744
  Confusion matrix:
[[165  21  13]
 [210  28 156]
 [ 17   5  75]]
  Time: 180s

================================================================================
CONFIG [thermal_map]: R6_ft_top40_30ep
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 46: early stopping
Restoring model weights from the end of the best epoch: 31.
  Stage 1 best: val_kappa=0.4216 at epoch 31/46
  Stage 2: unfreezing top 40% (77/191 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.4147 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4147, acc=0.3884, f1=0.3726
  Confusion matrix:
[[163  17  19]
 [196  28 170]
 [ 17   3  77]]
  Time: 135s

================================================================================
CONFIG [thermal_map]: R6_ft_top50_50ep
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.5, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 38: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 1 best: val_kappa=0.4257 at epoch 23/38
  Stage 2: unfreezing top 50% (96/191 layers, BN frozen)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 2 best: val_kappa=0.4284 at epoch 15/25
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4277, acc=0.3870, f1=0.3705
  Confusion matrix:
[[163  17  19]
 [180  26 188]
 [ 15   4  78]]
  Time: 143s

================================================================================
CONFIG [thermal_map]: R6_ft_top20_50ep
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.72, 0.341, 1.939]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 56: early stopping
Restoring model weights from the end of the best epoch: 41.
  Stage 1 best: val_kappa=0.4166 at epoch 53/56
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4214 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4214, acc=0.4058, f1=0.3958
  Confusion matrix:
[[166  17  16]
 [220  38 136]
 [ 16   5  76]]
  Time: 146s

  BEST from this round: R6_ft_top50_50ep (kappa=0.4277, s1_kappa=0.4257)

################################################################################
TOP 5 SELECTION: 5-FOLD VALIDATION OF BEST CONFIGS
################################################################################

Top 5 configs by fold-0 kappa:
  1. R5_aug_on_128 → kappa=0.4343
  2. R4_adamw_wd1e3 → kappa=0.4298
  3. R4_lr1e3_b64_cosine → kappa=0.4280
  4. R6_ft_top50_50ep → kappa=0.4277
  5. R4_lr1e3_b64_e100 → kappa=0.4269

  BEST from this round: R5_aug_on_128 (kappa=0.4343, s1_kappa=0.4145)

================================================================================
CONFIG [thermal_map]: TOP5_1_fold1
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.4519 at epoch 33/36
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4635 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4306, acc=0.4058, f1=0.3990
  Confusion matrix:
[[165  24  10]
 [221  40 133]
 [ 17   5  75]]
  Time: 126s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold2
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.3169 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.3267 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3267, acc=0.4061, f1=0.4106
  Confusion matrix:
[[111  42  25]
 [186  89  99]
 [  4  11  51]]
  Time: 118s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold3
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 47: early stopping
Restoring model weights from the end of the best epoch: 32.
  Stage 1 best: val_kappa=0.2757 at epoch 32/47
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.2948 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2948, acc=0.4725, f1=0.4495
  Confusion matrix:
[[ 90  53  34]
 [105 157 112]
 [  6  16  45]]
  Time: 161s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold4
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/278
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
  Stage 1 best: val_kappa=0.2673 at epoch 13/28
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 2 best: val_kappa=0.2877 at epoch 11/21
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2877, acc=0.3617, f1=0.3554
  Confusion matrix:
[[110  45  23]
 [144  74 161]
 [ 16   8  41]]
  Time: 138s

================================================================================
CONFIG [thermal_map]: TOP5_1_fold5
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.2520 at epoch 13/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.2637 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2637, acc=0.4594, f1=0.4527
  Confusion matrix:
[[ 97  67  13]
 [165 146  62]
 [ 18   8  40]]
  Time: 123s

  BEST from this round: R4_adamw_wd1e3 (kappa=0.4298, s1_kappa=0.4275)

================================================================================
CONFIG [thermal_map]: TOP5_2_fold1
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
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
  Stage 1 best: val_kappa=0.4503 at epoch 15/30
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 2 best: val_kappa=0.4423 at epoch 10/20
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4045, acc=0.4029, f1=0.3968
  Confusion matrix:
[[154  25  20]
 [189  51 154]
 [ 15   9  73]]
  Time: 373s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold2
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.3459 at epoch 8/23
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3235 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3235, acc=0.4211, f1=0.4247
  Confusion matrix:
[[ 87  19  13]
 [131  48  58]
 [ 18   3  41]]
  Time: 266s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold3
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.3918 at epoch 16/31
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 2 best: val_kappa=0.4196 at epoch 11/21
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4196, acc=0.4372, f1=0.4355
  Confusion matrix:
[[ 91  19   9]
 [103  44  90]
 [ 12   0  46]]
  Time: 375s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold4
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 41: early stopping
Restoring model weights from the end of the best epoch: 26.
  Stage 1 best: val_kappa=0.4348 at epoch 26/41
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 2 best: val_kappa=0.4745 at epoch 8/18
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4745, acc=0.4552, f1=0.4534
  Confusion matrix:
[[ 98  14   8]
 [114  47  74]
 [  3  12  43]]
  Time: 426s

================================================================================
CONFIG [thermal_map]: TOP5_2_fold5
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.3629 at epoch 17/32
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.3597 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3597, acc=0.3826, f1=0.3812
  Confusion matrix:
[[ 88  20  10]
 [131  29  76]
 [ 13   5  41]]
  Time: 379s

  BEST from this round: R4_lr1e3_b64_cosine (kappa=0.4280, s1_kappa=0.4209)

================================================================================
CONFIG [thermal_map]: TOP5_3_fold1
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/278
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
  Stage 1 best: val_kappa=0.4484 at epoch 23/38
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.4499 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4263, acc=0.4261, f1=0.4239
  Confusion matrix:
[[154  32  13]
 [182  64 148]
 [ 18   3  76]]
  Time: 372s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold2
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 60: early stopping
Restoring model weights from the end of the best epoch: 45.
  Stage 1 best: val_kappa=0.3397 at epoch 45/60
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3403 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3403, acc=0.4234, f1=0.4391
  Confusion matrix:
[[ 86  27   6]
 [142  51  44]
 [ 18   4  40]]
  Time: 512s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold3
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.4001 at epoch 16/31
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3934 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3934, acc=0.4589, f1=0.4583
  Confusion matrix:
[[74 41  4]
 [84 78 75]
 [12  8 38]]
  Time: 313s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold4
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.4409 at epoch 24/39
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 2 best: val_kappa=0.4780 at epoch 13/23
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4780, acc=0.4818, f1=0.4791
  Confusion matrix:
[[95 16  9]
 [94 61 80]
 [ 4 11 43]]
  Time: 443s

================================================================================
CONFIG [thermal_map]: TOP5_3_fold5
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 49: early stopping
Restoring model weights from the end of the best epoch: 34.
  Stage 1 best: val_kappa=0.3543 at epoch 34/49
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.3555 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3555, acc=0.4019, f1=0.4057
  Confusion matrix:
[[ 78  33   7]
 [112  51  73]
 [ 12  10  37]]
  Time: 444s

  BEST from this round: R6_ft_top50_50ep (kappa=0.4277, s1_kappa=0.4257)

================================================================================
CONFIG [thermal_map]: TOP5_4_fold1
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 50: early stopping
Restoring model weights from the end of the best epoch: 35.
  Stage 1 best: val_kappa=0.4610 at epoch 35/50
  Stage 2: unfreezing top 50% (96/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.4638 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4319, acc=0.4217, f1=0.4145
  Confusion matrix:
[[172  18   9]
 [228  47 119]
 [ 19   6  72]]
  Time: 162s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold2
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=1
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.2867 at epoch 8/23
  Stage 2: unfreezing top 50% (96/191 layers, BN frozen)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 23.
  Stage 2 best: val_kappa=0.2841 at epoch 23/33
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2833, acc=0.3997, f1=0.4056
  Confusion matrix:
[[101  55  22]
 [194 101  79]
 [  6  15  45]]
  Time: 151s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold3
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=2
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 41: early stopping
Restoring model weights from the end of the best epoch: 26.
  Stage 1 best: val_kappa=0.2763 at epoch 26/41
  Stage 2: unfreezing top 50% (96/191 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.2870 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2870, acc=0.4142, f1=0.4070
  Confusion matrix:
[[ 86  51  40]
 [110 119 145]
 [  4  12  51]]
  Time: 152s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold4
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=3
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/278
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
  Stage 1 best: val_kappa=0.2489 at epoch 13/28
  Stage 2: unfreezing top 50% (96/191 layers, BN frozen)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 2 best: val_kappa=0.3115 at epoch 11/21
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3121, acc=0.3376, f1=0.3442
  Confusion matrix:
[[103  39  36]
 [100  58 221]
 [  9   7  49]]
  Time: 134s

================================================================================
CONFIG [thermal_map]: TOP5_4_fold5
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=128, fold=4
  unfreeze_pct=0.5, finetune_lr=1e-05, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/278
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 53: early stopping
Restoring model weights from the end of the best epoch: 38.
  Stage 1 best: val_kappa=0.2435 at epoch 38/53
  Stage 2: unfreezing top 50% (96/191 layers, BN frozen)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 2 best: val_kappa=0.2386 at epoch 15/25
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 128, 128, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2386, acc=0.3847, f1=0.3896
  Confusion matrix:
[[100  59  18]
 [197  95  81]
 [ 17   7  42]]
  Time: 200s

  BEST from this round: R4_lr1e3_b64_e100 (kappa=0.4269, s1_kappa=0.4387)

================================================================================
CONFIG [thermal_map]: TOP5_5_fold1
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/278
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
  Stage 1 best: val_kappa=0.4482 at epoch 15/30
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.4373 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4110, acc=0.4275, f1=0.4248
  Confusion matrix:
[[163  25  11]
 [215  62 117]
 [ 21   6  70]]
  Time: 324s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold2
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.3398 at epoch 8/23
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 2 best: val_kappa=0.3303 at epoch 14/24
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3303, acc=0.4187, f1=0.4209
  Confusion matrix:
[[ 83  19  17]
 [120  49  68]
 [ 14   5  43]]
  Time: 340s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold3
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.4333 at epoch 16/31
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.3780 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3780, acc=0.4251, f1=0.4260
  Confusion matrix:
[[80 30  9]
 [92 55 90]
 [13  4 41]]
  Time: 311s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold4
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.4449 at epoch 11/26
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 19.
  Stage 2 best: val_kappa=0.4741 at epoch 19/29
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4741, acc=0.4407, f1=0.4345
  Confusion matrix:
[[103   9   8]
 [126  32  77]
 [  5   6  47]]
  Time: 398s

================================================================================
CONFIG [thermal_map]: TOP5_5_fold5
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.3558 at epoch 17/32
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.3301 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3301, acc=0.3632, f1=0.3601
  Confusion matrix:
[[ 85  21  12]
 [130  26  80]
 [ 14   6  39]]
  Time: 379s

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
  Alpha values (sum=3.0): [0.721, 0.341, 1.938]
  Trainable weights: 6/320
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
  Stage 1 best: val_kappa=0.4051 at epoch 17/32
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.3881 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.3636, acc=0.4203, f1=0.4184
  Confusion matrix:
[[128  45  26]
 [134 101 159]
 [ 12  24  61]]
  Time: 321s

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
  Alpha values (sum=3.0): [0.726, 0.345, 1.929]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.4309 at epoch 16/31
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.4255 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4255, acc=0.4569, f1=0.4562
  Confusion matrix:
[[ 91  15  13]
 [103  54  80]
 [  8   8  46]]
  Time: 316s

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
  Alpha values (sum=3.0): [0.723, 0.344, 1.933]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 28: early stopping
Restoring model weights from the end of the best epoch: 13.
  Stage 1 best: val_kappa=0.4365 at epoch 13/28
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.4133 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4133, acc=0.4372, f1=0.4362
  Confusion matrix:
[[83 28  8]
 [97 61 79]
 [ 6 15 37]]
  Time: 322s

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
  Alpha values (sum=3.0): [0.727, 0.346, 1.926]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 26: early stopping
Restoring model weights from the end of the best epoch: 11.
  Stage 1 best: val_kappa=0.5185 at epoch 11/26
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4915 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4915, acc=0.4915, f1=0.4958
  Confusion matrix:
[[83 24 13]
 [67 69 99]
 [ 1  6 51]]
  Time: 276s

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
  Alpha values (sum=3.0): [0.725, 0.345, 1.93]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(64, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.4340 at epoch 8/23
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.4343 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: thermal_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.4343, acc=0.3801, f1=0.3501
  Confusion matrix:
[[106   3   9]
 [139   3  94]
 [ 10   1  48]]
  Time: 256s

================================================================================
THERMAL MAP SEARCH COMPLETE — SUMMARY
================================================================================
  R1: Backbone+Freeze: R1_ResNet50V2_frozen → kappa=0.4036
  R2: Head: R2_large → kappa=0.4232
  R3: Loss+Reg+Alpha: R3_focal_g2_mixup02 → kappa=0.4260
  R4: Training+Optim: R4_adamw_wd1e3 → kappa=0.4298
  R5: Aug+ImgSize: R5_aug_on_128 → kappa=0.4343
  R6: FineTuning: R6_ft_top50_50ep → kappa=0.4277

────────────────────────────────────────────────────────────
TOP 5 CONFIGS — 5-FOLD RESULTS
────────────────────────────────────────────────────────────

  Rank   Config                           Fold0       Mean±Std
  ------------------------------------------------------------
  1      R4_lr1e3_b64_cosine             0.4280 0.3987±0.0497
  2      R4_adamw_wd1e3                  0.4298 0.3964±0.0517
  3      R4_lr1e3_b64_e100               0.4269 0.3847±0.0542
  4      R5_aug_on_128                   0.4343 0.3207±0.0585
  5      R6_ft_top50_50ep                0.4277 0.3106±0.0651

────────────────────────────────────────────────────────────
DETAILED RESULTS
────────────────────────────────────────────────────────────

#1: R4_lr1e3_b64_cosine (TOP5_3):
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=cosine, batch=64
  optimizer=adam, weight_decay=0.0
  epochs=60+30ft (unfreeze 20%), warmup=5
  aug=True, mixup=True(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4263     0.4261       0.4239
  Fold 2     0.3403     0.4234       0.4391
  Fold 3     0.3934     0.4589       0.4583
  Fold 4     0.4780     0.4818       0.4791
  Fold 5     0.3555     0.4019       0.4057
  ----------------------------------------
  Mean       0.3987     0.4384       0.4412
  Std        0.0497     0.0283       0.0257

#2: R4_adamw_wd1e3 (TOP5_2):
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=True(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4045     0.4029       0.3968
  Fold 2     0.3235     0.4211       0.4247
  Fold 3     0.4196     0.4372       0.4355
  Fold 4     0.4745     0.4552       0.4534
  Fold 5     0.3597     0.3826       0.3812
  ----------------------------------------
  Mean       0.3964     0.4198       0.4183
  Std        0.0517     0.0254       0.0261

#3: R4_lr1e3_b64_e100 (TOP5_5):
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=True(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4110     0.4275       0.4248
  Fold 2     0.3303     0.4187       0.4209
  Fold 3     0.3780     0.4251       0.4260
  Fold 4     0.4741     0.4407       0.4345
  Fold 5     0.3301     0.3632       0.3601
  ----------------------------------------
  Mean       0.3847     0.4150       0.4133
  Std        0.0542     0.0269       0.0269

#4: R5_aug_on_128 (TOP5_1):
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=True(alpha=0.2), img_size=128
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4306     0.4058       0.3990
  Fold 2     0.3267     0.4061       0.4106
  Fold 3     0.2948     0.4725       0.4495
  Fold 4     0.2877     0.3617       0.3554
  Fold 5     0.2637     0.4594       0.4527
  ----------------------------------------
  Mean       0.3207     0.4211       0.4134
  Std        0.0585     0.0402       0.0358

#5: R6_ft_top50_50ep (TOP5_4):
  backbone=ResNet50V2, freeze=frozen
  head=[256], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+50ft (unfreeze 50%), warmup=0
  aug=True, mixup=True(alpha=0.2), img_size=128
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.4319     0.4217       0.4145
  Fold 2     0.2833     0.3997       0.4056
  Fold 3     0.2870     0.4142       0.4070
  Fold 4     0.3121     0.3376       0.3442
  Fold 5     0.2386     0.3847       0.3896
  ----------------------------------------
  Mean       0.3106     0.3916       0.3922
  Std        0.0651     0.0298       0.0253

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
  Fold 1     0.3636     0.4203       0.4184
  Fold 2     0.4255     0.4569       0.4562
  Fold 3     0.4133     0.4372       0.4362
  Fold 4     0.4915     0.4915       0.4958
  Fold 5     0.4343     0.3801       0.3501
  ----------------------------------------
  Mean       0.4257     0.4372       0.4313
  Std        0.0410     0.0371       0.0481

────────────────────────────────────────────────────────────
STATISTICAL COMPARISON: #1 R4_lr1e3_b64_cosine vs BASELINE
────────────────────────────────────────────────────────────
  Mean Kappa diff:    -0.0269  (0.4257 → 0.3987)
  Mean Accuracy diff: +0.0012  (0.4372 → 0.4384)
  Mean F1 diff:       +0.0099  (0.4313 → 0.4412)

  Paired t-test on kappa (n=5 folds):
    t-statistic = -1.0063
    p-value     = 0.3712
    → NOT statistically significant (p >= 0.05)

Results saved to: /workspace/DFUMultiClassification/agent_communication/thermal_map_pipeline_audit/thermal_map_search_results.csv
Best config saved to: /workspace/DFUMultiClassification/agent_communication/thermal_map_pipeline_audit/thermal_map_best_config.json
