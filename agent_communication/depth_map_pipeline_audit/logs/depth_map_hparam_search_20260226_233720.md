Logging to: /workspace/DFUMultiClassification/agent_communication/depth_map_pipeline_audit/logs/depth_map_hparam_search_20260226_233720.log
================================================================================
DEPTH MAP HYPERPARAMETER SEARCH
================================================================================
Loaded 3108 samples for depth_map
FRESH START — backed up old results to /workspace/DFUMultiClassification/agent_communication/depth_map_pipeline_audit/depth_map_search_results.csv.bak

################################################################################
ROUND 1: BACKBONE + FREEZE STRATEGY
################################################################################

================================================================================
CONFIG [depth_map]: R1_EfficientNetB0_frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.1043 at epoch 7/22
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0751 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0803, acc=0.4822, f1=0.3427
  Confusion matrix:
[[145 245  32]
 [191 457  57]
 [ 23 105   6]]
  Time: 247s

================================================================================
CONFIG [depth_map]: R1_EfficientNetB0_partial_unfreeze
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 48/320
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 1 best: val_kappa=0.0628 at epoch 1/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0614, acc=0.2308, f1=0.2075
  Confusion matrix:
[[207   0 215]
 [304   0 401]
 [ 50   0  84]]
  Time: 138s

================================================================================
CONFIG [depth_map]: R1_EfficientNetB2_frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/448
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.0644 at epoch 9/24
  Stage 2: unfreezing top 20% (69/341 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.0578 at epoch 6/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0538, acc=0.4148, f1=0.3425
  Confusion matrix:
[[191 151  80]
 [250 313 142]
 [ 43  72  19]]
  Time: 389s

================================================================================
CONFIG [depth_map]: R1_EfficientNetB2_partial_unfreeze
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 68/448
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.0281 at epoch 2/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0282, acc=0.2776, f1=0.2772
  Confusion matrix:
[[128  63 231]
 [220 132 353]
 [ 30  14  90]]
  Time: 204s

================================================================================
CONFIG [depth_map]: R1_DenseNet121_frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
[1m       0/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 0s/step[1m   16384/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m2:36[0m 5us/step[1m   49152/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m2:03[0m 4us/step[1m   81920/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:55[0m 4us/step[1m  131072/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:24[0m 3us/step[1m  163840/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:17[0m 3us/step[1m  221184/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:03[0m 2us/step[1m  303104/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m51s[0m 2us/step [1m  376832/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m45s[0m 2us/step[1m  507904/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m36s[0m 1us/step[1m  589824/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m33s[0m 1us/step[1m  786432/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m26s[0m 1us/step[1m  901120/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m25s[0m 1us/step[1m 1179648/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m20s[0m 1us/step[1m 1310720/29084464[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m19s[0m 1us/step[1m 1753088/29084464[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1us/step[1m 1933312/29084464[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1us/step[1m 2613248/29084464[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m10s[0m 0us/step[1m 2777088/29084464[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m10s[0m 0us/step[1m 3760128/29084464[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m7s[0m 0us/step [1m 4038656/29084464[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m7s[0m 0us/step[1m 5341184/29084464[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m5s[0m 0us/step[1m 5718016/29084464[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m5s[0m 0us/step[1m 7667712/29084464[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m3s[0m 0us/step[1m 8224768/29084464[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m3s[0m 0us/step[1m10829824/29084464[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 0us/step[1m11567104/29084464[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 0us/step[1m14688256/29084464[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 0us/step[1m15695872/29084464[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 0us/step[1m18792448/29084464[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 0us/step[1m19783680/29084464[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step[1m22708224/29084464[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 0us/step[1m24412160/29084464[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 0us/step[1m26542080/29084464[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 0us/step[1m28246016/29084464[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 0us/step[1m29084464/29084464[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 0us/step
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.1235 at epoch 5/20
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1378 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1390, acc=0.4465, f1=0.3827
  Confusion matrix:
[[212 146  64]
 [243 320 142]
 [ 39  64  31]]
  Time: 379s

================================================================================
CONFIG [depth_map]: R1_DenseNet121_partial_unfreeze
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 78/612
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 24.
  Stage 1 best: val_kappa=0.0886 at epoch 27/39
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0830, acc=0.4496, f1=0.3673
  Confusion matrix:
[[182 204  36]
 [251 363  91]
 [ 49  63  22]]
  Time: 483s

================================================================================
CONFIG [depth_map]: R1_ResNet50V2_frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
[1m       0/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 0s/step[1m   16384/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m8:28[0m 5us/step[1m   49152/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m6:40[0m 4us/step[1m   81920/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m6:16[0m 4us/step[1m  131072/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m4:37[0m 3us/step[1m  172032/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m3:59[0m 3us/step[1m  229376/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m3:24[0m 2us/step[1m  327680/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m2:38[0m 2us/step[1m  393216/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m2:24[0m 2us/step[1m  540672/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:54[0m 1us/step[1m  606208/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:50[0m 1us/step[1m  851968/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:24[0m 1us/step[1m  925696/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:22[0m 1us/step[1m 1245184/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:04[0m 1us/step[1m 1327104/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m1:04[0m 1us/step[1m 1818624/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m49s[0m 1us/step [1m 1966080/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m47s[0m 1us/step[1m 2629632/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m37s[0m 0us/step[1m 2826240/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m36s[0m 0us/step[1m 3825664/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m27s[0m 0us/step[1m 4169728/94668760[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m26s[0m 0us/step[1m 5390336/94668760[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m21s[0m 0us/step[1m 5996544/94668760[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m19s[0m 0us/step[1m 7782400/94668760[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 0us/step[1m 8724480/94668760[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 0us/step[1m10895360/94668760[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m11s[0m 0us/step[1m12288000/94668760[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m10s[0m 0us/step[1m14835712/94668760[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m8s[0m 0us/step [1m16785408/94668760[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m7s[0m 0us/step[1m18882560/94668760[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m7s[0m 0us/step[1m20979712/94668760[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m6s[0m 0us/step[1m23068672/94668760[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m5s[0m 0us/step[1m24756224/94668760[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m5s[0m 0us/step[1m26886144/94668760[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m4s[0m 0us/step[1m28868608/94668760[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m4s[0m 0us/step[1m30769152/94668760[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m4s[0m 0us/step[1m32759808/94668760[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m4s[0m 0us/step[1m34619392/94668760[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m3s[0m 0us/step[1m36618240/94668760[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m3s[0m 0us/step[1m38469632/94668760[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m3s[0m 0us/step[1m40468480/94668760[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m3s[0m 0us/step[1m42311680/94668760[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m2s[0m 0us/step[1m44277760/94668760[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 0us/step[1m46145536/94668760[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 0us/step[1m48070656/94668760[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m2s[0m 0us/step[1m49987584/94668760[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m2s[0m 0us/step[1m51986432/94668760[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m2s[0m 0us/step[1m53821440/94668760[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m2s[0m 0us/step[1m55803904/94668760[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m1s[0m 0us/step[1m57655296/94668760[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 0us/step[1m59654144/94668760[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 0us/step[1m61489152/94668760[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 0us/step[1m63455232/94668760[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 0us/step[1m65331200/94668760[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 0us/step[1m67297280/94668760[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m1s[0m 0us/step[1m69148672/94668760[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m1s[0m 0us/step[1m71098368/94668760[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m1s[0m 0us/step[1m72982528/94668760[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 0us/step[1m74964992/94668760[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 0us/step[1m76824576/94668760[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 0us/step[1m78749696/94668760[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 0us/step[1m80650240/94668760[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 0us/step[1m82501632/94668760[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 0us/step[1m84475904/94668760[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 0us/step[1m86376448/94668760[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 0us/step[1m88285184/94668760[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 0us/step[1m90218496/94668760[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 0us/step[1m92094464/94668760[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 0us/step[1m93962240/94668760[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 0us/step[1m94668760/94668760[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 0us/step
  Trainable weights: 6/278
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.0563 at epoch 5/20
  Stage 2: unfreezing top 20% (39/191 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0499 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0485, acc=0.4830, f1=0.3481
  Confusion matrix:
[[111 276  35]
 [160 484  61]
 [ 26  94  14]]
  Time: 265s

================================================================================
CONFIG [depth_map]: R1_ResNet50V2_partial_unfreeze
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 42/278
  Using partial-unfreeze LR=0.0001
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.0713 at epoch 4/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0666, acc=0.4441, f1=0.3386
  Confusion matrix:
[[ 87 265  70]
 [131 444 130]
 [ 12  93  29]]
  Time: 185s

  BEST from this round: R1_DenseNet121_frozen (kappa=0.1390, s1_kappa=0.1235)

################################################################################
ROUND 2: HEAD ARCHITECTURE
################################################################################

================================================================================
CONFIG [depth_map]: R2_tiny
  backbone=DenseNet121, freeze=frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 45: early stopping
Restoring model weights from the end of the best epoch: 30.
  Stage 1 best: val_kappa=0.1380 at epoch 30/45
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1051 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1018, acc=0.3902, f1=0.3534
  Confusion matrix:
[[210 111 101]
 [259 239 207]
 [ 46  45  43]]
  Time: 614s

================================================================================
CONFIG [depth_map]: R2_small
  backbone=DenseNet121, freeze=frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1236 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.1575 at epoch 7/17
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1594, acc=0.4449, f1=0.3947
  Confusion matrix:
[[274 107  41]
 [351 250 104]
 [ 60  37  37]]
  Time: 410s

================================================================================
CONFIG [depth_map]: R2_medium
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.1118 at epoch 5/20
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1337 at epoch 3/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1300, acc=0.4148, f1=0.3644
  Confusion matrix:
[[247 106  69]
 [310 242 153]
 [ 50  50  34]]
  Time: 369s

================================================================================
CONFIG [depth_map]: R2_large
  backbone=DenseNet121, freeze=frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 42: early stopping
Restoring model weights from the end of the best epoch: 27.
  Stage 1 best: val_kappa=0.1146 at epoch 27/42
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1224 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1243, acc=0.4362, f1=0.3791
  Confusion matrix:
[[197 153  72]
 [238 317 150]
 [ 38  60  36]]
  Time: 585s

================================================================================
CONFIG [depth_map]: R2_two_layer
  backbone=DenseNet121, freeze=frozen
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
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 10/618
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 43: early stopping
Restoring model weights from the end of the best epoch: 28.
  Stage 1 best: val_kappa=0.1514 at epoch 28/43
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1494 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1489, acc=0.4211, f1=0.3763
  Confusion matrix:
[[212 136  74]
 [244 277 184]
 [ 41  51  42]]
  Time: 606s

  BEST from this round: R2_small (kappa=0.1594, s1_kappa=0.1236)

################################################################################
ROUND 3: LOSS + REGULARIZATION
################################################################################

================================================================================
CONFIG [depth_map]: R3_focal_g2_d03
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.0963 at epoch 4/19
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.1307 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1252, acc=0.4100, f1=0.3567
  Confusion matrix:
[[274  96  52]
 [368 212 125]
 [ 62  41  31]]
  Time: 388s

================================================================================
CONFIG [depth_map]: R3_focal_g3_d03
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=3.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 30: early stopping
Restoring model weights from the end of the best epoch: 15.
  Stage 1 best: val_kappa=0.1459 at epoch 15/30
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1255 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1232, acc=0.4155, f1=0.3760
  Confusion matrix:
[[210 126  86]
 [260 267 178]
 [ 45  42  47]]
  Time: 463s

================================================================================
CONFIG [depth_map]: R3_cce_d03
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=cce, gamma=0.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.1453 at epoch 4/19
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.1543 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1590, acc=0.4330, f1=0.3645
  Confusion matrix:
[[275 118  29]
 [368 249  88]
 [ 51  61  22]]
  Time: 380s

================================================================================
CONFIG [depth_map]: R3_focal_g2_d05
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.5, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.1601 at epoch 17/32
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1584 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1586, acc=0.4259, f1=0.3804
  Confusion matrix:
[[242 112  68]
 [286 254 165]
 [ 46  47  41]]
  Time: 501s

================================================================================
CONFIG [depth_map]: R3_focal_g2_d02
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.2, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.1317 at epoch 16/31
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1385 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1348, acc=0.4528, f1=0.3834
  Confusion matrix:
[[258 132  32]
 [341 289  75]
 [ 56  54  24]]
  Time: 482s

================================================================================
CONFIG [depth_map]: R3_focal_g2_ls01
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.1, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1134 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.1325 at epoch 12/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1335, acc=0.3513, f1=0.3267
  Confusion matrix:
[[258  48 116]
 [338 123 244]
 [ 51  21  62]]
  Time: 430s

================================================================================
CONFIG [depth_map]: R3_focal_g2_l2_1e3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.001
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 32: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 1 best: val_kappa=0.1376 at epoch 17/32
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.1303 at epoch 11/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1269, acc=0.4052, f1=0.3695
  Confusion matrix:
[[227 101  94]
 [272 236 197]
 [ 48  38  48]]
  Time: 563s

================================================================================
CONFIG [depth_map]: R3_focal_g2_mixup02
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.801, 0.321, 1.878]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 40: early stopping
Restoring model weights from the end of the best epoch: 25.
  Stage 1 best: val_kappa=0.1584 at epoch 25/40
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.1657 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1606, acc=0.4695, f1=0.3989
  Confusion matrix:
[[223 162  37]
 [262 340 103]
 [ 45  60  29]]
  Time: 591s

================================================================================
CONFIG [depth_map]: R3_focal_g2_alpha0
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
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
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1253 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1244 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1274, acc=0.5044, f1=0.3504
  Confusion matrix:
[[241 181   0]
 [307 395   3]
 [ 49  85   0]]
  Time: 359s

================================================================================
CONFIG [depth_map]: R3_focal_g2_alpha1
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=1.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=1.0): [0.267, 0.107, 0.626]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.1203 at epoch 14/29
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 2 best: val_kappa=0.1109 at epoch 8/18
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1080, acc=0.4132, f1=0.3674
  Confusion matrix:
[[218 121  83]
 [265 264 176]
 [ 51  44  39]]
  Time: 527s

================================================================================
CONFIG [depth_map]: R3_focal_g2_alpha5
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.1578 at epoch 4/19
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1630 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1614, acc=0.4750, f1=0.3814
  Confusion matrix:
[[246 154  22]
 [306 338  61]
 [ 43  76  15]]
  Time: 360s

================================================================================
CONFIG [depth_map]: R3_focal_g2_alpha8
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=8.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=8.0): [2.137, 0.856, 5.007]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.1327 at epoch 16/31
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1348 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1336, acc=0.4290, f1=0.3753
  Confusion matrix:
[[261 119  42]
 [350 248 107]
 [ 60  42  32]]
  Time: 476s

  BEST from this round: R3_focal_g2_alpha5 (kappa=0.1614, s1_kappa=0.1578)

################################################################################
ROUND 4: TRAINING DYNAMICS
################################################################################

================================================================================
CONFIG [depth_map]: R4_lr5e4_b64_plateau
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.0005, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.1275 at epoch 16/31
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.1345 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1352, acc=0.4259, f1=0.3821
  Confusion matrix:
[[199 146  77]
 [248 292 165]
 [ 39  49  46]]
  Time: 498s

================================================================================
CONFIG [depth_map]: R4_lr1e3_b64_plateau
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1583 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.1441 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1441, acc=0.4274, f1=0.3637
  Confusion matrix:
[[288  99  35]
 [384 226  95]
 [ 61  48  25]]
  Time: 380s

================================================================================
CONFIG [depth_map]: R4_lr3e3_b64_plateau
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.003, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1625 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1485 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1500, acc=0.4830, f1=0.3837
  Confusion matrix:
[[259 150  13]
 [318 337  50]
 [ 56  65  13]]
  Time: 349s

================================================================================
CONFIG [depth_map]: R4_lr1e3_b32_plateau
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1187 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.1732 at epoch 6/16

  POST-EVAL: kappa=0.1702, acc=0.4465, f1=0.3913
  Confusion matrix:
[[236 142  44]
 [294 292 119]
 [ 43  56  35]]
  Time: 415s

================================================================================
CONFIG [depth_map]: R4_lr1e3_b64_cosine
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=60, batch=64
  lr_schedule=cosine, warmup=5
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 48: early stopping
Restoring model weights from the end of the best epoch: 28.
  Stage 1 best: val_kappa=0.1542 at epoch 28/48
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1492 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1496, acc=0.4211, f1=0.3767
  Confusion matrix:
[[238 122  62]
 [291 252 162]
 [ 51  42  41]]
  Time: 645s

================================================================================
CONFIG [depth_map]: R4_lr1e3_b64_e100
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 34: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.1237 at epoch 17/34
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.1330 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1313, acc=0.4171, f1=0.3757
  Confusion matrix:
[[223 125  74]
 [285 259 161]
 [ 47  43  44]]
  Time: 524s

================================================================================
CONFIG [depth_map]: R4_adamw_wd1e4
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.0001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1142 at epoch 3/23
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.1468 at epoch 5/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1466, acc=0.4179, f1=0.3596
  Confusion matrix:
[[290  96  36]
 [389 209 107]
 [ 63  43  28]]
  Time: 422s

================================================================================
CONFIG [depth_map]: R4_adamw_wd1e3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.1246 at epoch 3/23
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.1567 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1621, acc=0.4290, f1=0.3732
  Confusion matrix:
[[284 101  37]
 [368 226 111]
 [ 59  44  31]]
  Time: 439s

  BEST from this round: R4_lr1e3_b32_plateau (kappa=0.1702, s1_kappa=0.1187)

################################################################################
ROUND 5: AUGMENTATION + IMAGE SIZE
################################################################################

================================================================================
CONFIG [depth_map]: R5_aug_on_128
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.0554 at epoch 5/20
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.0577 at epoch 5/14

  POST-EVAL: kappa=0.0566, acc=0.3466, f1=0.3114
  Confusion matrix:
[[217  93 112]
 [302 186 217]
 [ 52  48  34]]
  Time: 213s

================================================================================
CONFIG [depth_map]: R5_aug_off_128
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=128, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 128, 128, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 128, 128, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.0821 at epoch 4/19
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.0964 at epoch 11/17

  POST-EVAL: kappa=0.0911, acc=0.3973, f1=0.3524
  Confusion matrix:
[[193 139  90]
 [263 270 172]
 [ 39  57  38]]
  Time: 219s

================================================================================
CONFIG [depth_map]: R5_aug_on_256
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.1206 at epoch 5/20
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.1473 at epoch 5/15

  POST-EVAL: kappa=0.1469, acc=0.3965, f1=0.3664
  Confusion matrix:
[[220 103  99]
 [266 227 212]
 [ 37  44  53]]
  Time: 424s

================================================================================
CONFIG [depth_map]: R5_aug_off_256
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=False, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.0934 at epoch 14/29
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 2 best: val_kappa=0.1017 at epoch 10/20

  POST-EVAL: kappa=0.1015, acc=0.4401, f1=0.3754
  Confusion matrix:
[[203 155  64]
 [246 322 137]
 [ 49  55  30]]
  Time: 566s

  BEST from this round: R5_aug_on_256 (kappa=0.1469, s1_kappa=0.1206)

################################################################################
ROUND 6: FINE-TUNING STRATEGY
################################################################################

================================================================================
CONFIG [depth_map]: R6_ft_top20_30ep
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.1260 at epoch 16/31
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.1348 at epoch 5/15

  POST-EVAL: kappa=0.1357, acc=0.4005, f1=0.3599
  Confusion matrix:
[[241 109  72]
 [303 223 179]
 [ 52  41  41]]
  Time: 536s

================================================================================
CONFIG [depth_map]: R6_ft_top40_30ep
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.4, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 50: early stopping
Restoring model weights from the end of the best epoch: 35.
  Stage 1 best: val_kappa=0.1307 at epoch 35/50
  Stage 2: unfreezing top 40% (172/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1030 at epoch 9/11

  POST-EVAL: kappa=0.1036, acc=0.3918, f1=0.3543
  Confusion matrix:
[[216 112  94]
 [289 235 181]
 [ 44  47  43]]
  Time: 698s

================================================================================
CONFIG [depth_map]: R6_ft_top50_50ep
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.5, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.1344 at epoch 5/20
  Stage 2: unfreezing top 50% (214/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1406 at epoch 2/12

  POST-EVAL: kappa=0.1387, acc=0.4481, f1=0.3872
  Confusion matrix:
[[229 155  38]
 [312 306  87]
 [ 46  58  30]]
  Time: 416s

================================================================================
CONFIG [depth_map]: R6_ft_top20_50ep
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=5e-06, finetune_epochs=50
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.335, 0.535, 3.129]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Restoring model weights from the end of the best epoch: 36.
  Stage 1 best: val_kappa=0.1384 at epoch 36/50
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1336 at epoch 1/11

  POST-EVAL: kappa=0.1334, acc=0.4290, f1=0.3726
  Confusion matrix:
[[212 147  63]
 [264 296 145]
 [ 40  61  33]]
  Time: 683s

  BEST from this round: R6_ft_top50_50ep (kappa=0.1387, s1_kappa=0.1344)

################################################################################
TOP 5 SELECTION: 5-FOLD VALIDATION OF BEST CONFIGS
################################################################################

Top 5 configs by fold-0 kappa:
  1. R4_lr1e3_b32_plateau → kappa=0.1702
  2. R4_adamw_wd1e3 → kappa=0.1621
  3. R3_focal_g2_alpha5 → kappa=0.1614
  4. R3_focal_g2_mixup02 → kappa=0.1606
  5. R2_small → kappa=0.1594

  BEST from this round: R4_lr1e3_b32_plateau (kappa=0.1702, s1_kappa=0.1187)

================================================================================
CONFIG [depth_map]: TOP5_1_fold1
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.192, 0.584, 3.224]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.2164 at epoch 4/19
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 2 best: val_kappa=0.1849 at epoch 7/17

  POST-EVAL: kappa=0.1148, acc=0.4029, f1=0.3686
  Confusion matrix:
[[191 122 109]
 [236 265 204]
 [ 35  47  52]]
  Time: 433s

================================================================================
CONFIG [depth_map]: TOP5_1_fold2
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.26, 0.643, 3.097]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 1 best: val_kappa=0.1318 at epoch 9/24
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1156 at epoch 10/12

  POST-EVAL: kappa=0.1139, acc=0.4571, f1=0.3615
  Confusion matrix:
[[ 80  93  39]
 [141 276 129]
 [ 11  30  17]]
  Time: 429s

================================================================================
CONFIG [depth_map]: TOP5_1_fold3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.254, 0.555, 3.191]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.1126 at epoch 2/17
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 2 best: val_kappa=0.0651 at epoch 8/18

  POST-EVAL: kappa=0.0648, acc=0.3777, f1=0.3429
  Confusion matrix:
[[103  66  60]
 [149 131 100]
 [ 23  27  24]]
  Time: 420s

================================================================================
CONFIG [depth_map]: TOP5_1_fold4
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.134, 0.515, 3.35]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 35: early stopping
Restoring model weights from the end of the best epoch: 20.
  Stage 1 best: val_kappa=0.2159 at epoch 20/35
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1954 at epoch 2/12

  POST-EVAL: kappa=0.1888, acc=0.3774, f1=0.3670
  Confusion matrix:
[[ 91  60  39]
 [115 102 116]
 [ 22  34  41]]
  Time: 528s

================================================================================
CONFIG [depth_map]: TOP5_1_fold5
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=32
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.196, 0.576, 3.229]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 1 best: val_kappa=0.1136 at epoch 4/19
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0723 at epoch 1/11

  POST-EVAL: kappa=0.0723, acc=0.3739, f1=0.3274
  Confusion matrix:
[[ 48  52  18]
 [118 102  50]
 [ 13  22  13]]
  Time: 361s

  BEST from this round: R4_adamw_wd1e3 (kappa=0.1621, s1_kappa=0.1246)

================================================================================
CONFIG [depth_map]: TOP5_2_fold1
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.192, 0.584, 3.224]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.1775 at epoch 18/33
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 2 best: val_kappa=0.1238 at epoch 5/15
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0758, acc=0.4108, f1=0.3537
  Confusion matrix:
[[160 172  90]
 [203 323 179]
 [ 37  62  35]]
  Time: 515s

================================================================================
CONFIG [depth_map]: TOP5_2_fold2
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.26, 0.643, 3.097]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.1301 at epoch 10/25
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1320 at epoch 3/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1317, acc=0.5355, f1=0.3938
  Confusion matrix:
[[ 74 121  17]
 [130 350  66]
 [ 12  33  13]]
  Time: 401s

================================================================================
CONFIG [depth_map]: TOP5_2_fold3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.254, 0.555, 3.191]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 29: early stopping
Restoring model weights from the end of the best epoch: 14.
  Stage 1 best: val_kappa=0.0578 at epoch 14/29
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.0489 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0491, acc=0.4012, f1=0.3667
  Confusion matrix:
[[111  57  61]
 [166 136  78]
 [ 28  19  27]]
  Time: 453s

================================================================================
CONFIG [depth_map]: TOP5_2_fold4
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.134, 0.515, 3.35]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.2235 at epoch 10/25
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 16: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 2 best: val_kappa=0.2236 at epoch 6/16
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2203, acc=0.4048, f1=0.3904
  Confusion matrix:
[[114  48  28]
 [139  97  97]
 [ 31  26  40]]
  Time: 443s

================================================================================
CONFIG [depth_map]: TOP5_2_fold5
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=100, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adamw, weight_decay=0.001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.196, 0.576, 3.229]
  Trainable weights: 6/612
  Using AdamW (weight_decay=0.001)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.0511 at epoch 6/21
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.0739 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0665, acc=0.4174, f1=0.3622
  Confusion matrix:
[[ 66  45   7]
 [147 107  16]
 [ 23  16   9]]
  Time: 366s

  BEST from this round: R3_focal_g2_alpha5 (kappa=0.1614, s1_kappa=0.1578)

================================================================================
CONFIG [depth_map]: TOP5_3_fold1
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.192, 0.584, 3.224]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 42: early stopping
Restoring model weights from the end of the best epoch: 27.
  Stage 1 best: val_kappa=0.1710 at epoch 27/42
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 19: early stopping
Restoring model weights from the end of the best epoch: 9.
  Stage 2 best: val_kappa=0.1713 at epoch 9/19
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1176, acc=0.4171, f1=0.3730
  Confusion matrix:
[[189 132 101]
 [209 293 203]
 [ 36  54  44]]
  Time: 644s

================================================================================
CONFIG [depth_map]: TOP5_3_fold2
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.26, 0.643, 3.097]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.1517 at epoch 10/25
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1488 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1410, acc=0.5123, f1=0.3851
  Confusion matrix:
[[ 96 101  15]
 [173 312  61]
 [ 14  34  10]]
  Time: 413s

================================================================================
CONFIG [depth_map]: TOP5_3_fold3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.254, 0.555, 3.191]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.0618 at epoch 2/17
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0554 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0580, acc=0.4173, f1=0.3106
  Confusion matrix:
[[159  58  12]
 [241 124  15]
 [ 39  33   2]]
  Time: 325s

================================================================================
CONFIG [depth_map]: TOP5_3_fold4
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.134, 0.515, 3.35]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.2013 at epoch 10/25
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.2104 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2096, acc=0.4210, f1=0.3875
  Confusion matrix:
[[113  59  18]
 [137 122  74]
 [ 30  41  26]]
  Time: 405s

================================================================================
CONFIG [depth_map]: TOP5_3_fold5
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.0)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=5.0): [1.196, 0.576, 3.229]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.0492 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 27: early stopping
Restoring model weights from the end of the best epoch: 17.
  Stage 2 best: val_kappa=0.0914 at epoch 17/27
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0842, acc=0.3394, f1=0.3200
  Confusion matrix:
[[ 54  28  36]
 [104  74  92]
 [ 14  14  20]]
  Time: 484s

  BEST from this round: R3_focal_g2_mixup02 (kappa=0.1606, s1_kappa=0.1584)

================================================================================
CONFIG [depth_map]: TOP5_4_fold1
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=0
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.715, 0.35, 1.934]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 33: early stopping
Restoring model weights from the end of the best epoch: 18.
  Stage 1 best: val_kappa=0.2073 at epoch 18/33
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2094 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1471, acc=0.5178, f1=0.3985
  Confusion matrix:
[[126 266  30]
 [134 502  69]
 [ 20  89  25]]
  Time: 480s

================================================================================
CONFIG [depth_map]: TOP5_4_fold2
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.756, 0.386, 1.858]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 46: early stopping
Restoring model weights from the end of the best epoch: 31.
  Stage 1 best: val_kappa=0.1049 at epoch 31/46
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1186 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1248, acc=0.4363, f1=0.3573
  Confusion matrix:
[[109  71  32]
 [209 232 105]
 [ 15  28  15]]
  Time: 603s

================================================================================
CONFIG [depth_map]: TOP5_4_fold3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.752, 0.333, 1.914]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.0870 at epoch 7/22
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0792 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0846, acc=0.5227, f1=0.3633
  Confusion matrix:
[[107 116   6]
 [127 249   4]
 [ 26  47   1]]
  Time: 375s

================================================================================
CONFIG [depth_map]: TOP5_4_fold4
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.681, 0.309, 2.01]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 45: early stopping
Restoring model weights from the end of the best epoch: 30.
  Stage 1 best: val_kappa=0.2494 at epoch 30/45
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.2588 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2604, acc=0.4500, f1=0.4137
  Confusion matrix:
[[107  68  15]
 [113 144  76]
 [ 22  47  28]]
  Time: 614s

================================================================================
CONFIG [depth_map]: TOP5_4_fold5
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=True(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.717, 0.345, 1.937]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.0482 at epoch 7/22
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 13: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 2 best: val_kappa=0.0675 at epoch 3/13
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0686, acc=0.3463, f1=0.3116
  Confusion matrix:
[[ 52  42  24]
 [116  85  69]
 [ 15  19  14]]
  Time: 613s

  BEST from this round: R2_small (kappa=0.1594, s1_kappa=0.1236)

================================================================================
CONFIG [depth_map]: TOP5_5_fold1
  backbone=DenseNet121, freeze=frozen
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
  Alpha values (sum=3.0): [0.715, 0.35, 1.934]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 21.
  Stage 1 best: val_kappa=0.1712 at epoch 21/36
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1498 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1088, acc=0.4322, f1=0.3722
  Confusion matrix:
[[151 183  88]
 [169 353 183]
 [ 30  63  41]]
  Time: 833s

================================================================================
CONFIG [depth_map]: TOP5_5_fold2
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=1
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.756, 0.386, 1.858]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 1 best: val_kappa=0.1448 at epoch 2/17
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1443 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1339, acc=0.5159, f1=0.3862
  Confusion matrix:
[[ 92 105  15]
 [179 319  48]
 [ 11  37  10]]
  Time: 513s

================================================================================
CONFIG [depth_map]: TOP5_5_fold3
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=2
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.752, 0.333, 1.914]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.0734 at epoch 7/22
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0455 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0454, acc=0.3704, f1=0.3359
  Confusion matrix:
[[129  50  50]
 [200 101  79]
 [ 37  14  23]]
  Time: 557s

================================================================================
CONFIG [depth_map]: TOP5_5_fold4
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=3
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.681, 0.309, 2.01]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 25: early stopping
Restoring model weights from the end of the best epoch: 10.
  Stage 1 best: val_kappa=0.2315 at epoch 10/25
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
  Stage 2 best: val_kappa=0.2153 at epoch 4/14
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.2231, acc=0.4145, f1=0.3923
  Confusion matrix:
[[106  63  21]
 [134 118  81]
 [ 26  38  33]]
  Time: 669s

================================================================================
CONFIG [depth_map]: TOP5_5_fold5
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, epochs=50, batch=64
  lr_schedule=plateau, warmup=0
  optimizer=adam, weight_decay=0.0001
  augmentation=True, mixup=False(0.2)
  label_smooth=0.0, image_size=256, fold=4
  unfreeze_pct=0.2, finetune_lr=1e-05, finetune_epochs=30
================================================================================
  Diagnostic samples saved to /workspace/DFUMultiClassification/results/visualizations/diagnostic_samples/ (3 classes)
  Alpha values (sum=3.0): [0.717, 0.345, 1.937]
  Trainable weights: 6/612
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 18: early stopping
Restoring model weights from the end of the best epoch: 3.
  Stage 1 best: val_kappa=0.0755 at epoch 3/18
  Stage 2: unfreezing top 20% (86/428 layers, BN frozen)
Restoring model weights from the end of the best epoch: 26.
  Stage 2 best: val_kappa=0.0714 at epoch 26/30
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0781, acc=0.3257, f1=0.3095
  Confusion matrix:
[[ 51  33  34]
 [112  70  88]
 [ 14  13  21]]
  Time: 820s

################################################################################
BASELINE: EfficientNetB0 FROZEN ON ALL 5 FOLDS
################################################################################

================================================================================
CONFIG [depth_map]: BASELINE_fold1
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
  Alpha values (sum=3.0): [0.715, 0.35, 1.934]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 20: early stopping
Restoring model weights from the end of the best epoch: 5.
  Stage 1 best: val_kappa=0.0723 at epoch 5/20
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0508 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0669, acc=0.4623, f1=0.3468
  Confusion matrix:
[[211 176  35]
 [287 366  52]
 [ 46  82   6]]
  Time: 361s

================================================================================
CONFIG [depth_map]: BASELINE_fold2
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
  Alpha values (sum=3.0): [0.756, 0.386, 1.858]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 6.
  Stage 1 best: val_kappa=0.1584 at epoch 6/21
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1684 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1684, acc=0.4583, f1=0.3740
  Confusion matrix:
[[ 89  90  33]
 [155 265 126]
 [  6  32  20]]
  Time: 313s

================================================================================
CONFIG [depth_map]: BASELINE_fold3
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
  Alpha values (sum=3.0): [0.752, 0.333, 1.914]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 22: early stopping
Restoring model weights from the end of the best epoch: 7.
  Stage 1 best: val_kappa=0.1066 at epoch 7/22
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 12: early stopping
Restoring model weights from the end of the best epoch: 2.
  Stage 2 best: val_kappa=0.1064 at epoch 2/12
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1069, acc=0.4070, f1=0.3545
  Confusion matrix:
[[125  78  26]
 [186 137  57]
 [ 27  31  16]]
  Time: 359s

================================================================================
CONFIG [depth_map]: BASELINE_fold4
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
  Alpha values (sum=3.0): [0.681, 0.309, 2.01]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 23: early stopping
Restoring model weights from the end of the best epoch: 8.
  Stage 1 best: val_kappa=0.1630 at epoch 8/23
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.1725 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.1718, acc=0.4097, f1=0.3593
  Confusion matrix:
[[121  53  16]
 [163 117  53]
 [ 33  48  16]]
  Time: 359s

================================================================================
CONFIG [depth_map]: BASELINE_fold5
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
  Alpha values (sum=3.0): [0.717, 0.345, 1.937]
  Trainable weights: 6/320
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(32, 256, 256, 3))']
  warnings.warn(msg)
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(None, 256, 256, 3))']
  warnings.warn(msg)
Epoch 31: early stopping
Restoring model weights from the end of the best epoch: 16.
  Stage 1 best: val_kappa=0.0376 at epoch 16/31
  Stage 2: unfreezing top 20% (48/239 layers, BN frozen)
Epoch 11: early stopping
Restoring model weights from the end of the best epoch: 1.
  Stage 2 best: val_kappa=0.0160 at epoch 1/11
/venv/multimodal/lib/python3.11/site-packages/keras/src/models/functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: depth_map_input
Received: inputs=['Tensor(shape=(16, 256, 256, 3))']
  warnings.warn(msg)

  POST-EVAL: kappa=0.0193, acc=0.3073, f1=0.2943
  Confusion matrix:
[[ 43  36  39]
 [115  68  87]
 [ 18   7  23]]
  Time: 432s

================================================================================
DEPTH MAP SEARCH COMPLETE — SUMMARY
================================================================================
  R1: Backbone+Freeze: R1_DenseNet121_frozen → kappa=0.1390
  R2: Head: R2_small → kappa=0.1594
  R3: Loss+Reg+Alpha: R3_focal_g2_alpha5 → kappa=0.1614
  R4: Training+Optim: R4_lr1e3_b32_plateau → kappa=0.1702
  R5: Aug+ImgSize: R5_aug_on_256 → kappa=0.1469
  R6: FineTuning: R6_ft_top50_50ep → kappa=0.1387

────────────────────────────────────────────────────────────
TOP 5 CONFIGS — 5-FOLD RESULTS
────────────────────────────────────────────────────────────

  Rank   Config                           Fold0       Mean±Std
  ------------------------------------------------------------
  1      R3_focal_g2_mixup02             0.1606 0.1371±0.0677
  2      R3_focal_g2_alpha5              0.1614 0.1221±0.0521
  3      R2_small                        0.1594 0.1178±0.0604
  4      R4_lr1e3_b32_plateau            0.1702 0.1109±0.0440
  5      R4_adamw_wd1e3                  0.1621 0.1087±0.0623

────────────────────────────────────────────────────────────
DETAILED RESULTS
────────────────────────────────────────────────────────────

#1: R3_focal_g2_mixup02 (TOP5_4):
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0001
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=True(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.1471     0.5178       0.3985
  Fold 2     0.1248     0.4363       0.3573
  Fold 3     0.0846     0.5227       0.3633
  Fold 4     0.2604     0.4500       0.4137
  Fold 5     0.0686     0.3463       0.3116
  ----------------------------------------
  Mean       0.1371     0.4546       0.3689
  Std        0.0677     0.0644       0.0356

#2: R3_focal_g2_alpha5 (TOP5_3):
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0001
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.1176     0.4171       0.3730
  Fold 2     0.1410     0.5123       0.3851
  Fold 3     0.0580     0.4173       0.3106
  Fold 4     0.2096     0.4210       0.3875
  Fold 5     0.0842     0.3394       0.3200
  ----------------------------------------
  Mean       0.1221     0.4214       0.3552
  Std        0.0521     0.0548       0.0331

#3: R2_small (TOP5_5):
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=3.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adam, weight_decay=0.0001
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.2), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.1088     0.4322       0.3722
  Fold 2     0.1339     0.5159       0.3862
  Fold 3     0.0454     0.3704       0.3359
  Fold 4     0.2231     0.4145       0.3923
  Fold 5     0.0781     0.3257       0.3095
  ----------------------------------------
  Mean       0.1178     0.4118       0.3592
  Std        0.0604     0.0639       0.0317

#4: R4_lr1e3_b32_plateau (TOP5_1):
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, schedule=plateau, batch=32
  optimizer=adam, weight_decay=0.0
  epochs=50+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.1148     0.4029       0.3686
  Fold 2     0.1139     0.4571       0.3615
  Fold 3     0.0648     0.3777       0.3429
  Fold 4     0.1888     0.3774       0.3670
  Fold 5     0.0723     0.3739       0.3274
  ----------------------------------------
  Mean       0.1109     0.3978       0.3535
  Std        0.0440     0.0314       0.0159

#5: R4_adamw_wd1e3 (TOP5_2):
  backbone=DenseNet121, freeze=frozen
  head=[64], dropout=0.3, bn=True, l2=0.0
  loss=focal, gamma=2.0, alpha_sum=5.0
  lr=0.001, schedule=plateau, batch=64
  optimizer=adamw, weight_decay=0.001
  epochs=100+30ft (unfreeze 20%), warmup=0
  aug=True, mixup=False(alpha=0.0), img_size=256
  label_smoothing=0.0

  Fold        Kappa   Accuracy   F1 (macro)
  ----------------------------------------
  Fold 1     0.0758     0.4108       0.3537
  Fold 2     0.1317     0.5355       0.3938
  Fold 3     0.0491     0.4012       0.3667
  Fold 4     0.2203     0.4048       0.3904
  Fold 5     0.0665     0.4174       0.3622
  ----------------------------------------
  Mean       0.1087     0.4340       0.3733
  Std        0.0623     0.0511       0.0159

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
  Fold 1     0.0669     0.4623       0.3468
  Fold 2     0.1684     0.4583       0.3740
  Fold 3     0.1069     0.4070       0.3545
  Fold 4     0.1718     0.4097       0.3593
  Fold 5     0.0193     0.3073       0.2943
  ----------------------------------------
  Mean       0.1067     0.4089       0.3458
  Std        0.0588     0.0559       0.0272

────────────────────────────────────────────────────────────
STATISTICAL COMPARISON: #1 R3_focal_g2_mixup02 vs BASELINE
────────────────────────────────────────────────────────────
  Mean Kappa diff:    +0.0304  (0.1067 → 0.1371)
  Mean Accuracy diff: +0.0457  (0.4089 → 0.4546)
  Mean F1 diff:       +0.0231  (0.3458 → 0.3689)

  Paired t-test on kappa (n=5 folds):
    t-statistic = 1.1301
    p-value     = 0.3216
    → NOT statistically significant (p >= 0.05)

Results saved to: /workspace/DFUMultiClassification/agent_communication/depth_map_pipeline_audit/depth_map_search_results.csv
Best config saved to: /workspace/DFUMultiClassification/agent_communication/depth_map_pipeline_audit/depth_map_best_config.json
