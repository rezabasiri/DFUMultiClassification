# Depth RGB Hyperparameter Search Plan

## Goal
Find the best architecture + training configuration for `depth_rgb` standalone to maximize **val_cohen_kappa** (currently ~0.24 across 3 folds).

## Current Baseline
- **Backbone**: EfficientNetB0 (frozen), head-only training
- **Head**: Dense(128, relu) + BN + Dropout(0.3) + Dense(3, softmax)
- **Loss**: Focal loss (gamma=2.0, alpha=frequency-based, ordinal_weight=0.0)
- **LR**: 1e-3, ReduceLROnPlateau(factor=0.5, patience=7), EarlyStopping(patience=15)
- **Epochs**: 50 (Stage 1 only, no fine-tuning)
- **Augmentation**: General augmentation ON (brightness, contrast, saturation, noise)
- **Batch size**: 64
- **Image size**: 256x256
- **Val Kappa**: ~0.24 (3-fold average)

## Backbones Tested

| Backbone | Params | Why |
|----------|--------|-----|
| **SimpleCNN** | ~200K | No transfer learning, from-scratch baseline |
| **EfficientNetB0** | 4M | Current default, good efficiency/accuracy ratio |
| **EfficientNetB2** | 9M | Larger EfficientNet, more capacity |
| **DenseNet121** | 8M | Medical imaging standard — dense connections improve feature reuse |
| **ResNet50V2** | 25M | Classic baseline, pre-activation for better gradient flow |
| **MobileNetV3Large** | 5M | Lightweight, less overfitting risk with small datasets |

## Search Dimensions

### Round 1: Backbone + Freeze Strategy (11 configs)
- 5 pretrained backbones x 2 freeze strategies (frozen, partial_unfreeze) + 1 SimpleCNN
- Each frozen config also automatically gets a Stage 2 (partial fine-tune, 30 epochs, 1e-5 LR)

### Round 2: Head Architecture (5 configs)
- tiny: Dense(32), small: Dense(64), medium: Dense(128), large: Dense(256), two_layer: Dense(256)+Dense(64)
- All with BN + Dropout(0.3)

### Round 3: Loss + Regularization (8 configs)
- Focal gamma=2.0/3.0, CCE
- Dropout 0.2/0.3/0.5
- Label smoothing 0.1
- **L2 regularization** on head (1e-3)
- **Mixup augmentation** (alpha=0.2) — blends image pairs for better generalization

### Round 4: Training Dynamics (6 configs)
- LR: 5e-4, 1e-3, 3e-3
- Batch size: 32, 64
- **Cosine annealing** with 5-epoch warmup (vs ReduceLROnPlateau)
- Extended training: 100 epochs

### Round 5: Augmentation + Image Size (4 configs)
- Augmentation ON/OFF x Image size 224/256

## Search Strategy

**Sequential elimination** — each round takes the best config from the previous round and varies one dimension group:

1. **Round 1**: Find best backbone + freeze → ~11 configs
2. **Round 2**: Find best head → 5 configs
3. **Round 3**: Find best loss/regularization → 8 configs
4. **Round 4**: Find best training dynamics → 6 configs
5. **Round 5**: Find best aug/image-size → 4 configs
6. **Final**: Run best config on all 3 folds → 3 runs

**Total: ~37 configs on fold 1 + 3-fold validation of the winner.**

## Key Techniques Beyond Baseline
1. **DenseNet121** — widely used in medical imaging literature, dense connections help with limited data
2. **Cosine annealing** — smoother LR decay often outperforms step-based schedules
3. **Mixup** — virtual training examples via linear interpolation, strong regularizer for small datasets
4. **L2 regularization** — weight decay on head layers reduces overfitting
5. **Label smoothing** — prevents overconfident predictions
6. **Warmup** — gradual LR increase avoids early training instability

## Output
- CSV with all results: `results/depth_rgb_search_results.csv`
- Best config JSON: `results/depth_rgb_best_config.json`
- Best config's 3-fold mean kappa at the end
