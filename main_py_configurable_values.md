# Configurable Values in src/main.py

This document lists all hard-coded values in `src/main.py` that could be moved to a configuration file for easier tuning.

## Training Parameters (Lines 141-146)

```python
# Line 142
image_size = 64  # Image dimensions (64x64)

# Line 143
global_batch_size = 30  # Global batch size across all replicas

# Line 144
batch_size = global_batch_size // strategy.num_replicas_in_sync  # Per-replica batch size

# Line 145
n_epochs = 1000  # Number of training epochs
```

**Recommendation**: These are core training hyperparameters that should be configurable.

---

## Loss Function Parameters (Line 149-154)

```python
# create_focal_ordinal_loss_with_params function
def create_focal_ordinal_loss_with_params(ordinal_weight, gamma, alpha):
    # Function accepts parameters, but default values are passed when called
    # Typical values used: ordinal_weight=1.5, gamma=2.0-3.0, alpha=[0.598, 0.315, 1.597]
```

**Note**: Loss parameters are passed as function arguments, so they're somewhat configurable already.

---

## Attention Visualization Parameters (Lines 250, 278-281)

```python
# Line 250 - Visualization frequency
if (epoch + 1) % 1 == 0:  # Visualize every epoch

# Lines 278-281 - Attention heatmap scale ranges
model_vmin = 0.075  # Minimum for model attention weights
model_vmax = 0.275  # Maximum for model attention weights
class_vmin = 0.20   # Minimum for class attention weights
class_vmax = 0.45   # Maximum for class attention weights
```

**Recommendation**: These control visualization behavior and could be configurable.

---

## Attention Entropy Loss Parameters (Lines 608, 621-625, 643)

```python
# Line 608
epsilon = 1e-10  # Small value to prevent log(0)

# Lines 621-625 - Entropy calculation weighting
model_entropy = calculate_entropy(attention_scores['model_attention'])
class_entropy = calculate_entropy(attention_scores['class_attention'])
return 0.7 * model_entropy + 0.3 * class_entropy  # 70% model, 30% class

# Line 643 - Dynamic entropy weight
entropy_weight = 0.2 * (1.0 - tf.exp(-base_loss))  # Weight factor 0.2
```

**Recommendation**: These affect attention mechanism training and could be tuned.

---

## Learning Rate Scheduler Parameters (Lines 661-672, 710-712)

```python
# DynamicLRSchedule class defaults
def __init__(self,
             initial_lr=1e-3,        # Initial learning rate
             min_lr=1e-14,           # Minimum learning rate
             exploration_epochs=10,   # Number of exploration epochs
             cycle_length=30,         # Length of each cycle
             cycle_multiplier=2.0):   # Cycle length multiplier

# Lines 710-712 - Patience and decay
if self.patience_counter >= 5:  # Patience threshold
    self.initial_lr *= 0.8      # LR decay factor (20% reduction)
```

**Recommendation**: LR schedule parameters are critical for training and should be configurable.

---

## Gating Network Training (Lines 726, 766, 764, 790-807, 819, 830, 836)

```python
# Line 726 - Gating network architecture
num_heads=max(8, num_models+1),    # Number of attention heads (16 for 15 models)
key_dim=max(16, 2*(num_models+1)),  # Key dimension (32 for 15 models)

# Line 764 - Model compilation
model = ImprovedGatingNetwork(len(train_data[0]), 3)

# Line 766
optimizer=Adam(learning_rate=1e-3),  # Learning rate

# Lines 790-807 - Callbacks
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,        # LR reduction factor
    patience=5,        # Patience epochs
    min_lr=1e-9,       # Minimum LR
    min_delta=2e-3,    # Minimum change to qualify as improvement
    verbose=0,
    mode='min'
),
tf.keras.callbacks.EarlyStopping(
    monitor='val_weighted_accuracy',
    patience=20,       # Patience epochs
    restore_best_weights=True,
    verbose=2,
    min_delta=2e-2,    # Minimum change
    mode='max'
)

# Line 819
dataset = dataset.batch(64, drop_remainder=True)  # Batch size

# Line 830
epochs=1000,  # Number of epochs

# Line 836
verbose=0  # Training verbosity
```

**Recommendation**: Gating network training has many tunable hyperparameters.

---

## Model Combination Search Parameters (Lines 971, 975, 979, 998, 1003)

```python
# Line 971 - Manual exclusion
for i in [800000]:  # Models to exclude manually

# Line 975
num_models = len(available_indices) - 1 - 0  # Starting number of models

# Line 979 - Step calculation
steps = list(np.sort(range(min_models, num_models, max(int(0.20*(num_models)), 1)))) + [len(available_indices) - 1]
# Uses 20% step size

# Line 998
max_tries_min = min(max_tries, len(all_combinations))  # max_tries passed as parameter

# Line 1003
num_processes = min(3, os.cpu_count() or 1)  # Maximum 3 parallel processes
```

**Recommendation**: Search strategy parameters affect exploration breadth.

---

## Progress Saving Parameters (Lines 1105-1106, 1137-1138)

```python
# Lines 1105-1106 - load_progress
max_retries = 6      # Maximum retry attempts
retry_delay = 0.4    # Delay between retries (seconds)

# Lines 1137-1138 - save_progress
max_retries = 6      # Maximum retry attempts
retry_delay = 0.4    # Delay between retries (seconds)
```

**Recommendation**: File I/O retry parameters for robustness.

---

## Class Weight Calculation (Lines 1397-1399)

```python
# Hierarchical gating network class weights
class_weights = {
    0: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 0)),  # I
    1: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 1)),  # P
    2: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 2))   # R
}
# Formula: total / (num_classes * class_count)
```

**Note**: This is calculated dynamically, but the formula (division by 3) could be parameterized.

---

## Hierarchical Gating Network (Lines 1265, 1308-1312, 1405, 1415-1432)

```python
# Line 1265 - Architecture
def create_hierarchical_gating_network(num_models, num_classes, embedding_dim=32):
    # embedding_dim default = 32

# Lines 1308-1312 - Transformer parameters
transformer_output = TransformerBlock(
    embed_dim=embedding_dim,
    num_heads=2,           # Number of attention heads
    ff_dim=embedding_dim * 2  # Feed-forward dimension (2x embedding)
)(phase_embedded)

# Line 1405
optimizer=Adam(learning_rate=1e-3),  # Learning rate

# Lines 1415-1432 - Training parameters
epochs=500,         # Number of epochs
batch_size=32,      # Batch size

callbacks=[
    EarlyStopping(
        monitor='loss',
        patience=patience,  # patience parameter (default 20)
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,      # LR reduction factor
        patience=7,      # Patience
        min_lr=1e-12     # Minimum LR
    )
],
verbose=2  # Training verbosity
```

**Recommendation**: Hierarchical gating has separate hyperparameters from main gating network.

---

## Focal Loss for Gating (Line 1328-1342, 1402-1403)

```python
# Line 1328 - Focal loss function
def focal_loss_gating_network(gamma=2.0, alpha=None):  # Default gamma=2.0

# Lines 1402-1403 - Loss creation
loss = focal_loss_gating_network(gamma=3.0, alpha=None)  # gamma=3.0, no alpha
```

**Recommendation**: Focal loss gamma parameter for gating network.

---

## Cross-Validation Parameters (Lines 1344, 1370, 1382)

```python
# Line 1344 - train_hierarchical_gating_network
def train_hierarchical_gating_network(predictions_list, true_labels, n_splits=3, patience=20):
    # n_splits default = 3
    # patience default = 20

# Line 1370
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Line 1382
for fold, (train_idx, val_idx) in enumerate(skf.split(...)):
```

**Recommendation**: CV parameters for hierarchical gating.

---

## Grid Search Parameters (Lines 1583-1590)

```python
# Lines 1583-1590 - Parameter grid
param_grid = {
    'ordinal_weight': [1.0],
    'gamma': [2.0, 3.0],
    'alpha': [
        [0.598, 0.315, 1.597],  # Current weights
        [1, 0.5, 2],            # Alternative 1
        [1.5, 0.3, 1.5]         # Alternative 2
    ]
}
```

**Recommendation**: Grid search ranges for loss parameter tuning.

---

## Modality Search Configuration (Lines 1702-1724)

```python
# Lines 1702-1724 - Modality combinations
modalities = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']

excluded_combinations = []  # Can exclude specific combinations

included_combinations = [
    ('metadata',), ('depth_rgb',), ('depth_map',), ('thermal_map',),
    ('metadata','depth_rgb'), ('metadata','depth_map'),
    # ... many more combinations listed
]
```

**Note**: This defines which modality combinations to test. Usually left as code rather than config.

---

## Environment Variables (Lines 111-116)

```python
# Lines 111-116 - TensorFlow threading configuration
os.environ["OMP_NUM_THREADS"] = "2"         # OpenMP threads
os.environ['TF_NUM_INTEROP_THREADS'] = '2'  # TensorFlow inter-op threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'  # TensorFlow intra-op threads
os.environ['TF_DETERMINISTIC_OPS'] = '1'    # Deterministic ops
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Deterministic cuDNN
```

**Recommendation**: Performance tuning parameters that could be hardware-specific.

---

## Summary by Category

### High Priority (Core Training)
- `image_size` (142)
- `global_batch_size` (143)
- `n_epochs` (145)
- Gating network learning rate (766)
- Gating network epochs (830)
- Gating network batch size (819)

### Medium Priority (Training Details)
- Learning rate scheduler parameters (661-672)
- Early stopping patience (802, 1418)
- ReduceLROnPlateau parameters (790-797, 1424-1428)
- Hierarchical gating epochs/batch/lr (1405, 1415-1416)

### Low Priority (Fine-tuning)
- Attention visualization ranges (278-281)
- Entropy loss weights (621-625, 643)
- Focal loss gamma (1328, 1402)
- Grid search ranges (1583-1590)

### Infrastructure
- Number of parallel processes (1003)
- Retry parameters (1105-1106, 1137-1138)
- Threading configuration (111-116)

---

## Recommendation

**Production Config File Location**: `src/utils/config.py`

**Categories to Add**:
1. Training hyperparameters (image size, batch size, epochs, learning rate)
2. Model architecture (attention heads, embedding dimensions)
3. Optimization (LR scheduler, early stopping, callbacks)
4. Search parameters (max tries, parallel processes)
5. Infrastructure (threading, retries, verbosity)
