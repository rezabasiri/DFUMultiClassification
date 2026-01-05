# Quick Fixes Needed - Local Agent TODO

## Critical Fixes (Do First)

### 1. Fix Hardcoded Stage 1 Epoch Limit
**File**: `src/training/training_utils.py`
**Location**: Search for `stage1_epochs = 30`
**Fix**: Replace with config parameter
```python
# Current (hardcoded):
stage1_epochs = 30

# Change to:
stage1_epochs = config.get('stage1_epochs', 30)  # Use config or default to 30
```

### 2. Reduce Pre-training Epoch Print Frequency
**File**: `src/training/training_utils.py`
**Location**: Automatic pre-training section (around line 1209)
**Fix**: Use EPOCH_PRINT_INTERVAL for pre-training
```python
# In pretrain_callbacks, add:
PeriodicPrintCallback(interval=EPOCH_PRINT_INTERVAL)
```

### 3. Investigate "0 trainable weights"
**Location**: Stage 1 training output
**Issue**: Model says "Total model trainable weights: 0"
**Action**:
- Add debug prints to show which layers are trainable
- Check if metadata branch has any trainable weights
- Verify fusion layer architecture

**Add this debug code after line ~1240 (after freezing)**:
```python
vprint(f"  DEBUG: Trainable weights breakdown:", level=2)
for layer in model.layers:
    if layer.trainable_weights:
        vprint(f"    {layer.name}: {len(layer.trainable_weights)} weights", level=2)
total_trainable = sum([len(l.trainable_weights) for l in model.layers])
vprint(f"  Total trainable parameters: {total_trainable}", level=2)
```

## Testing Fixes

### Before Making Changes
1. Backup current files
2. Document what you're changing in INVESTIGATION_LOG.md

### After Making Changes
1. Test at 32x32 first (verify still works)
2. Then test at 128x128 (see if improves)
3. Save all logs with descriptive names

## Configuration Updates Needed

Edit `src/utils/production_config.py`:

```python
# Add these parameters:
STAGE1_EPOCHS = 30  # Stage 1 fusion training epochs (frozen image)
PRETRAIN_PRINT_INTERVAL = 20  # Print every N epochs during pre-training (0 = all)
```

## Ask Cloud Agent For Help With

- Adding EfficientNet backbone support (requires model architecture changes)
- Major refactoring of fusion architecture
- Adding trainable fusion layers
- Complex debugging requiring deep code analysis
