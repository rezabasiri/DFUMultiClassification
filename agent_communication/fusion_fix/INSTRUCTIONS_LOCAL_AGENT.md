# Fusion Investigation - Local Agent Instructions

## Mission
Investigate why fusion works at 32x32 (Kappa 0.316) but fails at 128x128 (Kappa 0.029).

## Environment
```bash
source /opt/miniforge3/bin/activate multimodal
cd /home/user/DFUMultiClassification
```

## Key Issues to Investigate

1. **"0 trainable weights" print** - Why does Stage 1 have 0 trainable weights? This prevents learning.
2. **Image size dependency** - 32x32 works, 128x128 fails
3. **Weak pre-training** - thermal_map gets Kappa 0.097 at 128x128 (expected ~0.14-0.20)
4. **P-class bias** - Model predicts Proliferative for 70% of samples

## Test Matrix

Run these tests systematically (save logs after each):

### Test 1: Verify 32x32 works (baseline)
```bash
# Edit production_config.py: IMAGE_SIZE=32, DATA_PERCENTAGE=50.0
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_50pct.txt
```
Expected: Kappa ~0.20-0.30 (reduced from 0.316 due to 50% data)

### Test 2: Test 64x64 (middle ground)
```bash
# Edit production_config.py: IMAGE_SIZE=64, DATA_PERCENTAGE=50.0
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_64x64_50pct.txt
```

### Test 3: Test 128x128 (currently failing)
```bash
# Edit production_config.py: IMAGE_SIZE=128, DATA_PERCENTAGE=50.0
python src/main.py --mode search --cv_folds 3 --verbosity 2 --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_128x128_50pct.txt
```

### Test 4-6: Try EfficientNet backbones (if simple CNN fails at 128x128)
Only if Test 3 fails. Test each backbone at 128x128:
- EfficientNetB0
- EfficientNetB2
- EfficientNetB3

## Configuration Changes Needed

### 1. Fix hardcoded 30 epoch limit (src/training/training_utils.py)
Find and replace hardcoded `stage1_epochs = 30` with config parameter.

### 2. Reduce epoch printing frequency
In automatic pre-training section, change:
- Current: Print every epoch
- Target: Print every 10-20 epochs (use `EPOCH_PRINT_INTERVAL` config)

### 3. Add EfficientNet option (if needed)
Add backbone parameter to model builder. **Ask cloud agent for help with this change.**

## Documentation Protocol

After each test run, update `agent_communication/fusion_fix/INVESTIGATION_LOG.md`:

```markdown
## Test X: [Config] - [Date]
**Settings**: IMAGE_SIZE=[X], DATA_PERCENTAGE=50%, BACKBONE=[simple/efficientnet]
**Results**:
- thermal_map pre-training: Kappa [X]
- Fusion Stage 1 (frozen): Kappa [X]
- Fusion Stage 2 (fine-tune): Kappa [X]
- Final fusion: Kappa [X]

**Observations**:
- [Key finding 1]
- [Key finding 2]

**Conclusion**: [Pass/Fail] - [Why]
```

## Key Prints to Monitor

1. `Total model trainable weights: [X]` - Should NOT be 0 in Stage 1!
2. `RF predictions sum to 1.0: [...]` - Should be [1.0, 1.0, 1.0]
3. Pre-training Kappa - Should be ~0.14-0.20 for thermal_map
4. Confusion matrix - Check for class bias

## Small Fixes (Do Yourself)

- Config parameter changes
- Logging/printing adjustments
- File path fixes
- Minor refactoring

## Ask Cloud Agent For

- Architecture changes (EfficientNet integration)
- Major code refactoring
- New model designs
- Complex debugging requiring code analysis

## Success Criteria

- Fusion Kappa > 0.20 at 128x128 (with 50% data)
- No "0 trainable weights" warning
- Pre-training achieves expected performance
- Results consistent across image sizes (or understand why not)
