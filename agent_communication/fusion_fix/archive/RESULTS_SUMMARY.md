# Results Summary - 128x128 Investigation

## Manual Saved Results (100% data, 3-fold CV)

### Metadata Only (RF)
- **Kappa**: 0.0897 ± 0.0732
- **Accuracy**: 0.479 ± 0.044
- **Macro F1**: 0.371 ± 0.051

### thermal_map Only (128x128 CNN)
- **Kappa**: 0.142 ± 0.037
- **Accuracy**: 0.446 ± 0.020
- **Macro F1**: 0.385 ± 0.029

### Fusion (metadata + thermal_map at 128x128)
- **Kappa**: 0.0965 ± 0.062
- **Accuracy**: 0.483 ± 0.049
- **Macro F1**: 0.375 ± 0.051

## Analysis

### Problem: Fusion Fails at 128x128
- **Fusion (0.0965) < thermal_map (0.142)** - Fusion is WORSE than image alone!
- **Fusion (0.0965) ≈ metadata (0.0897)** - Barely better than RF alone
- **Expected**: Fusion should be 0.20+ (combining 0.09 + 0.14 with 70/30 weights)

### Comparison to 32x32 (from previous test)
- **32x32 fusion**: Kappa 0.316 ✅ (works great!)
- **128x128 fusion**: Kappa 0.0965 ❌ (fails)
- **Degradation**: 69% drop in performance!

### Key Issues

1. **Image size dependency**
   - Something breaks between 32x32 and 128x128
   - Not just scale - fundamental architectural issue

2. **"0 trainable weights" in Stage 1**
   - Model can't learn with frozen architecture
   - Fixed weights (0.7*RF + 0.3*Image) produce bad results

3. **Weak components**
   - RF: Kappa 0.09 (expected ~0.25)
   - thermal_map: Kappa 0.14 (reasonable but not great)
   - Neither is strong enough to carry fusion

4. **P-class bias** (from terminal output)
   - Model predicts Proliferative for 70% of samples
   - Indicates systematic prediction error

## Investigation Priorities

1. **Why does RF get Kappa 0.09 instead of 0.25?**
   - Check if RF training is broken at 128x128
   - Verify feature selection is working

2. **Why does 32x32 work but 128x128 fails?**
   - Test intermediate sizes (64x64)
   - Try better CNN architectures (EfficientNet)

3. **Why "0 trainable weights" in Stage 1?**
   - Architecture uses fixed fusion (no learnable parameters)
   - Need to add trainable fusion layer

4. **Why P-class bias?**
   - Check class weighting in fusion
   - Verify RF probabilities are correct

## Next Steps

See `INSTRUCTIONS_LOCAL_AGENT.md` for detailed testing protocol.
