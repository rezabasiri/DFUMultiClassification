# FINAL ROOT CAUSE ANALYSIS

## The Pattern

All phases show model collapse to predicting ONE class:

| Phase | Alpha | Oversampling | Loss | Predicts | Train Acc | Min F1 |
|-------|-------|--------------|------|----------|-----------|--------|
| 2 | N/A | No | Cross-entropy | 100% P | 59.7% | 0.000 |
| 6 | [0.725, 0.344, 1.931] | No | Focal | 100% P | 60.5% | 0.000 |
| 7 | [0.725, 0.344, 1.931] | Yes | Focal | 100% R | 33.3% | 0.000 |
| 8 | [1.000, 1.000, 1.000] | Yes | Focal | 100% I | 33.3% | 0.000 |

## Critical Insight

**Phase 8 training accuracy = 33.3%** = Random guessing on balanced 3-class data

This means: **The model is NOT learning at all in Phase 8!**

Phases 2, 6, 7: Model "learns" to predict specific class (bad, but it's learning something)
Phase 8: Model is STUCK at initialization (worse - no learning at all)

## Why Model Can't Learn (Phase 8)

### Issue 1: Feature Scale Problems

```
Feature ranges: min=-0.24, max=7071.0
```

Features include:
- Bounding box coordinates (0-1000s)
- Patient age (0-100)
- Patient weight (0-200)
- Normalized coordinates (-1 to 1)

**Without normalization**: Gradients dominated by large-scale features, small-scale features ignored

### Issue 2: Focal Loss Too Aggressive

With balanced data ([1504, 1504, 1504]) + balanced alpha ([1, 1, 1]) + gamma=2.0:

**Early training**: Model outputs ~[0.33, 0.33, 0.33] for all samples
- For correct predictions: pt ≈ 0.33, (1-pt)^2 ≈ 0.45, focal weight ≈ 0.45
- **Focal loss suppresses easy examples** - but when model is uncertain (pt=0.33), everything is "hard"
- Gradients become very small → learning stalls

### Issue 3: Loss Value Too High

Phase 8: Training loss = 10.8, Validation loss = 11.5

For comparison:
- Phase 2 (cross-entropy): loss = 0.98 after 10 epochs
- Phase 6 (focal): loss = 6.73
- Phase 8 (focal + balanced): loss = 10.8

Higher loss → smaller gradients → slower/no learning

## The Real Fix

The model NEEDS:

1. **Feature Normalization** (StandardScaler): Scale all features to mean=0, std=1
2. **Plain Cross-Entropy** (not focal): With balanced data, we don't need focal loss
3. **Oversampling**: Keep the balanced data approach

## Why This Will Work

**With balanced data via oversampling**:
- Each class has equal representation (1504 samples each)
- Model sees equal examples during training
- No need for loss weighting (alpha=[1,1,1])
- No need for focal loss (data already balanced)

**With normalized features**:
- All features contribute equally to gradients
- Stable optimization
- Model can actually learn patterns

**With plain cross-entropy**:
- Simple, well-understood loss
- Works well with balanced data
- No gradient suppression issues

## Implementation

```python
from sklearn.preprocessing import StandardScaler

# 1. Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 2. Use plain cross-entropy
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # No focal, no ordinal
    metrics=['accuracy']
)

# 3. Train on balanced data (from oversampling)
model.fit(X_train_scaled, y_train_onehot, ...)
```

## Expected Result

- Training loss should decrease (not plateau at 10.8)
- Training accuracy should increase above 33.3%
- Model should predict all 3 classes
- Min F1 > 0.3 (actual learning!)

## Summary

**Root causes**:
1. ✅ Oversampling was disabled (fixed)
2. ✅ Double-correction bug (fixed)
3. ❌ **Features not normalized** (NEW - needs fix)
4. ❌ **Focal loss too aggressive for this case** (NEW - needs fix)

**Solution**: Normalize features + use simple cross-entropy + keep oversampling
