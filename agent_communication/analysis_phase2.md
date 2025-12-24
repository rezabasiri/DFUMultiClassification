# Phase 2 Analysis: Class Imbalance Issue Identified

## Key Finding

**Model predicts ONLY class P (Proliferation) - 100% of predictions**

- Test set: 622 samples
- Predicted class distribution:
  - Class 0 (I): 0 predictions
  - Class 1 (P): 622 predictions
  - Class 2 (R): 0 predictions

## Why This Happens

The minimal training script uses **plain categorical crossentropy with NO class weighting**:
```python
loss='categorical_crossentropy'
```

With imbalanced data (P=60.5%, I=28.7%, R=10.8%), the model learns that predicting P 100% of the time gives 60% accuracy, which is better than trying to learn the actual patterns.

## Critical Question

Does main.py's focal loss ACTUALLY apply class weights correctly?

### Focal Loss Implementation Review

**Current code (src/models/losses.py:130)**:
```python
focal_weight = alpha * tf.math.pow(1 - y_pred, gamma)
focal_loss = focal_weight * cross_entropy
```

**POTENTIAL BUG**: The alpha parameter should select the weight for the TRUE class, not broadcast across all predictions.

**Standard focal loss formula**:
```
FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
```

Where:
- `pt` = predicted probability of TRUE class
- `alpha_t` = weight for TRUE class (selected from alpha array based on y_true)

**Current implementation may be applying alpha incorrectly!**

## Next Steps

1. **Test focal loss with debug script** - Verify if alpha is applied correctly
2. **Fix focal loss if broken** - Proper per-class alpha weighting
3. **Re-run training** - Confirm fix resolves Min F1=0 issue

## Expected Result After Fix

With correct focal loss + alpha values:
- Model should predict all 3 classes
- Min F1 should be > 0
- Minority classes (I, R) should have reasonable F1 scores
