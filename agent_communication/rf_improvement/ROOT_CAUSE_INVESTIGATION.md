# ROOT CAUSE INVESTIGATION - Why Production Kappa (0.109) << Test Scripts Kappa (0.205)

## EXECUTIVE SUMMARY

The production pipeline adds an unnecessary **neural network training layer** on top of the optimized RF probabilities, which degrades performance from 0.205 to 0.109.

---

## KEY DIFFERENCE DISCOVERED

### Test Scripts (agent_communication/rf_improvement/solution_6_bayesian_optimization_fixed.py)

**Flow:**
1. Train RF with Bayesian params (646 trees, depth 14, min_samples_split 19)
2. Predict directly: `y_pred = model.predict(X_valid_norm)`
3. Calculate metrics: Kappa = 0.205 ± 0.057
4. **DONE** - No neural network training!

**Code (lines 260-267):**
```python
model = OrdinalRFClassifier(**best_params, random_state=42)
model.fit(X_train_norm, y_train)
y_pred = model.predict(X_valid_norm)  # Direct prediction!
kappa = cohen_kappa_score(y_valid, y_pred)
```

### Production Pipeline (src/main.py)

**Flow:**
1. Train RF with Bayesian params (646 trees, depth 14, min_samples_split 19)
2. Generate RF probabilities (rf_prob_I, rf_prob_P, rf_prob_R)
3. Pass through metadata branch (Cast → BatchNorm)
4. **Train neural network on top** (300 epochs, early stopping)
5. Use neural network predictions: Kappa = 0.109 ± 0.102
6. Neural network degrades RF performance by 46%!

**Evidence from v4 output:**
```
Total model trainable weights: 4  # Final classification layer
Metadata-only: Minimal training on final layer
Epoch 1/300 - loss: 1.2681 - val_loss: 0.9869  # Neural network training
Epoch 20/300 - loss: 1.1975 - val_loss: 0.7740
Restoring model weights from the end of the best epoch: 29.
Epoch 49: early stopping
```

---

## INVESTIGATION TASKS

### Task 1: Verify RF is Using Bayesian Parameters

**Goal:** Confirm RF is trained with correct parameters

**Steps:**
1. Check dataset_utils.py lines 900-925 (TFDF path)
2. Check dataset_utils.py lines 964-991 (sklearn path)
3. Look for:
   - num_trees/n_estimators = 646
   - max_depth = 14
   - min_examples/min_samples_split = 19
   - max_features = 'log2'

**Expected:** ✅ RF parameters are correct (already verified in code)

### Task 2: Confirm RF Probabilities are Generated Correctly

**Goal:** Verify RF produces 3 probabilities correctly

**Steps:**
1. Check dataset_utils.py lines 1024-1032
2. Look for:
   ```python
   prob_I = 1 - prob1
   prob_P = prob1 * (1 - prob2)
   prob_R = prob2

   split_data['rf_prob_I'] = prob_I
   split_data['rf_prob_P'] = prob_P
   split_data['rf_prob_R'] = prob_R
   ```

**Expected:** ✅ RF probabilities calculated correctly

### Task 3: Identify Where Neural Network Training Happens

**Goal:** Find the code that trains NN on RF probabilities

**Steps:**
1. Search for "Epoch.*loss.*val_loss" pattern in src/main.py
2. Find the model.fit() or model.compile() calls for metadata-only
3. Identify the final classification layer

**Questions:**
- Where is the final Dense(3) layer added?
- Why is it training for 300 epochs?
- Can we bypass this for metadata-only mode?

**File to check:** `src/main.py` around model building/training

### Task 4: Compare Predictions - RF vs NN

**Goal:** Measure performance degradation from adding NN layer

**Test Plan:**
```python
# After RF training, before NN training:
rf_predictions = np.argmax([rf_prob_I, rf_prob_P, rf_prob_R], axis=0)
rf_kappa = cohen_kappa_score(y_true, rf_predictions)
print(f"RF Direct Kappa: {rf_kappa}")  # Should be ~0.20

# After NN training:
nn_predictions = model.predict(...)
nn_kappa = cohen_kappa_score(y_true, nn_predictions)
print(f"NN Kappa: {nn_kappa}")  # Currently 0.109

print(f"Performance degradation: {(rf_kappa - nn_kappa) / rf_kappa * 100:.1f}%")
```

**Expected:** RF Direct Kappa ~0.20, NN Kappa ~0.109, Degradation ~45%

### Task 5: Check Test vs Train Predictions

**Goal:** Understand if NN is overfitting or underfitting

**Analysis:**
From v4 output, Fold 1:
```
Epoch 49: early stopping
Cohen's Kappa: 0.1308 (validation)
```

**Questions:**
- What's the training Kappa at epoch 49?
- Is the NN overfitting (train >> valid) or underfitting (train ≈ valid)?
- Does the NN help at all, or just add noise?

### Task 6: Understand Multi-Modal Architecture Purpose

**Goal:** Why does production use NN when metadata-only doesn't need it?

**Hypothesis:**
The NN layer exists for **multi-modal fusion** (combining metadata with images). When images are present, you need:
- Image branch → features
- Metadata branch → features
- Fusion layer → combine → final classification

But for **metadata-only**, you should:
- RF probabilities → argmax → predictions (NO NN needed!)

**Check:**
- Does src/main.py have a special path for metadata-only?
- Can we add: `if modalities == ['metadata']: use RF predictions directly`?

---

## PROPOSED SOLUTIONS

### Solution A: Bypass NN for Metadata-Only ✅ RECOMMENDED

**Concept:** When running metadata-only, skip NN training and use RF predictions directly

**Pseudocode:**
```python
if selected_modalities == ['metadata']:
    # Use RF predictions directly (like test scripts)
    predictions = np.argmax([rf_prob_I, rf_prob_P, rf_prob_R], axis=0)
    # Skip model building and NN training
else:
    # Build multi-modal model with fusion
    model = create_multimodal_model(...)
    model.fit(...)
    predictions = model.predict(...)
```

**Expected Impact:** Kappa 0.109 → 0.20 (+83%)

**Implementation Location:** `src/main.py` in training loop

### Solution B: Freeze RF Probabilities as Input ⚠️ PARTIAL

**Concept:** Train NN but with RF probabilities frozen (not normalized/transformed)

**Issues:**
- Still adds unnecessary complexity
- NN trained on 3107 samples may not help
- Test scripts prove NN not needed

**Not recommended** - Solution A is cleaner

### Solution C: Use Test Script Architecture ✅ ALTERNATIVE

**Concept:** Replace production metadata pipeline with test script approach

**Steps:**
1. When metadata-only detected, use OrdinalRFClassifier directly
2. Train RF, predict, calculate metrics
3. Skip all NN/model building code

**Pros:**
- Proven to work (Kappa 0.205)
- Simpler architecture
- Matches validated approach

**Cons:**
- Larger code change
- Loses multi-modal flexibility

---

## VALIDATION PLAN

After implementing Solution A or C:

**Test 1: Metadata-Only Performance**
```bash
# Set: included_modalities = [('metadata',)]
python src/main.py --mode search --cv_folds 5 --verbosity 2

# Expected:
# - NO "Epoch 1/300" messages
# - NO neural network training
# - Kappa: 0.20 ± 0.05
```

**Test 2: Multi-Modal Still Works**
```bash
# Set: included_modalities = [('metadata', 'depth_rgb')]
python src/main.py --mode search --cv_folds 3 --verbosity 2

# Expected:
# - YES neural network training (for fusion)
# - Model builds and trains correctly
# - No crashes
```

---

## INVESTIGATION SCRIPT

**File:** `agent_communication/rf_improvement/investigate_nn_degradation.py`

```python
"""
Investigate why production NN (Kappa 0.109) underperforms test RF (Kappa 0.205)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.model_selection import KFold

# Load data
df = pd.read_csv('data/Ulcer_data_revised_V2.csv')

# Setup (copy from solution_6)
# ... feature selection, imputation, normalization ...

# Train RF
print("Training RF with Bayesian params...")
# ... train RF models ...

# Get RF probabilities
prob_I = # ... calculate from RF1, RF2 ...
prob_P = # ...
prob_R = # ...

# Method 1: Direct RF prediction (like test scripts)
rf_predictions = np.argmax([prob_I, prob_P, prob_R], axis=0)
rf_kappa = cohen_kappa_score(y_valid, rf_predictions)
print(f"RF Direct Kappa: {rf_kappa:.4f}")

# Method 2: Pass through minimal NN (like production)
# Build minimal NN: Input(3) → Dense(3) → Softmax
# Train for 300 epochs with early stopping
# ... NN training ...
nn_predictions = # ... from trained NN ...
nn_kappa = cohen_kappa_score(y_valid, nn_predictions)
print(f"NN Kappa: {nn_kappa:.4f}")

# Analysis
degradation = (rf_kappa - nn_kappa) / rf_kappa * 100
print(f"\nPerformance degradation from adding NN: {degradation:.1f}%")
print(f"RF Kappa: {rf_kappa:.4f}")
print(f"NN Kappa: {nn_kappa:.4f}")

# Hypothesis test
if degradation > 20:
    print("\n❌ CONFIRMED: Neural network degrades RF performance significantly")
    print("   RECOMMENDATION: Use RF predictions directly for metadata-only mode")
else:
    print("\n✅ Neural network maintains RF performance")
```

---

## EXPECTED FINDINGS

Based on evidence from v3/v4 results:

1. **RF Parameters:** ✅ Correct (646 trees, depth 14, min_samples_split 19)
2. **RF Probabilities:** ✅ Correctly calculated
3. **RF Direct Kappa:** ~0.20 ± 0.05 (matches test scripts)
4. **NN Kappa:** ~0.109 ± 0.10 (current production)
5. **Degradation:** ~45-50% (from adding unnecessary NN layer)

**Root Cause:** Production pipeline trains neural network on top of RF probabilities, degrading performance by 45-50%.

**Solution:** Bypass neural network for metadata-only mode, use RF predictions directly.

**Expected Impact:** Kappa 0.109 → 0.20 (+83% improvement)

---

## NEXT STEPS

1. **Run Investigation Script:** Confirm RF direct predictions achieve Kappa ~0.20
2. **Implement Solution A:** Add metadata-only bypass in src/main.py
3. **Validate:** Run production with fix, verify Kappa ≥ 0.19
4. **Document:** Update architecture docs to explain metadata-only special case

**If Kappa ≥ 0.19 after Solution A** → ✅ **TASK COMPLETE - PRODUCTION READY**
