# Trainable Fusion Weights Implementation

**Purpose:** Make fusion weights learnable instead of fixed 70/30
**Benefit:** Model can adapt optimal RF/Image ratio to data quality

---

## Current Implementation (Fixed Fusion)

**File:** `src/models/builders.py`
**Lines:** 338-348 (for 2 modalities: metadata + 1 image)

```python
# CURRENT (Fixed weights):
rf_weight = 0.70  # RF contributes 70%
image_weight = 0.30  # Image contributes 30%

vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)

# Compute weighted average with FIXED weights
weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)

# Sum weighted predictions (always sums to 1.0)
output = Add(name='output')([weighted_rf, weighted_image])
```

**Issues:**
- Weights are FIXED (can't learn)
- Stage 1: 0 trainable parameters
- Can't adapt to different RF/image quality scenarios

---

## New Implementation (Trainable Fusion)

**Replace lines 338-348 with:**

```python
# TRAINABLE FUSION (adaptive weights):
vprint("  Using trainable fusion layer (adaptive RF/Image weighting)", level=2)

# Concatenate RF and image probabilities
# Shape: rf_probs (batch, 3), image_probs (batch, 3) → concatenated (batch, 6)
concatenated = Concatenate(name='concat_rf_image')([rf_probs, image_probs])

# Trainable fusion layer learns optimal weighting
# Input: 6 features (3 from RF + 3 from image)
# Output: 3 classes (I, P, R)
# The Dense layer will learn how to weight RF vs image predictions
fusion = Dense(
    3,
    activation='softmax',
    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
    name='trainable_fusion'
)(concatenated)

output = fusion
```

**Benefits:**
- Model LEARNS optimal RF/image weighting
- Stage 1 now has TRAINABLE parameters (6×3 + 3 = 21 params)
- Adapts to different data quality scenarios
- Can discover better than 70/30 split

---

## What the Model Will Learn

The Dense layer weights will effectively learn:
```python
# Learned weight matrix (6 inputs → 3 outputs):
# For each output class (I, P, R):
output[class] = w1*rf_prob[I] + w2*rf_prob[P] + w3*rf_prob[R] +
                w4*img_prob[I] + w5*img_prob[P] + w6*img_prob[R]
```

This is MORE flexible than fixed 70/30 because:
- Can weight different classes differently
- Can learn correlation patterns between RF and image predictions
- Can adapt if RF is weak for one class but strong for another

---

## Expected Initialization Behavior

With `RandomNormal(mean=0.0, stddev=0.01)`:
- Starts with weights close to zero
- Early behavior: approximately equal weighting of all inputs
- Gradually learns optimal combination through training

**Alternative (bias towards RF):**
```python
# Custom initializer that starts closer to 70/30:
def rf_biased_initializer(shape, dtype=None):
    weights = np.random.normal(0, 0.01, shape)
    # Bias towards RF columns (first 3 features)
    weights[:3, :] += 0.7 / 3  # RF gets 70% weight distributed
    weights[3:, :] += 0.3 / 3  # Image gets 30% weight distributed
    return tf.constant(weights, dtype=dtype)

fusion = Dense(
    3,
    activation='softmax',
    kernel_initializer=rf_biased_initializer,
    name='trainable_fusion'
)(concatenated)
```

---

## Implementation Steps

### Step 1: Backup Original File
```bash
cp src/models/builders.py src/models/builders.py.backup
```

### Step 2: Make Changes

**Option A: Simple Replacement (Recommended)**

Edit `src/models/builders.py` line 338-348:

```python
# Find this block:
                rf_weight = 0.70
                image_weight = 0.30
                vprint(f"  Fusion weights: RF={rf_weight:.2f}, Image={image_weight:.2f}", level=2)
                weighted_rf = Lambda(lambda x: x * rf_weight, name='weighted_rf')(rf_probs)
                weighted_image = Lambda(lambda x: x * image_weight, name='weighted_image')(image_probs)
                output = Add(name='output')([weighted_rf, weighted_image])

# Replace with:
                vprint("  Using trainable fusion layer (adaptive RF/Image weighting)", level=2)
                concatenated = Concatenate(name='concat_rf_image')([rf_probs, image_probs])
                fusion = Dense(
                    3,
                    activation='softmax',
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                    name='trainable_fusion'
                )(concatenated)
                output = fusion
```

**Option B: Make It Configurable**

Add to `src/utils/production_config.py`:
```python
# Fusion architecture
USE_TRAINABLE_FUSION = True  # True: trainable, False: fixed 70/30
```

Then in `builders.py`:
```python
from src.utils.production_config import USE_TRAINABLE_FUSION

if USE_TRAINABLE_FUSION:
    # Trainable fusion code...
else:
    # Fixed 70/30 code...
```

### Step 3: Verify Changes

```bash
# Check syntax
python -m py_compile src/models/builders.py

# Verify trainable params
python -c "
from src.models.builders import build_multimodal_model
model = build_multimodal_model(['metadata', 'thermal_map'], input_shape_dict={'thermal_map': (32, 32, 3)}, metadata_dim=50)
print(f'Total trainable params: {model.count_params()}')
model.summary()
"
```

Should show trainable params > 0 and 'trainable_fusion' layer.

---

## Testing

**Run fusion with trainable weights:**
```bash
# Config:
# IMAGE_SIZE = 32
# SAMPLING_STRATEGY = 'combined'
# INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]

python src/main.py --mode search --cv_folds 3 --verbosity 2 \
  --resume_mode fresh --device-mode multi \
  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_trainable.txt
```

**Look for:**
- Console prints: "Using trainable fusion layer"
- Training shows: "Total trainable parameters: X" (X > 0)
- Stage 1 shows improvement over fixed fusion
- Final Kappa > 0.17 (hopefully > 0.20!)

---

## Reverting Changes

If trainable fusion doesn't help:
```bash
cp src/models/builders.py.backup src/models/builders.py
```

---

## Expected Results

**Conservative Estimate:**
- Fixed fusion @ 100%: Kappa 0.17
- Trainable fusion @ 100%: Kappa 0.19-0.20 (+12-18%)

**Optimistic Estimate:**
- Trainable fusion @ 100%: Kappa 0.22+ (matches 50% data!)
- Model learns optimal weighting for data quality

**Key Benefit:**
- Solves "0 trainable params" issue
- Makes model adaptive to variations in RF/image quality
- Foundation for more advanced fusion strategies

---

## Next Steps After Implementation

1. **Compare fixed vs trainable:**
   - Run both versions @ 100% with 'combined' sampling
   - Measure improvement

2. **Analyze learned weights:**
   - Extract fusion layer weights after training
   - See what RF/image ratio model learned
   - Compare to fixed 70/30

3. **Test at different data percentages:**
   - Does trainable fusion help at 50% too?
   - Or is it specifically beneficial for 100% data?

4. **Consider more complex fusion:**
   - Add hidden layer: `Dense(16, relu) → Dense(3, softmax)`
   - Allows learning non-linear combinations
