# Investigation: CV Test Results vs Phase 9 Discrepancies

## Context

The comprehensive CV test completed successfully, but revealed two issues:

1. **metadata performance drop**: Phase 9: 97.6% acc → CV test: 50.1% acc
2. **thermal_map model leak warning**: Accuracy strictly increases [39.8%, 41.0%, 53.3%]

Your task: Investigate both issues and report findings.

---

## TASK 1: Investigate metadata Performance Drop

### The Problem

**Phase 9 results** (agent_communication/debug_09_normalized_features.py):
- Accuracy: 97.59%
- Min F1: 0.964
- Config: 20 epochs, simple Sequential model, metadata only

**CV Test results** (test_comprehensive_cv.py):
- Accuracy: 50.14%
- Min F1: 0.126
- Config: 150 epochs, full multimodal architecture, metadata only

**Question**: Why does CV test perform MUCH worse despite more epochs?

### Investigation Steps

#### Step 1: Compare Model Architectures

**A. Check Phase 9 model architecture:**
```bash
cd /home/rezab/projects/DFUMultiClassification
grep -A 20 "Build model" agent_communication/debug_09_normalized_features.py
```

Look for:
- Model type (Sequential vs Functional)
- Number of layers
- Layer sizes
- Loss function used

**B. Check CV test model architecture:**
```bash
grep -A 30 "create_multimodal_model" src/models/builders.py | head -50
```

Look for:
- How metadata branch is built
- Whether it uses the same architecture as Phase 9

**Finding to report**: Are the architectures different? If yes, describe how.

---

#### Step 2: Verify Feature Normalization is Running

**A. Check if StandardScaler is applied in CV test:**
```bash
# Search for StandardScaler usage in dataset_utils
grep -n "StandardScaler" src/data/dataset_utils.py

# Check if normalization block is still there (lines 965-995)
sed -n '965,995p' src/data/dataset_utils.py
```

**B. Create test script to verify normalization:**

Create: `agent_communication/verify_normalization.py`

```python
#!/usr/bin/env python3
"""Verify that feature normalization is applied during CV test"""

import pandas as pd
import numpy as np
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths

# Load data
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

data = prepare_dataset(
    depth_bb_file=data_paths['bb_depth_csv'],
    thermal_bb_file=data_paths['bb_thermal_csv'],
    csv_file=data_paths['csv_file'],
    selected_modalities=['metadata']
)

print(f"Data loaded: {len(data)} samples")
print(f"\nMetadata columns: {[col for col in data.columns if col not in ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']]}")

# Check numeric column ranges (before prepare_cached_datasets normalization)
numeric_cols = data.select_dtypes(include=[np.number]).columns
exclude = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']
check_cols = [col for col in numeric_cols if col not in exclude][:5]  # Check first 5

print(f"\nSample numeric column ranges (BEFORE prepare_cached_datasets):")
for col in check_cols:
    print(f"  {col}: [{data[col].min():.2f}, {data[col].max():.2f}], mean={data[col].mean():.2f}")

print("\n✅ If ranges are large (e.g., [0, 100]), normalization happens later in prepare_cached_datasets")
print("⚠️  If ranges are ~[-3, 3] with mean~0, normalization already applied (unexpected)")
```

**Run it:**
```bash
chmod +x agent_communication/verify_normalization.py
python agent_communication/verify_normalization.py > agent_communication/verify_normalization_output.txt 2>&1
cat agent_communication/verify_normalization_output.txt
```

**Finding to report**: Are features normalized? What are the value ranges?

---

#### Step 3: Compare Preprocessing Pipelines

**A. Phase 9 preprocessing:**
```bash
# Check Phase 9 normalization code (lines 40-60)
sed -n '40,80p' agent_communication/debug_09_normalized_features.py
```

**B. CV test preprocessing:**
```bash
# Check prepare_cached_datasets normalization (lines 965-995)
sed -n '950,1000p' src/data/dataset_utils.py
```

**Question**: Do both apply StandardScaler the same way?

**Finding to report**: Describe any differences in preprocessing.

---

#### Step 4: Check Training Parameters

**A. Compare loss functions:**

Phase 9:
```bash
grep -A 3 "loss=" agent_communication/debug_09_normalized_features.py
```

CV test:
```bash
grep -A 5 "get_focal_ordinal_loss" src/training/training_utils.py | head -10
```

**B. Compare optimizers:**

Phase 9:
```bash
grep "optimizer" agent_communication/debug_09_normalized_features.py
```

CV test:
```bash
grep "Adam" src/training/training_utils.py | grep "learning_rate"
```

**Finding to report**: Are loss functions and learning rates different?

---

#### Step 5: Run Quick Metadata-Only Test with Phase 9 Config

**Create simplified test:** `agent_communication/test_metadata_phase9_style.py`

```python
#!/usr/bin/env python3
"""Test metadata with Phase 9-style simple model (not full multimodal architecture)"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.evaluation.metrics import filter_frequent_misclassifications

# Load data
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

data = prepare_dataset(
    depth_bb_file=data_paths['bb_depth_csv'],
    thermal_bb_file=data_paths['bb_thermal_csv'],
    csv_file=data_paths['csv_file'],
    selected_modalities=['metadata']
)

data = filter_frequent_misclassifications(data, result_dir)

# Simple train/test split
from sklearn.model_selection import train_test_split

# Extract features (only numeric metadata)
exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
numeric_cols = data.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

X = data[feature_cols].values
y = data['Healing Phase Abs'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Classes: {np.unique(y_train, return_counts=True)}")

# Apply oversampling (like Phase 9)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
print(f"After oversampling: {np.unique(y_train_resampled, return_counts=True)}")

# Normalize (like Phase 9)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# One-hot encode
y_train_onehot = tf.keras.utils.to_categorical(y_train_resampled, num_classes=3)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Build SIMPLE model (like Phase 9)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTraining simple Sequential model (Phase 9 style)...")
history = model.fit(
    X_train_scaled, y_train_onehot,
    validation_data=(X_test_scaled, y_test_onehot),
    epochs=20,
    batch_size=32,
    verbose=2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)
y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)

from sklearn.metrics import classification_report
print(f"\n{'='*80}")
print("RESULTS:")
print(f"{'='*80}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['I', 'P', 'R']))

print(f"\n{'='*80}")
print("COMPARISON:")
print(f"{'='*80}")
print(f"Phase 9: 97.59% accuracy")
print(f"This test: {test_acc*100:.2f}% accuracy")
if test_acc > 0.9:
    print("✅ Matches Phase 9! Simple model works well on metadata.")
else:
    print("⚠️  Lower than Phase 9. Investigation needed.")
```

**Run it:**
```bash
chmod +x agent_communication/test_metadata_phase9_style.py
python agent_communication/test_metadata_phase9_style.py > agent_communication/test_metadata_phase9_style_output.txt 2>&1
tail -50 agent_communication/test_metadata_phase9_style_output.txt
```

**Finding to report**: Does simple Sequential model achieve ~97% accuracy like Phase 9?

---

### Summary for Task 1

After completing steps 1-5, create: `agent_communication/metadata_investigation_summary.txt`

Include:
1. Model architecture comparison (different?)
2. Normalization verification (applied correctly?)
3. Preprocessing differences (any?)
4. Training parameter differences (loss, optimizer, LR?)
5. Simple model test results (matches Phase 9?)

**Hypothesis to test**: CV test uses complex multimodal architecture which is suboptimal for metadata-only. Simple Sequential model (Phase 9 style) should perform much better.

---

## TASK 2: Investigate thermal_map Model Leak Warning

### The Problem

thermal_map accuracy strictly increases across folds:
- Fold 1: 39.8%
- Fold 2: 41.0%
- Fold 3: 53.3% ← Suspiciously high

**Possible causes:**
1. **Model leak**: Weights carrying over between folds
2. **Optimizer state leak**: Optimizer not reset between folds
3. **Natural variation**: Fold 3 legitimately easier (different patients)

### Investigation Steps

#### Step 1: Verify Model Reset Between Folds

**Check cross_validation_manual_split code:**
```bash
# Find where model is created for each fold
grep -n "create_multimodal_model" src/training/training_utils.py

# Check if model is recreated inside loop or reused
sed -n '1050,1070p' src/training/training_utils.py
```

**Look for:**
- Is `model = create_multimodal_model(...)` inside the fold/run loop?
- Or is model created once and reused?

**Finding to report**: Is model recreated for each fold?

---

#### Step 2: Check Optimizer Reset

```bash
# Find optimizer initialization
grep -n "model.compile" src/training/training_utils.py

# Check if it's inside fold loop
sed -n '1060,1080p' src/training/training_utils.py
```

**Finding to report**: Is optimizer reinitialized for each fold?

---

#### Step 3: Analyze Fold Data Distribution

**Create script:** `agent_communication/analyze_fold_distribution.py`

```python
#!/usr/bin/env python3
"""Analyze if fold 3 is genuinely easier than folds 1-2"""

import pandas as pd
import numpy as np
from src.data.image_processing import prepare_dataset
from src.utils.config import get_project_paths, get_data_paths
from src.training.training_utils import create_patient_folds

# Load data
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

data = prepare_dataset(
    depth_bb_file=data_paths['bb_depth_csv'],
    thermal_bb_file=data_paths['bb_thermal_csv'],
    csv_file=data_paths['csv_file'],
    selected_modalities=['thermal_map']
)

print(f"Total data: {len(data)} samples")
print(f"Class distribution: {data['Healing Phase Abs'].value_counts().sort_index()}")

# Create same folds as CV test
folds = create_patient_folds(data, n_folds=3, random_state=42)

print(f"\n{'='*80}")
print("FOLD ANALYSIS")
print(f"{'='*80}")

for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
    fold_data = data.iloc[val_idx]

    print(f"\nFold {fold_idx} (validation set):")
    print(f"  Samples: {len(fold_data)}")

    class_dist = fold_data['Healing Phase Abs'].value_counts().sort_index()
    print(f"  Class distribution:")
    for cls, count in class_dist.items():
        pct = count / len(fold_data) * 100
        print(f"    Class {cls}: {count} ({pct:.1f}%)")

    # Check class imbalance ratio
    max_count = class_dist.max()
    min_count = class_dist.min()
    imbalance_ratio = max_count / min_count
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio < 2.0:
        print(f"  → More balanced (easier to learn)")
    else:
        print(f"  → Imbalanced (harder to learn)")

print(f"\n{'='*80}")
print("HYPOTHESIS:")
print(f"{'='*80}")
print("If fold 3 has lower imbalance ratio, it's genuinely easier.")
print("If all folds have similar imbalance, the increasing accuracy suggests a leak.")
```

**Run it:**
```bash
chmod +x agent_communication/analyze_fold_distribution.py
python agent_communication/analyze_fold_distribution.py > agent_communication/fold_distribution_analysis.txt 2>&1
cat agent_communication/fold_distribution_analysis.txt
```

**Finding to report**: Are fold imbalance ratios similar, or is fold 3 more balanced?

---

#### Step 4: Check Cache/Checkpoint Reuse

```bash
# Check if checkpoints are properly deleted between folds
grep -n "cleanup_for_resume_mode\|clear_cache" agent_communication/test_comprehensive_cv.py

# Verify cleanup happens at start
sed -n '340,360p' agent_communication/test_comprehensive_cv.py
```

**Finding to report**: Are caches cleared before the test?

---

#### Step 5: Reproduce thermal_map Test in Isolation

**Create:** `agent_communication/test_thermal_map_isolated.py`

```python
#!/usr/bin/env python3
"""Run thermal_map test in isolation to check for leak"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.utils.config import cleanup_for_resume_mode

# Clean everything before test
print("Cleaning all checkpoints and caches...")
cleanup_for_resume_mode('fresh')

# Now run CV test for thermal_map only
from test_comprehensive_cv import run_modality_test, QUICK_TEST_CONFIG

print(f"\n{'='*80}")
print("Running thermal_map 3-fold CV (isolated)")
print(f"{'='*80}\n")

result = run_modality_test(['thermal_map'], QUICK_TEST_CONFIG)

print(f"\n{'='*80}")
print("RESULTS:")
print(f"{'='*80}")
print(f"Fold accuracies: {result['fold_accuracies']}")
print(f"Average: {result['avg_accuracy']:.4f}")

fold_accs = result['fold_accuracies']
if fold_accs == sorted(fold_accs) and fold_accs[2] > fold_accs[0] * 1.2:
    print("\n⚠️  Accuracy still strictly increases with large gap")
    print("   This suggests either:")
    print("   1. Model/optimizer leak (weights carrying over)")
    print("   2. Fold 3 is genuinely much easier")
else:
    print("\n✅ No strict increase pattern in isolated test")
    print("   Original pattern may have been due to test order effects")
```

**Run it:**
```bash
cd /home/rezab/projects/DFUMultiClassification
python agent_communication/test_thermal_map_isolated.py > agent_communication/thermal_isolated_results.txt 2>&1
tail -30 agent_communication/thermal_isolated_results.txt
```

**Finding to report**: Does isolated test show same increasing pattern?

---

### Summary for Task 2

After completing steps 1-5, create: `agent_communication/thermal_leak_investigation_summary.txt`

Include:
1. Model reset verification (yes/no?)
2. Optimizer reset verification (yes/no?)
3. Fold distribution analysis (imbalance ratios)
4. Cache cleanup verification (clean?)
5. Isolated test results (pattern reproduced?)

**Conclusion**: Is this a genuine leak or natural variation?

---

## Deliverables

After completing both investigations, commit results:

```bash
cd /home/rezab/projects/DFUMultiClassification

# Add all investigation outputs
git add agent_communication/metadata_investigation_summary.txt
git add agent_communication/thermal_leak_investigation_summary.txt
git add agent_communication/verify_normalization_output.txt
git add agent_communication/test_metadata_phase9_style_output.txt
git add agent_communication/fold_distribution_analysis.txt
git add agent_communication/thermal_isolated_results.txt

# Add any new test scripts
git add agent_communication/verify_normalization.py
git add agent_communication/test_metadata_phase9_style.py
git add agent_communication/analyze_fold_distribution.py
git add agent_communication/test_thermal_map_isolated.py

git commit -m "Investigation results: metadata performance drop and thermal_map leak analysis"
```

Then notify: "Investigation complete - awaiting manual push"

---

## Expected Timeline

- Task 1 (metadata investigation): ~30 minutes
- Task 2 (thermal_map leak): ~20 minutes
- **Total**: ~50 minutes

Focus on **collecting evidence**, not fixing issues yet. The remote agent will analyze your findings and decide on fixes.
