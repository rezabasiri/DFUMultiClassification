# Misclassification-Based Data Filtering Guide

## Overview

The system tracks frequently misclassified samples and allows you to exclude them in subsequent runs using **per-class thresholds** to handle data imbalance intelligently.

## How It Works

### Step 1: Fresh Training (Identify Problematic Cases)

During the first training run, the system automatically tracks misclassifications:

```python
# Happens automatically during training in training_utils.py lines 1207, 1236
track_misclassifications(y_true, y_pred, sample_ids, selected_modalities, result_dir)
```

**CSV Files Created:**
- `results/frequent_misclassifications_{modality}.csv` - Per modality (e.g., metadata, depth_rgb)
- `results/frequent_misclassifications_total.csv` - Across all modalities combined

**CSV Format:**
```csv
Patient,Appointment,DFU,True_Label,Predicted_Label,Sample_ID,Misclass_Count
42,1,1,I,P,P042A01D1,5
103,2,1,P,R,P103A02D1,12
...
```

- **Misclass_Count**: How many times this specific sample was misclassified with this true→predicted pattern
- Counts accumulate across multiple training runs

### Step 2: Analyze Misclassifications

After the first run, examine the CSV files:

```bash
# View the most frequently misclassified samples
head -20 results/frequent_misclassifications_total.csv

# Or programmatically
python -c "
import pandas as pd
df = pd.read_csv('results/frequent_misclassifications_total.csv')
print('\\nMisclassifications by class:')
for phase in ['I', 'P', 'R']:
    phase_data = df[df['True_Label'] == phase]
    print(f'{phase}: {len(phase_data)} unique samples, avg count: {phase_data[\"Misclass_Count\"].mean():.2f}')
    print(f'   Max count: {phase_data[\"Misclass_Count\"].max()}')
"
```

### Step 3: Set Class-Specific Thresholds

The key insight: **Use lower thresholds for dominant classes to reduce imbalance**

**Current class distribution** (approximate):
- I (Inflammatory): 30%
- P (Proliferative): 60% ← **DOMINANT**
- R (Remodeling): 10%

**Recommended threshold strategy:**

```python
thresholds = {
    'I': 5,   # Higher threshold (keep more) - minority class
    'P': 3,   # Lower threshold (exclude more) - dominant class
    'R': 8    # Highest threshold (keep most) - rarest class
}
```

**Logic:**
- **P (dominant)**: Low threshold = exclude more misclassified P samples → reduces class imbalance
- **I (minority)**: Medium threshold = keep more I samples → preserves representation
- **R (rare)**: High threshold = keep almost all R samples → protects rarest class

### Step 4: Save the Misclassification File

The filter function looks for a **saved** version (not the live one):

```bash
# After first run completes, save the CSV for filtering
cp results/frequent_misclassifications_total.csv \
   results/frequent_misclassifications_saved.csv
```

**Why separate file?**
- The live CSV (`_total.csv`) keeps accumulating during training
- The saved CSV (`_saved.csv`) is your baseline for filtering
- This prevents the filter from changing mid-training

### Step 5: Run with Filtering Enabled

**Option A: Modify main.py directly**

Edit `src/main.py` line 1844:

```python
# Before (uses defaults: I=12, P=9, R=12)
data = filter_frequent_misclassifications(data, result_dir)

# After (custom thresholds)
data = filter_frequent_misclassifications(
    data,
    result_dir,
    thresholds={'I': 5, 'P': 3, 'R': 8}  # Lower P threshold to reduce dominance
)
```

**Option B: Add command-line arguments**

For more flexibility, modify main.py to accept threshold arguments:

```python
# Add to argument parser (around line 1700-1800)
parser.add_argument('--threshold_I', type=int, default=12,
                    help='Misclassification threshold for Inflammatory class')
parser.add_argument('--threshold_P', type=int, default=9,
                    help='Misclassification threshold for Proliferative class')
parser.add_argument('--threshold_R', type=int, default=12,
                    help='Misclassification threshold for Remodeling class')

# Then use in filtering (line 1844)
data = filter_frequent_misclassifications(
    data,
    result_dir,
    thresholds={
        'I': args.threshold_I,
        'P': args.threshold_P,
        'R': args.threshold_R
    }
)
```

Then run:
```bash
python src/main.py --threshold_I 5 --threshold_P 3 --threshold_R 8
```

## Complete Workflow Example

### First Run (Fresh Training)

```bash
# Clean start
python -c "from src.utils.config import cleanup_for_resume_mode; cleanup_for_resume_mode('fresh')"

# Train and collect misclassifications
python src/main.py --modalities metadata depth_rgb --cv_folds 5 --n_runs 3

# Examine results
python -c "
import pandas as pd
df = pd.read_csv('results/frequent_misclassifications_total.csv')
for phase in ['I', 'P', 'R']:
    p_df = df[df['True_Label'] == phase]
    print(f'{phase}: {len(p_df)} samples, max={p_df[\"Misclass_Count\"].max()}, mean={p_df[\"Misclass_Count\"].mean():.2f}')
"

# Save the baseline
cp results/frequent_misclassifications_total.csv \
   results/frequent_misclassifications_saved.csv
```

### Second Run (With Filtering)

```bash
# Method 1: Direct modification
# Edit src/main.py line 1844 to add thresholds={'I': 5, 'P': 3, 'R': 8}

# Method 2: Command-line (if you added arguments)
python src/main.py \
    --modalities metadata depth_rgb \
    --cv_folds 5 \
    --n_runs 3 \
    --threshold_I 5 \
    --threshold_P 3 \
    --threshold_R 8 \
    --resume_mode from_data  # Start fresh training but keep data processing

# Check filtering results
# The output will show:
# Excluding X frequently misclassified samples:
# Class I: Y samples
# Class P: Z samples (should be highest)
# Class R: W samples (should be lowest)
```

## Advanced: Automatic Threshold Selection

Create a helper script to analyze and suggest thresholds:

```python
# scripts/suggest_thresholds.py
import pandas as pd
import numpy as np

def suggest_thresholds(csv_file='results/frequent_misclassifications_total.csv'):
    """Suggest thresholds based on class distribution and misclass patterns."""
    df = pd.read_csv(csv_file)

    suggestions = {}
    for phase in ['I', 'P', 'R']:
        phase_df = df[df['True_Label'] == phase]

        # Calculate statistics
        count = len(phase_df)
        max_misclass = phase_df['Misclass_Count'].max()
        median_misclass = phase_df['Misclass_Count'].median()
        q75 = phase_df['Misclass_Count'].quantile(0.75)

        # Suggest threshold based on 75th percentile
        # But adjust based on class size (lower for larger classes)
        base_threshold = int(q75)

        # Adjustment factor based on class size
        if count > 1000:  # Large class (likely P)
            adjustment = 0.7  # Lower threshold
        elif count < 300:  # Small class (likely R)
            adjustment = 1.5  # Higher threshold
        else:  # Medium class (likely I)
            adjustment = 1.0  # No adjustment

        suggested = max(2, int(base_threshold * adjustment))

        suggestions[phase] = suggested

        print(f"\n{phase} (Inflammatory/Proliferative/Remodeling):")
        print(f"  Samples: {count}")
        print(f"  Max misclass count: {max_misclass}")
        print(f"  Median misclass count: {median_misclass:.1f}")
        print(f"  75th percentile: {q75:.1f}")
        print(f"  → Suggested threshold: {suggested}")
        print(f"    (Will exclude ~{len(phase_df[phase_df['Misclass_Count'] >= suggested])} samples)")

    print(f"\n\nRecommended command:")
    print(f"python src/main.py --threshold_I {suggestions['I']} --threshold_P {suggestions['P']} --threshold_R {suggestions['R']}")

    return suggestions

if __name__ == '__main__':
    suggest_thresholds()
```

Run it:
```bash
python scripts/suggest_thresholds.py
```

## Monitoring Impact

Track how filtering affects class balance:

```python
# Before filtering
original_counts = data['Healing Phase Abs'].value_counts().sort_index()

# After filtering
filtered_counts = filtered_data['Healing Phase Abs'].value_counts().sort_index()

print("Class distribution change:")
for phase_idx, phase_name in enumerate(['I', 'P', 'R']):
    orig = original_counts.get(phase_idx, 0)
    filt = filtered_counts.get(phase_idx, 0)
    change = ((filt - orig) / orig * 100) if orig > 0 else 0
    print(f"{phase_name}: {orig} → {filt} ({change:+.1f}%)")
```

## Tips

1. **Start conservative**: Use high thresholds first, then gradually lower
2. **Monitor minority classes**: Don't exclude too many R (Remodeling) samples
3. **Iterate**: After each filtered run, check if performance improves
4. **Balance vs accuracy tradeoff**: More filtering = better balance but less data
5. **Class-specific goals**:
   - P: Can afford to lose samples (60% of data)
   - I: Be careful (30% of data)
   - R: Protect heavily (only 10% of data)

## Expected Outcomes

With proper threshold tuning:
- ✅ Reduced class imbalance (P dominance decreases)
- ✅ Better minority class detection (R F1 score improves)
- ✅ More balanced confusion matrix
- ✅ Higher macro F1 (treats all classes equally)
- ⚠️ Slightly lower overall accuracy (acceptable tradeoff)

---

**File**: `docs/MISCLASSIFICATION_FILTERING_GUIDE.md`
**Purpose**: Guide for using per-class misclassification filtering to improve class balance
**Key Feature**: Lower thresholds for dominant classes, higher for minority classes
