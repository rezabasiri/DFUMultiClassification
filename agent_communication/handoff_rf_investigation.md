# Handoff Document: RF Metadata Classification Investigation

## Date: 2026-02-14
## Objective: Independent investigation of why Random Forest metadata classification performance is low

---

## System Environment

### Hardware
- **GPUs**: 2x NVIDIA RTX A5000 (24 GB VRAM each)
- **OS**: Ubuntu 20.04, Linux 5.4.0-216-generic, x86_64

### Software
- **Python**: `/venv/multimodal/bin/python` (Python 3.11.14)
- **TensorFlow**: 2.18.1 (Keras 3.13.2)
- **Key libraries**: numpy 1.26.4, pandas 3.0.0, scikit-learn 1.8.0, imbalanced-learn 0.14.1
- **tfdf (TensorFlow Decision Forests)**: NOT installed — all RF code uses sklearn RandomForestClassifier

### How to run
```bash
/venv/multimodal/bin/python src/main.py --data_percentage 100 --device-mode multi --verbosity 2 --resume_mode fresh
# --fold N    Run only fold N (1-indexed)
# --cv_folds 3   Number of patient-level CV folds (default: 3)
```

### Project location
- Working directory: `/workspace/DFUMultiClassification`
- Git branch: `claude/fix-trainable-parameters-UehxS`

---

## Project Overview

Diabetic Foot Ulcer (DFU) multi-classification into 3 healing phases:
- **0 = I (Inflammation)**
- **1 = P (Proliferation)** — majority class
- **2 = R (Remodeling)** — minority class

The project uses multimodal fusion: clinical metadata (via RF) + image modalities (via CNNs).

---

## Data Files

| File | Description |
|------|-------------|
| `data/raw/DataMaster_Processed_V12_WithMissing.csv` | Raw metadata CSV (890 rows, 72 columns, 268 patients) |
| `data/raw/bounding_box_depth.csv` | Depth image bounding boxes |
| `data/raw/bounding_box_thermal.csv` | Thermal image bounding boxes |
| `results/best_matching.csv` | Joined dataset: metadata + image paths (3108 rows, 84 columns, 233 patients) |

### Dataset structure
- `best_matching.csv` is created by `src/data/preprocessing.py:create_best_matching_dataset()` which joins depth bounding box rows with metadata via (Patient#, Appt#, DFU#).
- Each depth image row produces one entry, so the same patient/appointment/DFU metadata is duplicated across multiple image rows.
- 648 unique (Patient#, Appt#, DFU#) combos → 3108 rows (average 4.8 rows per combo, max 18).

### Class distribution (best_matching.csv)
| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| I (Inflammation) | 0 | 893 | 28.7% |
| P (Proliferation) | 1 | 1880 | 60.5% |
| R (Remodeling) | 2 | 335 | 10.8% |

### Patient heterogeneity
- 233 unique patients
- 73 out of 233 patients (31%) have rows belonging to multiple healing phase classes
- Example: Patient 4 has 63 rows: 51 P, 9 I, 3 R
- Same patient metadata (age, sex, BMI, etc.) is identical across all that patient's rows

### Missing data (raw CSV, 890 rows)
- `Type of Pain2` and `Type of Pain_Grouped2`: 78% missing
- `Exudate Appearance`: 19.7% missing
- `Foot Score`, `Abnormally High Arch`: ~6-7% missing
- `Onset (Days)`: 5.5% missing
- Most other features: 0-4% missing
- Identifiers and image paths: 0% missing

---

## Code Architecture

### Key files
| File | Purpose |
|------|---------|
| `src/main.py` | Main entry point, orchestrates full pipeline |
| `src/main_original.py` | Original/reference implementation (before refactoring) |
| `src/data/dataset_utils.py` | Data loading, preprocessing, RF training, dataset creation |
| `src/data/preprocessing.py` | Image preprocessing, best_matching dataset creation |
| `src/models/builders.py` | Neural network model architecture (branches, fusion) |
| `src/training/training_utils.py` | Training loops, weight transfer, evaluation |
| `src/utils/production_config.py` | All hyperparameters and configuration |

### Pipeline flow
1. Load `best_matching.csv` (3108 rows)
2. Patient-level stratified split: ~80% train, ~20% val (no patient in both)
3. Compute class weights on unique cases before oversampling
4. Oversample training data (RandomOverSampler: under-sample majority to middle, then over-sample minority to middle → balanced classes)
5. Feature engineering, encoding, imputation, scaling
6. RF training and prediction
7. Create TF datasets
8. Neural network training (pre-train image branches, then fusion)

---

## Metadata Preprocessing Pipeline (dataset_utils.py)

All metadata preprocessing happens inside `prepare_cached_datasets()` (line 600) → nested `preprocess_split()` (line 1098).

### Step-by-step transformations (when `'metadata'` is in selected modalities):

**1. Image column removal (line 1118)**
```python
image_related_columns = ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
metadata_df = split_data.drop(columns=[col for col in image_related_columns if col in split_data.columns])
```

**2. Feature engineering (lines 1122-1131)**
```python
metadata_df['BMI'] = metadata_df['Weight (Kg)'] / ((metadata_df['Height (cm)'] / 100) ** 2)
metadata_df['Age above 60'] = (metadata_df['Age'] > 60).astype(int)
metadata_df['Age Bin'] = pd.cut(metadata_df['Age'], bins=range(0, int(metadata_df['Age'].max()) + 20, 20), ...)
metadata_df['Weight Bin'] = pd.cut(metadata_df['Weight (Kg)'], bins=range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20), ...)
metadata_df['Height Bin'] = pd.cut(metadata_df['Height (cm)'], bins=range(0, int(metadata_df['Height (cm)'].max()) + 10, 10), ...)
```

**3. Categorical encoding (lines 1134-1150)**

Auto-encoded via `pd.Categorical().codes` (alphabetical ordering):
- `Sex (F:0, M:1)` → F=0, M=1
- `Side (Left:0, Right:1)` → Left=0, Right=1
- `Foot Aspect` → Dorsal=0, Lateral=1, Medial=2, Plantar=3
- `Odor` → NoOdor=0, Unpleasant=1
- `Type of Pain Grouped` → ChronicPain=0, GeneralAches=1, LocalizedPain=2, NoPain=3, PhantomAndUnusualSensations=4, PressureAndMovement=5, SharpAndIntensePain=6, ShootingPain=7, ThrobbingPain=8

Explicit mappings:
- `Location Grouped` → Hallux=0, toes=1, middle=2, Heel=3, ankle=4
- `Dressing Grouped` → NoDressing=0, BandAid=1, BasicDressing=1, AbsorbantDressing=2, Antiseptic=3, AdvanceMethod=4, other=4
- `Exudate Appearance` → Serous=0, Haemoserous=1, Bloody=2, Thick=3

**4. Feature dropping (lines 1152-1175)**
```python
features_to_drop = [
    'ID', 'Location', 'Healing Phase', 'Phase Confidence (%)', 'DFU#', 'Appt#',
    'Appt Days', 'Type of Pain2', 'Type of Pain_Grouped2', 'Type of Pain',
    'Peri-Ulcer Temperature (°C)', 'Wound Centre Temperature (°C)', 'Dressing',
    'Dressing Grouped', 'No Offloading', 'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW', 'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast', 'Offloading: Crutches, Walkers or Wheelchairs',
    'Offloading Score'
]
```

**5. Integer casting (lines 1161-1192)**

`integer_columns` list contains ~56 column names. After imputation, these are cast to int.
Note: 13 columns appear in both `features_to_drop` and `integer_columns` (e.g., Type of Pain, Dressing, Dressing Grouped, all Offloading columns). These are dropped at step 4 before the int cast at step 5.

**6. Imputation (line 1183)**
```python
imputer = KNNImputer(n_neighbors=5)
source_df[columns_to_impute] = imputer.fit_transform(source_df[columns_to_impute])
metadata_df[columns_to_impute] = imputer.transform(metadata_df[columns_to_impute])
```
`columns_to_impute` = all remaining columns except image-related and `Healing Phase Abs`.
Note: `Patient#` is included in `columns_to_impute`.

**7. Scaling (lines 1194-1197)**
```python
scaler = StandardScaler()
source_df[columns_to_impute] = scaler.fit_transform(source_df[columns_to_impute])
metadata_df[columns_to_impute] = scaler.transform(metadata_df[columns_to_impute])
```
Note: `Patient#` is included in the columns being scaled.

**8. Feature selection (lines 1199-1259) — currently DISABLED**
Controlled by `RF_FEATURE_SELECTION` in `production_config.py` (currently `False`).
When enabled: Mutual Information classification selects top K features (default 40).

**9. RF training (lines 1261-1317)**

RF uses ordinal decomposition with 2 binary classifiers:
- RF1: I(0) vs P+R(1) — `y_bin1 = (y > 0).astype(int)`
- RF2: I+P(0) vs R(1) — `y_bin2 = (y > 1).astype(int)`

Current hyperparameters (reverted to original defaults):
```python
RandomForestClassifier(n_estimators=300, random_state=42+run*(run+3), class_weight=cw_dict, n_jobs=-1)
```

Class weights computed on unique cases before oversampling (lines 1060-1087):
```python
class_weights_binary1 = compute_class_weight('balanced', classes=np.array([0, 1]), y=unique_cases['label_bin1'])
class_weights_binary2 = compute_class_weight('balanced', classes=np.array([0, 1]), y=unique_cases['label_bin2'])
```

For training data: uses 5-fold out-of-fold predictions (lines 1283-1307).
For validation data: uses final RF models trained on all training data (lines 1314-1317).

**10. Probability combination (lines 1319-1330)**
```python
prob_I_unnorm = 1 - prob1
prob_P_unnorm = prob1 * (1 - prob2)
prob_R_unnorm = prob2
total = prob_I_unnorm + prob_P_unnorm + prob_R_unnorm
prob_I = prob_I_unnorm / total  # normalize to sum to 1.0
prob_P = prob_P_unnorm / total
prob_R = prob_R_unnorm / total
```

**11. RF standalone metrics are printed (lines 1332-1341)**

After probability computation, accuracy, F1-macro, and Kappa are computed and printed for both training (OOF) and validation splits.

---

## Oversampling Details (lines 769-1058)

`apply_mixed_sampling_to_df()` applies combined under+over sampling:
1. Under-sample majority classes to middle class count
2. Over-sample minority class to match

Example: I:599, P:1164, R:158 → undersample P to 599 → oversample R to 599 → balanced 599:599:599

Uses `RandomOverSampler` (creates exact duplicate rows, not SMOTE synthetic samples).

Oversampling happens BEFORE RF training (line 1096 is before the `preprocess_split` call).

---

## Differences Between Current Code and Original (main_original.py)

| Aspect | Original (main_original.py) | Current (dataset_utils.py) |
|--------|---------------------------|--------------------------|
| RF library | tfdf with sklearn fallback | sklearn only (tfdf code removed) |
| RF hyperparameters | n_estimators=300, all other defaults | n_estimators=300, all other defaults (reverted) |
| KNN imputer | n_neighbors=5 | n_neighbors=5 (reverted) |
| Feature selection | None (all features used) | Configurable, currently disabled |
| RF training predictions | In-sample (predict on own training data) | Out-of-fold 5-fold CV |
| Metadata-only model output | `Activation('softmax')` on RF probs | `Lambda(tf.identity)` (identity, no distortion) |
| Columns dropped before RF | `['Patient#', 'Healing Phase Abs']` | `['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']` |

---

## Current RF Validation Metrics

From the latest run (fold 1):
```
RF standalone metrics (VALIDATION):
    Accuracy: 0.4124  |  F1-macro: 0.3985  |  Kappa: 0.0773
```

---

## What to Investigate

The user wants an independent investigation into why RF validation performance is low, covering:

1. **Raw dataset quality** — class distribution, missing values, feature informativeness
2. **Feature preprocessing** — are encodings correct? Are important features being dropped? Are unimportant ones kept?
3. **Imputation** — is KNN imputation appropriate? Is Patient# contaminating imputation?
4. **Scaling** — RF is tree-based and scale-invariant. Is scaling before RF necessary or harmful?
5. **RF parameters** — are defaults appropriate? Should class weights be computed differently?
6. **Ordinal decomposition** — is the 2-binary-classifier approach optimal vs direct 3-class?
7. **Oversampling interaction** — training data is oversampled before RF. Does duplicating rows help or hurt RF?
8. **Class weights** — computed on unique cases but RF trains on oversampled (duplicated) data. Is there a mismatch?
9. **Data leakage** — Patient# in imputation columns. Any other leakage?
10. **Dropped features** — all 6 Offloading columns + score are dropped. Type of Pain is dropped. Appt Days is dropped. Dressing Grouped is encoded then dropped. Are these clinically relevant?
11. **Row duplication** — same metadata repeated ~4.8x per case (once per image). How does this affect RF training?

The user specifically wants factual findings, not opinions. They want to understand the actual state of the data and code.

---

## Key Code Locations

| What | File | Lines |
|------|------|-------|
| Main entry point | `src/main.py` | Full file |
| Original implementation | `src/main_original.py` | Full file |
| Data preprocessing + RF | `src/data/dataset_utils.py` | 600-1420 |
| Oversampling | `src/data/dataset_utils.py` | 769-1058 |
| Feature engineering | `src/data/dataset_utils.py` | 1122-1131 |
| Categorical encoding | `src/data/dataset_utils.py` | 1134-1150 |
| Features dropped | `src/data/dataset_utils.py` | 1152-1175 |
| Integer columns list | `src/data/dataset_utils.py` | 1161-1173 |
| KNN imputation | `src/data/dataset_utils.py` | 1183-1185 |
| Scaling | `src/data/dataset_utils.py` | 1194-1197 |
| Feature selection | `src/data/dataset_utils.py` | 1199-1259 |
| RF training + OOF | `src/data/dataset_utils.py` | 1261-1317 |
| RF probability combination | `src/data/dataset_utils.py` | 1319-1330 |
| RF metrics printout | `src/data/dataset_utils.py` | 1332-1341 |
| Class weight computation | `src/data/dataset_utils.py` | 1059-1094 |
| Patient-level split | `src/data/dataset_utils.py` | 641-676 |
| Production config | `src/utils/production_config.py` | Full file |
| Model architecture | `src/models/builders.py` | 359-557 |
| LearnableFusionWeights | `src/models/builders.py` | 26-60 |
| best_matching creation | `src/data/preprocessing.py` | 49-101 |
