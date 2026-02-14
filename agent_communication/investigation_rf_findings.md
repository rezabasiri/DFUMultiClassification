# RF Metadata Classification Investigation: Findings

## Date: 2026-02-14
## Status: Investigation complete, findings verified with experiments

---

## Executive Summary

The RF metadata classifier achieves **99%+ OOF training accuracy** but only **~40-55% validation accuracy** (Kappa 0.01-0.10). This is not an RF configuration problem. The **root cause is that the metadata features fundamentally cannot predict healing phase across patients**. The data contains ~54 features, but most are patient-level demographics (age, sex, BMI, etc.) that are constant across all of a patient's visits. Since patients have multiple healing phases, these features are non-discriminative. The OOF procedure does not fix the core problem because duplicate rows within the oversampled training set mean the OOF "held-out" fold still contains exact copies of training samples.

---

## Finding 1: OOF on Oversampled Duplicated Data Is Ineffective

**Severity: CRITICAL**

### The Problem
The 5-fold OOF procedure is designed to prevent in-sample predictions. However, it operates on the **oversampled** training data, which contains exact duplicate rows (RandomOverSampler creates copies). When 5-fold splits this data, the "held-out" fold contains rows that are **identical** to rows in the training fold.

### Evidence
| Experiment | Train OOF Acc | Train OOF Kappa | Val Kappa | Gap |
|---|---|---|---|---|
| A: Current (OOF + oversample) | 0.991 | 0.987 | 0.018 | +0.969 |
| E: In-sample (no OOF) | 1.000 | 1.000 | 0.018 | +0.982 |

OOF accuracy is 99.1% — barely different from in-sample (100%). The OOF is **not doing its job** because duplicates leak across folds.

### Root Cause Chain
1. `best_matching.csv` has ~3.6 rows per unique (Patient#, Appt#, DFU#) combination (different image crops)
2. ALL metadata columns are **identical** across these rows (confirmed by diagnostic)
3. Oversampling (combined strategy) creates additional exact copies
4. After oversampling, training has 2184 rows but only ~502 unique metadata patterns
5. 5-fold OOF: each fold sees ~437 rows, but their exact copies exist in the other 4 folds
6. RF memorizes the training patterns and the OOF "predictions" are just memory lookups

### Validation
Experiment I (deduplicated training) confirms this: deduplication reduces train OOF accuracy from 99.1% to 65.7%, and the train/val kappa gap shrinks from 0.969 to 0.251. The 65.7% train OOF is much closer to the true generalization ability.

---

## Finding 2: Metadata Features Are Fundamentally Non-Discriminative Across Patients

**Severity: CRITICAL (fundamental limitation)**

### The Problem
73 out of 233 patients (31%) have rows in multiple healing phase classes. The same patient with identical metadata (age, sex, BMI, diabetes type, foot anatomy, etc.) appears as class I in one appointment and class P or R in another. Patient-level features cannot distinguish healing phases because **the same patient transitions through all phases over time**.

### Evidence from Mutual Information
- Top MI feature: Wound Centre Temperature Normalized (0.573)
- Temperature features dominate the top 5 (4 out of 5)
- 19 features have MI < 0.001 (effectively zero): Bunion, Heart Conditions, Sensory Peripheral, Sex, Side, Foot Aspect, Odor, Exudate Appearance, etc.
- Median MI across all 64 features: 0.006

### What This Means
The only features with meaningful discriminative power are the **temperature measurements** and **Onset (Days)** — these are visit-specific, not patient-level. Most of the remaining ~50 features are patient demographics and static clinical characteristics that do not change between visits and therefore cannot predict which healing phase a wound is in.

### Validation
Even the best experimental configuration (Experiment D: no oversampling, direct 3-class, balanced weights) only achieves validation Kappa = 0.105. This is near-chance agreement. The RF is not "broken" — the features simply don't carry the signal needed.

---

## Finding 3: Categorical Encoding Bug (pd.Categorical().codes)

**Severity: MEDIUM — Actively corrupting features**

### The Problem
The code encodes `Foot Aspect` and `Type of Pain Grouped` using `pd.Categorical(col).codes`, which assigns codes based on **which categories are present** in the data. Since train and validation are processed separately (`preprocess_split` is called once for train, once for val), different categories present in each split produce **different code mappings**.

### Evidence
```
Foot Aspect: MISMATCH!
  Train: {'Dorsal': 0, 'Lateral': 1, 'Plantar': 2}
  Val:   {'Dorsal': 0, 'Medial': 1, 'Plantar': 2}
```
- Train has 'Lateral' (code 1) but not 'Medial'
- Val has 'Medial' (code 1) but not 'Lateral'
- 'Plantar' gets code 2 in both, but 'Lateral' in train and 'Medial' in val both get code 1 — **the same numerical code represents different categories**

```
Type of Pain Grouped: MISMATCH!
  Train: 9 categories (codes 0-8)
  Val:   7 categories (codes 0-6, missing SharpAndIntensePain and ShootingPain)
```
- 'ThrobbingPain': train code = 8, val code = 6

### Impact
The RF trains on one numerical encoding and predicts on a different one. For these two features, the RF's learned splits are based on incorrect values at validation time. However, since both features have very low MI scores, the practical impact is small relative to the fundamental discrimination problem.

### Fix
Use `pd.Categorical(col, categories=GLOBAL_CATEGORY_LIST).codes` with a fixed category list, or use `OrdinalEncoder` fitted on the full dataset before splitting.

---

## Finding 4: Class Weight Mismatch

**Severity: LOW**

### The Problem
Class weights are computed on **unique cases before oversampling** but applied to RF training on **oversampled data**. After oversampling, the 3-class distribution is balanced (728:728:728), but the binary decomposition is not:
- Binary1 (I vs P+R): 728 vs 1456 (1:2 ratio)
- Binary2 (I+P vs R): 1456 vs 728 (2:1 ratio)

### Evidence
```
Weights USED (from unique cases):     Bin1: {0:1.560, 1:0.736}  Bin2: {0:0.567, 1:4.246}
Weights CORRECT for oversampled data: Bin1: {0:1.500, 1:0.750}  Bin2: {0:0.750, 1:1.500}
```

Binary2 has a major mismatch: weight for class 1 (R) is 4.246 (from unique cases) vs 1.500 (correct for oversampled). This over-weights R predictions by ~2.8x.

### Impact
The RF's binary classifier for I+P vs R gives too much weight to R samples, potentially causing more R predictions than warranted. However, given the fundamental feature limitation, this is secondary.

---

## Finding 5: Dressing Grouped Reaches RF Despite Handoff Suggesting Otherwise

**Severity: LOW (informational)**

### The Facts
- `'Dressing'` IS in `features_to_drop` → dropped
- `'Dressing Grouped'` is NOT in `features_to_drop` → **reaches RF**
- Dressing Grouped is encoded (mapped to 0-4) AND included as an RF feature
- It has 11% missing values, making it one of the higher-missing features
- The handoff document's `features_to_drop` list does not include 'Dressing Grouped'

This is not necessarily wrong — Dressing Grouped may contain useful clinical signal. But it contradicts the comment that "Dressing Grouped is encoded then dropped."

---

## Finding 6: Patient# in KNN Imputation

**Severity: LOW (no measurable impact)**

### The Problem
`Patient#` (range 2-270, std=66.9) is included in `columns_to_impute` for KNN imputation. This means patients with numerically close IDs get more similar imputed values, even though Patient# is an arbitrary identifier.

### Evidence
Experiment G (removing Patient# from imputation) shows **identical results** to Experiment A: both get val Kappa = 0.018. The KNN imputer uses all features including Patient#, but since Patient# has only moderate weight among 54 features, its influence is negligible.

---

## Finding 7: Temperature Features Are the Only Real Signal

**Severity: INFORMATIONAL**

The mutual information analysis reveals a stark divide:
- **Temperature features**: MI 0.17-0.57 (the only strong signal)
- **Onset (Days)**: MI 0.47 (visit-specific)
- **Everything else**: MI < 0.12

The 6 temperature-related features are also highly correlated with each other (r > 0.7 for several pairs), meaning the RF effectively has ~2-3 independent informative features out of 54.

### The Temperature Features
These are the temperature measurements (wound centre, peri-ulcer, intact skin) in both raw and normalized forms. They're visit-specific and directly related to wound healing phase — inflammation raises temperature, proliferation/remodeling may lower it. But they also have 3% missing values and the normalized versions have near-zero variance after scaling.

---

## Finding 8: Oversampling Hurts More Than It Helps for RF

**Severity: MEDIUM**

### Evidence
| Config | Val Kappa | Description |
|---|---|---|
| A (oversample + ordinal) | 0.018 | Current pipeline |
| B (no oversample + ordinal) | 0.076 | 4.3x better |
| C (oversample + direct) | 0.054 | |
| D (no oversample + direct) | 0.105 | **Best RF result** |

Removing oversampling consistently improves validation kappa. Combined oversampling creates duplicate rows that RF memorizes. The `class_weight='balanced'` parameter already handles class imbalance internally within RF — external oversampling is counterproductive.

---

## Finding 9: Ordinal Decomposition Hurts Compared to Direct 3-Class

**Severity: MEDIUM**

### Evidence
| Config | Val Kappa |
|---|---|
| A (ordinal + OS) | 0.018 |
| C (direct + OS) | 0.054 |
| B (ordinal, no OS) | 0.076 |
| D (direct, no OS) | 0.105 |

Direct 3-class RF outperforms ordinal decomposition in both oversampled and non-oversampled settings. The ordinal decomposition forces a sequential decision structure (I vs not-I, then P vs R) that may not match the actual class boundaries. Additionally, the class weight mismatch (Finding 4) specifically degrades the ordinal approach.

---

## Finding 10: Feature Selection Makes Things Worse

**Severity: LOW**

Experiment F (top 40 MI features) achieves val Kappa = -0.001 (worse than chance), vs 0.018 for the full feature set. Mutual information computed on the oversampled training data is inflated and selects features that don't generalize.

---

## Experiment Summary Table

| Exp | Description | Train Kappa | Val Kappa | Gap |
|---|---|---|---|---|
| A | Current (OOF+OS+Ordinal) | 0.987 | 0.018 | 0.969 |
| B | No Oversampling | 0.997 | 0.076 | 0.921 |
| C | Direct 3-class (OS+OOF) | 0.988 | 0.054 | 0.935 |
| D | Direct 3-class (no OS) | 0.997 | 0.105 | 0.893 |
| E | In-sample (memorized) | 1.000 | 0.018 | 0.982 |
| F | Feature selection (top 40) | 0.990 | -0.001 | 0.991 |
| G | No Patient# in impute | 0.987 | 0.018 | 0.969 |
| H | No scaling | 0.990 | 0.005 | 0.985 |
| I | Deduplicated train | 0.337 | 0.086 | 0.251 |
| J | GradientBoosting | 0.982 | 0.082 | 0.900 |

**Key takeaway**: The best achievable RF validation Kappa with this feature set is ~0.10 (Experiment D). All configurations show massive train/val gaps because the features fundamentally lack cross-patient discriminative power.

---

## Recommendations (Ordered by Expected Impact)

### 1. Accept that metadata alone cannot reliably predict healing phase
The patient-level features (demographics, medical history) don't change between visits, but healing phase does. RF metadata is best viewed as a weak prior, not a reliable classifier.

### 2. If keeping RF metadata in fusion, use direct 3-class RF without oversampling
Switch from ordinal decomposition to direct 3-class classification, remove oversampling, and let `class_weight='balanced'` handle imbalance. Expected improvement: Kappa 0.018 → 0.105 (+0.087).

### 3. Deduplicate training data before RF
Drop duplicate metadata rows before RF training. Each unique (Patient#, Appt#, DFU#) should contribute one row. This reduces training rows from 2184 to ~502 and makes OOF predictions meaningful.

### 4. Fix the categorical encoding bug
Use fixed category lists for `pd.Categorical` encoding to ensure consistent train/val mappings for Foot Aspect and Type of Pain Grouped.

### 5. Fix class weight computation
Compute class weights on the data RF actually trains on (after oversampling), not on unique cases before oversampling. Or better: remove oversampling entirely and use `class_weight='balanced'` which handles this automatically.

### 6. Consider dropping low-MI features
The 19 features with MI < 0.001 add noise without signal. Consider reducing the feature set to the ~15-20 most informative features (primarily temperatures, onset, weight, age, and wound assessment scores).

### 7. Reconsider the fusion architecture
Given that RF metadata Kappa is at best ~0.10, the `FUSION_INIT_RF_WEIGHT = 0.70` initialization heavily biases fusion toward a weak signal. Consider lowering this to 0.30-0.40, or making fusion initialization data-driven based on pre-training validation performance.
