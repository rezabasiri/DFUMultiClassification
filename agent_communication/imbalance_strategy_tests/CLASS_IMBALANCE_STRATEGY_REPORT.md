# Class Imbalance Strategy Comparison Report

**Date:** 2024-12-24
**Dataset:** DFU Multi-Classification (Full Dataset)
**Objective:** Identify the most effective strategy for handling class imbalance

---

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Samples | 3,107 |
| Unique Patients | 233 |
| Features | 61 (metadata) |
| Class I (Inflammatory) | 892 (28.7%) |
| Class P (Proliferative) | 1,880 (60.5%) |
| Class R (Remodeling) | 335 (10.8%) |
| Imbalance Ratio | 5.6:1 |

---

## Methodology

- **Evaluation:** 3-fold patient-level cross-validation (no patient overlap between folds)
- **Model:** Dense neural network (128→64→3 with dropout)
- **Metrics:** F1 Macro, F1 Weighted, Min F1 (worst class), Cohen's Kappa, Balanced Accuracy
- **Baseline:** No class balancing applied
- **Alpha Calculation:** Class weights computed from training data BEFORE any sampling

---

## Results: Percentage Improvement from Baseline

### Baseline Performance
| Metric | Value |
|--------|-------|
| F1 Macro | 0.3640 |
| F1 Weighted | 0.5238 |
| Min F1 | 0.0945 |
| Cohen's Kappa | 0.0768 |
| Balanced Accuracy | 0.3781 |

### All Strategies Ranked by Average Improvement

| Rank | Strategy | F1 Macro | F1 Weighted | Min F1 | Kappa | Bal Acc | **AVG** |
|------|----------|----------|-------------|--------|-------|---------|---------|
| 1 | **RandomOverSampler** | +21.8% | -1.0% | +197% | +124% | +26.7% | **+73.7%** |
| 2 | SMOTE+Alpha(sqrt) | +9.0% | -20.9% | +233% | +84% | +31.4% | +67.3% |
| 3 | Alpha (balanced) | +15.5% | -10.5% | +225% | +76% | +23.9% | +65.9% |
| 4 | Borderline-SMOTE | +14.6% | -10.3% | +215% | +64% | +23.4% | +61.3% |
| 5 | Oversample+Alpha(sqrt) | +6.2% | -23.5% | +199% | +86% | +31.7% | +59.9% |
| 6 | Oversample+Alpha | +8.5% | -21.1% | +229% | +55% | +25.8% | +59.5% |
| 7 | ADASYN | +15.8% | -5.6% | +224% | +35% | +19.5% | +57.7% |
| 8 | Focal Loss | +13.2% | -8.2% | +181% | +76% | +20.7% | +56.4% |
| 9 | SMOTE | +13.9% | -8.3% | +184% | +64% | +22.8% | +55.4% |
| 10 | RandomUnderSampler | +11.2% | -10.9% | +206% | +44% | +19.8% | +53.9% |
| 11 | SMOTE+Tomek | +15.4% | -2.6% | +166% | +69% | +16.7% | +52.9% |
| 12 | SMOTE+ENN | +0.3% | -28.6% | +196% | +45% | +26.0% | +47.7% |
| 13 | SMOTE+Tomek+Alpha | +0.4% | -27.1% | +190% | +36% | +23.8% | +44.4% |
| 14 | SMOTE+Focal | -5.0% | -34.7% | +159% | +66% | +28.1% | +42.7% |
| 15 | SMOTE+Alpha | -0.2% | -31.1% | +164% | +45% | +25.3% | +40.7% |
| 16 | Oversample+Focal | -6.0% | -36.4% | +125% | +51% | +25.8% | +32.0% |
| 17 | Alpha (sqrt) | +3.8% | -2.7% | +25% | +6% | +2.0% | +6.7% |

---

## Top 3 Strategies (Detailed)

### 🥇 #1: RandomOverSampler
**Average Improvement: +73.7%**

| Metric | Improvement | Absolute Value |
|--------|-------------|----------------|
| F1 Macro | +21.8% | 0.4436 |
| F1 Weighted | -1.0% | 0.5187 |
| Min F1 | +197.2% | 0.2808 |
| Cohen's Kappa | +123.9% | 0.1719 |
| Balanced Accuracy | +26.7% | 0.4789 |

**Per-class F1:** I=0.471, P=0.579, R=0.281

**Why it wins:**
- Highest F1 Macro improvement (+21.8%)
- Highest Kappa improvement (+124%)
- Minimal F1 Weighted sacrifice (-1.0%)
- Simple to implement

---

### 🥈 #2: SMOTE+Alpha(sqrt)
**Average Improvement: +67.3%**

| Metric | Improvement | Absolute Value |
|--------|-------------|----------------|
| F1 Macro | +9.0% | 0.3967 |
| F1 Weighted | -20.9% | 0.4141 |
| Min F1 | +232.5% | 0.3142 |
| Cohen's Kappa | +84.4% | 0.1416 |
| Balanced Accuracy | +31.4% | 0.4967 |

**Per-class F1:** I=0.469, P=0.406, R=0.314

**Why it's notable:**
- **Best Min F1 improvement (+233%)** - best for minority class R
- Highest Balanced Accuracy improvement (+31.4%)
- Uses sqrt-scaled weights (less aggressive than balanced)

---

### 🥉 #3: Alpha (balanced)
**Average Improvement: +65.9%**

| Metric | Improvement | Absolute Value |
|--------|-------------|----------------|
| F1 Macro | +15.5% | 0.4204 |
| F1 Weighted | -10.5% | 0.4688 |
| Min F1 | +224.6% | 0.3067 |
| Cohen's Kappa | +76.2% | 0.1353 |
| Balanced Accuracy | +23.9% | 0.4686 |

**Per-class F1:** I=0.440, P=0.514, R=0.307

**Why it's notable:**
- No sampling required - just class weights
- Good balance across all metrics
- Simple to implement in any framework

---

## Key Findings

### 1. Simple strategies outperform complex ones
RandomOverSampler (simplest) beats SMOTE+Tomek+Alpha (complex) by a large margin with full data.

### 2. Combining sampling + alpha can hurt performance
| Strategy | AVG Improvement |
|----------|-----------------|
| RandomOverSampler alone | +73.7% |
| Oversample+Alpha | +59.5% |
| Oversample+Alpha(sqrt) | +59.9% |

Adding alpha weights to oversampling reduces overall improvement by ~14 percentage points.

### 3. F1 Weighted often decreases - this is expected
Most strategies sacrifice F1 Weighted to improve minority class performance. This is a valid trade-off when class balance matters.

### 4. Best strategy depends on priority

| Priority | Best Strategy | Key Metric |
|----------|---------------|------------|
| Overall balance | RandomOverSampler | +73.7% avg |
| Minority class (R) | SMOTE+Alpha(sqrt) | Min F1: 0.314 |
| Simplicity | Alpha (balanced) | No sampling needed |
| F1 Weighted preservation | RandomOverSampler | Only -1.0% |

---

## Recommendations

### Primary Recommendation
**Use RandomOverSampler alone** for the best overall improvement with minimal complexity.

### Alternative for Minority Class Focus
If the R class (Remodeling) is clinically most important, use **SMOTE+Alpha(sqrt)** which achieves the highest Min F1 (0.314 vs 0.281).

### Implementation Notes
1. Apply oversampling to **training data only** (never to validation/test)
2. Calculate class weights from **original training distribution** before sampling
3. Use sqrt-scaled weights if combining with sampling (less aggressive)

---

## Appendix: Strategy Descriptions

| Strategy | Description |
|----------|-------------|
| RandomOverSampler | Randomly duplicate minority class samples |
| SMOTE | Synthetic Minority Over-sampling Technique |
| ADASYN | Adaptive Synthetic Sampling |
| Borderline-SMOTE | SMOTE focused on borderline samples |
| RandomUnderSampler | Randomly remove majority class samples |
| SMOTE+Tomek | SMOTE + Tomek links cleaning |
| SMOTE+ENN | SMOTE + Edited Nearest Neighbors cleaning |
| Alpha (balanced) | Class weights inversely proportional to frequency |
| Alpha (sqrt) | Class weights with sqrt scaling (less aggressive) |
| Focal Loss | Loss function that down-weights easy examples |
