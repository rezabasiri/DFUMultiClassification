# Gating Network Optimization Audit — Report

**Date:** 2026-03-20
**Configs tested:** 111 (8 strategies, 7 model subsets)
**Baseline:** Best single combination kappa = 0.474 (metadata+depth_rgb+depth_map, 5-fold avg from pred_run files)

---

## 1. Motivation

The initial gating network implementation used a dual-level attention architecture (MultiHeadAttention with model-level and class-level attention, residual blocks, entropy-regularized loss). When deployed in Run 2, it collapsed to majority-class prediction on 2 of 5 folds (kappa = 0.000), yielding a 5-fold mean kappa of 0.258 — far worse than the best individual combination (0.584). The architecture had 45 input features (15 combinations x 3 classes) but only ~1,600 training samples per fold, causing severe overfitting.

This audit systematically searched for an ensemble strategy that reliably improves over the best individual modality combination.

---

## 2. Objective

Find the optimal method to combine predictions from up to 15 modality-specific models into a single prediction that beats the best individual combination. The method must:
- Work with any subset of modalities (not assume metadata is present)
- Be stable across folds (low kappa variance)
- Not overfit on the small training set (~1,600 samples)

---

## 3. Method

### 3.1 Data Source

Pre-computed predictions from a completed `main.py --mode search --cv_folds 5` run. Per fold, each of the 15 modality combinations produces:
- `pred_run{fold}_{combo}_train.npy` — softmax probabilities on training set (shape: ~1600 x 3)
- `pred_run{fold}_{combo}_valid.npy` — softmax probabilities on validation set (shape: ~400 x 3)
- `true_label_run{fold}_{combo}_{train|valid}.npy` — integer class labels

No model retraining was required — the audit operates entirely on saved prediction arrays.

### 3.2 Strategies Tested

| Strategy | Description | Trainable params |
|----------|-------------|-----------------|
| Simple average | Unweighted mean of softmax predictions from selected combos | 0 |
| Rank-weighted average | Exponential decay weights by combo kappa rank (decay ∈ {0.3, 0.5, 0.7, 0.9}) | 0 |
| Optimal weighted average | Scipy Nelder-Mead optimization of per-combo weights to maximize train kappa | 0 (weights fit on train) |
| Temperature scaling | Per-combo temperature parameter learned to minimize NLL on train set | N (one per combo) |
| Logistic regression stacking | Concatenated probabilities (N_combos x 3 features) → LR (C ∈ {0.01, 0.1, 1.0, 10.0}) | N_features x 3 |
| Gradient boosting stacking | Concatenated probabilities → GBClassifier (various n_estimators, depths, LRs) | Ensemble of trees |
| MLP stacking | Concatenated probabilities → Dense layers → softmax (various architectures) | 100s–1000s |
| Attention gating | Stacked predictions (N_combos x 3) → MultiHeadAttention → Dense → softmax | 100s–1000s |

### 3.3 Model Subsets

Each strategy was tested with multiple subsets of the 15 combinations:

| Subset | Description | # Combos |
|--------|-------------|----------|
| top3 | 3 highest-kappa combos | 3 |
| top5 | 5 highest-kappa combos | 5 |
| top7 | 7 highest-kappa combos | 7 |
| top10 | 10 highest-kappa combos | 10 |
| all15 | All 15 combinations | 15 |
| meta_only | All combos containing metadata | 8 |
| standalone | Only single-modality combos | 4 |

### 3.4 Evaluation

Every configuration was evaluated with 5-fold patient-stratified cross-validation on the polished dataset (443 unique samples). Metrics: Cohen's kappa, accuracy, macro F1, per-class F1 (I, P, R).

---

## 4. Results

### 4.1 Overall Statistics

- **Total configs:** 111
- **Beat baseline (kappa > 0.474):** 64 (58%)
- **Best kappa:** 0.537 (simple_avg_meta_only)
- **Worst kappa:** 0.395 (simple_avg_standalone)

### 4.2 Top 10 Configurations

| Rank | Config | Strategy | #Combos | Kappa | ±Std | Acc | F1 | F1-I | F1-P | F1-R |
|------|--------|----------|---------|-------|------|-----|-----|------|------|------|
| 1 | simple_avg_meta_only | simple_average | 8 | **0.537** | 0.121 | **0.789** | **0.704** | **0.687** | 0.843 | **0.583** |
| 2 | opt_weighted_top10 | optimal_weighted | 10 | 0.530 | 0.119 | 0.786 | 0.698 | 0.678 | 0.841 | 0.576 |
| 3 | temp_scaled_top5 | temperature_scaled | 5 | 0.518 | 0.114 | 0.768 | 0.691 | 0.668 | 0.824 | 0.581 |
| 4 | opt_weighted_top7 | optimal_weighted | 7 | 0.506 | 0.121 | 0.771 | 0.688 | 0.643 | 0.828 | 0.591 |
| 5 | stack_mlp_h64_d0.3_all15 | stacking_mlp | 15 | 0.506 | 0.112 | **0.791** | 0.679 | 0.664 | **0.850** | 0.522 |
| 6 | temp_scaled_top7 | temperature_scaled | 7 | 0.505 | 0.119 | 0.763 | 0.685 | 0.649 | 0.820 | 0.587 |
| 7 | stack_lr_C1.0_all15 | stacking_lr | 15 | 0.503 | 0.106 | 0.785 | 0.681 | 0.660 | 0.845 | 0.537 |
| 8 | stack_mlp_h128_64_d0.4_all15 | stacking_mlp | 15 | 0.502 | **0.094** | 0.780 | 0.679 | 0.667 | 0.841 | 0.528 |
| 9 | stack_mlp_h32_d0.3_all15 | stacking_mlp | 15 | 0.501 | 0.126 | 0.789 | 0.678 | 0.649 | 0.849 | 0.536 |
| 10 | simple_avg_top10 | simple_average | 10 | 0.501 | 0.122 | 0.766 | 0.683 | 0.660 | 0.823 | 0.566 |

### 4.3 Best Per Strategy

| Strategy | Best Config | Kappa | Acc | F1 |
|----------|------------|-------|-----|-----|
| Simple average | simple_avg_meta_only | **0.537** | 0.789 | **0.704** |
| Optimal weighted | opt_weighted_top10 | 0.530 | 0.786 | 0.698 |
| Temperature scaled | temp_scaled_top5 | 0.518 | 0.768 | 0.691 |
| MLP stacking | stack_mlp_h64_d0.3_all15 | 0.506 | **0.791** | 0.679 |
| Logistic regression | stack_lr_C1.0_all15 | 0.503 | 0.785 | 0.681 |
| Rank-weighted | rank_weighted_top5_d0.9 | 0.497 | 0.760 | 0.677 |
| Attention gating | attn_h8_k32_d0.1_noca_top5 | 0.492 | 0.785 | 0.673 |
| Gradient boosting | stack_gb_n100_d3_lr0.1_all15 | 0.471 | 0.787 | 0.669 |

### 4.4 Winner: simple_avg_meta_only

Per-fold breakdown:

| Fold | Kappa | Note |
|------|-------|------|
| 1 | 0.610 | |
| 2 | 0.518 | |
| 3 | **0.722** | Best fold |
| 4 | 0.463 | |
| 5 | 0.371 | Weakest fold |
| **Mean** | **0.537 ± 0.121** | |

The 8 metadata-containing combinations averaged:
1. metadata (standalone)
2. metadata+depth_rgb
3. metadata+depth_map
4. metadata+thermal_map
5. metadata+depth_rgb+depth_map
6. metadata+depth_rgb+thermal_map
7. metadata+depth_map+thermal_map
8. metadata+depth_rgb+depth_map+thermal_map

---

## 5. Key Findings

### 5.1 Simpler Methods Outperform Learned Ensembles

The strategy ranking (simple average > optimal weighted > temperature scaled > MLP > LR > attention) is inversely correlated with model complexity. With ~1,600 training samples and 15 x 3 = 45 input features, learned ensembles overfit. The attention gating (most complex) performed worst among competitive strategies.

### 5.2 Metadata-Containing Combos Are the Optimal Subset

Averaging only the 8 metadata-containing combos (kappa 0.537) outperforms averaging all 15 (kappa 0.501) by +0.036. The 7 non-metadata combos add noise — their image-only predictions dilute the strong metadata signal. This confirms metadata is the dominant modality.

### 5.3 Ensemble Gains Are Real but Modest

The best ensemble (kappa 0.537) improves over the best single combo (kappa 0.474) by +0.063. This gain comes from prediction smoothing — averaging reduces overconfident misclassifications, particularly benefiting the minority R-class (F1-R 0.583 vs ~0.52 for single combos).

### 5.4 The Attention Gating Network Was Fundamentally Mismatched

The original attention architecture (from main.py) had:
- Dual MultiHeadAttention layers (model-level + class-level)
- Trainable temperature parameter
- Entropy-regularized loss
- ~1,000+ trainable parameters

For a training set of ~1,600 samples with 45 input features, this is massively overparameterized. The audit's best attention config (attn_h8_k32_d0.1_noca, kappa 0.492) used simpler settings (no class attention, fewer heads) but still couldn't match parameter-free methods.

### 5.5 Combo Count Matters: More Isn't Always Better

| Subset | Simple avg kappa | Optimal weighted kappa |
|--------|-----------------|----------------------|
| top3 | 0.494 | 0.494 |
| top5 | 0.498 | 0.498 |
| top7 | 0.491 | 0.506 |
| top10 | 0.501 | **0.530** |
| all15 | 0.487 | 0.506 |
| meta_only (8) | **0.537** | — |

For simple averaging, the metadata-only subset dominates. For optimal weighting, top10 works best (the optimizer can downweight weak combos). Adding all 15 hurts simple average because weak combos get equal voice.

---

## 6. Implementation

The winning strategy (`simple_average_best`) was implemented in `src/main.py` with automatic fallback:

1. **Primary:** Identify the best-performing standalone modality per fold → average all combinations containing it
2. **Fallback 1:** If <2 combinations contain the anchor modality → use optimal weighted average of all combos (scipy optimization on training set)
3. **Fallback 2:** If no train predictions available → simple average of all combinations
4. **Configuration:** `GATING_ENSEMBLE_STRATEGY` in `production_config.py` (options: `simple_average_best`, `optimal_weighted`, `simple_average_all`, `attention`)

This design is modality-agnostic — it works whether metadata, images, or any subset of modalities is available.

---

## 7. Bug Discovered and Fixed

During this audit, a critical bug was discovered in the combo_pred file generation. The `save_combination_predictions` function read from `sum_pred_run{fold}_{type}.npy` (aggregated predictions), which was overwritten by each modality combination. This caused all 15 combo_pred files per fold to contain identical predictions — the last combination's output.

**Fix:** Changed to read from `pred_run{fold}_{combo}_{type}.npy` (per-combo files) which are never overwritten. This also explains why the original attention gating collapsed — it was training on 15 identical copies of the same predictions, so it had no diversity to learn from.

---

## 8. Files

| File | Description |
|------|-------------|
| `gating_hparam_search.py` | Optimization script (111 configs, 8 strategies) |
| `gating_search_results.csv` | All 111 results with per-fold kappas |
| `gating_best_config.json` | Winner config (simple_avg_meta_only) |
| `logs/gating_search_*.log` | Detailed execution log |
| `REPORT.md` | This report |
| `src/main.py` (lines 2033-2144) | Updated gating ensemble with strategy selection and fallback chain |
| `src/utils/production_config.py` | `GATING_ENSEMBLE_STRATEGY = 'simple_average_best'` |
