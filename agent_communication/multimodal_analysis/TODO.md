# Modality Error Correlation Analysis

## Goal
Measure whether standalone modalities make errors on different samples (complementary) or the same samples (redundant). This predicts whether fusion will improve over the best standalone.

## Prerequisites
Completed standalone runs with saved predictions:
- `('depth_rgb',)` — expected kappa ~0.44
- `('thermal_map',)` — expected kappa ~0.44
- `('depth_map',)` — expected kappa ~0.18 (optional, near-random)

## Data Location
Per-fold validation predictions saved by `save_run_predictions()` in `training_utils.py`:
```
results/<combo_dir>/fold_<N>/
  pred_run1_<config>_valid.npy       # softmax probs (N, 3)
  true_label_run1_<config>_valid.npy # true labels (N,)
  sample_ids_run1_<config>_valid.npy # [Patient, Appt, DFU] (N, 3)
```

## Approach
1. Load validation predictions + sample_ids for each standalone modality, per fold
2. Match samples across modalities by sample_id (same CV split = same val samples)
3. Compute per-sample correctness vectors: `correct_A[i] = (pred_A[i] == true[i])`

## Metrics to Compute
- **Error overlap rate**: `P(both wrong) / P(at least one wrong)` — lower = more complementary
- **Cohen's kappa between error vectors**: agreement on which samples are misclassified — lower = more complementary
- **Q-statistic**: `(N11*N00 - N01*N10) / (N11*N00 + N01*N10)` where Nij = count(correct_A=i, correct_B=j) — closer to 0 = more diverse
- **Disagreement measure**: `(N01 + N10) / N_total` — higher = more complementary
- **Conditional error rate**: `P(A wrong | B correct)` vs `P(A wrong)` — if conditional < marginal, B catches A's mistakes

## Interpretation
| Q-statistic | Meaning |
|---|---|
| ~1.0 | Same errors, fusion won't help |
| ~0.5 | Moderate correlation, fusion may help slightly |
| ~0.0 | Independent errors, fusion should help |
| < 0.0 | Negatively correlated, fusion will likely help significantly |

## Script Location
Create: `scripts/modality_error_correlation.py`
