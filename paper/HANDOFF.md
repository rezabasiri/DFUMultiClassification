# Agent Handoff Document

**Project:** DFU Multimodal Classification (FUSE4DFU)
**Date:** 2026-03-28
**Status:** Paper revision in progress. Code complete. Final run done.

---

## 0. Environment and Setup

### Machine
| Item | Value |
|------|-------|
| Working directory | `/workspace/DFUMultiClassification` |
| Git branch | `claude/fix-trainable-parameters-UehxS` |
| Python environment | `/venv/multimodal` (Python 3.11.14) |
| Python binary | `/venv/multimodal/bin/python3` |
| GPU | NVIDIA RTX A5000 (24 GB VRAM) |
| OS | Linux 5.4.0-216-generic (x86_64) |

### Key Library Versions
| Library | Version |
|---------|---------|
| TensorFlow | 2.18.1 |
| scikit-learn | 1.5.2 |
| NumPy | 1.26.4 |
| Pandas | 2.3.3 |
| diffusers | (for SDXL generative augmentation) |
| scikit-optimize | (for Bayesian optimization) |

### How to Run
```bash
# Main training (all 15 modality combos, 5-fold CV, with gating + gen aug)
/venv/multimodal/bin/python3 src/main.py --mode search --cv_folds 5

# Data polishing (Phase 1 misclass tracking + Phase 2 threshold optimization)
/venv/multimodal/bin/python3 scripts/auto_polish_dataset_v2.py --phase2-modalities metadata+depth_rgb+thermal_map

# Compute statistics for paper
/venv/multimodal/bin/python3 paper/compute_statistics.py

# Generate figures for paper
/venv/multimodal/bin/python3 paper/generate_figures.py

# Joint optimization audit (already complete, do not re-run unless needed)
/venv/multimodal/bin/python3 agent_communication/joint_optimization_audit/joint_hparam_search.py --n-trials 100 --top-n 10

# Gating network audit (already complete, do not re-run unless needed)
/venv/multimodal/bin/python3 agent_communication/gating_network_audit/gating_hparam_search.py
```

### Current Production Config Values
```
IMAGE_SIZE = 128
FUSION_STRATEGY = 'feature_concat'
FUSION_IMAGE_PROJECTION_DIM = 0
STAGE1_LR = 0.001716
STAGE1_EPOCHS = 500
STAGE2_FINETUNE_EPOCHS = 30
STAGE2_UNFREEZE_PCT = 0.05
USE_GENERATIVE_AUGMENTATION = True
GENERATIVE_AUG_PROB = 0.15
USE_GATING_NETWORK = True
GATING_ENSEMBLE_STRATEGY = 'simple_average_best'
USE_CORE_DATA = True (to turn off: remove 'or True' from the env var check)
THRESHOLD_I = 53
THRESHOLD_P = 84
THRESHOLD_R = 70
RGB_BACKBONE = 'DenseNet121'
MAP_BACKBONE = 'DenseNet121'
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 10
RF_FEATURE_SELECTION_K = 80
GLOBAL_BATCH_SIZE = 64
N_EPOCHS = 200
LABEL_SMOOTHING = 0.0
SAMPLING_STRATEGY = 'none'
TRACK_MISCLASS = 'none'
INCLUDED_COMBINATIONS = [
    ('metadata', 'depth_rgb', 'thermal_map'),
    ('metadata', 'thermal_map'),
    ('metadata', 'depth_rgb'),
    ('metadata',)
]
```
Note: When running `main.py --mode search`, INCLUDED_COMBINATIONS is ignored and all 15 combinations of the 4 modalities are tested. INCLUDED_COMBINATIONS is used for targeted runs.

### Files to Clean Before a Fresh Run
```bash
# Predictions and checkpoints
rm -f results/checkpoints/*.npy results/checkpoints/*.json
# Model weights
rm -f results/models/*.weights.h5 results/models/*.h5
# CSV results
rm -f results/csv/modality_results_run_*.csv results/csv/modality_results_averaged.csv
rm -f results/csv/modality_combination_results.csv results/csv/gating_*.csv
# Logs
rm -f results/logs/training*.log
# Diagnostic images
rm -rf results/visualizations/diagnostic_samples/*
rm -f results/visualizations/gen*.png
# TF records cache (optional, large)
rm -rf results/tf_records/*
# Keep: results/generative_aug_cache/ (reusable if same SDXL model + resolution)
# Keep: results/misclassifications_saved/ (polish thresholds)
```

### User's Specific Instructions (MUST follow)
1. Do NOT modify `main_original.py`
2. Ignore `agent_communication/depth_map_pipeline_audit/` folder (no audit was performed for depth_map)
3. Generative augmentation only applies to the depth_rgb modality (other modalities are measurement-based)
4. Do not use the word "curation" or "polishing" in the paper. Say "multimodal matching process" and "screening out corrupted and mismatched images"
5. No hyphens or dashes in paper sentences (use colons, commas, or restructure)
6. No "macro"/"micro" terminology in the paper (use "class-averaged" instead)
7. Percentages with whole numbers: 1 decimal place (75.4%). Fractions without whole numbers: 2 decimal places (0.61)
8. Abbreviations defined at first use in BODY text (abstract definitions do not count for the body). Only abbreviate if term is used again later.
9. No abbreviation definitions in the abstract except DFU
10. Frame results as RELATIVE comparisons on a uniform dataset, not absolute performance claims
11. All new citations must be logged in `paper/new_citations.md`
12. `paper/references.bib` must be updated when adding new citations
13. Do not use emojis
14. Be concise and direct. Not wordy.
15. Contributions should be in numbered lists or bullet points

---

## 1. Project Overview

This project implements multimodal deep learning for Diabetic Foot Ulcer (DFU) healing phase classification (3 classes: Inflammatory/Proliferative/Remodeling) using 4 modalities: clinical metadata, RGB imaging, thermal mapping, and depth sensing. The model is called **FUSE4DFU** (Fusion for Diabetic Foot Ulcer using 4 modalities).

The paper is being revised for resubmission to **IEEE Journal of Biomedical and Health Informatics (JBHI)**. Reviewer comments have been received and responses drafted.

---

## 2. Current State

### Code
- **Production code is complete and stable.** All experiments have been run.
- Main entry point: `src/main.py --mode search --cv_folds 5`
- All configuration in: `src/utils/production_config.py`
- Current settings: 15% generative augmentation ON, gating network ON, USE_CORE_DATA ON (thresholds I=53, P=84, R=70)

### Paper
- **LaTeX file:** `paper/main.tex` (actively being edited)
- **References:** `paper/references.bib`
- **Figures:** `paper/figures/` (generated by `paper/generate_figures.py`)
- **Statistics:** `paper/statistics/` (generated by `paper/compute_statistics.py`)
- **TODO list:** `paper/TODO_paper_updates.md` (89 items, 72 resolved, 14 placeholder, 3 merged)
- **Reviewer response:** `paper/reviewer_response_draft.md`
- **New citations tracking:** `paper/new_citations.md`
- **Figure diagram descriptions:** `paper/figure_descriptions_for_diagrams.md` (for AI agent to generate Figures 1 and 3)
- **Notes from user:** `paper/Notes.md` (25 numbered notes, all applied)

### Results (Final Run: 15% gen aug)
- Results CSV: `results/csv/modality_combination_results.csv`
- Gating results: `results/csv/gating_network_averaged_results.csv`
- Prediction files: `results/checkpoints/pred_run*_valid.npy` and `pred_run*_train.npy` (75 each, all 15 combos x 5 folds)
- Training logs: `results/logs/training_fold*.log`
- Generated images cache: `results/generative_aug_cache/` (144 images per phase)

---

## 3. Architecture Summary

### Model (FUSE4DFU)
- **Metadata branch:** Random Forest (300 trees, depth 10, 80 MI-selected features) producing 3-class OOF probabilities. Zero trainable neural parameters.
- **Image branches:** DenseNet121 (ImageNet pretrained) for all image modalities. Projection heads: depth_rgb [128,32], thermal_map [128], depth_map [128]. Input size 128x128.
- **Fusion:** Feature concatenation. RF probs (3-dim) concatenated with projected image features, then fully connected output layer with 3 outputs.
- **Training:** Two-stage. Stage 1: frozen backbones, 500 epochs, LR 1.72e-3, early stopping. Stage 2: top 5% backbone unfrozen, 30 epochs, LR 5e-6.
- **Loss:** Focal loss (gamma=2.0, frequency-based class weights)
- **Ensemble:** Simple probability averaging of metadata-containing combinations. Optimized 4-combo subset: metadata, metadata+depth, metadata+thermal, metadata+RGB+depth. Kappa 0.58, accuracy 80.6%.
- **Generative augmentation:** Fine-tuned SDXL 1.0 on CPU. Pre-cached 144 images/phase to disk. Injected at 15% probability during training of depth_rgb-containing combos only.

### Gating Network
- Simple averaging of predictions from metadata-containing modality combinations
- Selected from 111 configurations across 8 strategies (simple avg, weighted avg, temperature scaling, logistic regression, MLP, gradient boosting, rank-weighted, multi-head attention)
- Attention-based gating collapsed on 2/5 folds; simple averaging was most robust
- Implementation in `src/main.py` function `run_gating_ensemble()`

### Data Pipeline
- Raw data: `data/raw/` (CSV + image folders)
- Matched dataset: 3,108 rows, 443 unique assessments, 233 patients
- After screening (USE_CORE_DATA=True): thresholds I=53, P=84, R=70 applied to misclassification counts
- Patient-stratified 5-fold CV. Folds run as isolated subprocesses to prevent TF/CUDA leakage.

---

## 4. Key Experimental Results

### Best Individual Combination: metadata + RGB + thermal
- Kappa: 0.61 +/- 0.09, Accuracy: 71.2%, F1-I: 0.69, F1-P: 0.76, F1-R: 0.54

### Optimized Ensemble (4 combos)
- Kappa: 0.58 +/- 0.14, Accuracy: 80.6%, F1-I: 0.73, F1-P: 0.86, F1-R: 0.59

### Single Modalities
- Metadata: kappa 0.45, RGB: 0.51, Thermal: 0.50, Depth: 0.19

### Generative Augmentation Dose-Response
- 0%: kappa 0.57, F1-R 0.52
- 6%: kappa 0.59, F1-R 0.54
- 15%: kappa 0.58, F1-R 0.55 (optimal)
- 25%: kappa 0.58, F1-R 0.55 (declining)

### Statistical Tests (in `paper/statistics/`)
- ANOVA: F=5.56, p=0.002 across modality group sizes
- Linear regression: slope=0.07 kappa/modality, R-squared=0.40, p=0.01
- McNemar: fusion vs metadata chi2=4.56, p=0.03; fusion vs depth chi2=84.50, p<0.001
- Paired t-test: fusion vs metadata p=0.34 (not significant at fold level, significant at sample level via McNemar)
- ROC/AUC: best fusion AUC 0.87, per-class 0.91/0.80/0.90

---

## 5. Optimization History

### Phase 1: Standalone Modality Audits
- `agent_communication/depth_rgb_pipeline_audit/` (135 configs, BASELINE EfficientNetB0 won)
- `agent_communication/thermal_map_pipeline_audit/` (135 configs, EfficientNetB2 with mixup won)
- **Outcome:** These results were later superseded by joint optimization

### Phase 2: Fusion Audit
- `agent_communication/fusion_pipeline_audit/` (206 configs, 9 rounds)
- Tested: prob_concat, feature_concat, feature_concat_attn, gated, hybrid, residual, stacking
- **Outcome:** feature_concat won (kappa 0.303), but still below metadata-only (0.333)

### Phase 3: Joint Bayesian Optimization (KEY BREAKTHROUGH)
- `agent_communication/joint_optimization_audit/` (100 Bayesian trials, 23 hyperparameters)
- **Discovered:** DenseNet121 > EfficientNet, 128px > 256px, Stage 2 helps when jointly optimized
- **Outcome:** First time fusion beat metadata (kappa 0.369 vs 0.307)
- Report: `agent_communication/joint_optimization_audit/REPORT.md`

### Phase 4: Gating Network Optimization
- `agent_communication/gating_network_audit/` (111 configs, 8 strategies)
- **Discovered:** Simple averaging of metadata combos beats all learned ensemble methods
- Report: `agent_communication/gating_network_audit/REPORT.md`

### Phase 5: Data Polishing
- `scripts/auto_polish_dataset_v2.py`
- Misclassification tracking across 7 modality combos x 3 runs x 5 folds
- Bayesian optimization of per-class thresholds
- Best: I=53, P=84, R=70 (68.4% retention)

### Phase 6: Final Runs (4 configurations)
- Run 1: Baseline (no gating, no gen aug) — kappa 0.61
- Run 2: + Gating network — kappa 0.58 (ensemble), 0.54 (gating kappa)
- Run 3a: + Gen aug 6% — kappa 0.59, F1-R 0.54
- Run 3b: + Gen aug 15% — kappa 0.58, F1-R 0.55, ensemble F1-R 0.59
- Run 3c: + Gen aug 25% — declining performance

---

## 6. Paper TODO (Remaining Items)

### Placeholder Items (require additional work)
From `paper/TODO_paper_updates.md`:

**Figures needing generation/update:**
- Item 69: Figure 1 (framework_overview.png) needs manual redraw. Description in `paper/figure_descriptions_for_diagrams.md`.
- Item 70: Figure 3 (gaman_architecture.png) needs manual redraw. Description in same file.
- Item 71: Figure 4 (phase_f1_scores.png) — already generated at `paper/figures/phase_f1_scores.png`
- Item 72: Figure 5 (modality_agreement) — can reuse `agent_communication/multimodal_analysis/figures/modality_agreement_regions.png`
- Item 73: Figure 6 (performance_progression.png) — already generated at `paper/figures/performance_progression.png`
- Item 74: Figure 7 (dose_response.png) — already generated at `paper/figures/dose_response.png`
- Item 76: Cross-run comparison figure — generated at `paper/figures/cross_run_comparison.png`

**Statistical analyses (already computed but may need re-verification with final 15% run):**
- Item 35: ROC/AUC — computed in `paper/statistics/roc_auc_values.csv`
- Item 36: ANOVA/linear regression — computed in `paper/statistics/anova_results.json` and `linear_regression_results.json`
- Item 42: ROC/AUC values in tex — already inserted
- Item 44: ANOVA p-value in tex — already inserted
- Item 86: Paired t-tests — computed in `paper/statistics/paired_ttests.csv`

### Reviewer Response
- Draft at `paper/reviewer_response_draft.md`
- All 3 reviewers' comments addressed
- Reviewer comments source: `paper/reviewers comments.txt`

---

## 7. Important User Preferences and Guidelines

### Writing Style
1. No hyphens/dashes in sentences (use colons, commas, or restructure)
2. No "macro"/"micro" terminology (use "class-averaged" instead)
3. Percentages with whole numbers: 1 decimal place (75.4%)
4. Fractions without whole numbers: 2 decimal places (0.61)
5. Abbreviations spelled out at first use in body text (not just abstract). Only abbreviate if used again later.
6. No abbreviation definitions in the abstract except DFU
7. Concise, direct, scientific language. Not wordy.
8. Numbered lists or bullet points for contributions and key findings
9. Do not use emojis

### Content Guidelines
1. Do NOT mention "polish v2," "outlier removal," or "curation." Describe dataset preparation as multimodal matching + screening corrupted images.
2. Frame results as RELATIVE comparisons (how modalities combine on a uniform dataset) rather than absolute performance claims. The point is multimodal insight, not chasing highest numbers.
3. The absolute values depend on dataset size and can improve with more data. This should be stated.
4. All new citations must be logged in `paper/new_citations.md` with verification status.
5. `paper/references.bib` must be updated when adding new citations.
6. Figures 1 and 3 need manual creation (architecture diagrams). Descriptions are in `paper/figure_descriptions_for_diagrams.md`.
7. The user's previous papers should be cited where relevant (metadata power, diffusion models for DFU, vision-language models).

### Code Guidelines
1. Do not modify `main_original.py`
2. Ignore `depth_map_pipeline_audit/` folder
3. Generative augmentation only applies to depth_rgb modality
4. Folds run as subprocesses for GPU memory isolation
5. USE_CORE_DATA can be toggled off by removing `or True` from the env var check in production_config.py
6. GATING_ENABLED in production_config controls ensemble on/off

---

## 8. File Map

### Core Code
| File | Purpose |
|------|---------|
| `src/main.py` | Main training script, search mode, gating ensemble |
| `src/utils/production_config.py` | All hyperparameters and settings |
| `src/models/builders.py` | Model architecture (DenseNet121 branches, fusion) |
| `src/training/training_utils.py` | Training loop, 2-stage training, prediction saving |
| `src/data/dataset_utils.py` | Data pipeline, image loading, augmentation |
| `src/data/generative_augmentation_v3.py` | SDXL generative augmentation with pre-caching |
| `src/utils/gpu_config.py` | GPU memory management |
| `scripts/auto_polish_dataset_v2.py` | Two-phase data polishing (misclass tracking + Bayesian threshold optimization) |

### Paper
| File | Purpose |
|------|---------|
| `paper/main.tex` | LaTeX manuscript (actively editing) |
| `paper/references.bib` | Bibliography |
| `paper/TODO_paper_updates.md` | 89-item TODO with resolution status |
| `paper/reviewer_response_draft.md` | Response to 3 reviewers |
| `paper/reviewers comments.txt` | Original reviewer comments |
| `paper/new_citations.md` | Tracking new references added |
| `paper/Notes.md` | User's 25 numbered editing notes (all applied) |
| `paper/figure_descriptions_for_diagrams.md` | Descriptions for Figures 1 and 3 |
| `paper/experiment_report.md` | Comprehensive experiment report |
| `paper/compute_statistics.py` | Script to compute ROC/AUC, ANOVA, t-tests |
| `paper/generate_figures.py` | Script to generate paper figures |
| `paper/statistics/` | Computed statistics (JSON, CSV) |
| `paper/figures/` | Generated figures (PNG) |
| `paper/dataset_statistics.txt` | Raw dataset statistics reference |

### Experiment Reports
| File | Purpose |
|------|---------|
| `agent_communication/joint_optimization_audit/REPORT.md` | Joint optimization experiment report |
| `agent_communication/gating_network_audit/REPORT.md` | Gating network experiment report |
| `agent_communication/INVESTIGATION_image_modality_underperformance.md` | Why images underperform metadata |
| `agent_communication/TODO_image_modality_investigation.md` | Investigation tracking |

### Results
| File | Purpose |
|------|---------|
| `results/csv/modality_combination_results.csv` | Final 15-combo results |
| `results/csv/gating_network_*.csv` | Gating ensemble results |
| `results/checkpoints/pred_run*_*.npy` | Per-fold predictions (75 valid + 75 train) |
| `results/misclassifications_saved/` | Polish v2 outputs, thresholds |
| `results/generative_aug_cache/` | Pre-cached SDXL images (I/P/R x 144) |
| `agent_communication/multimodal_analysis/` | Statistical analysis figures and tables |

---

## 9. Known Issues and Gotchas

1. **combo_pred files were corrupted in an earlier run** (all contained identical data). Fixed by reading from `pred_run*` files directly. Old combo_pred files were deleted.
2. **Generative augmentation and TF cannot share GPU.** SDXL runs on CPU with pre-caching. Images generated to `results/generative_aug_cache/` with a manifest file for cache validation.
3. **tf.io.decode_jpeg handles PNG correctly** (auto-detects format). This was investigated and confirmed not a bug.
4. **The gating network in main.py had a bug** where it was called per-combination instead of once across all combinations. Fixed: per-combo gating is skipped (correctly says "only 1 model"), cross-combo gating runs at the end.
5. **Train predictions must be saved for gating** (controlled by always saving train preds regardless of TRACK_MISCLASS setting). This was a fix applied to training_utils.py.
6. **The appendix table (Table II) uses `\begin{table}[H]`** requiring the `float` package. Already added to preamble.
7. **Decimal convention:** Percentages at 1 decimal, fractions at 2 decimals. Already applied throughout.

---

## 10. What Remains To Do

### Paper (Priority Order)
1. **Generate Figures 1 and 3** (architecture diagrams) using descriptions in `paper/figure_descriptions_for_diagrams.md`
2. **Review and finalize** `paper/main.tex` end-to-end for consistency, flow, and any remaining TODO comments
3. **Verify all citations** in `paper/new_citations.md` are correct
4. **Finalize reviewer response** `paper/reviewer_response_draft.md`
5. **Proofread** for minor typographical issues, spacing, comma usage
6. **Check IEEE JBHI formatting** requirements (page limit, figure resolution 300 DPI, reference format)
7. **Update experiment_report.md** if any final changes are made to the paper

### Optional Future Work
- External validation on DFU2020 or Plantar Thermogram Database
- Clinician evaluation of synthetic image quality
- Longitudinal healing trajectory prediction
- Multi-site validation
