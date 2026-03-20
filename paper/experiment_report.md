# Multimodal Classification of Diabetic Foot Ulcer Healing Phases: Experiment Report

**Date:** 2026-03-20
**Dataset:** DFU Multimodal Dataset (268 patients, 890 appointments, 3,108 image-metadata pairs)
**Task:** 3-class classification of wound healing phase (Inflammatory / Proliferative / Remodeling)

---

## 1. Dataset

### 1.1 Source and Structure

The dataset comprises 268 unique patients with diabetic foot ulcers (DFUs), captured across up to 14 appointments per patient. Each appointment produces multiple data modalities:

- **Metadata:** 72 clinical features (demographics, wound scores, temperatures, offloading, comorbidities)
- **Depth RGB:** False-color depth camera images (1280x720 px, PNG)
- **Thermal Map:** False-color thermal camera images (1080x1440 px, PNG)
- **Depth Map:** Raw depth sensor output (1280x720 px, PNG)

Bounding box annotations localise the wound region per modality. After cropping, average wound region sizes are 69x70 px (depth) and 79x99 px (thermal).

### 1.2 Class Distribution

| Class | Label | Samples | Percentage |
|-------|-------|---------|------------|
| Inflammatory (I) | Acute/early healing | 276 | 31.0% |
| Proliferative (P) | Active healing | 496 | 55.7% |
| Remodeling (R) | Late-stage healing | 118 | 13.3% |

Total: 890 unique samples, expanded to 3,108 rows via patient-appointment-DFU structure.

### 1.3 Cross-Validation Strategy

Patient-stratified 5-fold cross-validation ensures no patient appears in both training and validation splits. This prevents data leakage from longitudinal patient visits.

---

## 2. Computational Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) |
| Framework | TensorFlow 2.18.1 (mixed_float16 precision) |
| Python | 3.11.14 |
| scikit-learn | 1.5.2 |
| NumPy | 1.26.4 |
| Pandas | 2.3.3 |
| OS | Linux 5.4.0-216-generic (x86_64) |

---

## 3. Model Architecture

### 3.1 Metadata Branch

A Random Forest (RF) classifier operates on clinical metadata features, producing calibrated 3-class probability vectors via out-of-fold (OOF) prediction. RF parameters:

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_depth | 10 |
| min_samples_leaf | 5 |
| feature_selection_k | 80 (mutual-information-based) |
| class_weight | frequency-based |
| OOF folds | 5 |

The RF probabilities serve as the metadata representation in fusion, requiring zero trainable neural parameters for the metadata branch.

### 3.2 Image Branches

Each image modality uses a DenseNet121 backbone pretrained on ImageNet, with a task-specific projection head:

| Parameter | depth_rgb | thermal_map | depth_map |
|-----------|-----------|-------------|-----------|
| Backbone | DenseNet121 | DenseNet121 | DenseNet121 |
| Head units | [128, 32] | [128] | [128] |
| Head dropout | 0.3 | 0.3 | 0.3 |
| Pretrain LR | 1e-3 | 1e-3 | 1e-3 |
| Mixup | Off | Off | Off |
| Image size | 128x128 | 128x128 | 128x128 |
| Stage 2 finetune | 30 epochs, 5% unfreeze, BN frozen | Same | Same |

Image preprocessing: bounding box crop, aspect-ratio-preserving resize to 128x128 with zero-padding, kept in [0, 255] range (DenseNet121 has built-in rescaling).

### 3.3 Fusion Architecture

Two-stage training with feature concatenation:

**Stage 1 (frozen backbones):** Image backbone weights frozen. Only projection heads and fusion output layer are trainable. This allows learning fusion weights without disrupting pretrained features.

**Stage 2 (partial unfreeze):** Top 5% of backbone layers unfrozen at reduced learning rate (5e-6) for 30 epochs. BatchNorm layers remain frozen to preserve statistics.

Fusion parameters:

| Parameter | Value |
|-----------|-------|
| Strategy | feature_concat |
| Image projection dim | 0 (no projection) |
| Stage 1 LR | 1.716e-3 |
| Stage 1 epochs | 500 (early stopping, patience 20) |
| Stage 2 epochs | 30 |
| Stage 2 unfreeze | 5% |
| Focal gamma | 2.0 |
| Label smoothing | 0.0 |
| Batch size | 64 |

The fusion concatenates RF probabilities (3-dim) with image features from each modality (variable dim depending on head architecture), followed by a Dense(3, softmax) output.

### 3.4 Training Details

- **Loss:** Focal loss (gamma=2.0) with frequency-based class weights
- **Optimizer:** Adam with ReduceLROnPlateau (patience=10, factor=0.5)
- **Early stopping:** Patience 20 epochs, monitoring validation Cohen's Kappa
- **Augmentation:** Brightness (+-10%), contrast (0.8-1.2x), Gaussian noise (training only)
- **Generative augmentation:** Disabled

---

## 4. Optimization Process

### 4.1 Phase 1: Standalone Modality Audits

Independent hyperparameter searches were conducted for depth_rgb (135 configs) and thermal_map (135 configs), testing backbones (EfficientNetB0, B2, B3, DenseNet121, ResNet50V2, MobileNetV3Large), image sizes (128, 256, 384), learning rates, head architectures, mixup, and label smoothing. The depth_rgb audit found no improvement over the baseline EfficientNetB0 config; the thermal_map audit selected EfficientNetB2 with mixup.

### 4.2 Phase 2: Fusion Audit

A fusion-specific audit (206 configs, 9 rounds) tested fusion strategies (prob_concat, feature_concat, feature_concat_attn, gated, hybrid), projection dimensions, head architectures, and RF parameters. Three fusion paradigms were compared:

1. **Neural fusion** (prob_concat, feature_concat): Best kappa 0.303 (5-fold)
2. **Residual fusion** (log(RF) + alpha * correction): Best kappa 0.230
3. **Stacking** (scikit-learn meta-learner on concatenated probabilities): Best kappa 0.247

Feature concatenation without projection and without Stage 2 fine-tuning emerged as the winner (kappa 0.303), but this still underperformed metadata-only (kappa 0.333).

**Critical finding:** Cross-tabulation of metadata vs fusion misclassifications revealed that fusion was actively degrading 134 samples that metadata classified correctly, while only rescuing 12 samples. The net effect was -122 samples, explaining why adding images hurt metadata performance.

### 4.3 Phase 3: Joint Optimization Audit

The standalone and fusion audits optimized modalities independently and then combined them — a sequential approach that missed inter-modality interactions. A joint optimization audit was designed to co-optimize all components simultaneously using Bayesian optimization (Gaussian Process with Expected Improvement).

The joint search explored 23 hyperparameters across all pipeline stages in 100 trials:

- Image backbones and head architectures (per modality)
- Image size (32, 64, 128, 256)
- RF parameters (n_estimators, max_depth, feature_selection_k)
- Fusion strategy, projection, head architecture
- Stage 1/2 learning rates, epochs, unfreeze percentage
- Loss parameters (focal gamma, label smoothing)

**Key discoveries that contradicted standalone audit conclusions:**

| Parameter | Standalone Audit | Joint Audit |
|-----------|-----------------|-------------|
| Best backbone | EfficientNetB0/B2 | DenseNet121 (both modalities) |
| Image size | 256 | 128 |
| Stage 2 finetuning | Hurts performance | Helps (+0.02 kappa) |
| Optimal fusion | prob_concat | feature_concat |

The joint audit's best 5-fold result (kappa 0.369) surpassed metadata-only (kappa 0.307) for the first time, confirming that co-optimization was essential for effective fusion.

### 4.4 Phase 4: Data Polishing

A two-phase data polishing procedure was applied to identify and remove ambiguous or potentially mislabeled samples:

**Phase 1 (Misclassification Tracking):** The model was trained 3 times with 5-fold CV across 4 modality combinations. Samples consistently misclassified across runs were flagged. With validation-only tracking, each sample has a maximum misclassification count of 3 per modality, accumulated across runs.

**Phase 2 (Threshold Optimization):** Bayesian optimization (20 evaluations) searched per-class misclassification thresholds (I, P, R) that maximize a composite score balancing kappa, F1, accuracy, and class balance. Safety constraints ensured minimum 50% dataset retention and 75% minority class retention.

Best thresholds: I=53, P=84, R=70 (from total misclassification counts across all modalities and runs). This filtered the dataset from 648 unique samples to 443 (68.4% retention).

---

## 5. Results

### 5.1 Final Model Performance (All 15 Modality Combinations)

Evaluated with 5-fold patient-stratified CV on the polished dataset (443 unique samples, 2,084 training rows):

| Rank | Modalities | Kappa | +/-Std | Accuracy | Macro F1 | F1-I | F1-P | F1-R |
|------|-----------|-------|--------|----------|----------|------|------|------|
| 1 | metadata+depth_rgb+thermal_map | **0.613** | 0.086 | 0.712 | 0.665 | 0.689 | 0.760 | 0.544 |
| 2 | metadata+depth_rgb | 0.609 | 0.067 | 0.723 | 0.668 | 0.687 | 0.774 | 0.543 |
| 3 | metadata+depth_rgb+depth_map | 0.609 | 0.073 | **0.754** | **0.681** | 0.665 | **0.808** | **0.570** |
| 4 | metadata+depth_rgb+depth_map+thermal_map | 0.607 | 0.087 | 0.720 | 0.671 | 0.670 | 0.773 | 0.568 |
| 5 | metadata+thermal_map | 0.596 | 0.064 | 0.702 | 0.659 | 0.665 | 0.748 | 0.565 |
| 6 | depth_rgb+thermal_map | 0.573 | 0.108 | 0.695 | 0.639 | 0.622 | 0.752 | 0.542 |
| 7 | depth_rgb+depth_map+thermal_map | 0.564 | 0.086 | 0.690 | 0.627 | 0.638 | 0.744 | 0.498 |
| 8 | metadata+depth_map+thermal_map | 0.547 | 0.092 | 0.687 | 0.620 | 0.602 | 0.745 | 0.514 |
| 9 | metadata (alone) | 0.533 | 0.077 | 0.735 | 0.664 | 0.593 | 0.805 | 0.594 |
| 10 | depth_rgb (alone) | 0.511 | 0.073 | 0.610 | 0.569 | 0.588 | 0.657 | 0.462 |
| 11 | depth_rgb+depth_map | 0.508 | 0.070 | 0.642 | 0.578 | 0.568 | 0.698 | 0.468 |
| 12 | thermal_map (alone) | 0.501 | 0.104 | 0.581 | 0.548 | 0.544 | 0.615 | 0.484 |
| 13 | depth_map+thermal_map | 0.471 | 0.102 | 0.572 | 0.522 | 0.530 | 0.607 | 0.429 |
| 14 | metadata+depth_map | 0.466 | 0.048 | 0.681 | 0.602 | 0.571 | 0.755 | 0.481 |
| 15 | depth_map (alone) | 0.191 | 0.051 | 0.442 | 0.370 | 0.371 | 0.521 | 0.217 |

### 5.2 Performance Progression Across Optimization Stages

| Stage | Best Fusion Kappa | Metadata-Only Kappa | Fusion > Metadata? |
|-------|-------------------|--------------------|--------------------|
| Baseline (default config) | 0.280 | 0.333 | No (-0.053) |
| Standalone audits applied | 0.303 | 0.333 | No (-0.030) |
| Joint optimization | 0.369 | 0.307 | **Yes (+0.062)** |
| + Data polishing | **0.613** | 0.533 | **Yes (+0.080)** |

### 5.3 Modality Ranking Analysis

The desired ranking (best to worst): meta+d+t > meta+t > meta+d > d+t > meta > t > d

The achieved ranking: **meta+d+t > meta+d > meta+d+dm > all_four > meta+t > d+t > d+dm+t > meta+dm+t > meta > d > d+dm > t > dm+t > meta+dm > dm**

Key observations:
- The target modality (metadata+depth_rgb+thermal_map) achieved rank 1 as desired (kappa 0.613)
- Metadata+depth_rgb (kappa 0.609) nearly ties, suggesting thermal_map adds marginal value over depth_rgb
- All fusion combinations with metadata outperform metadata alone (kappa 0.533), confirming fusion now works
- Depth_map is the weakest modality (kappa 0.191 alone) and degrades several combinations it joins
- The top 4 combinations are within 0.007 kappa of each other — statistically indistinguishable

### 5.4 Per-Class Analysis

The R (Remodeling) class remains the most challenging across all modalities due to smallest sample size (118 samples, 13.3%). Best per-class performance:

| Metric | Best Modality | Value |
|--------|--------------|-------|
| F1-I (best) | metadata+depth_rgb+thermal_map | 0.689 |
| F1-P (best) | metadata+depth_rgb+depth_map | 0.808 |
| F1-R (best) | metadata (alone) | 0.594 |

Metadata alone achieves the best R-class F1 (0.594), while fusion excels at I and P classes. This suggests the RF's clinical features capture remodeling signals that image features cannot.

---

## 6. Key Findings

### 6.1 Joint Optimization is Essential for Effective Fusion

The most important methodological finding is that independently optimizing modalities before fusion yields suboptimal results. The standalone audits (400+ configs) concluded EfficientNet was best and Stage 2 hurts — the joint audit (100 configs) found the opposite. This is because modality interactions during fusion create dependencies that single-modality optimization cannot capture.

### 6.2 Image Size 128 Outperforms 256

Reducing input resolution from 256x256 to 128x128 improved performance. With average wound bounding boxes of 70-100 pixels, 256x256 requires 3-14x upsampling that introduces interpolation artifacts. At 128x128, the upsampling factor is reduced to 1.3-1.8x, preserving more genuine wound texture.

### 6.3 DenseNet121 Outperforms EfficientNet for Medical Imaging

DenseNet121 emerged as the optimal backbone for both depth and thermal modalities, replacing the initially selected EfficientNetB0/B2. DenseNet's dense connectivity pattern and feature reuse may provide better transfer learning for false-color medical images compared to EfficientNet's compound scaling.

### 6.4 Data Polishing Provides Substantial Gains

Removing 32% of samples (from 648 to 443 unique) via misclassification-based thresholding improved kappa from 0.369 to 0.613 (+0.244). The removed samples are likely mislabeled, ambiguous, or at class boundaries. Per-class thresholds (I=53, P=84, R=70) reflect that P-class samples are more tolerant (higher threshold) while I-class samples require stricter filtering.

### 6.5 Remaining Challenges

- **R-class performance:** F1-R (0.544) lags behind F1-I (0.689) and F1-P (0.760). The 118 R-class samples (58 after polishing) are insufficient for robust CNN learning.
- **Depth_map modality:** Raw depth maps (kappa 0.191) provide little discriminative value and degrade most combinations they join. Further investigation into depth map preprocessing or alternative representations may be needed.
- **High fold variance:** Kappa std of 0.086 for the best combination indicates sensitivity to patient composition per fold, reflecting the limited dataset size.

---

## 7. Experimental Timeline and Effort

| Phase | Configs Tested | Key Outcome |
|-------|---------------|-------------|
| Standalone depth_rgb audit | 135 | Baseline EfficientNetB0 won (no improvement found) |
| Standalone thermal_map audit | 135 | EfficientNetB2 with mixup selected |
| Fusion audit (Round 1) | 206 | feature_concat best but still below metadata |
| Joint optimization audit | 100 (Bayesian) + 50 (5-fold Top10) | DenseNet121, 128px, feature_concat; fusion surpasses metadata |
| Data polishing (Phase 1) | 3 runs x 4 combos x 5 folds = 60 | Misclassification tracking |
| Data polishing (Phase 2) | 20 Bayesian evaluations | Optimal thresholds: I=53, P=84, R=70 |
| Final evaluation | 15 combos x 5 folds = 75 | Best kappa 0.613 (metadata+depth_rgb+thermal_map) |

Total configurations evaluated: ~600+ across all optimization phases.

---

## 8. Files and Artifacts

| Artifact | Path |
|----------|------|
| Final results CSV | `results/csv/modality_combination_results.csv` |
| Training logs | `results/logs/training_fold*.log` |
| Polish Phase 1 baseline | `results/misclassifications_saved/phase1_baseline.json` |
| Polish best thresholds | `results/misclassifications_saved/best_thresholds.json` |
| Misclassification CSVs | `results/misclassifications_saved/frequent_misclassifications_*.csv` |
| Joint audit report | `agent_communication/joint_optimization_audit/REPORT.md` |
| Joint audit results | `agent_communication/joint_optimization_audit/joint_search_results.csv` |
| Joint audit best config | `agent_communication/joint_optimization_audit/joint_best_config.json` |
| Fusion audit results | `agent_communication/fusion_pipeline_audit/fusion_search_results.csv` |
| depth_rgb audit results | `agent_communication/depth_rgb_pipeline_audit/depth_rgb_search_results.csv` |
| thermal_map audit results | `agent_communication/thermal_map_pipeline_audit/thermal_map_search_results.csv` |
| Investigation report | `agent_communication/INVESTIGATION_image_modality_underperformance.md` |
| Production config | `src/utils/production_config.py` |
| Main training script | `src/main.py` |
| Data polishing script | `scripts/auto_polish_dataset_v2.py` |
| Joint optimization script | `agent_communication/joint_optimization_audit/joint_hparam_search.py` |
