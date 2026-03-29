# Response to Reviewer Comments

**Manuscript:** Multimodal Healing Phase Classification of Diabetic Foot Ulcer Using Jointly Optimized Fusion with Generative Augmentation

**Journal:** IEEE Journal of Biomedical and Health Informatics

We thank the Associate Editor and all reviewers for their thorough and constructive feedback. Every major concern has been addressed through substantial revisions including architectural simplification, comprehensive ablation studies, expanded statistical analyses, and medical insight into modality contributions. Below we address each comment.

---

## Reviewer 1

**R1.1:Lack of fundamental innovation**

The architecture has been fundamentally restructured. The original complex system (attention mechanisms, gating networks, EfficientNetB3) is replaced with DFU-MFNet: feature concatenation fusion with DenseNet121 backbones and a parameter-free ensemble. The central innovation is now the joint Bayesian optimization methodology (100 trials, 23 hyperparameters) that co-optimizes all pipeline components simultaneously, improving kappa from 0.30 (sequential tuning) to 0.61 (joint optimization). A dedicated ensemble audit (111 configurations, 8 strategies) confirmed that simple averaging outperforms all learned alternatives, including attention-based gating which collapsed to majority-class prediction on 2 of 5 folds. [Sections 2.3, 2.4, 2.6, 4.2]

**R1.2:Poorly defined baselines; unjustified complexity**

Single-modality baselines are now clearly defined with identical evaluation protocol: metadata (kappa 0.45, acc 70.55%), RGB (0.51, 60.96%), thermal (0.50, 58.13%), depth (0.19, 44.21%). All improvements reported as absolute values. McNemar's test confirms fusion vs metadata significance at the sample level (chi-squared = 4.56, p = 0.03). The architecture uses no attention mechanisms and no learned ensemble weights. [Sections 3.1, 3.2, Table I]

**R1.3:No ablation studies**

Ablation is provided through: (1) 100 Bayesian trials comparing 6 backbones, 4 image sizes, 2 fusion strategies, and 8 ensemble methods; (2) 111 ensemble configurations confirming simple averaging over complex alternatives; (3) dose-response across 3 augmentation probabilities; (4) ANOVA (F = 5.56, p = 0.002) and linear regression (slope = 0.07 kappa/modality, R-squared = 0.40, p = 0.01) quantifying modality contributions. [Sections 2.6, 3.2, 4.2]

**R1.4:Insufficient dataset; no external validation**

The matched dataset comprises 3,108 data points from 443 assessments across 233 patients. Five-fold patient-stratified CV prevents leakage. Architectural choices reflect data constraints (simpler models validated over complex ones). External validation on public datasets (DFU2020, Plantar Thermogram Database) is identified as future work. Code and model weights are publicly available. [Sections 2.2, 4.5]

**R1.5:Insufficient synthetic image validation**

Expanded evaluation includes FID (220.13), SSIM (0.87 +/- 0.05), LPIPS (0.41 +/- 0.10) with per-phase breakdown. Elevated FID is contextualized as a known limitation of ImageNet-pretrained metrics for medical domains. Dose-response across 6%, 15%, 25% injection rates shows inverted-U pattern: F1-R improves from 0.52 to 0.55 at 15%, declines at 25%. Augmentation is restricted to RGB (thermal/depth require biophysical constraint preservation). Clinical expert validation is identified as future work. [Section 3.1, 4.3]

**R1.6:Superficial comparison with existing methods**

Section 4.2 discusses DFU_VIRNet and related approaches. Direct comparison is limited by task differences (binary vs three-class) and dataset differences. Our three-class formulation provides higher clinical value by informing phase-specific therapeutic interventions. The finding that simple fusion outperforms complex attention approaches aligns with broader literature on limited-data regimes. [Section 4.2]

**R1.7:No clinical validation**

Clinical significance is reframed around evidence-based deployment guidance rather than outcome claims. Tiered implementation: metadata-only (kappa 0.45, no hardware), metadata + RGB (kappa 0.61, single camera), full multimodal (accuracy 78.86%, specialized equipment). The modality analysis provides medical insight: metadata captures systemic healing factors, RGB captures wound morphology distinguishing inflammatory from proliferative phases, thermal captures subsurface vascular activity identifying remodeling phase, and depth alone is not phase-discriminative. Prospective clinical trials are identified as future work. [Sections 4.1, 4.4, 4.5]

---

## Reviewer 2

**R2.1:Insufficient ablation studies**

Addressed through 100 Bayesian optimization trials and 111 ensemble configurations (see R1.3). Each component's contribution is quantified: ANOVA shows significant modality group differences (p = 0.002), and each additional modality adds 0.07 kappa (linear regression, p = 0.01). [Sections 2.6, 3.2]

**R2.2:Misleading baseline; overstated improvement**

All improvements reported as absolute metrics. Prior work achieving 69% accuracy on a larger dataset is acknowledged explicitly. Current metadata achieves 70.55%, fusion improves to 75.38%. McNemar's test confirms significance (p = 0.03). No percentage-based improvement claims. [Sections 3.2, 4.4]

**R2.3:Low input resolution (64x64)**

Increased to 128x128, selected through joint optimization across 32, 64, 128, 256. Justified by average wound bounding box size (70x70 pixels): 128x128 provides 1.3-1.8x upsampling, preserving wound texture; 256x256 causes 3-14x upsampling with interpolation artifacts. Backbone changed from EfficientNetB3 to DenseNet121. [Sections 2.3, 4.2]

**R2.4:Dataset limitations**

Architecture simplified to match data regime: feature concatenation (not attention), parameter-free ensemble (not learned gating), DenseNet121 (not EfficientNetB3). Class imbalance addressed with focal loss, per-class F1 reported (Table I), ensemble improves F1-R from 0.54 to 0.59. [Sections 2.3, 3.3, Table I]

**R2.5:Generalizability and selection bias**

All 15 modality combinations are evaluated, not just the full four-modality setup. Results show effective performance with partial modalities: metadata + RGB alone achieves kappa 0.57. The modular architecture adapts to available resources. Equipment cost ($3,000-$5,000) refers to a combined multi-sensor device. [Sections 3.2, 4.4]

**R2.6:Loss function and calibration**

Focal loss is now standard (Equation 1), with the ordinal component removed after ablation showed no benefit. Expected Calibration Error (ECE) is reported: best fusion ECE = 0.077, indicating well-calibrated predictions. Per-modality ECE ranges from 0.058 to 0.164. [Sections 2.3, 3.2]

---

## Reviewer 3

**R3.1:External validation**

Acknowledged as limitation. Code and weights are publicly available for independent validation. Public DFU datasets are single-modality, preventing full multimodal evaluation, but the modular architecture supports single-modality deployment. Multi-site validation is a priority future direction. [Section 4.5]

**R3.2:FID insufficient; need t-SNE or clinician review**

SSIM and LPIPS added alongside FID, with per-phase breakdown. The dose-response analysis (3 probabilities) provides empirical evidence of useful training signal. Expert clinical validation is identified as future work. [Section 3.1]

**R3.3:Comparison with recent multimodal approaches**

DFU_VIRNet and related approaches discussed. Task differences (binary vs three-class) prevent direct numerical comparison. Our contribution is positioned as the first systematic multimodal three-class healing phase study with joint optimization. [Section 4.2]

**R3.4:Attention heatmaps or feature importance**

The revised architecture uses feature concatenation (no attention maps). Instead, we provide: modality agreement analysis (Figure 5) showing per-sample correctness patterns, error correlation metrics (Q-statistic, disagreement) between modalities, and a quantified modality contribution hierarchy from all 15 combinations. [Sections 3.3, 4.1]

**R3.5:Clearer description of pipeline**

Section 2 restructured: two-stage training protocol fully specified (500 epochs frozen, 30 epochs 5% unfreeze), missing modality handling explained (separate model per combination, ensemble adapts to available modalities), training hyperparameters listed. Figure 3 shows detailed architecture. [Section 2]

**R3.6:Effect sizes and confidence intervals**

ANOVA with pairwise comparisons (Bonferroni corrected), linear regression (slope = 0.07, R-squared = 0.40), McNemar's test (chi-squared = 4.56 and 84.50), and standard deviations for all metrics across 5-fold CV. [Sections 3.2, 3.3]

**R3.7:Clinical workflow integration**

Tiered deployment framework in Section 4.4: (1) metadata-only using EHR data, no hardware; (2) metadata + RGB with single depth camera; (3) full multimodal with ensemble. Each tier with quantified performance. Medical insight: metadata captures systemic factors, RGB captures wound morphology, thermal captures vascular activity informing remodeling detection. [Sections 4.1, 4.4]

**R3.8:Per-class analysis**

Table I includes per-class F1 (I, P, R) for all combinations. Ensemble specifically addresses R-class: F1-R improves from 0.54 to 0.59. Confusion matrices provided in supplementary figures. [Table I, Section 3.3]

**R3 Minor Comments 1-8**

All addressed: (1) Methods restructured with shorter paragraphs. (2) Figure captions expanded. (3) All abbreviations defined at first use. (4) Two decimal places throughout. (5) DOIs added. (6) Framework overview in Figure 1. (7) Figures at 300 DPI. (8) Proofreading completed.

---

## Summary of Changes

| Concern | Resolution |
|---------|-----------|
| Complex architecture | Simplified to feature concatenation + parameter-free ensemble |
| No ablation | 100 Bayesian trials + 111 ensemble configs + 3-level dose-response |
| Weak baselines | Absolute metrics with identical evaluation protocol |
| Overstated claims | McNemar's p = 0.03; no percentage improvements |
| Small dataset | Architecture matched to data regime; simpler models validated |
| No external validation | Acknowledged; code/weights public; future direction |
| Synthetic validation | FID + SSIM + LPIPS + dose-response; clinical validation as future work |
| No calibration | ECE reported (0.058-0.164 range) |
| Low resolution | 128x128, justified by bounding box analysis |
| Missing statistics | ANOVA, McNemar's, paired t-tests, linear regression, effect sizes |
| Clinical relevance | Tiered deployment; modality-to-biology insight; decision-making guidance |
| Medical insight | Modality results mapped to wound biology (vascular, morphological, systemic) |
