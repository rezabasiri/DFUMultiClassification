# Feature Engineering & Selection Plan
**Date Created**: 2025-12-24
**Project**: DFU Multi-Classification
**Current Baseline**: ~47-50% accuracy (patient-level CV, metadata only)
**Goal**: Improve generalization performance through systematic feature engineering and selection

---

## Executive Summary

### Problem Statement
Current metadata-only model achieves ~47-50% accuracy with proper patient-level cross-validation (down from inflated 97.6% with data leakage). This represents the **true baseline** for generalizable wound healing phase classification. Feature engineering is necessary to capture complex clinical patterns not evident in raw features.

### Objectives
1. **Develop comprehensive feature engineering strategies** based on clinical domain knowledge
2. **Implement systematic feature selection** to identify most informative features
3. **Prevent overfitting** through proper patient-level CV and validation
4. **Improve model performance** beyond current ~50% baseline
5. **Maintain interpretability** for clinical relevance

### Current Dataset
- **Patients**: 233 unique patients
- **Samples**: 891 total samples (after filtering)
- **Raw Features**: 65+ features across demographics, clinical scores, wound characteristics, temporal info
- **Class Distribution**: I=24%, P=57%, R=19% (imbalanced)
- **Modalities**: metadata (tabular), depth_rgb, depth_map, thermal_map (images)

---

## Feature Engineering Strategies

### Strategy 1: Temperature-Based Features 🌡️
**Rationale**: Paper shows thermal modality has superior inflammatory phase performance (F1=0.34 for I class). Temperature gradients indicate inflammation and perfusion status.

**Features to Create**:
- [ ] `temp_wound_to_periulcer_diff` = Wound Centre Temp - Peri-Ulcer Temp
- [ ] `temp_wound_to_intact_diff` = Wound Centre Temp - Intact Skin Temp
- [ ] `temp_periulcer_to_intact_diff` = Peri-Ulcer Temp - Intact Skin Temp
- [ ] `temp_wound_to_periulcer_ratio` = Wound Centre Temp / Peri-Ulcer Temp (avoid div by zero)
- [ ] `temp_gradient_normalized` = (Wound - Intact) / Intact (normalized gradient)
- [ ] `temp_inflammation_index` = Combine wound-peri difference with erythema/edema binary flags
- [ ] `temp_variance` = Std dev across all three temperature measurements (if multiple readings)
- [ ] `temp_abnormality_flag` = Binary: 1 if wound temp > peri-ulcer + threshold

**Implementation Priority**: HIGH (thermal modality showed strong inflammatory detection)

---

### Strategy 2: Temporal/Longitudinal Features 📅
**Rationale**: Healing is a temporal process. Visit patterns and time since onset may indicate healing trajectory.

**Features to Create**:
- [ ] `days_per_visit` = Appt Days / (Appt# + 1) (average days between visits)
- [ ] `visit_frequency_category` = Binned: frequent (<7 days), normal (7-14), infrequent (>14)
- [ ] `onset_to_first_appt` = Days for initial presentation (if Appt#==0)
- [ ] `healing_duration_group` = Onset binned: acute (<30 days), subacute (30-90), chronic (>90)
- [ ] `visit_progression_ratio` = Appt# / max(Appt#) for patient (normalized visit position)
- [ ] `time_acceleration` = Difference in days between consecutive visits (requires patient history)

**Implementation Priority**: MEDIUM (requires careful handling of patient-level grouping)

**Note**: Be cautious with patient-specific temporal aggregations - must not leak across CV folds!

---

### Strategy 3: Composite Clinical Scores & Indices 🏥
**Rationale**: Paper mentions Habits Score, Clinical Score, Foot Score, Leg Score, Wound Score, Offloading Score. Create additional composite indices.

**Features to Create**:
- [ ] `total_clinical_burden` = Sum of all existing scores (Habits + Clinical + Foot + Leg + Wound + Offloading)
- [ ] `foot_deformity_count` = Sum of Bunion + Claw + Hammer + Charcot + Flat Arch + High Arch
- [ ] `periulcer_condition_count` = Sum of Erythema + Edema + Pale + Maceration
- [ ] `foot_abnormality_count` = Sum of Hair Loss + Dry Skin + Fissures + Callus + Thickened Nail + Fungal Nails
- [ ] `offloading_diversity` = Number of different offloading methods used (0-6)
- [ ] `wound_severity_index` = Weighted combo: Exudate Amount × 2 + Tunneling × 3 + Odor × 1.5
- [ ] `inflammation_indicator` = Erythema + Edema + (temp_wound_to_periulcer_diff > threshold)
- [ ] `comorbidity_count` = Heart Conditions + Cancer History + Sensory Peripheral

**Implementation Priority**: HIGH (clinically interpretable, combines domain knowledge)

---

### Strategy 4: Biomechanical & Anatomical Features 🦶
**Rationale**: Location and foot structure affect healing. Paper shows location grouping (Hallux, Toes, Middle, Heel, Ankle).

**Features to Create**:
- [ ] `location_risk_score` = Risk weights based on literature: Heel=3, Hallux=2, Toes=2, Middle=1.5, Ankle=1
- [ ] `plantar_vs_other` = Binary: 1 if Foot Aspect contains "Plantar" (weight-bearing area)
- [ ] `toe_region` = Binary: 1 if Location Grouped in {Hallux, Toes}
- [ ] `weight_bearing_zone` = Binary: combine Plantar aspect + Heel/Hallux location
- [ ] `arch_deformity_present` = Binary: 1 if NOT(No Arch Deformities)
- [ ] `structural_instability` = Charcot + arch_deformity_present + foot_deformity_count
- [ ] `offloading_adequacy` = Offloading Score / wound_severity_index (is offloading sufficient?)

**Implementation Priority**: MEDIUM-HIGH (anatomical context is clinically relevant)

---

### Strategy 5: Patient Risk Profile Features 👤
**Rationale**: Patient demographics and comorbidities affect healing capacity.

**Features to Create**:
- [ ] `bmi` = Weight (kg) / (Height (m))^2
- [ ] `bmi_category` = Binned: underweight (<18.5), normal (18.5-25), overweight (25-30), obese (>30)
- [ ] `age_group` = Binned: young (<40), middle (40-65), elderly (>65)
- [ ] `high_risk_patient` = (Age > 65) OR (Heart Conditions) OR (BMI > 30) OR (Sensory Peripheral)
- [ ] `diabetes_duration_proxy` = Number of DFUs (multiple ulcers suggest long-standing diabetes)
- [ ] `lifestyle_risk` = Habits Score (already exists, but ensure proper use)
- [ ] `age_bmi_interaction` = Age × BMI (older + obese = high risk)
- [ ] `age_diabetes_interaction` = Age × Type of Diabetes
- [ ] `sensory_foot_deformity` = Sensory Peripheral × foot_deformity_count (neuropathy + deformity)

**Implementation Priority**: MEDIUM (patient-level features are stable but may not vary enough)

---

### Strategy 6: Wound Characteristics Ratios 🩹
**Rationale**: Relative measures may be more informative than absolute values.

**Features to Create**:
- [ ] `pain_to_exudate_ratio` = Pain Level / (Exudate Amount + 1) (pain should decrease with drainage)
- [ ] `exudate_to_size_proxy` = Exudate Amount / (assume wound size not directly available, use temp variance as proxy?)
- [ ] `odor_exudate_interaction` = Odor × Exudate Appearance (thick+odor = infection)
- [ ] `wound_complexity` = Tunneling + (Exudate Amount > 2) + (Odor > 0) + periulcer_condition_count
- [ ] `infection_risk_score` = (Odor × 3) + (Exudate Appearance >= 3 × 2) + (temp_inflammation_index)
- [ ] `healing_inhibitor_score` = infection_risk_score + foot_deformity_count + (No Offloading)

**Implementation Priority**: MEDIUM (interpretable but may need validation)

---

### Strategy 7: Polynomial & Interaction Features 🔢
**Rationale**: Non-linear relationships may exist between continuous features.

**Features to Create**:
- [ ] `age_squared` = Age^2
- [ ] `onset_squared` = Onset (Days)^2
- [ ] `temp_diff_squared` = (temp_wound_to_periulcer_diff)^2
- [ ] `bmi_temp_interaction` = BMI × temp_wound_to_intact_diff (obesity affects thermoregulation)
- [ ] `age_onset_interaction` = Age × Onset (Days) (older patients with chronic wounds)
- [ ] `exudate_pain_interaction` = Exudate Amount × Pain Level
- [ ] `temp_exudate_interaction` = temp_inflammation_index × Exudate Amount

**Implementation Priority**: LOW-MEDIUM (risk of overfitting, use feature selection carefully)

**Note**: Use polynomial features sparingly - validate with feature importance analysis.

---

### Strategy 8: Missing Data Indicators 🚩
**Rationale**: Paper mentions 2.5% average missing data, imputed with 5-NN. Missingness patterns may be informative.

**Features to Create**:
- [ ] `temp_data_missing` = Binary: 1 if any temperature measurement was originally missing
- [ ] `wound_data_missing` = Binary: 1 if exudate/odor/tunneling was missing
- [ ] `patient_data_missing` = Binary: 1 if demographics (age/weight/height) missing
- [ ] `missing_feature_count` = Total count of originally missing features (track before imputation)
- [ ] `critical_missing` = Binary: 1 if healing phase-related features (temp, exudate) were missing

**Implementation Priority**: LOW (may not add much signal, but worth testing)

**Note**: Need to track missingness BEFORE imputation in preprocessing pipeline.

---

### Strategy 9: Domain-Specific Clinical Ratios 🔬
**Rationale**: Based on wound healing biology from paper (inflammatory → proliferative → remodeling phases).

**Features to Create**:
- [ ] `inflammatory_signature` = temp_inflammation_index + Erythema + Edema + Pain Level
- [ ] `proliferative_signature` = (Exudate Amount == 1 or 2) × (Pain Level < 5) × (1 - Odor) (clean drainage, lower pain)
- [ ] `remodeling_signature` = (Exudate Amount == 0) × (Pain Level < 3) × (No Peri-ulcer Conditions) (dry, minimal pain)
- [ ] `phase_transition_score` = Weighted: inflammatory_sig × 1 + proliferative_sig × 0 + remodeling_sig × (-1) (progression axis)
- [ ] `perfusion_proxy` = temp_wound_to_intact_diff (positive = hyperperfusion/inflammation, negative = ischemia)
- [ ] `neuropathy_severity` = Sensory Peripheral × (Pain Level == 0) (neuropathy with no pain despite wound)

**Implementation Priority**: HIGH (directly aligned with target classes - interpretable!)

**Clinical Validation**: These should be reviewed by domain expert if possible.

---

## Feature Selection Strategy

### Approach 1: Filter Methods (Fast, Univariate)
**Methods**:
- [ ] Mutual Information (sklearn.feature_selection.mutual_info_classif)
- [ ] ANOVA F-statistic (sklearn.feature_selection.f_classif)
- [ ] Chi-squared test for categorical features
- [ ] Correlation matrix analysis (remove highly correlated features > 0.95)

**Threshold**: Keep top 50-100 features based on MI/F-score

**Priority**: HIGH (fast baseline selection)

---

### Approach 2: Wrapper Methods (Iterative, Multivariate)
**Methods**:
- [ ] Recursive Feature Elimination (RFE) with Random Forest
- [ ] Forward Selection (add features iteratively)
- [ ] Backward Elimination (remove features iteratively)
- [ ] Boruta algorithm (all-relevant feature selection)

**Validation**: Use patient-level CV, track Min F1 to ensure minority class learning

**Priority**: MEDIUM (computationally expensive but robust)

---

### Approach 3: Embedded Methods (Model-Based)
**Methods**:
- [ ] L1 Regularization (Lasso) - sparse feature weights
- [ ] Random Forest Feature Importance (Gini importance or permutation importance)
- [ ] XGBoost Feature Importance (gain, cover, frequency)
- [ ] ElasticNet (L1 + L2 regularization)

**Validation**: Patient-level CV with multiple random seeds

**Priority**: HIGH (built into model training)

---

### Approach 4: Stability Selection
**Method**:
- [ ] Resample data multiple times (bootstrap or subsampling)
- [ ] Run feature selection on each resample
- [ ] Select features that appear in > 80% of runs (stable features)

**Rationale**: Prevents selection of spurious features that work on one split but not others

**Priority**: MEDIUM-HIGH (critical for small dataset with 233 patients)

---

### Approach 5: Clinical Validation
**Method**:
- [ ] Review selected features with domain knowledge from paper
- [ ] Ensure features align with wound healing biology
- [ ] Flag features that seem spurious or data-leaky
- [ ] Prioritize interpretable features over black-box combinations

**Priority**: HIGH (maintain clinical relevance)

---

## Implementation Checklist

### Phase 1: Setup & Data Exploration ✅
- [x] Read paper/main.tex for domain knowledge
- [x] Read raw CSV for available features
- [x] Understand current preprocessing pipeline
- [x] Document current performance baseline (~47-50%)
- [ ] Create feature engineering utility module

### Phase 2: Implement Feature Engineering Strategies
- [ ] **Strategy 1**: Temperature-based features (8 features)
- [ ] **Strategy 2**: Temporal features (6 features - requires patient grouping)
- [ ] **Strategy 3**: Composite scores (8 features)
- [ ] **Strategy 4**: Biomechanical features (7 features)
- [ ] **Strategy 5**: Patient risk profiles (9 features)
- [ ] **Strategy 6**: Wound characteristic ratios (6 features)
- [ ] **Strategy 7**: Polynomial features (7 features - use sparingly)
- [ ] **Strategy 8**: Missing data indicators (5 features - if needed)
- [ ] **Strategy 9**: Clinical signatures (6 features - HIGH PRIORITY)

**Total New Features**: ~60-70 engineered features (combining with 65 raw = ~130 total before selection)

### Phase 3: Feature Selection Implementation
- [ ] Create feature selection module (src/features/feature_selection.py)
- [ ] Implement mutual information filtering
- [ ] Implement correlation analysis
- [ ] Implement Random Forest feature importance
- [ ] Implement RFE with cross-validation
- [ ] Implement stability selection
- [ ] Create feature selection report generator

### Phase 4: Validation & Testing
- [ ] Run feature engineering on full dataset
- [ ] Verify no data leakage (patient-level CV maintained)
- [ ] Test each feature strategy independently
- [ ] Run comprehensive feature selection comparison
- [ ] Generate feature importance visualizations
- [ ] Create final selected feature set

### Phase 5: Model Training with Engineered Features
- [ ] Train metadata-only model with selected features
- [ ] Compare performance vs baseline (47-50%)
- [ ] Analyze per-class F1 scores (especially minority class R)
- [ ] Run ablation study (remove feature groups, measure impact)
- [ ] Document results in agent_communication/feature_engineering/

### Phase 6: Integration with Multimodal Pipeline
- [ ] Integrate feature engineering into src/data/dataset_utils.py
- [ ] Ensure compatibility with existing preprocessing
- [ ] Test with image modalities (depth_rgb, depth_map, thermal_map)
- [ ] Run comprehensive CV test with engineered features
- [ ] Update production_config.py with feature engineering flags

---

## Technical Implementation Notes

### Code Structure
```
src/
  features/
    __init__.py
    feature_engineering.py    # All feature creation functions
    feature_selection.py      # Selection algorithms
    feature_utils.py         # Helper functions
agent_communication/
  feature_engineering/
    PLAN.md                  # This file
    PROGRESS.md              # Track completed tasks
    feature_analysis.ipynb   # Exploratory analysis (optional)
    results/                 # Feature importance plots, selection reports
```

### Key Functions to Create
```python
# feature_engineering.py
def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame
def create_composite_scores(df: pd.DataFrame) -> pd.DataFrame
def create_biomechanical_features(df: pd.DataFrame) -> pd.DataFrame
def create_patient_risk_features(df: pd.DataFrame) -> pd.DataFrame
def create_wound_ratios(df: pd.DataFrame) -> pd.DataFrame
def create_polynomial_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame
def create_clinical_signatures(df: pd.DataFrame) -> pd.DataFrame
def engineer_all_features(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame

# feature_selection.py
def select_features_mutual_info(X, y, k: int = 50) -> List[str]
def select_features_rfe(X, y, n_features: int = 50, cv_folds: int = 3) -> List[str]
def select_features_rf_importance(X, y, threshold: float = 0.01) -> List[str]
def select_features_stability(X, y, n_iterations: int = 100, threshold: float = 0.8) -> List[str]
def select_features_comprehensive(X, y, methods: List[str]) -> Dict[str, List[str]]
```

### Data Leakage Prevention Rules
1. **Feature engineering**: Apply AFTER patient-level CV split (inside fold loop)
2. **Temporal features**: Only use information from CURRENT and PAST visits, never future
3. **Patient aggregations**: Compute only on training set patients, never validation set
4. **Scaling/normalization**: Fit on train, transform on validation (already implemented)
5. **Feature selection**: Run INSIDE CV loop, not on full dataset

### Validation Requirements
- **ALL experiments** must use patient-level CV (create_patient_folds from training_utils.py)
- **Report metrics** per fold: Accuracy, F1 Macro, F1 per class [I, P, R], Min F1
- **Track variance** across folds (high variance = overfitting)
- **Min F1 threshold**: Must be > 0.15 (ensure minority class R is learned)

---

## Success Criteria

### Primary Goal
- **Improve accuracy** from baseline ~47-50% to **>55-60%** while maintaining patient-level generalization

### Secondary Goals
- **Min F1 > 0.20** (minority class R properly learned)
- **F1 Macro > 0.45** (balanced performance across classes)
- **Fold variance < 0.05** (consistent generalization)
- **Feature interpretability** maintained (explainable to clinicians)

### Failure Modes to Avoid
- ❌ Overfitting to training data (high train, low val performance)
- ❌ Data leakage (patient overlap between train/val)
- ❌ Spurious features (work on one split, fail on others)
- ❌ Loss of minority class learning (Min F1 drops to 0)
- ❌ Feature explosion without selection (>200 features = curse of dimensionality)

---

## Collaboration Notes

### For Local Agent
- **Feature engineering code** will be in src/features/ module
- **Testing scripts** will be in agent_communication/feature_engineering/
- **Results** saved to agent_communication/feature_engineering/results/
- **Progress tracking** via PROGRESS.md (checkboxes updated as tasks complete)

### For Remote Agent (Claude Code)
- **Strategic planning** and **code review** focus
- **Domain knowledge integration** from paper
- **Mathematical validation** of feature formulas
- **Documentation** and **result interpretation**

### Handoff Protocol
1. Create feature engineering module skeleton
2. Implement one strategy at a time, test independently
3. Commit after each strategy with clear message
4. Update PROGRESS.md checkboxes
5. Generate feature importance report before moving to next strategy

---

## Timeline Estimate (Flexible)

**Note**: No strict deadlines - focus on quality and validation

1. **Phase 1** (Setup): 1-2 hours
2. **Phase 2** (Feature Engineering): 4-6 hours (implement all 9 strategies)
3. **Phase 3** (Feature Selection): 3-4 hours (implement and test selection methods)
4. **Phase 4** (Validation): 2-3 hours (run CV experiments)
5. **Phase 5** (Model Training): 2-3 hours (evaluate performance gains)
6. **Phase 6** (Integration): 1-2 hours (merge into main pipeline)

**Total**: ~13-20 hours of development work

---

## References

1. **Paper**: paper/main.tex - GAMAN architecture, modality analysis, clinical context
2. **Raw Data**: data/raw/DataMaster_Processed_V12_WithMissing.csv - 65+ features, 233 patients
3. **Baseline**: agent_communication/test_metadata_phase9_cv_output.txt - True performance ~47%
4. **CV Implementation**: src/training/training_utils.py:create_patient_folds - Patient-level splitting
5. **Preprocessing**: src/data/dataset_utils.py:prepare_cached_datasets - Current feature pipeline

---

## Questions for Clarification

- [ ] Should we prioritize interpretability over performance? (Clinical acceptance)
- [ ] Are there specific clinical features known to be important from literature?
- [ ] What is acceptable computational cost for feature selection? (RFE can be slow)
- [ ] Should we create a feature engineering config file for reproducibility?
- [ ] Do we want automatic feature engineering (AutoFeat, tsfresh) or manual only?

---

**Last Updated**: 2025-12-24
**Status**: Ready to begin implementation
**Next Step**: Create src/features/ module skeleton and implement Strategy 1 (Temperature features)
