"""
Solution 11: Extensive Feature Engineering

Creates domain-specific engineered features to improve RF performance beyond raw metadata.
Tests interaction features, polynomial features, ratios, and clinical indices.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from src.utils.config import get_data_paths, get_project_paths
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SOLUTION 11: Extensive Feature Engineering")
print("="*80)

# Load data
_, _, root = get_project_paths()
df_raw = pd.read_csv(get_data_paths(root)['csv_file'])

print(f"Dataset: {len(df_raw)} samples")
print(f"Original features: {len(df_raw.columns)} columns")

# Features to exclude (data leakage + identifiers)
features_to_drop = [
    'Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
    'ID', 'Location', 'Healing Phase', 'Phase Confidence (%)',  # Data leakage!
    'Appt Days', 'Type of Pain2', 'Type of Pain_Grouped2', 'Type of Pain',
    'Peri-Ulcer Temperature (°C)', 'Wound Centre Temperature (°C)',
    'Dressing', 'Dressing Grouped',
    'No Offloading', 'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW', 'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast', 'Offloading: Crutches, Walkers or Wheelchairs',
    'Offloading Score',
    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'
]

# Extract base numeric features
base_feature_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns
                     if c not in features_to_drop]

# Copy dataframe for feature engineering
df = df_raw.copy()

print(f"\nBase numeric features: {len(base_feature_cols)}")
print("Creating engineered features...")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# 1. BMI (Body Mass Index) - clinical importance for DFU
if 'Weight (Kg)' in df.columns and 'Height (cm)' in df.columns:
    df['BMI'] = df['Weight (Kg)'] / ((df['Height (cm)'] / 100) ** 2)
    print("  ✓ BMI")

# 2. Age-related interactions
if 'Age' in df.columns:
    if 'Weight (Kg)' in df.columns:
        df['Age_x_Weight'] = df['Age'] * df['Weight (Kg)']
    if 'Clinical Score' in df.columns:
        df['Age_x_ClinicalScore'] = df['Age'] * df['Clinical Score']
    if 'Onset (Days)' in df.columns:
        df['Age_x_Onset'] = df['Age'] * df['Onset (Days)']
    # Age bins (young, middle, senior)
    df['Age_Squared'] = df['Age'] ** 2
    print("  ✓ Age interactions (4 features)")

# 3. Onset (healing time) - critical for progression
if 'Onset (Days)' in df.columns:
    df['Onset_Log'] = np.log1p(df['Onset (Days)'])  # Log transform (right-skewed)
    df['Onset_Squared'] = df['Onset (Days)'] ** 2
    if 'Weight (Kg)' in df.columns:
        df['Onset_x_Weight'] = df['Onset (Days)'] * df['Weight (Kg)']
    if 'Clinical Score' in df.columns:
        df['Onset_x_ClinicalScore'] = df['Onset (Days)'] * df['Clinical Score']
    if 'Wound Score' in df.columns:
        df['Onset_x_WoundScore'] = df['Onset (Days)'] * df['Wound Score']
    print("  ✓ Onset interactions (5 features)")

# 4. Temperature-related features (if normalized temps available)
if all(col in df.columns for col in ['Wound Centre Temperature Normalized (°C)',
                                       'Peri-Ulcer Temperature Normalized (°C)',
                                       'Intact Skin Temperature (°C)']):
    df['Temp_Wound_Periulcer_Diff'] = (df['Wound Centre Temperature Normalized (°C)'] -
                                         df['Peri-Ulcer Temperature Normalized (°C)'])
    df['Temp_Wound_Intact_Diff'] = (df['Wound Centre Temperature Normalized (°C)'] -
                                      df['Intact Skin Temperature (°C)'])
    df['Temp_Periulcer_Intact_Diff'] = (df['Peri-Ulcer Temperature Normalized (°C)'] -
                                          df['Intact Skin Temperature (°C)'])
    df['Temp_Wound_Periulcer_Ratio'] = (df['Wound Centre Temperature Normalized (°C)'] /
                                          (df['Peri-Ulcer Temperature Normalized (°C)'] + 0.001))
    print("  ✓ Temperature differences & ratios (4 features)")

# 5. Clinical severity index (combined scores)
score_cols = [c for c in df.columns if 'Score' in c and c != 'Offloading Score' and c not in features_to_drop]
if len(score_cols) >= 2:
    df['Total_Clinical_Severity'] = df[score_cols].sum(axis=1)
    print(f"  ✓ Total Clinical Severity (sum of {len(score_cols)} scores)")

# 6. Wound-related interactions
if 'Wound Score' in df.columns:
    if 'Pain Level' in df.columns:
        df['WoundScore_x_Pain'] = df['Wound Score'] * df['Pain Level']
    if 'Exudate Amount (None:0,Minor,Medium,Severe:3)' in df.columns:
        df['WoundScore_x_Exudate'] = df['Wound Score'] * df['Exudate Amount (None:0,Minor,Medium,Severe:3)']
    print("  ✓ Wound interactions (2 features)")

# 7. Deformity count (sum of foot/toe deformities)
deformity_cols = ['No Toes Deformities', 'Bunion', 'Claw', 'Hammer',
                  'Charcot Arthropathy', 'Flat (Pes Planus) Arch',
                  'Abnormally High Arch']
deformity_cols = [c for c in deformity_cols if c in df.columns]
if len(deformity_cols) > 0:
    df['Total_Deformities'] = df[deformity_cols].sum(axis=1)
    print(f"  ✓ Total Deformities (sum of {len(deformity_cols)} deformity indicators)")

# 8. Abnormality count (foot abnormalities)
abnorm_cols = ['Foot Hair Loss', 'Foot Dry Skin', 'Foot Fissure Cracks',
               'Foot Callus', 'Thickened Toenail']
abnorm_cols = [c for c in abnorm_cols if c in df.columns]
if len(abnorm_cols) > 0:
    df['Total_Foot_Abnormalities'] = df[abnorm_cols].sum(axis=1)
    print(f"  ✓ Total Foot Abnormalities (sum of {len(abnorm_cols)} indicators)")

# 9. Peri-ulcer condition count
periulcer_cols = ['Erythema at Peri-ulcer', 'Edema at Peri-ulcer',
                  'Pale Colour at Peri-ulcer', 'Maceration at Peri-ulcer']
periulcer_cols = [c for c in periulcer_cols if c in df.columns]
if len(periulcer_cols) > 0:
    df['Total_Periulcer_Conditions'] = df[periulcer_cols].sum(axis=1)
    print(f"  ✓ Total Peri-ulcer Conditions (sum of {len(periulcer_cols)} indicators)")

# 10. Comorbidity count
comorbid_cols = ['Type of Diabetes', 'Heart Conditions', 'Cancer History',
                 'Sensory Peripheral']
comorbid_cols = [c for c in comorbid_cols if c in df.columns]
if len(comorbid_cols) > 0:
    df['Total_Comorbidities'] = df[comorbid_cols].sum(axis=1)
    print(f"  ✓ Total Comorbidities (sum of {len(comorbid_cols)} conditions)")

# 11. Risk interaction: Smoking × Clinical severity
if 'Smoking' in df.columns and 'Clinical Score' in df.columns:
    df['Smoking_x_ClinicalScore'] = df['Smoking'] * df['Clinical Score']
    print("  ✓ Smoking × Clinical Score")

# 12. Wound severity composite
if 'Wound Score' in df.columns and 'Pain Level' in df.columns:
    df['Wound_Pain_Composite'] = df['Wound Score'] + df['Pain Level']
    print("  ✓ Wound-Pain Composite")

# Extract all numeric features (base + engineered)
all_numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in features_to_drop]

X = df[all_numeric_cols].values
y = df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values

n_base = len(base_feature_cols)
n_engineered = len(all_numeric_cols) - n_base

print(f"\n{'='*80}")
print(f"Feature Summary:")
print(f"  Base features:       {n_base}")
print(f"  Engineered features: {n_engineered}")
print(f"  Total features:      {len(all_numeric_cols)}")
print(f"{'='*80}")

# ============================================================================
# EVALUATION WITH PATIENT-LEVEL CV
# ============================================================================

patients = df['Patient#'].unique()
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\nPatient-level {n_folds}-fold CV:")
print(f"Total patients: {len(patients)}\n")

kappas = []
accs = []
f1_macros = []
f1_per_class_all = []

for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf.split(patients)):
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    X_train = train_df[all_numeric_cols].values
    X_valid = valid_df[all_numeric_cols].values
    y_train = train_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values
    y_valid = valid_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    # Normalization (critical for engineered features with different scales)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_valid_norm = scaler.transform(X_valid_imp)

    # Train RF1 (I vs P+R)
    y_train_bin1 = (y_train > 0).astype(int)
    classes1 = np.array([0, 1])
    weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_bin1)
    rf1 = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=10, max_features='sqrt',
        random_state=42, class_weight={0: weights1[0], 1: weights1[1]}, n_jobs=-1
    )
    rf1.fit(X_train_norm, y_train_bin1)

    # Train RF2 (I+P vs R)
    y_train_bin2 = (y_train > 1).astype(int)
    classes2 = np.array([0, 1])
    weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_bin2)
    rf2 = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=10, max_features='sqrt',
        random_state=42, class_weight={0: weights2[0], 1: weights2[1]}, n_jobs=-1
    )
    rf2.fit(X_train_norm, y_train_bin2)

    # Predict probabilities
    prob1 = rf1.predict_proba(X_valid_norm)[:, 1]  # P(not I)
    prob2 = rf2.predict_proba(X_valid_norm)[:, 1]  # P(R)

    # Convert to 3-class
    prob_I = 1 - prob1
    prob_R = prob2
    prob_P = prob1 * (1 - prob2)
    probs = np.column_stack([prob_I, prob_P, prob_R])
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)

    # Metrics
    kappa = cohen_kappa_score(y_valid, y_pred)
    acc = accuracy_score(y_valid, y_pred)
    f1_macro = f1_score(y_valid, y_pred, average='macro')
    f1_per_class = f1_score(y_valid, y_pred, average=None)

    kappas.append(kappa)
    accs.append(acc)
    f1_macros.append(f1_macro)
    f1_per_class_all.append(f1_per_class)

    print(f"Fold {fold_idx+1}: Kappa={kappa:.4f}, Acc={acc:.4f}, F1={f1_macro:.4f}, F1_per_class={f1_per_class}")

# Aggregate results
f1_per_class_avg = np.mean(f1_per_class_all, axis=0)
f1_per_class_std = np.std(f1_per_class_all, axis=0)

print("\n" + "="*80)
print("FINAL RESULTS - Feature Engineering")
print("="*80)
print(f"Kappa:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
print(f"\nPer-class F1 scores:")
print(f"  Class I: {f1_per_class_avg[0]:.4f} ± {f1_per_class_std[0]:.4f}")
print(f"  Class P: {f1_per_class_avg[1]:.4f} ± {f1_per_class_std[1]:.4f}")
print(f"  Class R: {f1_per_class_avg[2]:.4f} ± {f1_per_class_std[2]:.4f}")
print(f"  Min F1:  {f1_per_class_avg.min():.4f}")
print(f"\nFeature breakdown:")
print(f"  Base features:       {n_base}")
print(f"  Engineered features: {n_engineered}")
print(f"  Total:               {len(all_numeric_cols)}")
print("="*80)

# Feature importance analysis (optional - from last fold)
feature_importances_rf1 = rf1.feature_importances_
feature_importances_rf2 = rf2.feature_importances_
combined_importance = (feature_importances_rf1 + feature_importances_rf2) / 2

# Sort by importance
importance_pairs = sorted(zip(all_numeric_cols, combined_importance),
                         key=lambda x: x[1], reverse=True)

print("\nTop 20 Most Important Features:")
for i, (fname, importance) in enumerate(importance_pairs[:20], 1):
    marker = "🆕" if fname not in base_feature_cols else "  "
    print(f"{i:2d}. {marker} {fname:45s} ({importance:.4f})")
print("\n🆕 = Engineered feature")
print("="*80)
