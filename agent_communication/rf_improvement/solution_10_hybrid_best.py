"""
Solution 10: Hybrid approach combining best elements from all solutions

After running solutions 6-9, manually configure this to combine:
- Best imputation method (from solution 8)
- Best feature selection k (from solution 7)
- Best RF hyperparameters (from solution 6)
- Best decomposition strategy (from solution 9)

INSTRUCTIONS: Run solutions 6-9 first, then update the config below with best settings.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE AFTER RUNNING SOLUTIONS 6-9
# ============================================================================
# Best imputation (from solution 8)
BEST_IMPUTER = KNNImputer(n_neighbors=5)  # Update based on solution 8 results

# Best feature selection k (from solution 7)
BEST_K_FEATURES = 40  # Update based on solution 7 results (or None for all features)

# Best RF hyperparameters (from solution 6)
BEST_RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Best decomposition strategy (from solution 9)
BEST_STRATEGY = 'current'  # Options: 'current', 'reverse', 'sequential', 'ovr'
# ============================================================================

# Load data
df = pd.read_csv('data/processed/best_matching.csv')

exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
X = df.drop(columns=exclude_cols, errors='ignore')
y = df['Healing Phase Abs'].values

print("="*80)
print("SOLUTION 10: Hybrid Best-of-All Approach")
print("="*80)
print(f"Dataset: {len(df)} samples, {X.shape[1]} features")
print(f"\nConfiguration:")
print(f"  Imputer: {BEST_IMPUTER}")
print(f"  Feature selection k: {BEST_K_FEATURES}")
print(f"  RF params: {BEST_RF_PARAMS}")
print(f"  Decomposition: {BEST_STRATEGY}")

# Feature selection preprocessing (if enabled)
if BEST_K_FEATURES is not None and BEST_K_FEATURES < X.shape[1]:
    print(f"\nPerforming feature selection (k={BEST_K_FEATURES})...")
    imputer_temp = BEST_IMPUTER
    X_temp = imputer_temp.fit_transform(X)
    scaler_temp = StandardScaler()
    X_scaled_temp = scaler_temp.fit_transform(X_temp)
    mi_scores = mutual_info_classif(X_scaled_temp, y, random_state=42)
    top_k_indices = np.argsort(mi_scores)[-BEST_K_FEATURES:]
    feature_mask = top_k_indices
    print(f"Selected top {BEST_K_FEATURES} features based on mutual information")
else:
    feature_mask = None
    print("\nUsing all features (no selection)")

patients = df['Patient#'].unique()
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

kappas = []
accs = []
f1_macros = []
f1_per_class_all = []

for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf.split(patients)):
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    X_train = train_df.drop(columns=exclude_cols, errors='ignore')
    X_valid = valid_df.drop(columns=exclude_cols, errors='ignore')
    y_train = train_df['Healing Phase Abs'].values
    y_valid = valid_df['Healing Phase Abs'].values

    # Imputation
    X_train_imp = BEST_IMPUTER.fit_transform(X_train)
    X_valid_imp = BEST_IMPUTER.transform(X_valid)

    # Normalization
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_valid_norm = scaler.transform(X_valid_imp)

    # Feature selection
    if feature_mask is not None:
        X_train_norm = X_train_norm[:, feature_mask]
        X_valid_norm = X_valid_norm[:, feature_mask]

    # Apply decomposition strategy
    if BEST_STRATEGY == 'current':
        # Current: RF1=(I vs P+R), RF2=(I+P vs R)
        y_train_bin1 = (y_train > 0).astype(int)
        y_train_bin2 = (y_train > 1).astype(int)

        classes1 = np.array([0, 1])
        weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_bin1)
        rf1 = RandomForestClassifier(**BEST_RF_PARAMS, class_weight={0: weights1[0], 1: weights1[1]})
        rf1.fit(X_train_norm, y_train_bin1)

        classes2 = np.array([0, 1])
        weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_bin2)
        rf2 = RandomForestClassifier(**BEST_RF_PARAMS, class_weight={0: weights2[0], 1: weights2[1]})
        rf2.fit(X_train_norm, y_train_bin2)

        prob1 = rf1.predict_proba(X_valid_norm)[:, 1]
        prob2 = rf2.predict_proba(X_valid_norm)[:, 1]
        prob_I = 1 - prob1
        prob_R = prob2
        prob_P = prob1 * (1 - prob2)

    # Note: Add other strategies if needed based on solution 9 results

    # Normalize and predict
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

    print(f"\nFold {fold_idx+1}:")
    print(f"  Kappa:    {kappa:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 per class (I/P/R): {f1_per_class}")

# Aggregate results
f1_per_class_avg = np.mean(f1_per_class_all, axis=0)
f1_per_class_std = np.std(f1_per_class_all, axis=0)

print("\n" + "="*80)
print("FINAL RESULTS - Hybrid Best-of-All")
print("="*80)
print(f"Kappa:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
print(f"\nPer-class F1 scores:")
print(f"  Class I: {f1_per_class_avg[0]:.4f} ± {f1_per_class_std[0]:.4f}")
print(f"  Class P: {f1_per_class_avg[1]:.4f} ± {f1_per_class_std[1]:.4f}")
print(f"  Class R: {f1_per_class_avg[2]:.4f} ± {f1_per_class_std[2]:.4f}")
print(f"  Min F1:  {f1_per_class_avg.min():.4f}")
print("="*80)
