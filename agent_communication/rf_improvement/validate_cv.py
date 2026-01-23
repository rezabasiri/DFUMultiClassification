"""CV Validation: Patient-level 5-fold CV (NO DATA LEAKAGE)"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from src.utils.config import get_data_paths, get_project_paths

print("PATIENT-LEVEL 5-FOLD CV VALIDATION")
print("="*60)

_, _, root = get_project_paths()
df = pd.read_csv(get_data_paths(root)['csv_file'])

# Map labels
df['label'] = df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2})

# Get unique patients
patients = df['Patient#'].unique()
np.random.seed(42)
np.random.shuffle(patients)

# 5-fold patient-level CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf.split(patients)):
    print(f"\nFold {fold_idx+1}/5...")

    # Split patients
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    # Split data by patients (NO LEAKAGE)
    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    print(f"  Train: {len(train_patients)} patients, {len(train_df)} samples")
    print(f"  Valid: {len(valid_patients)} patients, {len(valid_df)} samples")

    # Verify no patient overlap
    assert len(set(train_patients) & set(valid_patients)) == 0, "Patient leakage detected!"

    # Extract labels
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values

    # Extract features (NO VALIDATION DATA IN PREPROCESSING)
    exclude = ['Healing Phase Abs', 'Patient#', 'Appt#', 'DFU#', 'label']
    feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]

    X_train = train_df[feature_cols].values
    X_valid = valid_df[feature_cols].values

    # Imputation: FIT on train ONLY
    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)  # Fit on train
    X_valid = imputer.transform(X_valid)      # Transform valid (NO FIT)

    # Normalization: FIT on train ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Fit on train
    X_valid = scaler.transform(X_valid)       # Transform valid (NO FIT)

    # Class weights: TRAIN DATA ONLY (unique cases)
    unique_cases = train_df[['Patient#', 'Appt#', 'DFU#', 'label']].drop_duplicates()
    unique_cases['bin1'] = (unique_cases['label'] > 0).astype(int)
    unique_cases['bin2'] = (unique_cases['label'] > 1).astype(int)

    w1 = compute_class_weight('balanced', classes=np.array([0,1]), y=unique_cases['bin1'])
    w2 = compute_class_weight('balanced', classes=np.array([0,1]), y=unique_cases['bin2'])
    cw1, cw2 = dict(zip([0,1], w1)), dict(zip([0,1], w2))

    # Binary labels
    y_bin1_train = (y_train > 0).astype(int)
    y_bin2_train = (y_train > 1).astype(int)

    # Tuned RF (TRAIN DATA ONLY)
    rf1 = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
        class_weight=cw1,
        random_state=42,
        n_jobs=-1
    )
    rf2 = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
        class_weight=cw2,
        random_state=42,
        n_jobs=-1
    )

    rf1.fit(X_train, y_bin1_train)
    rf2.fit(X_train, y_bin2_train)

    # Predict on VALIDATION ONLY
    prob1 = rf1.predict_proba(X_valid)[:, 1]
    prob2 = rf2.predict_proba(X_valid)[:, 1]

    prob_I = 1 - prob1
    prob_P = prob1 * (1 - prob2)
    prob_R = prob2

    final_pred = np.argmax(np.column_stack([prob_I, prob_P, prob_R]), axis=1)

    # Metrics
    acc = accuracy_score(y_valid, final_pred)
    kappa = cohen_kappa_score(y_valid, final_pred)
    f1_macro = f1_score(y_valid, final_pred, average='macro')
    f1_per_class = f1_score(y_valid, final_pred, average=None)

    fold_results.append({
        'fold': fold_idx+1,
        'accuracy': acc,
        'kappa': kappa,
        'macro_f1': f1_macro,
        'f1_I': f1_per_class[0],
        'f1_P': f1_per_class[1],
        'f1_R': f1_per_class[2]
    })

    print(f"  Acc: {acc:.3f}, Kappa: {kappa:.3f}, F1: I={f1_per_class[0]:.3f} P={f1_per_class[1]:.3f} R={f1_per_class[2]:.3f}")

# Summary
results_df = pd.DataFrame(fold_results)
print("\n" + "="*60)
print("PATIENT-LEVEL CV RESULTS (5-fold, NO LEAKAGE)")
print("="*60)
print(f"\nAccuracy:  {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
print(f"Kappa:     {results_df['kappa'].mean():.3f} ± {results_df['kappa'].std():.3f}")
print(f"Macro F1:  {results_df['macro_f1'].mean():.3f} ± {results_df['macro_f1'].std():.3f}")
print(f"\nPer-class F1:")
print(f"  I: {results_df['f1_I'].mean():.3f} ± {results_df['f1_I'].std():.3f}")
print(f"  P: {results_df['f1_P'].mean():.3f} ± {results_df['f1_P'].std():.3f}")
print(f"  R: {results_df['f1_R'].mean():.3f} ± {results_df['f1_R'].std():.3f}")

# Save
results_df.to_csv('cv_validation.csv', index=False)
print("\n✓ Saved to cv_validation.csv")

# Validation check
mean_kappa = results_df['kappa'].mean()
if mean_kappa > 0.2:
    print(f"\n✅ VALIDATED: Kappa {mean_kappa:.3f} > 0.2 (target met)")
    print("   RECOMMEND: Implement tuned RF in production")
elif mean_kappa > 0.15:
    print(f"\n⚠️  MARGINAL: Kappa {mean_kappa:.3f} (0.15-0.2)")
    print("   Consider further improvements")
else:
    print(f"\n❌ FAILED: Kappa {mean_kappa:.3f} < 0.15")
    print("   Parameters not robust - reject")
