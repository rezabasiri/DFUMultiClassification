"""
Solution 9: Alternative ordinal decomposition strategies

Current: RF1=(I vs P+R), RF2=(I+P vs R)
Test alternatives:
  - Strategy A: RF1=(I+P vs R), RF2=(I vs P)  [reverse order]
  - Strategy B: RF1=(I vs P), RF2=(P vs R)    [sequential]
  - Strategy C: One-vs-Rest (3 binary classifiers)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed/best_matching.csv')

# Extract metadata features
exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']

print("="*80)
print("SOLUTION 9: Alternative Ordinal Decomposition Strategies")
print("="*80)

patients = df['Patient#'].unique()
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

strategies = {
    'Current (I vs P+R, I+P vs R)': 'current',
    'Strategy A (I+P vs R, I vs P)': 'reverse',
    'Strategy B (I vs P, P vs R)': 'sequential',
    'Strategy C (One-vs-Rest x3)': 'ovr'
}

results = []

for strategy_name, strategy_type in strategies.items():
    print(f"\n{'='*80}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*80}")

    kappas = []
    accs = []
    f1_macros = []

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
        imputer = KNNImputer(n_neighbors=5)
        X_train_imp = imputer.fit_transform(X_train)
        X_valid_imp = imputer.transform(X_valid)

        # Normalization
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_imp)
        X_valid_norm = scaler.transform(X_valid_imp)

        if strategy_type == 'current':
            # Current: RF1=(I vs P+R), RF2=(I+P vs R)
            y_train_bin1 = (y_train > 0).astype(int)
            y_train_bin2 = (y_train > 1).astype(int)

            # Train RF1
            classes1 = np.array([0, 1])
            weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_bin1)
            rf1 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42,
                                         class_weight={0: weights1[0], 1: weights1[1]}, n_jobs=-1)
            rf1.fit(X_train_norm, y_train_bin1)

            # Train RF2
            classes2 = np.array([0, 1])
            weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_bin2)
            rf2 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42,
                                         class_weight={0: weights2[0], 1: weights2[1]}, n_jobs=-1)
            rf2.fit(X_train_norm, y_train_bin2)

            # Predict
            prob1 = rf1.predict_proba(X_valid_norm)[:, 1]  # P(not I)
            prob2 = rf2.predict_proba(X_valid_norm)[:, 1]  # P(R)
            prob_I = 1 - prob1
            prob_R = prob2
            prob_P = prob1 * (1 - prob2)

        elif strategy_type == 'reverse':
            # Strategy A: RF1=(I+P vs R), RF2=(I vs P)
            y_train_bin1 = (y_train > 1).astype(int)  # R vs (I+P)
            y_train_bin2 = (y_train[y_train <= 1] > 0).astype(int)  # I vs P (subset)
            X_train_subset = X_train_norm[y_train <= 1]

            # Train RF1
            classes1 = np.array([0, 1])
            weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_bin1)
            rf1 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42,
                                         class_weight={0: weights1[0], 1: weights1[1]}, n_jobs=-1)
            rf1.fit(X_train_norm, y_train_bin1)

            # Train RF2
            classes2 = np.array([0, 1])
            weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_bin2)
            rf2 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42,
                                         class_weight={0: weights2[0], 1: weights2[1]}, n_jobs=-1)
            rf2.fit(X_train_subset, y_train_bin2)

            # Predict
            prob_R = rf1.predict_proba(X_valid_norm)[:, 1]
            prob_IP = 1 - prob_R
            prob_P_given_IP = rf2.predict_proba(X_valid_norm)[:, 1]
            prob_P = prob_IP * prob_P_given_IP
            prob_I = prob_IP * (1 - prob_P_given_IP)

        elif strategy_type == 'sequential':
            # Strategy B: RF1=(I vs P), RF2=(P vs R)
            mask1 = y_train <= 1  # I or P
            mask2 = y_train >= 1  # P or R

            # Train RF1: I vs P
            X_train_IP = X_train_norm[mask1]
            y_train_IP = (y_train[mask1] > 0).astype(int)
            classes1 = np.unique(y_train_IP)
            weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_IP)
            rf1 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42,
                                         class_weight={classes1[i]: weights1[i] for i in range(len(classes1))}, n_jobs=-1)
            rf1.fit(X_train_IP, y_train_IP)

            # Train RF2: P vs R
            X_train_PR = X_train_norm[mask2]
            y_train_PR = (y_train[mask2] > 1).astype(int)
            classes2 = np.unique(y_train_PR)
            weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_PR)
            rf2 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42,
                                         class_weight={classes2[i]: weights2[i] for i in range(len(classes2))}, n_jobs=-1)
            rf2.fit(X_train_PR, y_train_PR)

            # Predict
            prob_P_given_IP = rf1.predict_proba(X_valid_norm)[:, 1] if rf1.predict_proba(X_valid_norm).shape[1] > 1 else rf1.predict_proba(X_valid_norm)[:, 0]
            prob_R_given_PR = rf2.predict_proba(X_valid_norm)[:, 1] if rf2.predict_proba(X_valid_norm).shape[1] > 1 else rf2.predict_proba(X_valid_norm)[:, 0]

            # Combine (simple heuristic)
            prob_I = (1 - prob_P_given_IP)
            prob_R = prob_R_given_PR
            prob_P = 1 - prob_I - prob_R
            prob_P = np.clip(prob_P, 0, 1)

        elif strategy_type == 'ovr':
            # Strategy C: One-vs-Rest
            rf_I = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42, n_jobs=-1)
            rf_P = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42, n_jobs=-1)
            rf_R = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                         max_features='sqrt', random_state=42, n_jobs=-1)

            y_I = (y_train == 0).astype(int)
            y_P = (y_train == 1).astype(int)
            y_R = (y_train == 2).astype(int)

            rf_I.fit(X_train_norm, y_I)
            rf_P.fit(X_train_norm, y_P)
            rf_R.fit(X_train_norm, y_R)

            prob_I = rf_I.predict_proba(X_valid_norm)[:, 1]
            prob_P = rf_P.predict_proba(X_valid_norm)[:, 1]
            prob_R = rf_R.predict_proba(X_valid_norm)[:, 1]

        # Normalize probabilities
        probs = np.column_stack([prob_I, prob_P, prob_R])
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = np.argmax(probs, axis=1)

        # Metrics
        kappa = cohen_kappa_score(y_valid, y_pred)
        acc = accuracy_score(y_valid, y_pred)
        f1_macro = f1_score(y_valid, y_pred, average='macro')

        kappas.append(kappa)
        accs.append(acc)
        f1_macros.append(f1_macro)

        print(f"Fold {fold_idx+1}: Kappa={kappa:.4f}, Acc={acc:.4f}, F1={f1_macro:.4f}")

    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1_macros)

    results.append({
        'strategy': strategy_name,
        'kappa_mean': mean_kappa,
        'kappa_std': std_kappa,
        'acc_mean': mean_acc,
        'f1_mean': mean_f1
    })

    print(f"\n{strategy_name} Average: Kappa={mean_kappa:.4f}±{std_kappa:.4f}")

# Find best strategy
best_result = max(results, key=lambda x: x['kappa_mean'])

print("\n" + "="*80)
print("FINAL RESULTS - Ordinal Decomposition Comparison")
print("="*80)
print(f"Best strategy: {best_result['strategy']}")
print(f"Kappa:    {best_result['kappa_mean']:.4f} ± {best_result['kappa_std']:.4f}")
print(f"Accuracy: {best_result['acc_mean']:.4f}")
print(f"F1 Macro: {best_result['f1_mean']:.4f}")
print("\nAll results (sorted by Kappa):")
for r in sorted(results, key=lambda x: x['kappa_mean'], reverse=True):
    print(f"  {r['strategy']:35s}: Kappa={r['kappa_mean']:.4f}±{r['kappa_std']:.4f}")
print("="*80)
