"""
Solution 8: Compare different imputation strategies

Tests: KNN, Median, Mean, and Iterative imputation.
Current implementation uses KNN(k=5), but other methods might work better.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
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
X = df.drop(columns=exclude_cols, errors='ignore')
y = df['Healing Phase Abs'].values

print("="*80)
print("SOLUTION 8: Imputation Strategy Comparison")
print("="*80)
print(f"Dataset: {len(df)} samples, {X.shape[1]} features")

# Check missing values
missing_count = X.isnull().sum().sum()
missing_pct = 100 * missing_count / (X.shape[0] * X.shape[1])
print(f"Missing values: {missing_count} ({missing_pct:.2f}%)")

# Define imputation strategies
imputers = {
    'KNN_k5': KNNImputer(n_neighbors=5),
    'KNN_k3': KNNImputer(n_neighbors=3),
    'KNN_k10': KNNImputer(n_neighbors=10),
    'Median': SimpleImputer(strategy='median'),
    'Mean': SimpleImputer(strategy='mean'),
    'Iterative': IterativeImputer(random_state=42, max_iter=10)
}

patients = df['Patient#'].unique()
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

results = []

for imputer_name, imputer_obj in imputers.items():
    print(f"\n{'='*80}")
    print(f"Testing: {imputer_name}")
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

        # Imputation with current strategy
        X_train_imp = imputer_obj.fit_transform(X_train)
        X_valid_imp = imputer_obj.transform(X_valid)

        # Normalization
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_imp)
        X_valid_norm = scaler.transform(X_valid_imp)

        # Train RF1
        y_train_bin1 = (y_train > 0).astype(int)
        classes1 = np.array([0, 1])
        weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_bin1)
        class_weight_dict1 = {0: weights1[0], 1: weights1[1]}
        rf1 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                     max_features='sqrt', random_state=42,
                                     class_weight=class_weight_dict1, n_jobs=-1)
        rf1.fit(X_train_norm, y_train_bin1)

        # Train RF2
        y_train_bin2 = (y_train > 1).astype(int)
        classes2 = np.array([0, 1])
        weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_bin2)
        class_weight_dict2 = {0: weights2[0], 1: weights2[1]}
        rf2 = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10,
                                     max_features='sqrt', random_state=42,
                                     class_weight=class_weight_dict2, n_jobs=-1)
        rf2.fit(X_train_norm, y_train_bin2)

        # Predict probabilities
        prob1 = rf1.predict_proba(X_valid_norm)[:, 1]
        prob2 = rf2.predict_proba(X_valid_norm)[:, 1]

        # Convert to 3-class
        prob_I = 1 - prob1
        prob_R = prob2
        prob_P = prob1 * (1 - prob2)
        probs = np.column_stack([prob_I, prob_P, prob_R])
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
        'method': imputer_name,
        'kappa_mean': mean_kappa,
        'kappa_std': std_kappa,
        'acc_mean': mean_acc,
        'f1_mean': mean_f1
    })

    print(f"\n{imputer_name} Average: Kappa={mean_kappa:.4f}±{std_kappa:.4f}, Acc={mean_acc:.4f}, F1={mean_f1:.4f}")

# Find best method
best_result = max(results, key=lambda x: x['kappa_mean'])

print("\n" + "="*80)
print("FINAL RESULTS - Imputation Strategy Comparison")
print("="*80)
print(f"Best method: {best_result['method']}")
print(f"Kappa:    {best_result['kappa_mean']:.4f} ± {best_result['kappa_std']:.4f}")
print(f"Accuracy: {best_result['acc_mean']:.4f}")
print(f"F1 Macro: {best_result['f1_mean']:.4f}")
print("\nAll results (sorted by Kappa):")
for r in sorted(results, key=lambda x: x['kappa_mean'], reverse=True):
    print(f"  {r['method']:12s}: Kappa={r['kappa_mean']:.4f}±{r['kappa_std']:.4f}, Acc={r['acc_mean']:.4f}")
print("="*80)
