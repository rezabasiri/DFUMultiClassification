"""
Solution 6: Bayesian Hyperparameter Optimization for RF1 and RF2 separately

Systematically search for optimal hyperparameters using Bayesian optimization.
Optimizes both RF classifiers independently with different search spaces.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from src.utils.config import get_data_paths, get_project_paths
import warnings
warnings.filterwarnings('ignore')

# Load data
_, _, root = get_project_paths()
df = pd.read_csv(get_data_paths(root)['csv_file'])

# Map labels
df['label'] = df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2})

# Create binary labels for ordinal decomposition
y_bin1 = (df['label'] > 0).astype(int)  # I vs (P+R)
y_bin2 = (df['label'] > 1).astype(int)  # (I+P) vs R

# Extract metadata features (73 features)
exclude_cols = ['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs', 'label',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
X = df.drop(columns=exclude_cols, errors='ignore')

print("="*80)
print("SOLUTION 6: Bayesian Hyperparameter Optimization (RF1 and RF2 separately)")
print("="*80)
print(f"Dataset: {len(df)} samples, {X.shape[1]} features")
print(f"RF1 (I vs P+R) distribution: {y_bin1.value_counts().to_dict()}")
print(f"RF2 (I+P vs R) distribution: {y_bin2.value_counts().to_dict()}")

# Define search spaces for both RFs
search_space_rf1 = {
    'n_estimators': Integer(200, 1000),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(5, 30),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'max_samples': Real(0.5, 1.0),
}

search_space_rf2 = {
    'n_estimators': Integer(200, 1000),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(5, 30),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'max_samples': Real(0.5, 1.0),
}

# Patient-level CV setup
patients = df['Patient#'].unique()
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\nPatient-level {n_folds}-fold CV:")
print(f"Total patients: {len(patients)}")

# Function to compute kappa (for optimization)
def kappa_scorer(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

kappa_score = make_scorer(kappa_scorer)

# Step 1: Optimize RF1
print("\n" + "="*80)
print("Step 1: Optimizing RF1 (I vs P+R)")
print("="*80)

rf1_scores = []
for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf.split(patients)):
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    feature_cols_cv = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols and c != 'label']
    X_train = train_df[feature_cols_cv]
    X_valid = valid_df[feature_cols_cv]
    y_train_label = train_df['label'].values
    y_valid_label = valid_df['label'].values
    y_train = (y_train_label > 0).astype(int)
    y_valid = (y_valid_label > 0).astype(int)

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    # Normalization
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_valid_norm = scaler.transform(X_valid_imp)

    # Class weights
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    # Bayesian optimization (10 iterations for speed)
    if fold_idx == 0:  # Only optimize on first fold
        print(f"\nRunning Bayesian optimization on fold 1 (10 iterations)...")
        bayes_search = BayesSearchCV(
            RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_jobs=-1),
            search_space_rf1,
            n_iter=10,
            cv=3,
            scoring=kappa_score,
            n_jobs=1,
            random_state=42,
            verbose=0
        )
        bayes_search.fit(X_train_norm, y_train)
        best_params_rf1 = bayes_search.best_params_
        print(f"Best RF1 params: {best_params_rf1}")
        print(f"Best CV kappa: {bayes_search.best_score_:.4f}")

    # Train with best params and evaluate
    rf1 = RandomForestClassifier(**best_params_rf1, random_state=42, class_weight=class_weight_dict, n_jobs=-1)
    rf1.fit(X_train_norm, y_train)
    pred = rf1.predict(X_valid_norm)
    kappa = cohen_kappa_score(y_valid, pred)
    acc = accuracy_score(y_valid, pred)

    rf1_scores.append(kappa)
    print(f"Fold {fold_idx+1}: Kappa={kappa:.4f}, Acc={acc:.4f}")

print(f"\nRF1 Average Kappa: {np.mean(rf1_scores):.4f} ± {np.std(rf1_scores):.4f}")

# Step 2: Optimize RF2
print("\n" + "="*80)
print("Step 2: Optimizing RF2 (I+P vs R)")
print("="*80)

rf2_scores = []
for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf.split(patients)):
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    feature_cols_cv = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols and c != 'label']
    X_train = train_df[feature_cols_cv]
    X_valid = valid_df[feature_cols_cv]
    y_train_label = train_df['label'].values
    y_valid_label = valid_df['label'].values
    y_train = (y_train_label > 1).astype(int)
    y_valid = (y_valid_label > 1).astype(int)

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    # Normalization
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_valid_norm = scaler.transform(X_valid_imp)

    # Class weights
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}

    # Bayesian optimization (10 iterations for speed)
    if fold_idx == 0:  # Only optimize on first fold
        print(f"\nRunning Bayesian optimization on fold 1 (10 iterations)...")
        bayes_search = BayesSearchCV(
            RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_jobs=-1),
            search_space_rf2,
            n_iter=10,
            cv=3,
            scoring=kappa_score,
            n_jobs=1,
            random_state=42,
            verbose=0
        )
        bayes_search.fit(X_train_norm, y_train)
        best_params_rf2 = bayes_search.best_params_
        print(f"Best RF2 params: {best_params_rf2}")
        print(f"Best CV kappa: {bayes_search.best_score_:.4f}")

    # Train with best params and evaluate
    rf2 = RandomForestClassifier(**best_params_rf2, random_state=42, class_weight=class_weight_dict, n_jobs=-1)
    rf2.fit(X_train_norm, y_train)
    pred = rf2.predict(X_valid_norm)
    kappa = cohen_kappa_score(y_valid, pred)
    acc = accuracy_score(y_valid, pred)

    rf2_scores.append(kappa)
    print(f"Fold {fold_idx+1}: Kappa={kappa:.4f}, Acc={acc:.4f}")

print(f"\nRF2 Average Kappa: {np.mean(rf2_scores):.4f} ± {np.std(rf2_scores):.4f}")

# Step 3: End-to-end evaluation with optimized RF1 and RF2
print("\n" + "="*80)
print("Step 3: End-to-end 3-class evaluation")
print("="*80)

kappas = []
accs = []
f1_macros = []

for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf.split(patients)):
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    feature_cols_cv = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols and c != 'label']
    X_train = train_df[feature_cols_cv]
    X_valid = valid_df[feature_cols_cv]
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    # Normalization
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_valid_norm = scaler.transform(X_valid_imp)

    # Train RF1
    y_train_bin1 = (y_train > 0).astype(int)
    classes1 = np.array([0, 1])
    weights1 = compute_class_weight('balanced', classes=classes1, y=y_train_bin1)
    class_weight_dict1 = {0: weights1[0], 1: weights1[1]}
    rf1 = RandomForestClassifier(**best_params_rf1, random_state=42, class_weight=class_weight_dict1, n_jobs=-1)
    rf1.fit(X_train_norm, y_train_bin1)

    # Train RF2
    y_train_bin2 = (y_train > 1).astype(int)
    classes2 = np.array([0, 1])
    weights2 = compute_class_weight('balanced', classes=classes2, y=y_train_bin2)
    class_weight_dict2 = {0: weights2[0], 1: weights2[1]}
    rf2 = RandomForestClassifier(**best_params_rf2, random_state=42, class_weight=class_weight_dict2, n_jobs=-1)
    rf2.fit(X_train_norm, y_train_bin2)

    # Predict probabilities
    prob1 = rf1.predict_proba(X_valid_norm)[:, 1]  # P(not I)
    prob2 = rf2.predict_proba(X_valid_norm)[:, 1]  # P(R)

    # Convert to 3-class probabilities
    prob_I = 1 - prob1
    prob_R = prob2
    prob_P = prob1 * (1 - prob2)

    # Predict class
    probs = np.column_stack([prob_I, prob_P, prob_R])
    y_pred = np.argmax(probs, axis=1)

    # Metrics
    kappa = cohen_kappa_score(y_valid, y_pred)
    acc = accuracy_score(y_valid, y_pred)
    f1_macro = f1_score(y_valid, y_pred, average='macro')

    kappas.append(kappa)
    accs.append(acc)
    f1_macros.append(f1_macro)

    print(f"Fold {fold_idx+1}: Kappa={kappa:.4f}, Acc={acc:.4f}, F1_macro={f1_macro:.4f}")

print("\n" + "="*80)
print("FINAL RESULTS - Bayesian Optimized RF")
print("="*80)
print(f"Kappa:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
print("\nOptimized RF1 params:", best_params_rf1)
print("Optimized RF2 params:", best_params_rf2)
print("="*80)
