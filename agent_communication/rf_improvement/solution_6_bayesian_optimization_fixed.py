"""
Solution 6 (FIXED): Bayesian Hyperparameter Optimization for 3-class performance

CRITICAL FIX: Optimizes END-TO-END 3-class Kappa (not binary classifiers separately).
Previous version optimized RF1 and RF2 independently, which degraded combined performance.
Now optimizes unified hyperparameters that maximize final 3-class Kappa.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
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

# Extract metadata features - exclude data leakage columns
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

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in features_to_drop]
X = df[feature_cols].values
y = df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values

print("="*80)
print("SOLUTION 6: Bayesian Optimization (Fixed - End-to-End 3-class)")
print("="*80)
print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
print(f"Excluded {len(features_to_drop)} columns (Phase Confidence removed - data leakage)")
print(f"Target: Optimize unified RF params for final 3-class Kappa")

# Define unified search space (same params for both RF1 and RF2)
search_space = {
    'n_estimators': Integer(200, 1000),
    'max_depth': Integer(5, 20),  # Lower max to avoid overfitting
    'min_samples_split': Integer(5, 30),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
}

print("\nSearch space:")
for param, space in search_space.items():
    print(f"  {param}: {space}")

# Custom estimator that wraps the ordinal RF pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

class OrdinalRFClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal RF using two binary classifiers: RF1=(I vs P+R), RF2=(I+P vs R)"""

    def __init__(self, n_estimators=500, max_depth=10, min_samples_split=10,
                 min_samples_leaf=1, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Binary labels
        y_bin1 = (y > 0).astype(int)  # I vs (P+R)
        y_bin2 = (y > 1).astype(int)  # (I+P) vs R

        # Class weights
        classes1 = np.array([0, 1])
        weights1 = compute_class_weight('balanced', classes=classes1, y=y_bin1)
        class_weight_dict1 = {0: weights1[0], 1: weights1[1]}

        classes2 = np.array([0, 1])
        weights2 = compute_class_weight('balanced', classes=classes2, y=y_bin2)
        class_weight_dict2 = {0: weights2[0], 1: weights2[1]}

        # Train RF1
        self.rf1_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight=class_weight_dict1,
            n_jobs=-1
        )
        self.rf1_.fit(X, y_bin1)

        # Train RF2
        self.rf2_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight=class_weight_dict2,
            n_jobs=-1
        )
        self.rf2_.fit(X, y_bin2)

        return self

    def predict(self, X):
        # Get probabilities from both RFs
        prob1 = self.rf1_.predict_proba(X)[:, 1]  # P(not I)
        prob2 = self.rf2_.predict_proba(X)[:, 1]  # P(R)

        # Convert to 3-class probabilities
        prob_I = 1 - prob1
        prob_R = prob2
        prob_P = prob1 * (1 - prob2)

        # Normalize and predict
        probs = np.column_stack([prob_I, prob_P, prob_R])
        probs = probs / probs.sum(axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """Return 3-class Kappa score"""
        y_pred = self.predict(X)
        return cohen_kappa_score(y, y_pred)

# Patient-level CV setup
patients = df['Patient#'].unique()
n_folds = 3  # Reduced for speed (Bayesian optimization is expensive)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\nPatient-level {n_folds}-fold CV for Bayesian optimization")
print(f"Total patients: {len(patients)}")

# Prepare data for optimization (use first fold for speed)
print("\n" + "="*80)
print("Running Bayesian Optimization on Fold 1")
print("="*80)

fold_idx = 0
train_patient_idx, valid_patient_idx = list(kf.split(patients))[fold_idx]
train_patients = patients[train_patient_idx]
valid_patients = patients[valid_patient_idx]

train_df = df[df['Patient#'].isin(train_patients)].copy()
valid_df = df[df['Patient#'].isin(valid_patients)].copy()

X_train = train_df[feature_cols].values
X_valid = valid_df[feature_cols].values
y_train = train_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values
y_valid = valid_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values

# Imputation
imputer = KNNImputer(n_neighbors=5)
X_train_imp = imputer.fit_transform(X_train)
X_valid_imp = imputer.transform(X_valid)

# Normalization
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_imp)
X_valid_norm = scaler.transform(X_valid_imp)

# Bayesian optimization (15 iterations)
print("Running Bayesian search (15 iterations, 3 inner CV folds)...")
print("Optimizing for END-TO-END 3-class Kappa\n")

kappa_scorer = make_scorer(cohen_kappa_score)

bayes_search = BayesSearchCV(
    OrdinalRFClassifier(random_state=42),
    search_space,
    n_iter=15,
    cv=3,  # Inner CV
    scoring=kappa_scorer,
    n_jobs=1,  # OrdinalRF already uses n_jobs=-1 internally
    random_state=42,
    verbose=1
)

bayes_search.fit(X_train_norm, y_train)

best_params = bayes_search.best_params_
print(f"\n" + "="*80)
print("Best Parameters Found:")
print("="*80)
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"\nBest inner CV Kappa: {bayes_search.best_score_:.4f}")

# Evaluate best params on validation set
best_model = bayes_search.best_estimator_
y_pred_valid = best_model.predict(X_valid_norm)
kappa_valid = cohen_kappa_score(y_valid, y_pred_valid)
acc_valid = accuracy_score(y_valid, y_pred_valid)
f1_valid = f1_score(y_valid, y_pred_valid, average='macro')

print(f"\nValidation Set Performance:")
print(f"  Kappa:    {kappa_valid:.4f}")
print(f"  Accuracy: {acc_valid:.4f}")
print(f"  F1 Macro: {f1_valid:.4f}")

# Now evaluate across all folds with best params
print("\n" + "="*80)
print("Evaluating Best Params Across All 5 Folds")
print("="*80)

kf_full = KFold(n_splits=5, shuffle=True, random_state=42)
kappas = []
accs = []
f1_macros = []

for fold_idx, (train_patient_idx, valid_patient_idx) in enumerate(kf_full.split(patients)):
    train_patients = patients[train_patient_idx]
    valid_patients = patients[valid_patient_idx]

    train_df = df[df['Patient#'].isin(train_patients)].copy()
    valid_df = df[df['Patient#'].isin(valid_patients)].copy()

    X_train = train_df[feature_cols].values
    X_valid = valid_df[feature_cols].values
    y_train = train_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values
    y_valid = valid_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    # Normalization
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    X_valid_norm = scaler.transform(X_valid_imp)

    # Train with best params
    model = OrdinalRFClassifier(**best_params, random_state=42)
    model.fit(X_train_norm, y_train)

    # Predict
    y_pred = model.predict(X_valid_norm)

    # Metrics
    kappa = cohen_kappa_score(y_valid, y_pred)
    acc = accuracy_score(y_valid, y_pred)
    f1_macro = f1_score(y_valid, y_pred, average='macro')

    kappas.append(kappa)
    accs.append(acc)
    f1_macros.append(f1_macro)

    print(f"Fold {fold_idx+1}: Kappa={kappa:.4f}, Acc={acc:.4f}, F1={f1_macro:.4f}")

print("\n" + "="*80)
print("FINAL RESULTS - Bayesian Optimized (End-to-End 3-class)")
print("="*80)
print(f"Kappa:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
print("\nOptimized Parameters (unified for RF1 and RF2):")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print("\nNote: These params optimize COMBINED 3-class performance, not binary tasks separately")
print("="*80)
