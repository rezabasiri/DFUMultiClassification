"""Diagnostic: Compare LOO baseline Kappa with random vs patient-grouped OOF folds.

Tests whether the 0.26 LOO Kappa is real signal or patient memorization.
"""
import sys
sys.path.insert(0, '/workspace/DFUMultiClassification')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('results/best_matching.csv')
data['Healing Phase Abs'] = data['Healing Phase Abs'].astype(str).map({'I': 0, 'P': 1, 'R': 2})

# --- Preprocess (same as _preprocess_for_loo) ---
df = data.copy()
image_cols = ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
              'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
              'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']
df = df.drop(columns=[c for c in image_cols if c in df.columns])

df['BMI'] = df['Weight (Kg)'] / ((df['Height (cm)'] / 100) ** 2)
df['Age above 60'] = (df['Age'] > 60).astype(int)
df['Age Bin'] = pd.cut(df['Age'], bins=range(0, int(df['Age'].max()) + 20, 20),
                       right=False, labels=range(len(range(0, int(df['Age'].max()) + 20, 20)) - 1))
df['Weight Bin'] = pd.cut(df['Weight (Kg)'], bins=range(0, int(df['Weight (Kg)'].max()) + 20, 20),
                          right=False, labels=range(len(range(0, int(df['Weight (Kg)'].max()) + 20, 20)) - 1))
df['Height Bin'] = pd.cut(df['Height (cm)'], bins=range(0, int(df['Height (cm)'].max()) + 10, 10),
                          right=False, labels=range(len(range(0, int(df['Height (cm)'].max()) + 10, 10)) - 1))

_FIXED_CATEGORIES = {
    'Sex (F:0, M:1)': ['F', 'M'],
    'Side (Left:0, Right:1)': ['Left', 'Right'],
    'Foot Aspect': ['Dorsal', 'Lateral', 'Medial', 'Plantar'],
    'Odor': ['NoOdor', 'Unpleasant'],
    'Type of Pain Grouped': ['ChronicPain', 'GeneralAches', 'LocalizedPain', 'NoPain',
                              'PhantomAndUnusualSensations', 'PressureAndMovement',
                              'SharpAndIntensePain', 'ShootingPain', 'ThrobbingPain'],
}
for col, cats in _FIXED_CATEGORIES.items():
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=cats).codes

categorical_mappings = {
    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {'ankle': 4, 'Heel': 3, 'middle': 2, 'toes': 1, 'Hallux': 0},
    'Dressing Grouped': {'NoDressing': 0, 'BandAid': 1, 'BasicDressing': 1, 'AbsorbantDressing': 2, 'Antiseptic': 3, 'AdvanceMethod': 4, 'other': 4},
    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {'Serous': 0, 'Haemoserous': 1, 'Bloody': 2, 'Thick': 3}
}
for col, mapping in categorical_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

features_to_drop = [
    'ID', 'Location', 'Healing Phase', 'Phase Confidence (%)', 'DFU#', 'Appt#',
    'Appt Days', 'Type of Pain2', 'Type of Pain_Grouped2', 'Type of Pain',
    'Dressing',
    'No Offloading', 'Offloading: Therapeutic Footwear',
    'Offloading: Scotcast Boot or RCW', 'Offloading: Half Shoes or Sandals',
    'Offloading: Total Contact Cast', 'Offloading: Crutches, Walkers or Wheelchairs',
    'Offloading Score'
]
df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

# Keep Patient# for grouping before dropping
patients = df['Patient#'].values
y = df['Healing Phase Abs'].values
X = df.drop(['Patient#', 'Healing Phase Abs'], axis=1, errors='ignore')
X = X.select_dtypes(include=[np.number])

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# --- Deduplicate ---
X_str = X.astype(str).apply(lambda row: '||'.join(row), axis=1)
unique_mask = ~X_str.duplicated(keep='first')
unique_indices = np.where(unique_mask)[0]

X_unique = X.iloc[unique_indices].values
y_unique = y[unique_indices]
patients_unique = patients[unique_indices]

print(f"Total rows: {len(X)}, Unique patterns: {len(X_unique)}")
print(f"Unique patients: {len(np.unique(patients_unique))}")
print(f"Class distribution: I={sum(y_unique==0)}, P={sum(y_unique==1)}, R={sum(y_unique==2)}")
print()

y_bin1 = (y_unique > 0).astype(int)
y_bin2 = (y_unique > 1).astype(int)


def compute_oof_kappa(X, y_b1, y_b2, y_3c, splitter, groups=None, n_est=300):
    """Compute OOF Kappa with a given splitter."""
    prob1 = np.zeros(len(X))
    prob2 = np.zeros(len(X))

    split_args = (X,) if groups is None else (X, None, groups)

    for tr, val in splitter.split(*split_args):
        rf1 = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                     class_weight='balanced', n_jobs=-1)
        rf2 = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                     class_weight='balanced', n_jobs=-1)
        rf1.fit(X[tr], y_b1[tr])
        rf2.fit(X[tr], y_b2[tr])
        prob1[val] = rf1.predict_proba(X[val])[:, 1]
        prob2[val] = rf2.predict_proba(X[val])[:, 1]

    p_I = 1 - prob1
    p_P = prob1 * (1 - prob2)
    p_R = prob2
    preds = np.argmax(np.column_stack([p_I, p_P, p_R]), axis=1)
    return cohen_kappa_score(y_3c, preds)


# --- Test 1: Random KFold (what LOO currently uses) ---
print("=" * 60)
print("Test 1: Random KFold OOF (same as LOO baseline)")
print("=" * 60)
for n_folds in [3, 5]:
    for n_est in [50, 300]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        kappa = compute_oof_kappa(X_unique, y_bin1, y_bin2, y_unique, kf, n_est=n_est)
        print(f"  {n_folds}-fold, {n_est} trees: Kappa = {kappa:.4f}")

# --- Test 2: Patient-Grouped KFold (no patient leakage) ---
print()
print("=" * 60)
print("Test 2: Patient-Grouped KFold OOF (no patient leakage)")
print("=" * 60)
for n_folds in [3, 5]:
    for n_est in [50, 300]:
        gkf = GroupKFold(n_splits=n_folds)
        kappa = compute_oof_kappa(X_unique, y_bin1, y_bin2, y_unique, gkf,
                                  groups=patients_unique, n_est=n_est)
        print(f"  {n_folds}-fold, {n_est} trees: Kappa = {kappa:.4f}")

# --- Test 3: True train/val split (mimicking the actual pipeline) ---
print()
print("=" * 60)
print("Test 3: Actual 80/20 patient split (mimicking pipeline)")
print("=" * 60)
np.random.seed(42)
all_patients = np.unique(patients_unique)
np.random.shuffle(all_patients)
n_train = int(len(all_patients) * 0.8)
train_pats = set(all_patients[:n_train])
val_pats = set(all_patients[n_train:])

train_mask = np.array([p in train_pats for p in patients_unique])
val_mask = ~train_mask

X_train, X_val = X_unique[train_mask], X_unique[val_mask]
y_train, y_val = y_unique[train_mask], y_unique[val_mask]
y_b1_train = (y_train > 0).astype(int)
y_b2_train = (y_train > 1).astype(int)

print(f"  Train: {train_mask.sum()} patterns ({len(train_pats)} patients)")
print(f"  Val: {val_mask.sum()} patterns ({len(val_pats)} patients)")

for n_est in [50, 300]:
    rf1 = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf1.fit(X_train, y_b1_train)
    rf2.fit(X_train, y_b2_train)

    prob1 = rf1.predict_proba(X_val)[:, 1]
    prob2 = rf2.predict_proba(X_val)[:, 1]
    p_I = 1 - prob1
    p_P = prob1 * (1 - prob2)
    p_R = prob2
    preds = np.argmax(np.column_stack([p_I, p_P, p_R]), axis=1)
    kappa = cohen_kappa_score(y_val, preds)
    print(f"  {n_est} trees: Val Kappa = {kappa:.4f}")

# --- Test 4: Same as Test 3 but with MI feature selection (top 40) ---
print()
print("=" * 60)
print("Test 4: 80/20 patient split + MI feature selection (top 40)")
print("=" * 60)
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_unique[train_mask], y_unique[train_mask], random_state=42)
top_40 = np.argsort(mi_scores)[-40:]
feature_names = X.columns.tolist()
print(f"  Top 5 MI features: {[feature_names[i] for i in np.argsort(mi_scores)[-5:][::-1]]}")

X_train_sel = X_train[:, top_40]
X_val_sel = X_val[:, top_40]

for n_est in [50, 300]:
    rf1 = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf1.fit(X_train_sel, y_b1_train)
    rf2.fit(X_train_sel, y_b2_train)

    prob1 = rf1.predict_proba(X_val_sel)[:, 1]
    prob2 = rf2.predict_proba(X_val_sel)[:, 1]
    p_I = 1 - prob1
    p_P = prob1 * (1 - prob2)
    p_R = prob2
    preds = np.argmax(np.column_stack([p_I, p_P, p_R]), axis=1)
    kappa = cohen_kappa_score(y_val, preds)
    print(f"  {n_est} trees, 40 features: Val Kappa = {kappa:.4f}")

# --- Test 5: Direct 3-class RF (no ordinal decomposition) ---
print()
print("=" * 60)
print("Test 5: Direct 3-class RF (no ordinal decomposition)")
print("=" * 60)
for n_est in [50, 300]:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    kappa = cohen_kappa_score(y_val, preds)
    print(f"  {n_est} trees, all features: Val Kappa = {kappa:.4f}")

    # With feature selection
    rf_sel = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                    class_weight='balanced', n_jobs=-1)
    rf_sel.fit(X_train_sel, y_train)
    preds_sel = rf_sel.predict(X_val_sel)
    kappa_sel = cohen_kappa_score(y_val, preds_sel)
    print(f"  {n_est} trees, 40 features: Val Kappa = {kappa_sel:.4f}")
