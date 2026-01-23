"""Solution 2: SMOTE for class balancing"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from src.utils.config import get_data_paths, get_project_paths

print("SOLUTION 2: SMOTE")
print("="*60)

_, _, root = get_project_paths()
df = pd.read_csv(get_data_paths(root)['csv_file'])

train_df = df.sample(frac=0.8, random_state=42).copy()
valid_df = df.drop(train_df.index).copy()

y_train = train_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values
y_valid = valid_df['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}).values

exclude = ['Healing Phase Abs', 'Patient#', 'Appt#', 'DFU#']
feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]
X_train, X_valid = train_df[feature_cols].values, valid_df[feature_cols].values

imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# SMOTE on binary labels
y_bin1_train = (y_train > 0).astype(int)
y_bin2_train = (y_train > 1).astype(int)

smote1 = SMOTE(random_state=42)
smote2 = SMOTE(random_state=42)

X_train_sm1, y_bin1_sm = smote1.fit_resample(X_train, y_bin1_train)
X_train_sm2, y_bin2_sm = smote2.fit_resample(X_train, y_bin2_train)

# Train RFs on SMOTE data
rf1 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf1.fit(X_train_sm1, y_bin1_sm)
rf2.fit(X_train_sm2, y_bin2_sm)

prob1 = rf1.predict_proba(X_valid)[:, 1]
prob2 = rf2.predict_proba(X_valid)[:, 1]

prob_I = 1 - prob1
prob_P = prob1 * (1 - prob2)
prob_R = prob2

final_pred = np.argmax(np.column_stack([prob_I, prob_P, prob_R]), axis=1)

acc = accuracy_score(y_valid, final_pred)
kappa = cohen_kappa_score(y_valid, final_pred)
f1_macro = f1_score(y_valid, final_pred, average='macro')
f1_per_class = f1_score(y_valid, final_pred, average=None)

print(f"\nAccuracy: {acc:.3f}")
print(f"Kappa: {kappa:.3f}")
print(f"Macro F1: {f1_macro:.3f}")
print(f"Per-class F1: I={f1_per_class[0]:.3f}, P={f1_per_class[1]:.3f}, R={f1_per_class[2]:.3f}")
print(classification_report(y_valid, final_pred, target_names=['I', 'P', 'R']))

results = pd.DataFrame([{
    'method': 'smote',
    'accuracy': acc,
    'kappa': kappa,
    'macro_f1': f1_macro,
    'f1_I': f1_per_class[0],
    'f1_P': f1_per_class[1],
    'f1_R': f1_per_class[2]
}])

if os.path.exists('results.csv'):
    results.to_csv('results.csv', mode='a', header=False, index=False)
else:
    results.to_csv('results.csv', index=False)
print("\n✓ Appended to results.csv")
