"""Solution 5: Ensemble of RFs with different seeds"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, f1_score
from src.utils.config import get_data_paths, get_project_paths

print("SOLUTION 5: ENSEMBLE (5 RFs)")
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

unique_cases = train_df[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates()
unique_cases['label_bin1'] = (unique_cases['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}) > 0).astype(int)
unique_cases['label_bin2'] = (unique_cases['Healing Phase Abs'].map({'I':0, 'P':1, 'R':2}) > 1).astype(int)

weights1 = compute_class_weight('balanced', classes=np.array([0,1]), y=unique_cases['label_bin1'])
weights2 = compute_class_weight('balanced', classes=np.array([0,1]), y=unique_cases['label_bin2'])
cw1, cw2 = dict(zip([0,1], weights1)), dict(zip([0,1], weights2))

y_bin1_train = (y_train > 0).astype(int)
y_bin2_train = (y_train > 1).astype(int)

# Train 5 RFs with different seeds, average predictions
seeds = [42, 123, 456, 789, 1011]
all_probs_I = []
all_probs_P = []
all_probs_R = []

for seed in seeds:
    rf1 = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight=cw1, n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight=cw2, n_jobs=-1)
    rf1.fit(X_train, y_bin1_train)
    rf2.fit(X_train, y_bin2_train)

    prob1 = rf1.predict_proba(X_valid)[:, 1]
    prob2 = rf2.predict_proba(X_valid)[:, 1]

    all_probs_I.append(1 - prob1)
    all_probs_P.append(prob1 * (1 - prob2))
    all_probs_R.append(prob2)

# Average probabilities from all models
prob_I = np.mean(all_probs_I, axis=0)
prob_P = np.mean(all_probs_P, axis=0)
prob_R = np.mean(all_probs_R, axis=0)

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
    'method': 'ensemble',
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
