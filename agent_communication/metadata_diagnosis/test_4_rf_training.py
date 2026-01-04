"""Test 4: RF Ordinal Classifier Training (CRITICAL TEST)"""
import sys
sys.path.insert(0, '/home/user/DFUMultiClassification')

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
from src.utils.config import get_data_paths, get_project_paths

print("="*60)
print("TEST 4: RF TRAINING (ORDINAL APPROACH)")
print("="*60)

# Load data
_, _, root = get_project_paths()
data_paths = get_data_paths(root)
df = pd.read_csv(data_paths['metadata']).copy()

# Simple 80/20 split
train_df = df.sample(frac=0.8, random_state=42).copy()
valid_df = df.drop(train_df.index).copy()

print(f"\n✓ Data split: {len(train_df)} train, {len(valid_df)} valid")

# Labels
y_train = train_df['Healing Phase Abs'].values
y_valid = valid_df['Healing Phase Abs'].values

print(f"\n✓ Train label distribution: {np.bincount(y_train)}")
print(f"✓ Valid label distribution: {np.bincount(y_valid)}")

# Calculate class weights on unique cases
unique_cases = train_df[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates()
unique_cases['label_bin1'] = (unique_cases['Healing Phase Abs'] > 0).astype(int)  # I vs (P+R)
unique_cases['label_bin2'] = (unique_cases['Healing Phase Abs'] > 1).astype(int)  # (I+P) vs R

weights_bin1 = compute_class_weight('balanced', classes=np.array([0, 1]), y=unique_cases['label_bin1'])
weights_bin2 = compute_class_weight('balanced', classes=np.array([0, 1]), y=unique_cases['label_bin2'])
class_weight_dict_binary1 = dict(zip([0, 1], weights_bin1))
class_weight_dict_binary2 = dict(zip([0, 1], weights_bin2))

print(f"\n✓ Class weights binary1 (I vs P+R): {class_weight_dict_binary1}")
print(f"✓ Class weights binary2 (I+P vs R): {class_weight_dict_binary2}")

# Prepare features (minimal preprocessing for speed)
exclude = ['Healing Phase Abs', 'Patient#', 'Appt#', 'DFU#']
feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]

X_train = train_df[feature_cols].values
X_valid = valid_df[feature_cols].values

# Quick imputation + normalization
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

print(f"\n✓ Feature matrix shape: train={X_train.shape}, valid={X_valid.shape}")

# Create binary labels
y_train_bin1 = (y_train > 0).astype(int)
y_train_bin2 = (y_train > 1).astype(int)
y_valid_bin1 = (y_valid > 0).astype(int)
y_valid_bin2 = (y_valid > 1).astype(int)

print(f"\n✓ Binary1 train: {np.bincount(y_train_bin1)} (I vs P+R)")
print(f"✓ Binary2 train: {np.bincount(y_train_bin2)} (I+P vs R)")

# Train RF classifiers (sklearn for simplicity)
from sklearn.ensemble import RandomForestClassifier

print(f"\n⏳ Training RF1 (I vs P+R)...")
rf1 = RandomForestClassifier(n_estimators=100, random_state=42,
                             class_weight=class_weight_dict_binary1, n_jobs=-1)
rf1.fit(X_train, y_train_bin1)
pred_bin1_train = rf1.predict(X_train)
pred_bin1_valid = rf1.predict(X_valid)
acc1_train = accuracy_score(y_train_bin1, pred_bin1_train)
acc1_valid = accuracy_score(y_valid_bin1, pred_bin1_valid)
print(f"✓ RF1 accuracy: train={acc1_train:.3f}, valid={acc1_valid:.3f}")

print(f"\n⏳ Training RF2 (I+P vs R)...")
rf2 = RandomForestClassifier(n_estimators=100, random_state=42,
                             class_weight=class_weight_dict_binary2, n_jobs=-1)
rf2.fit(X_train, y_train_bin2)
pred_bin2_train = rf2.predict(X_train)
pred_bin2_valid = rf2.predict(X_valid)
acc2_train = accuracy_score(y_train_bin2, pred_bin2_train)
acc2_valid = accuracy_score(y_valid_bin2, pred_bin2_valid)
print(f"✓ RF2 accuracy: train={acc2_train:.3f}, valid={acc2_valid:.3f}")

# Calculate 3-class probabilities
prob1_train = rf1.predict_proba(X_train)[:, 1]
prob2_train = rf2.predict_proba(X_train)[:, 1]
prob1_valid = rf1.predict_proba(X_valid)[:, 1]
prob2_valid = rf2.predict_proba(X_valid)[:, 1]

# Convert to 3-class probabilities (ordinal approach)
prob_I_train = 1 - prob1_train
prob_P_train = prob1_train * (1 - prob2_train)
prob_R_train = prob2_train

prob_I_valid = 1 - prob1_valid
prob_P_valid = prob1_valid * (1 - prob2_valid)
prob_R_valid = prob2_valid

# Get final predictions
final_pred_train = np.argmax(np.column_stack([prob_I_train, prob_P_train, prob_R_train]), axis=1)
final_pred_valid = np.argmax(np.column_stack([prob_I_valid, prob_P_valid, prob_R_valid]), axis=1)

# Evaluate
print(f"\n{'='*60}")
print("VALIDATION RESULTS:")
print("="*60)
print(classification_report(y_valid, final_pred_valid, target_names=['I', 'P', 'R']))
kappa = cohen_kappa_score(y_valid, final_pred_valid)
acc = accuracy_score(y_valid, final_pred_valid)
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Accuracy: {acc:.4f}")

# Check for failure
if acc < 0.20:
    print(f"\n⚠️  CRITICAL FAIL: Accuracy {acc:.1%} < 20%!")
    sys.exit(1)

if kappa < 0:
    print(f"\n⚠️  CRITICAL FAIL: Negative Kappa {kappa:.4f}!")
    sys.exit(1)

print(f"\n{'='*60}")
print("TEST 4: PASSED ✓")
print(f"Validation accuracy: {acc:.1%}, Kappa: {kappa:.4f}")
print("="*60)
